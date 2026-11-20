"""A string index of SynthTab excerpts, as validation music for the note model.

Its string labels should not be used to score strings: on these excerpts the audio often does not sound like
the labelled string (the pipeline gets 0.59 string accuracy against 0.94 to 0.99 on real direct recordings).

SynthTab (Zang et al., ICASSP 2024, CC BY-NC 4.0, https://synthtab.dev) renders DadaGP tabs string by string with
Ample Sound sampled guitars (acoustic: four guitars played with pick, fingers or thumb; electric: seven guitars, DI,
several pickups), mixed to one mono 44.1 kHz FLAC per track and timbre. Its audio sits on UR Box in one zip of about
30 GB per timbre (acoustic timbres in four alphabetical parts), so only the chosen renders are fetched: each zip's
central directory is read through HTTP range requests and the chosen FLAC members are inflated from their byte
ranges. The labels come from SynthTab's JAMS archive, read in place: per string (1 = high e), notes in ticks of 960
per quarter, rendered at the track's first tempo.

Selection (seeded, written to `selection` and reused when it exists): six-string tracks in standard tuning with one
tempo, whose DadaGP file has no capo (SynthTab's labels and renders ignore it) and whose song Stage 2 did not train on;
`per_timbre` tracks per timbre, no song twice in the whole set; each track gives one excerpt of `excerpt_s` starting
on a note, dense enough, without long rests, and with few notes of `drop_effects`, which are left out of the labels.

Labels move per timbre by the median lag of its excerpts against GuitarSet's, as build_index.py does for a group:
the onset envelope's at first, and the lag to Stage 1's notes once note_lags.py has measured it.

Usage:
  uv run scripts/stage1/strings/build_synthtab_index.py
  uv run scripts/stage1/strings/note_lags.py index_dir=data/string_index_synthtab "datasets=[synthtab]" \
      stage1_checkpoint=models/finetuned/z8_step_0045000.pth
  uv run scripts/stage1/strings/build_synthtab_index.py
"""

import io
import json
import re
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import guitarpro
import hydra
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1 import string_data as sd

TICKS_PER_QUARTER = 960


class RemoteFile(io.RawIOBase):
    """A file behind an HTTP URL that honours Range requests, read in blocks of `block` bytes."""

    def __init__(self, url, block):
        self.link, self.block, self.pos, self.start, self.data = url, block, 0, 0, b''
        self.resolve()

    def resolve(self):
        # Box redirects to a signed URL that expires, so it is fetched again when a request fails.
        r = urllib.request.urlopen(urllib.request.Request(self.link, method='HEAD'), timeout=60)
        self.url, self.size = r.url, int(r.headers['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def fetch(self, first, last):
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={'Range': f'bytes={first}-{last}'})
                return urllib.request.urlopen(req, timeout=120).read()
            except OSError:
                time.sleep(2**attempt)
                self.resolve()
        raise OSError(f'{self.link}: bytes {first}-{last} failed')

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        if not self.start <= self.pos < self.start + len(self.data):
            last = min(self.pos + max(self.block, len(buffer)), self.size) - 1
            self.start, self.data = self.pos, self.fetch(self.pos, last)
        chunk = self.data[self.pos - self.start : self.pos - self.start + len(buffer)]
        buffer[: len(chunk)] = chunk
        self.pos += len(chunk)
        return len(chunk)


def remote_zip(cfg, zip_id):
    return zipfile.ZipFile(io.BufferedReader(RemoteFile(cfg.box_url.format(id=zip_id), cfg.block_bytes)))


def listing(cfg):
    """Every rendered FLAC of the timbres' zips: timbre, zip, member, folder (the track)."""
    path = REPO_ROOT / cfg.synthtab_dir / 'listing.csv'
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for timbre, t in cfg.timbres.items():
        for zip_id in t.zips:
            members = [m for m in remote_zip(cfg, zip_id).namelist() if m.endswith('.flac')]
            rows += [{'timbre': timbre, 'zip': zip_id, 'member': m, 'folder': m.split('/')[-2]} for m in members]
            info(f'{timbre} zip {zip_id}: {len(members)} renders')
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def jams_names(z):
    """{track folder: its JAMS member} of SynthTab's label archive."""
    return {n.split('/')[1]: n for n in z.namelist() if n.endswith('.jams') and n.count('/') == 2}


def jams_notes(jam):
    """onset, offset (s), pitch, string, fret, effects of every note, at the track's first tempo."""
    bpm = next(a for a in jam['annotations'] if a['namespace'] == 'tempo')['data'][0]['value']
    sec = 60 / (TICKS_PER_QUARTER * bpm)
    rows = []
    for a in jam['annotations']:
        if a['namespace'] != 'note_tab':
            continue
        s, open_pitch = a['sandbox']['string_index'], a['sandbox']['open_tuning']
        prev = None
        for n in sorted(a['data'], key=lambda n: n['time']):
            v = n['value']
            # SynthTab's renderer sometimes drops a note that repeats the previous one on its string with no gap.
            repeat = (
                prev is not None and prev['value']['fret'] == v['fret'] and prev['time'] + prev['duration'] == n['time']
            )
            rows.append({'onset': n['time'] * sec, 'offset': (n['time'] + n['duration']) * sec,
                         'pitch': open_pitch + v['fret'], 'string': s, 'fret': v['fret'],
                         'effects': sorted(set(v) - {'fret', 'velocity'}), 'repeat': repeat})  # fmt: skip
            prev = n
    return pd.DataFrame(rows).sort_values(['onset', 'string'], ignore_index=True)


def tracks(cfg, z, names):
    """One row per JAMS of the wanted instruments: its tuning, number of strings and tempos, notes and length."""
    path = REPO_ROOT / cfg.synthtab_dir / 'tracks.csv.gz'
    if path.exists():
        return pd.read_csv(path)
    ends = tuple(f' - {i}__midi' for i in cfg.instruments)
    rows = []
    for folder, name in names.items():
        if not folder.endswith(ends):
            continue
        jam = json.loads(z.read(name))
        strings = sorted((a['sandbox']['string_index'], a['sandbox']['open_tuning'], len(a['data']))
                         for a in jam['annotations'] if a['namespace'] == 'note_tab')  # fmt: skip
        tempos = next(a for a in jam['annotations'] if a['namespace'] == 'tempo')['data']
        rows.append({'folder': folder, 'strings': len(strings), 'tuning': ','.join(str(p) for _, p, _ in strings),
                     'tempos': len(tempos), 'bpm': tempos[0]['value'], 'notes': sum(n for *_, n in strings),
                     'length_s': jam['file_metadata']['duration'] * 60 / (TICKS_PER_QUARTER * tempos[0]['value'])})  # fmt: skip
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def dadagp_files(root):
    """{SynthTab's name of a DadaGP file ('.' -> ' - ', so 'A - B.gp4' is 'A - B - gp4'): path under root}."""
    files = {}
    for p in sorted(root.glob('*/*/*.gp[345]')):
        files.setdefault(p.name.replace('.', ' - '), p.relative_to(root))
    return files


def song_key(name):
    """Artist and title, without the '(2)' or '(Acoustic)' that tell DadaGP's versions of one song apart."""
    return re.sub(r'[^a-z0-9]', '', re.sub(r'\([^)]*\)', '', name.lower()))


def window(notes, cfg, rng):
    """A seeded excerpt start (s) among those that pass the density, rest and effect limits; None if none does."""
    onsets = notes['onset'].to_numpy()
    dropped = notes['effects'].map(lambda e: bool(set(e) & set(cfg.drop_effects))).to_numpy()
    length = notes['offset'].max()
    ok = []
    for start in np.unique(onsets) - cfg.preroll_s:
        end = start + cfg.excerpt_s
        if start < 0 or end > length + 1:
            continue
        inside = (onsets >= start) & (onsets < end)
        times = np.r_[np.unique(onsets[inside]), end]
        if (inside.sum() >= cfg.min_notes_per_s * cfg.excerpt_s and np.diff(times).max() <= cfg.max_rest_s
                and dropped[inside].mean() <= cfg.max_dropped_fraction):  # fmt: skip
            ok.append(start)
    return float(rng.choice(ok)) if ok else None


def select(cfg, z, names):
    path = REPO_ROOT / cfg.selection
    if path.exists():
        return pd.read_csv(path)
    tr = tracks(cfg, z, names)
    tr = tr[(tr['strings'] == 6) & (tr['tuning'] == ','.join(str(p) for p in cfg.tuning)) & (tr['tempos'] == 1)]
    tr = tr[tr['length_s'] >= cfg.excerpt_s]
    present = listing(cfg)
    present = present[present['folder'].isin(tr['folder'])]
    root = REPO_ROOT / cfg.dadagp_dir
    files = dadagp_files(root)
    trained = {
        json.loads(line)['filename'] for p in cfg.stage2_train for line in (REPO_ROOT / p).read_text().splitlines()
    }
    rng = np.random.default_rng(cfg.seed)
    orders = {t: iter(rng.permutation(g['folder'].unique())) for t, g in present.groupby('timbre')}
    skipped, checked, used, rows = {}, {}, set(), []

    def check(folder):
        song, label, _ = folder.rsplit('__', 2)
        gp = files.get(song)
        if gp is None:
            return 'no DadaGP file'
        if f'{Path(str(gp).replace("/", "_").replace(" ", "_")).stem}.json' in trained:
            return 'song in Stage 2 training'
        try:
            capo = guitarpro.parse(str(root / gp)).tracks[int(label.split(' - ')[0]) - 1].offset
        except Exception:
            return 'DadaGP file unreadable'
        if capo:
            return 'capo'
        start = window(jams_notes(json.loads(z.read(names[folder]))), cfg, rng)
        return 'no excerpt passes' if start is None else {'song': song_key(gp.stem), 'gp': str(gp), 'start_s': start}

    for _ in range(cfg.per_timbre):
        for timbre in cfg.timbres:
            for folder in orders[timbre]:
                if folder not in checked:
                    checked[folder] = check(folder)
                c = checked[folder]
                if isinstance(c, str):
                    skipped[c] = skipped.get(c, 0) + 1
                    continue
                if c['song'] in used:
                    continue
                used.add(c['song'])
                member = present[(present['timbre'] == timbre) & (present['folder'] == folder)].iloc[0]
                rows.append({'rec': f'synthtab/{timbre}/{sd.slug(folder.rsplit("__", 1)[0])}', 'timbre': timbre,
                             'folder': folder, 'zip': member['zip'], 'member': member['member'], 'song': c['song'],
                             'gp': c['gp'], 'start_s': round(c['start_s'], 4)})  # fmt: skip
                break
    info(f'{len(rows)} excerpts chosen; tracks passed over: {skipped}')
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def download(cfg, chosen):
    """Each chosen render's full FLAC under <synthtab_dir>/audio/<timbre>/, fetched once."""
    paths = []
    for zip_id, group in chosen.groupby('zip', sort=False):
        z = None
        for r in group.itertuples():
            path = REPO_ROOT / cfg.synthtab_dir / 'audio' / r.timbre / f'{r.folder}.flac'
            paths.append((r.Index, path))
            if path.exists():
                continue
            z = z or remote_zip(cfg, zip_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix('.part')
            part.write_bytes(z.read(r.member))
            part.rename(path)
            info(f'fetched {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1e6:.1f} MB)')
    return pd.Series(dict(paths)).reindex(chosen.index)


def measure(job):
    audio, onsets = job
    y, _ = librosa.load(REPO_ROOT / audio, sr=sd.ALIGN_SR, mono=True)
    lag, dist = sd.alignment(y, onsets)
    ok = ~np.isnan(dist)
    return lag * 1000, float(ok.mean()), float(np.median(np.abs(dist[ok])) * 1000) if ok.any() else np.nan


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='synthtab_index')
def main(cfg):
    z = zipfile.ZipFile(REPO_ROOT / cfg.jams_zip)
    names = jams_names(z)
    chosen = select(cfg, z, names)
    full = download(cfg, chosen)
    recs, notes, effects = [], [], {}
    for r, src in zip(chosen.itertuples(), full, strict=True):
        n = jams_notes(json.loads(z.read(names[r.folder])))
        end = r.start_s + cfg.excerpt_s
        n = n[(n['onset'] >= r.start_s) & (n['onset'] < end)].copy()
        drop = n['effects'].map(lambda e: bool(set(e) & set(cfg.drop_effects)))
        for e in n['effects']:
            for k in e:
                effects[k] = effects.get(k, 0) + 1
        n = n[~drop]
        n[['onset', 'offset']] -= r.start_s
        n['offset'] = n['offset'].clip(upper=cfg.excerpt_s)
        info_sf = sf.info(src)
        y, sr = sf.read(src, start=round(r.start_s * info_sf.samplerate), stop=round(end * info_sf.samplerate))
        audio = REPO_ROOT / cfg.synthtab_dir / 'excerpts' / f'{r.rec.removeprefix("synthtab/")}.flac'
        audio.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio, y if y.ndim == 1 else y.mean(axis=1), sr, subtype='PCM_24')
        chain = cfg.timbres[r.timbre].chain
        recs.append({'rec': r.rec, 'dataset': 'synthtab', 'audio': sd.rel(audio), 'player': 'synthtab',
                     'guitar': f'synthtab-{r.timbre}', 'chain': chain, 'content': 'pieces', 'split': 'val',
                     'duration': len(y) / sr, 'source': r.folder, 'start_s': r.start_s,
                     'dropped_effect_notes': int(drop.sum())})  # fmt: skip
        notes.append(n.assign(rec=r.rec)[[*sd.NOTE_COLUMNS, 'fret', 'repeat']])
    recs = pd.DataFrame(recs)
    notes = pd.concat(notes, ignore_index=True)
    notes, dropped = sd.clean(recs, notes)

    with ProcessPoolExecutor(cfg.workers) as pool:
        jobs = [(a, notes.loc[notes['rec'] == r, 'onset'].to_numpy()) for r, a in zip(recs['rec'], recs['audio'], strict=True)]  # fmt: skip
        recs[['lag_ms', 'matched', 'residual_ms']] = list(pool.map(measure, jobs))
    bias = json.loads((REPO_ROOT / cfg.bias_report).read_text())['detector_bias_ms']
    off = bias - recs.groupby('guitar')['lag_ms'].transform('median')
    lags_path = REPO_ROOT / cfg.out_dir / 'note_lags.csv'
    by_notes = cfg.note_bias_ms is not None and lags_path.exists()
    if by_notes:
        lag = recs['rec'].map(pd.read_csv(lags_path).set_index('rec')['lag_raw_ms'])
        off = lag.groupby(recs['guitar']).transform('median') - cfg.note_bias_ms
    recs['shift_ms'] = np.where(off.abs() > cfg.shift_threshold_ms, off, 0.0).round(1)
    shift = notes['rec'].map(recs.set_index('rec')['shift_ms']) / 1000
    notes['onset'] += shift
    notes['offset'] += shift
    notes = notes[(notes['onset'] >= 0) & (notes['onset'] < cfg.excerpt_s)].reset_index(drop=True)

    out = REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out / 'recordings.csv', index=False)
    notes.to_csv(out / 'notes.csv.gz', index=False, float_format='%.5f')
    per_timbre = recs.groupby('guitar').agg(chain=('chain', 'first'), excerpts=('rec', 'size'),
                                            lag_ms=('lag_ms', 'median'), shift_ms=('shift_ms', 'first'),
                                            residual_ms=('residual_ms', 'median'))  # fmt: skip
    per_timbre['notes'] = notes.groupby(notes['rec'].map(recs.set_index('rec')['guitar'])).size()
    report = {
        'excerpts': len(recs),
        'minutes': round(recs['duration'].sum() / 60, 1),
        'notes': len(notes),
        'repeat_notes': int(notes['repeat'].sum()),
        'effects_in_excerpts': effects,
        'dropped_effect_notes': int(recs['dropped_effect_notes'].sum()),
        'dropped_by_clean': dropped,
        'shift_source': 'Stage 1 notes (note_lags.csv)' if by_notes else 'onset envelope',
        'detector_bias_ms': bias,
        'note_bias_ms': cfg.note_bias_ms,
        'per_timbre': per_timbre.reset_index().to_dict('records'),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=float))
    info(
        f'{len(recs)} excerpts, {report["minutes"]} min, {len(notes)} notes -> {out}\n{per_timbre.round(1).to_string()}'
    )


if __name__ == '__main__':
    main()

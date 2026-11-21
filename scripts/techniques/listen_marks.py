"""Listening clips for judging whether a tab's technique marks (hammer-on or pull-off, slide, bend, vibrato) were played.

Notes are sampled per group from the technique index: notes carrying one mark, and unmarked notes as controls,
among them ones that follow another fret on their string closely (where an unmarked hammer-on or slide would be).
Each is cut with its surroundings into an MP3 and a spectrogram with the note marked.

Pitch marks are also checked without ears. For every note carrying one of `pitch.marks` and a sample of unmarked
ones, where no other note starts while the note sounds, a pitch track (pYIN, searched from below the tab pitch to a
major third above) gives three figures in semitones: `rise`, how far the pitch gets above the tab pitch, `travel`,
how far it moves during the note, and `wobble`, how far it swings around its own slow course. Prints them per
dataset, player and mark. Writes <id>_di.mp3, <id>.png, clips.json and pitch.csv.

Usage:
  uv run scripts/techniques/listen_marks.py
  uv run scripts/techniques/listen_marks.py --config-name listen_vibrato
"""

import json

import hydra
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from hydra.core.hydra_config import HydraConfig
from scipy.ndimage import median_filter

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.listening import clip, labels_around, spectrogram


def load_notes(cfg):
    """The index's notes of the configured datasets and splits, with their recording's columns, the marks that
    count (`mark`, comma-separated) and, per note, the gap to the previous note on its string and that note's
    fret and marks."""
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv').set_index('rec')
    notes = pd.read_csv(index / 'notes.csv.gz', keep_default_na=False).join(
        recs[['dataset', 'player', 'guitar', 'chain', 'split', 'audio']], on='rec'
    )
    notes = notes[notes['chain'] == 'di'].reset_index(drop=True)
    ignore = set(cfg.ignore_marks)
    notes['mark'] = [','.join(sorted(set(t.split(',')) - ignore - {''})) for t in notes['techniques']]
    on_string = notes.sort_values(['rec', 'string', 'onset'], kind='stable').groupby(['rec', 'string'], sort=False)
    notes['since_previous'] = notes['onset'] - on_string['onset'].shift()
    notes['previous_fret'] = on_string['fret'].shift()
    notes['previous_mark'] = on_string['techniques'].shift().fillna('')
    return notes[notes['dataset'].isin(list(cfg.datasets)) & notes['split'].isin(list(cfg.splits))]


def follows(notes, cfg):
    """Unmarked notes that start soon after a note at another fret of their string, itself picked and held."""
    return (
        (notes['since_previous'] < cfg.follows_s)
        & (notes['previous_fret'] != notes['fret'])
        & ~notes['previous_mark'].str.contains('hammer|slide|dead|tie')
    )


def pick(notes, cfg, rng):
    """[(note id, group)]: each group's notes in random order, at most `max_per_rec` of a recording and none close
    to a note already picked."""
    picked = []
    for g in cfg.groups:
        pool = notes[(notes['dataset'] == g.dataset) & (notes['mark'] == g.mark) & (notes['onset'] >= cfg.before_s)]
        if g.get('follows') is not None:
            pool = pool[follows(pool, cfg) == g.follows]
        if g.get('players') is not None:
            pool = pool[pool['player'].isin(list(g.players))]
        if g.get('recs') is not None:
            pool = pool[pool['rec'].isin(list(g.recs))]
        if g.get('min_s') is not None:
            pool = pool[pool['offset'] - pool['onset'] >= g.min_s]
        taken = {}
        for i, n in pool.sample(frac=1, random_state=int(rng.integers(1 << 31))).iterrows():
            near = [notes.loc[j] for j, _ in picked if notes.at[j, 'rec'] == n.rec]
            if (
                sum(taken.values()) < g.n
                and taken.get(n.rec, 0) < cfg.max_per_rec
                and all(abs(n.onset - m.onset) >= cfg.min_gap_s for m in near)
            ):
                picked.append((i, g))
                taken[n.rec] = taken.get(n.rec, 0) + 1
    return picked


def pitch_figures(y, sr, note, cfg):
    """(rise, travel, wobble) in semitones from the note's pitch track, or NaNs where too little of it is voiced.
    The wobble is the swing left after a running median over `wobble_frames` is taken out."""
    seg = y[int(note.onset * sr) : int(min(note.offset, note.onset + cfg.pitch.max_s) * sr)]
    if len(seg) < cfg.pitch.frame:
        return np.nan, np.nan, np.nan
    f0, voiced, _ = librosa.pyin(
        seg,
        sr=sr,
        fmin=librosa.midi_to_hz(note.pitch - cfg.pitch.below),
        fmax=librosa.midi_to_hz(note.pitch + cfg.pitch.above),
        frame_length=cfg.pitch.frame,
        hop_length=cfg.pitch.hop,
    )
    midi = librosa.hz_to_midi(f0[voiced])
    if len(midi) < cfg.pitch.min_frames:
        return np.nan, np.nan, np.nan
    low, high = np.percentile(midi, [5, 95])
    swing = np.percentile(midi - median_filter(midi, size=cfg.pitch.wobble_frames, mode='nearest'), [5, 95])
    return high - note.pitch, high - low, swing[1] - swing[0]


def pitch_check(notes, picked, cfg, rng):
    """Pitch figures of notes carrying one of `pitch.marks`, a sample of unmarked ones and every picked note, for
    notes long enough and with no other note starting while they sound."""
    long = notes[notes['offset'] - notes['onset'] >= cfg.pitch.min_s]
    plain = [
        g.sample(min(len(g), cfg.pitch.plain_per_dataset), random_state=int(rng.integers(1 << 31)))
        for _, g in long[long['mark'] == ''].groupby('dataset')
    ]
    chosen = pd.concat([long[long['mark'].isin(list(cfg.pitch.marks))], *plain, notes.loc[[i for i, _ in picked]]])
    chosen = chosen[~chosen.index.duplicated()]
    rows = []
    for rec, group in chosen.groupby('rec', sort=False):
        onsets = notes.loc[notes['rec'] == rec, 'onset'].to_numpy()
        y, sr = librosa.load(REPO_ROOT / group['audio'].iloc[0], sr=cfg.pitch.sr, mono=True)
        for i, n in group.iterrows():
            end = min(n.offset, n.onset + cfg.pitch.max_s)
            alone = ((onsets > n.onset - cfg.together_s) & (onsets < end)).sum() == 1
            rise, travel, wobble = pitch_figures(y, sr, n, cfg) if alone else (np.nan, np.nan, np.nan)
            rows.append({'note': i, 'alone': alone, 'rise': rise, 'travel': travel, 'wobble': wobble})
    return chosen.join(pd.DataFrame(rows).set_index('note'))


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='listen_marks')
def main(cfg):
    out = HydraConfig.get().runtime.output_dir
    notes = load_notes(cfg)
    rng = np.random.default_rng(cfg.seed)
    picked = pick(notes, cfg, rng)

    pitch = pitch_check(notes, picked, cfg, rng)
    columns = ['rec', 'onset', 'offset', 'pitch', 'string', 'fret', 'dataset', 'player', 'split', 'mark']
    pitch[[*columns, 'alone', 'rise', 'travel', 'wobble']].to_csv(f'{out}/pitch.csv')
    measured = pitch[pitch['mark'].isin([*cfg.pitch.marks, '']) & pitch['rise'].notna()]
    measured = measured.assign(moves=measured[['rise', 'travel']].max(axis=1))
    table = measured.groupby(['dataset', 'player', 'mark']).agg(
        notes=('moves', 'size'),
        rise=('rise', 'median'),
        travel=('travel', 'median'),
        wobble=('wobble', 'median'),
        **{f'moves >= {t}': ('moves', lambda m, t=t: (m >= t).mean()) for t in cfg.pitch.thresholds},
        **{f'wobble >= {t}': ('wobble', lambda m, t=t: (m >= t).mean()) for t in cfg.pitch.wobble_thresholds},
    )
    marked = pitch[pitch['mark'].isin(list(cfg.pitch.marks))]
    info(
        f'{"/".join(cfg.pitch.marks)}-marked notes: {len(marked)}, measured (alone and voiced): {marked["rise"].notna().sum()}'
    )
    info(f'pitch movement in semitones\n{table.round(2).to_string()}')

    order = rng.permutation(len(picked))
    by_rec = dict(tuple(notes.groupby('rec', sort=False)))
    clips = []
    for number, (i, g) in enumerate((picked[k] for k in order), 1):
        n = notes.loc[i]
        cid = f'm{number:02d}'
        y, sr = librosa.load(REPO_ROOT / n.audio, sr=None, mono=True)
        seg, start = clip(y, sr, n.onset, cfg.before_s, cfg.after_s)
        sf.write(f'{out}/{cid}_di.mp3', seg.astype(np.float32), sr, format='MP3')
        axes = spectrogram(seg, sr, n, start, cfg, f'{out}/{cid}.png')
        figures = pitch.loc[i]
        clips.append({
            'id': cid, 'group': g.name, 'dataset': n.dataset, 'player': n.player, 'guitar': n.guitar, 'rec': n.rec, 'split': n.split,
            'onset': round(float(n.onset), 3), 'duration': round(float(n.offset - n.onset), 3),
            'pitch': int(n.pitch), 'name': librosa.midi_to_note(int(n.pitch), unicode=False),
            'string': int(n.string), 'fret': int(n.fret), 'marks': n.techniques,
            'previous': None if np.isnan(n.since_previous) else
                {'fret': int(n.previous_fret), 'ago': round(float(n.since_previous), 2), 'marks': n.previous_mark},
            'rise': None if np.isnan(figures.rise) else round(float(figures.rise), 2),
            'travel': None if np.isnan(figures.travel) else round(float(figures.travel), 2),
            'wobble': None if np.isnan(figures.wobble) else round(float(figures.wobble), 2),
            'lead_s': cfg.before_s, 'axes': axes, 'span': [-cfg.before_s, cfg.after_s],
            'labels': labels_around(by_rec[n.rec], n, cfg.before_s, cfg.after_s),
        })  # fmt: skip
    (REPO_ROOT / out / 'clips.json').write_text(json.dumps({'clips': clips}, indent=1))
    groups = pd.Series([c['group'] for c in clips]).value_counts()
    info(f'{len(clips)} clips in {out}\n{groups.to_string()}')


if __name__ == '__main__':
    main()

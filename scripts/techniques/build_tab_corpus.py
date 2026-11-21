"""The notes and technique marks of every guitar track in DadaGP, for a prior over techniques that reads only tabs.

Each Guitar Pro file is read with the parser that reads GOAT's tabs (gtp.stage2.guitarpro), so the marks mean the
same. Kept are the tracks with a guitar program, `strings` strings and at least `min_notes` notes. Artists are
split into training and validation by a hash of their name.

Writes <out_dir>/tracks.csv (one row per track, in the order of notes.npz) and notes.npz: per note its track, start
and end in seconds, string, fret and marks, the marks as bits in the order of gtp.stage2.guitarpro.MARKS.

Usage:
  uv run scripts/techniques/build_tab_corpus.py
"""

import zlib
from multiprocessing import Pool

import guitarpro as gp
import hydra
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage2.guitarpro import MARKS, track_notes

BIT = {m: 1 << i for i, m in enumerate(MARKS)}


def read_file(job):
    path, programs, strings, min_notes = job
    try:
        song = gp.parse(str(path))
    except Exception:  # pyguitarpro raises many kinds of errors on damaged files
        return path, None
    tracks = []
    for i, track in enumerate(song.tracks):
        guitar = programs[0] <= track.channel.instrument <= programs[1] and not track.isPercussionTrack
        if not guitar or len(track.strings) != strings:
            continue
        try:
            tab = track_notes(song, track, techniques=True)
        except Exception:
            continue
        notes = sorted(tab['notes'], key=lambda n: (n['start'], n['string']))
        if len(notes) < min_notes:
            continue
        arrays = {
            'start': np.array([n['start'] for n in notes], dtype=np.float32),
            'end': np.array([n['end'] for n in notes], dtype=np.float32),
            'string': np.array([n['string'] for n in notes], dtype=np.int8),
            'fret': np.array([min(n['fret'], 127) for n in notes], dtype=np.int8),
            'marks': np.array([sum(BIT[m] for m in n['techniques']) for n in notes], dtype=np.uint32),
        }
        row = {'track_idx': i, 'name': track.name, 'program': track.channel.instrument, 'notes': len(notes)}
        tracks.append(({**row, 'tuning': ' '.join(map(str, tab['tuning']))}, arrays))
    return path, tracks


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='tab_corpus')
def main(cfg):
    root = REPO_ROOT / cfg.dadagp_dir
    files = sorted(p.with_name(p.name.removesuffix('.tokens.txt')) for p in root.glob('*/*/*.tokens.txt'))
    files = [f for f in files if f.exists()][: cfg.limit or None]
    jobs = [(f, tuple(cfg.programs), cfg.strings, cfg.min_notes) for f in files]
    rows, parts, unread = [], [], 0
    with Pool(cfg.workers) as pool:
        for done, (path, tracks) in enumerate(pool.imap(read_file, jobs, chunksize=16), 1):
            if tracks is None:
                unread += 1
                continue
            artist = path.parent.name
            val = zlib.crc32(artist.encode()) % 10_000 < cfg.val_share * 10_000
            for row, arrays in tracks:
                file = str(path.relative_to(root))
                rows.append({'file': file, 'artist': artist, 'split': 'val' if val else 'train', **row})
                parts.append({**arrays, 'track': np.full(row['notes'], len(rows) - 1, dtype=np.int32)})
            if done % 2000 == 0:
                info(f'{done}/{len(files)} files, {len(rows)} tracks')
    out = REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tracks = pd.DataFrame(rows)
    tracks.to_csv(out / 'tracks.csv', index_label='track')
    notes = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(out / 'notes.npz', **notes)
    carried = {m: int((notes['marks'] & BIT[m] > 0).sum()) for m in MARKS}
    info(
        f'{len(files)} files ({unread} unreadable), {len(tracks)} tracks, {len(notes["track"])} notes -> {out}\n'
        f'{tracks.groupby("split")["notes"].agg(["count", "sum"])}\n{carried}'
    )


if __name__ == '__main__':
    main()

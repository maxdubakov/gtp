"""Harmonic levels (gtp.stage1.envelope) for every note in the string index.

Writes <index_dir>/cache/levels.npz: `x` (notes, n_harmonics) float16 dB aligned to notes.csv.gz rows, NaN where a
harmonic is unusable.

Usage:
  uv run scripts/stage1/strings/harmonic_levels.py
  uv run scripts/stage1/strings/harmonic_levels.py limit=3
"""

import json
from concurrent.futures import ProcessPoolExecutor

import hydra
import librosa
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.envelope import note_levels


def job_levels(job):
    audio, onsets, offsets, pitches, cfg = job
    y, sr = librosa.load(REPO_ROOT / audio, sr=cfg.sample_rate, mono=True)
    return note_levels(y, sr, onsets, offsets, pitches, cfg).astype(np.float16)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='harmonic_levels')
def main(cfg):
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    notes = pd.read_csv(index / 'notes.csv.gz')
    if cfg.limit:
        recs = recs.groupby('dataset').head(cfg.limit)
    by_rec = notes.groupby('rec').indices
    idx = [by_rec[r] for r in recs['rec']]
    cols = [notes[c].to_numpy() for c in ('onset', 'offset', 'pitch')]
    jobs = [(a, *(c[i] for c in cols), cfg) for a, i in zip(recs['audio'], idx, strict=True)]
    x = np.full((len(notes), cfg.n_harmonics), np.nan, np.float16)
    with ProcessPoolExecutor(cfg.workers) as pool:
        for n, (i, levels) in enumerate(zip(idx, pool.map(job_levels, jobs, chunksize=4), strict=True)):
            x[i] = levels
            if (n + 1) % 500 == 0:
                info(f'{n + 1}/{len(jobs)} recordings')
    out = index / 'cache' / ('levels_limit.npz' if cfg.limit else 'levels.npz')
    np.savez(out, x=x)
    out.with_suffix('.json').write_text(json.dumps(dict(cfg) | {'n_notes': len(notes)}, indent=1, default=str))
    info(f'Wrote {out}: {len(notes)} notes, {np.isfinite(x).sum(axis=1).mean():.1f} usable harmonics per note')


if __name__ == '__main__':
    main()

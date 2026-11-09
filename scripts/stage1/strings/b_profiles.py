"""Inharmonicity likelihood profiles (gtp.stage1.bprofile) for every note in the string index.

Writes <index_dir>/cache/bprof.npz: `x` (notes, frames, grid) float16 aligned to notes.csv.gz rows, and the grid of
log10(B) values.

Usage:
  uv run scripts/stage1/strings/b_profiles.py
  uv run scripts/stage1/strings/b_profiles.py limit=3
"""

import json
from concurrent.futures import ProcessPoolExecutor

import hydra
import librosa
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.bprofile import grid, recording_profiles


def job_profiles(job):
    audio, onsets, offsets, pitches, cfg = job
    y, sr = librosa.load(REPO_ROOT / audio, sr=cfg.sample_rate, mono=True)
    return recording_profiles(y, sr, onsets, offsets, pitches, cfg).astype(np.float16)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='b_profiles')
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
    x = np.zeros((len(notes), cfg.n_frames, len(grid(cfg))), np.float16)
    done = np.zeros(len(notes), bool)
    with ProcessPoolExecutor(cfg.workers) as pool:
        for n, (i, prof) in enumerate(zip(idx, pool.map(job_profiles, jobs, chunksize=4), strict=True)):
            x[i], done[i] = prof, True
            if (n + 1) % 500 == 0:
                info(f'{n + 1}/{len(jobs)} recordings')
    out = index / 'cache' / ('bprof_limit.npz' if cfg.limit else 'bprof.npz')
    np.savez(out, x=x, log_b=grid(cfg), done=done)
    out.with_suffix('.json').write_text(json.dumps(dict(cfg) | {'n_notes': int(done.sum())}, indent=1, default=str))
    info(f'Wrote {out}: {int(done.sum())} notes')


if __name__ == '__main__':
    main()

"""Partial features (gtp.stage1.partials) for every note in the string index.

Writes <index_dir>/cache/harm.npz with `x` (notes, features) float32 aligned to notes.csv.gz rows, the feature
names, and which notes were extracted (`done`).

Usage:
  uv run scripts/stage1/strings/harmonic_features.py
  uv run scripts/stage1/strings/harmonic_features.py limit=20
"""

import json
from concurrent.futures import ProcessPoolExecutor

import hydra
import librosa
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.partials import feature_names, note_features


def recording_features(job):
    audio, onsets, pitches, cfg = job
    y, sr = librosa.load(REPO_ROOT / audio, sr=cfg.sample_rate, mono=True)
    return np.stack([note_features(y, sr, o, p, cfg) for o, p in zip(onsets, pitches, strict=True)])


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='harmonic_features')
def main(cfg):
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    notes = pd.read_csv(index / 'notes.csv.gz')
    if cfg.limit:
        recs = recs.groupby('dataset').head(cfg.limit)
    by_rec = notes.groupby('rec').indices
    idx = [by_rec[r] for r in recs['rec']]
    jobs = [
        (a, notes['onset'].to_numpy()[i], notes['pitch'].to_numpy()[i], cfg)
        for a, i in zip(recs['audio'], idx, strict=True)
    ]
    names = feature_names(cfg)
    x = np.full((len(notes), len(names)), np.nan, np.float32)
    with ProcessPoolExecutor(cfg.workers) as pool:
        for n, (i, feats) in enumerate(zip(idx, pool.map(recording_features, jobs, chunksize=4), strict=True)):
            x[i] = feats
            if (n + 1) % 500 == 0:
                info(f'{n + 1}/{len(jobs)} recordings')
    done = np.isfinite(x).any(axis=1)
    out = index / 'cache' / ('harm.npz' if not cfg.limit else 'harm_limit.npz')
    np.savez(out, x=x, names=np.array(names), done=done)
    (out.with_suffix('.json')).write_text(json.dumps(dict(cfg) | {'n_notes': int(done.sum())}, indent=1, default=str))
    info(f'Wrote {out}: {int(done.sum())} notes x {len(names)} features')


if __name__ == '__main__':
    main()

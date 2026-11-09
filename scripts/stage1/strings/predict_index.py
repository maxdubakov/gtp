"""Score the real recordings of the string index with trained index runs (train_index.py).

Writes <run>/index_probs.npz per run: `note` (rows of notes.csv.gz) and the calibrated `probs` (N, 6), for every
note of the requested splits. Later analyses join on `note` instead of re-deriving the test order of a run.

Usage:
  uv run scripts/stage1/strings/predict_index.py "runs=['results/strings/runs/<run>']"
"""

import hydra
import numpy as np

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.string_data import load_cached_index
from gtp.stage1.strings import StringClassifier


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='predict_index')
def main(cfg):
    device = auto_device()
    for run in cfg.runs:
        clf = StringClassifier.from_run(REPO_ROOT / run, device)
        notes, rows, x, _ = load_cached_index(cfg.index_dir, clf.feature)
        keep = np.flatnonzero((rows['variant'] == 0) & rows['note'].map(notes['split']).isin(list(cfg.splits)))
        note = rows['note'].to_numpy()[keep]
        order = np.argsort(keep)
        probs = clf.predict_windows(x[keep[order]], notes['pitch'].to_numpy()[note[order]])
        np.savez(REPO_ROOT / run / 'index_probs.npz', note=note[order], probs=probs)
        hit = probs.argmax(1) == notes['string'].to_numpy()[note[order]] - 1
        info(f'{run}: {len(note)} notes, accuracy {hit.mean():.3f}')


if __name__ == '__main__':
    main()

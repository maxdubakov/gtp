"""Score the real recordings of the string index with trained index runs (train_index.py).

Writes <run>/index_probs.npz per run: `note` (rows of notes.csv.gz) and the calibrated `probs` (N, 6), for every
note of the requested splits. Later analyses join on `note` instead of re-deriving the test order of a run. With
`out_dir`, each run's file goes to <out_dir>/<run name>/ instead, with accuracy.csv beside them: per run, dataset and
chain, the accuracy and NLL next to the run's pitch prior on the same notes.

Usage:
  uv run scripts/stage1/strings/predict_index.py "runs=['results/strings/runs/<run>']"
  uv run scripts/stage1/strings/predict_index.py "runs=['results/strings/runs/<run>']" "splits=[val]" \
      out_dir=results/strings/predictions/<name>
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.string_data import load_cached_index
from gtp.stage1.strings import N_STRINGS, StringClassifier, summarize


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='predict_index')
def main(cfg):
    device = auto_device()
    table = []
    for run in cfg.runs:
        clf = StringClassifier.from_run(REPO_ROOT / run, device)
        notes, rows, x, _ = load_cached_index(cfg.index_dir, clf.feature)
        keep = np.flatnonzero((rows['variant'] == 0) & rows['note'].map(notes['split']).isin(list(cfg.splits)))
        note = rows['note'].to_numpy()[keep]
        order = np.argsort(keep)
        windows, pitches = x[keep[order]], notes['pitch'].to_numpy()[note[order]]
        if cfg.adapt_bn or clf.centre:
            probs = np.zeros((len(order), N_STRINGS), np.float32)
            rec = notes['rec'].to_numpy()[note[order]]
            for i in pd.Series(rec).groupby(rec).indices.values():
                probs[i] = clf.predict_windows(windows[i], pitches[i], adapt=windows[i] if cfg.adapt_bn else None)
        else:
            probs = clf.predict_windows(windows, pitches)
        out = REPO_ROOT / (run if cfg.out_dir is None else Path(cfg.out_dir) / Path(run).name)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / 'index_probs.npz', note=note[order], probs=probs)
        frame = notes.iloc[note[order]].reset_index(drop=True)
        prior = clf.prior(frame['pitch'].to_numpy())
        for (dataset, chain), g in frame.groupby(['dataset', 'chain']):
            label = g['string'].to_numpy() - 1
            m, p = summarize(probs[g.index], label), summarize(prior[g.index], label)
            table.append({'run': Path(run).name, 'dataset': dataset, 'chain': chain, 'n': m['n'], 'acc': m['acc'],
                          'prior_acc': p['acc'], 'nll_bits': m['nll_bits'], 'prior_nll_bits': p['nll_bits']})  # fmt: skip
        hit = probs.argmax(1) == frame['string'].to_numpy() - 1
        info(f'{run}: {len(note)} notes, accuracy {hit.mean():.3f}')
    table = pd.DataFrame(table)
    info(f'\n{table.round(3).to_string(index=False)}')
    if cfg.out_dir is not None:
        table.to_csv(REPO_ROOT / cfg.out_dir / 'accuracy.csv', index=False)


if __name__ == '__main__':
    main()

"""Confusion matrices of technique-classifier runs on one validation group: what each labelled class was named.

Rows are the labels, columns what the classifier named them; counts are averaged over the matching runs (seeds),
with each row's shares next to them. The heads of `by_frets_heads` are also split by how many frets lie between the
note and the one before it on its string, since a hammer-on and a slide over one fret sound much alike.

Usage:
  uv run scripts/techniques/confusion.py "pattern=a_base_s*"
"""

import fnmatch

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.string_data import load_cached_index
from gtp.stage1.techniques import head_labels


def matrix(labels, predicted, classes):
    counts = np.zeros((len(classes), len(classes)))
    np.add.at(counts, (labels[labels >= 0], predicted[labels >= 0]), 1)
    return counts


def shown(counts, classes):
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    cells = [[f'{c:.0f} ({s:.0%})' for c, s in zip(cr, sr, strict=True)] for cr, sr in zip(counts, share, strict=True)]
    return pd.DataFrame(cells, index=[f'is {c}' for c in classes], columns=[f'named {c}' for c in classes]).to_string()


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='confusion')
def main(cfg):
    runs = [
        r
        for r in sorted((REPO_ROOT / cfg.runs_dir).iterdir())
        if fnmatch.fnmatch(r.name.split('_', 2)[-1], cfg.pattern) and (r / 'val_logits.npz').exists()
    ]
    trained = OmegaConf.load(runs[0] / '.hydra' / 'config.yaml')
    notes, _, _, _ = load_cached_index(trained.index_dir, trained.feature, trained.cache)
    labels = head_labels(notes, trained.heads, trained.unlabelled)
    ordered = notes.sort_values(['rec', 'string', 'onset'], kind='stable')
    frets = (
        (ordered['fret'] - ordered.groupby(['rec', 'string'], sort=False)['fret'].shift()).abs().reindex(notes.index)
    )
    inside = np.logical_and.reduce([(notes[k] == v).to_numpy() for k, v in cfg.group.items()])

    info(f'{len(runs)} runs matching {cfg.pattern}, notes of {dict(cfg.group)}')
    for head, classes in ((name, list(h.classes)) for name, h in trained.heads.items()):
        splits = {'all': np.ones(len(notes), dtype=bool)}
        if head in cfg.by_frets_heads:
            splits |= {
                f'{lo} to {hi} frets from the note before': frets.between(lo, hi).to_numpy() for lo, hi in cfg.by_frets
            }
        for name, part in splits.items():
            counts = []
            for run in runs:
                z = np.load(run / 'val_logits.npz')
                keep = (inside & part)[z['note']]
                counts.append(matrix(labels[head][z['note']][keep], z[head].argmax(axis=1)[keep], classes))
            info(f'{head}, {name}\n{shown(np.mean(counts, axis=0), classes)}')


if __name__ == '__main__':
    main()

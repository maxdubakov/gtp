"""Confusion matrices of technique-classifier runs on one validation group: what each labelled class was named.

Rows are the labels, columns what the classifier named them; counts are averaged over the matching runs (seeds),
with each row's shares next to them and each class's F1 below. The heads of `by_frets_heads` are also split by how
many frets lie between the note and the one before it on its string, since a hammer-on and a slide over one fret
sound much alike. `joint` reads two heads together as one, e.g. the symbol a tab prints from how a note is attacked
and whether the pitch slides into it.

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
    table = pd.DataFrame(cells, index=[f'is {c}' for c in classes], columns=[f'named {c}' for c in classes])
    f1 = 2 * np.diag(counts) / np.maximum(counts.sum(axis=0) + counts.sum(axis=1), 1)
    return f'{table.to_string()}\nF1 ' + ', '.join(f'{c} {v:.3f}' for c, v in zip(classes, f1, strict=True))


def joined(first, second, sizes, classes):
    """Class of the joint head per note from its two heads' classes (-1 where either is): `classes` lists, per
    joint class, the pairs of the two heads' class indices that make it."""
    table = np.full(sizes, -1)
    for k, pairs in enumerate(classes):
        for i, j in pairs:
            table[i, j] = k
    return np.where((first >= 0) & (second >= 0), table[first, second], -1)


def logits(run):
    return np.load(run / 'val_logits.npz')


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
    names = {name: list(h.classes) for name, h in trained.heads.items()}
    named = [{name: z[name].argmax(axis=1) for name in names} | {'note': z['note']} for z in map(logits, runs)]
    for name, joint in cfg.joint.items():
        a, b = joint.heads
        if a not in names or b not in names:
            continue
        pairs = [[(names[a].index(i), names[b].index(j)) for i, j in of] for of in joint.classes.values()]
        sizes = (len(names[a]), len(names[b]))
        labels[name] = joined(labels[a], labels[b], sizes, pairs)
        for run in named:
            run[name] = joined(run[a], run[b], sizes, pairs)
        names[name] = list(joint.classes)
    for head, classes in names.items():
        splits = {'all': np.ones(len(notes), dtype=bool)}
        if head in cfg.by_frets_heads:
            splits |= {
                f'{lo} to {hi} frets from the note before': frets.between(lo, hi).to_numpy() for lo, hi in cfg.by_frets
            }
        for name, part in splits.items():
            counts = []
            for run in named:
                keep = (inside & part)[run['note']]
                counts.append(matrix(labels[head][run['note']][keep], run[head][keep], classes))
            info(f'{head}, {name}\n{shown(np.mean(counts, axis=0), classes)}')


if __name__ == '__main__':
    main()

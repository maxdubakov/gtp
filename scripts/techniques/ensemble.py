"""Average the validation answers of several technique-classifier runs and score the average like a run of its own.

The runs' log-probabilities are averaged per head. The result is written as a run directory named `name` next to
the others (val_logits.npz, metrics.json and the first run's config), so compare.py, confusion.py and fuse_prior.py
read it as any other run.

Usage:
  uv run scripts/techniques/ensemble.py "patterns=[d_ex_s*]" name=d_ex_all
"""

import fnmatch
import json
import shutil

import hydra
import numpy as np
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.string_data import load_cached_index
from gtp.stage1.tab_prior import log_softmax
from gtp.stage1.techniques import grouped_scores, head_labels


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='ensemble')
def main(cfg):
    runs = [
        r
        for r in sorted((REPO_ROOT / cfg.runs_dir).iterdir())
        if any(fnmatch.fnmatch(r.name.split('_', 2)[-1], p) for p in cfg.patterns) and (r / 'val_logits.npz').exists()
    ]
    trained = OmegaConf.load(runs[0] / '.hydra' / 'config.yaml')
    notes, _, _, _ = load_cached_index(trained.index_dir, trained.feature, trained.cache)
    heads = {name: list(head.classes) for name, head in trained.heads.items()}
    labels = head_labels(notes, trained.heads, trained.unlabelled)
    saved = [np.load(r / 'val_logits.npz') for r in runs]
    note = saved[0]['note']
    if any(not np.array_equal(z['note'], note) for z in saved):
        raise ValueError('the runs scored different notes')
    mean = {h: np.mean([log_softmax(z[h]) for z in saved], axis=0) for h in heads}
    day, hour, _ = runs[-1].name.split('_', 2)
    out = runs[0].parent / f'{day}_{hour}_{cfg.name}'
    (out / '.hydra').mkdir(parents=True, exist_ok=True)
    shutil.copy(runs[0] / '.hydra' / 'config.yaml', out / '.hydra' / 'config.yaml')
    frame = notes.iloc[note].reset_index(drop=True)
    val = {h: grouped_scores(frame, labels[h][note], mean[h].argmax(axis=1), c) for h, c in heads.items()}
    np.savez(out / 'val_logits.npz', note=note, **mean)
    (out / 'metrics.json').write_text(json.dumps({'heads': heads, 'val': val, 'of': [r.name for r in runs]}, indent=1))
    info(f'{len(runs)} runs -> {out.name}')


if __name__ == '__main__':
    main()

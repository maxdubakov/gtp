"""Fuse technique-classifier runs with the tab prior and score the result like a run of its own.

Per head, the fused log-probability of a class is the classifier's plus `weight` times what the prior adds to the
class's share in its training tabs: log p_tab(class | the tab around the note) - log share(class). The prior reads
the index's own notes (strings, frets, label times), every recording as one tab; a tied note keeps the classifier's
answer, and so does every head not in `heads`.

For each run and weight this writes a run directory next to the classifier's, named as it with `_tab<weight x 10>`
before the seed (`_tabonly` for the prior with no audio), holding val_logits.npz, metrics.json and the classifier's
config, so that compare.py and confusion.py read it as any other run.

Usage:
  uv run scripts/techniques/fuse_prior.py "pattern=d_ex_s*" prior=results/techniques/priors/<run>
"""

import fnmatch
import json
import re
import shutil

import hydra
import numpy as np
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.string_data import load_cached_index
from gtp.stage1.tab_prior import index_prior, log_softmax
from gtp.stage1.techniques import grouped_scores, head_labels


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='fuse_prior')
def main(cfg):
    runs = [
        r
        for r in sorted((REPO_ROOT / cfg.runs_dir).iterdir())
        if fnmatch.fnmatch(r.name.split('_', 2)[-1], cfg.pattern)
        and (r / 'val_logits.npz').exists()
        and not (r / 'fused.json').exists()
    ]
    trained = OmegaConf.load(runs[0] / '.hydra' / 'config.yaml')
    notes, _, _, _ = load_cached_index(trained.index_dir, trained.feature, trained.cache)
    heads = {name: list(head.classes) for name, head in trained.heads.items()}
    labels = head_labels(notes, trained.heads, trained.unlabelled)
    gains, alone = index_prior(REPO_ROOT / cfg.prior, notes, heads, auto_device())

    def write(run, tag, logits, note):
        day, hour, name = run.name.split('_', 2)
        out = run.parent / f'{day}_{hour}_{re.sub(r"(_s\d+)$", rf"_{tag}\1", name)}'
        (out / '.hydra').mkdir(parents=True, exist_ok=True)
        shutil.copy(run / '.hydra' / 'config.yaml', out / '.hydra' / 'config.yaml')
        frame = notes.iloc[note].reset_index(drop=True)
        val = {h: grouped_scores(frame, labels[h][note], logits[h].argmax(axis=1), c) for h, c in heads.items()}
        np.savez(out / 'val_logits.npz', note=note, **logits)
        (out / 'metrics.json').write_text(json.dumps({'heads': heads, 'val': val}, indent=1))
        (out / 'fused.json').write_text(json.dumps({'classifier': run.name, 'prior': str(cfg.prior), 'as': tag}))
        return out.name

    for run in runs:
        z = np.load(run / 'val_logits.npz')
        note = z['note']
        heard = {h: log_softmax(z[h]) for h in heads}
        made = []
        for w in cfg.weights:
            fused = {h: heard[h] + (w * gains[h][note] if h in cfg.heads else 0) for h in heads}
            made.append(write(run, f'tab{round(w * 10):02d}', fused, note))
        if cfg.alone:
            read = {h: np.where(np.isnan(alone[h][note]), heard[h], alone[h][note]) for h in heads}
            made.append(write(run, 'tabonly', read, note))
        info(f'{run.name} -> {made}')


if __name__ == '__main__':
    main()

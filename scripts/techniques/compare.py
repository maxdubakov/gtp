"""Compare technique-classifier runs: one row per setting (seeds averaged), one column per validation figure.

A run's setting is its name without the timestamp and a trailing _s<seed>. Each column is a class's F1 (or P, R)
in one validation group of the trainer's metrics.json.

Usage:
  uv run scripts/techniques/compare.py "patterns=[reg_*]"
"""

import fnmatch
import json
import re

import hydra
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='compare')
def main(cfg):
    rows = []
    for run in sorted((REPO_ROOT / cfg.runs_dir).iterdir()):
        name = run.name.split('_', 2)[-1]
        if not (run / 'metrics.json').exists() or not any(fnmatch.fnmatch(name, p) for p in cfg.patterns):
            continue
        val = json.loads((run / 'metrics.json').read_text())['val']
        row = {'setting': re.sub(r'_s\d+$', '', name)}
        for c in cfg.columns:
            row[c.name] = val[c.head].get(c.group, {}).get(c['class'], {}).get(cfg.metric, float('nan'))
        rows.append(row)
    runs = pd.DataFrame(rows)
    table = runs.groupby('setting', sort=False).mean().assign(runs=runs.groupby('setting', sort=False).size())
    spread = runs.groupby('setting', sort=False).std()
    with pd.option_context('display.width', 250, 'display.float_format', '{:.3f}'.format):
        info(f'{cfg.metric}, mean over runs\n{table.to_string()}')
        info(f'spread between runs (standard deviation)\n{spread.to_string()}')


if __name__ == '__main__':
    main()

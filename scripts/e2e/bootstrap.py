"""Confidence intervals for an index.py run: recordings resampled with replacement within each group.

Tab F1 (matched notes on the right string over predicted and reference notes) and string accuracy on matched notes,
per method and group, with percentile intervals from `resamples` bootstrap draws of the group's recordings. The
difference of each method from Stage 2 alone is resampled on the same draws, so its interval is paired.
Writes <run>/bootstrap.csv.

Usage:
  uv run scripts/e2e/bootstrap.py run=results/e2e/index/<run>
"""

import hydra
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info


def scores(per_rec, idx, methods):
    s = per_rec.iloc[idx].sum()
    out = {}
    for m in methods:
        out[f'{m}_tab_f1'] = 2 * s[f'{m}_ok'] / max(s['n_est'] + s['n_ref'], 1)
        out[f'{m}_acc'] = s[f'{m}_ok'] / max(s['n_matched'], 1)
    return out


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'e2e'), config_name='bootstrap')
def main(cfg):
    run = REPO_ROOT / cfg.run
    notes = pd.read_csv(run / 'notes.csv.gz')
    notes = notes[(notes['input'] == cfg.input) & notes['ref_string'].notna()]
    counts = pd.read_csv(run / 'counts.csv')
    counts = counts[counts['input'] == cfg.input].set_index('rec')
    ok = notes.assign(**{f'{m}_ok': notes[m] == notes['ref_string'] for m in cfg.methods})
    per_rec = counts.join(ok.groupby('rec')[[f'{m}_ok' for m in cfg.methods]].sum()).fillna(0)
    rng = np.random.default_rng(cfg.seed)
    lo, hi = 100 * (1 - cfg.level) / 2, 100 * (1 + cfg.level) / 2
    rows = []
    for key, g in per_rec.groupby(list(cfg.by)):
        point = scores(g, np.arange(len(g)), cfg.methods)
        draws = pd.DataFrame([scores(g, rng.integers(0, len(g), len(g)), cfg.methods) for _ in range(cfg.resamples)])
        row = dict(zip(cfg.by, key, strict=True)) | {'recordings': len(g)}
        for name, value in point.items():
            row[name] = value
            row[f'{name}_lo'], row[f'{name}_hi'] = np.percentile(draws[name], [lo, hi])
        gain = draws['pipeline_tab_f1'] - draws['stage2_tab_f1']
        row['gain_tab_f1_lo'], row['gain_tab_f1_hi'] = np.percentile(gain, [lo, hi])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(run / 'bootstrap.csv', index=False)
    show = [*cfg.by, 'recordings', 'stage2_tab_f1', 'pipeline_tab_f1', 'pipeline_tab_f1_lo', 'pipeline_tab_f1_hi',
            'gain_tab_f1_lo', 'gain_tab_f1_hi']  # fmt: skip
    info(f'\n{table[show].round(3).to_string(index=False)}')


if __name__ == '__main__':
    main()

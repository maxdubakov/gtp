"""Whether the notes a tab marks as not picked are played that way, per recording.

A pick stroke leaves a burst in the upper spectrum at the onset (gtp.stage1.techniques.attack_burst); a hammer-on, a
pull-off or a legato slide leaves next to none. For each recording with enough marked notes this gives the chance
that a marked note has a smaller burst than a picked one of the same recording (AUC: 1 = always, 0.5 = the marked
notes look picked). A player who picks the notes a tab marks as hammered shows up as a recording near 0.5.

Writes `out`: rec, dataset, player, split, notes, marked, auc, and the median burst of each kind.

Usage:
  uv run scripts/techniques/attack_check.py
"""

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.string_data import load_cached_index
from gtp.stage1.techniques import attack_burst, head_labels


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='attack_check')
def main(cfg):
    notes, rows, x, meta = load_cached_index(cfg.index_dir, cfg.feature, cfg.cache)
    trained = OmegaConf.load(REPO_ROOT / cfg.train_config)
    marked = head_labels(notes, {cfg.head: trained.heads[cfg.head]}, [])[cfg.head]
    real = rows[rows['variant'] == 0]
    burst = np.full(len(notes), np.nan)
    for start in range(0, len(real), 20000):
        part = real.iloc[start : start + 20000]
        burst[part['note']] = attack_burst(
            x[part.index.to_numpy()][:, 0].astype(np.float32), meta['layout']['pre_frames']
        )
    notes = notes.assign(burst=burst, marked=marked)
    out = []
    for rec, of in notes[notes['marked'] >= 0].groupby('rec', sort=False):
        picked, led = of.loc[of['marked'] == 0, 'burst'], of.loc[of['marked'] == 1, 'burst']
        if len(led) < cfg.min_marked or picked.empty:
            continue
        first = of.iloc[0]
        out.append(
            {
                'rec': rec,
                'dataset': first['dataset'],
                'player': first['player'],
                'split': first['split'],
                'notes': len(of),
                'marked': len(led),
                'auc': roc_auc_score(of['marked'], -of['burst']),
                'picked_burst': picked.median(),
                'marked_burst': led.median(),
            }
        )
    out = pd.DataFrame(out).sort_values(['dataset', 'player', 'auc'])
    path = REPO_ROOT / cfg.out
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    with pd.option_context('display.width', 250, 'display.max_rows', 500, 'display.float_format', '{:.2f}'.format):
        info(
            f'{len(out)} recordings with at least {cfg.min_marked} marked notes -> {path}\n{out.to_string(index=False)}'
        )


if __name__ == '__main__':
    main()

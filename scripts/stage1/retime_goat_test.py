"""Re-time GOAT's evaluation labels to each render's audio, the way its training labels were (refine_labels.py).

The same frozen checkpoint and settings that re-timed the training manifest are applied to every render (DI and the
five amps) of the `split` items, so training and evaluation labels are timed alike. Notes are neither added nor
dropped and keep their pitch and order; `notes.csv.gz` has one row per shipped note.

Usage:
  uv run scripts/stage1/retime_goat_test.py checkpoint=runs/stage1/<run>/checkpoints/step_0010000.pth
"""

import hydra
import pandas as pd

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1 import benchmarks
from gtp.stage1.inference import PianoTranscription
from gtp.stage1.retiming import detected, retime


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='retime_goat_test')
def main(cfg):
    model = PianoTranscription(str(REPO_ROOT / cfg.checkpoint), device=cfg.device or auto_device())
    model.batch_size = cfg.batch
    notes, report = [], []
    for r in benchmarks.goat(list(cfg.renders), cfg.split):
        est = detected(model, benchmarks.load(r.audio), cfg.onset_threshold)
        new, lag, moved = retime(r.notes, est, cfg.lag_window_s, cfg.max_shift_s)
        notes.append(pd.DataFrame(new, columns=['onset', 'offset', 'pitch']).assign(rec=r.rec))
        report.append({'rec': r.rec, 'lag_ms': 1000 * lag, 'moved': moved,
                       'mean_abs_change_ms': 1000 * abs(new[:, 0] - r.notes[:, 0]).mean()})  # fmt: skip
    out = REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(notes)[['rec', 'onset', 'offset', 'pitch']].to_csv(out / 'notes.csv.gz', index=False)
    report = pd.DataFrame(report)
    report.to_csv(out / 'report.csv', index=False)
    info(f'{len(report)} recordings re-timed -> {out}\n{report.describe().round(3).to_string()}')


if __name__ == '__main__':
    main()

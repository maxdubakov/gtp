"""Note P/R/F1 of one Stage 1 checkpoint against the onset tolerance, on the benchmarks of benchmark.py.

If F1 keeps rising past the published 50 ms, the misses there are onsets the model places near a note but outside the
window: label timing error more than model error. Scores are benchmark.py's (mean of per-recording F1, plus micro F1)
from the same cached model outputs, at each tolerance in `tolerances`.

Writes per_recording.csv and summary.csv (one row per benchmark and tolerance) to the Hydra run dir.

Usage:
  uv run scripts/stage1/tolerance_curve.py
  uv run scripts/stage1/tolerance_curve.py checkpoint=runs/stage1/<run>/checkpoints/step_0030000.pth "benchmarks=[egdb]"
"""

from functools import partial

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1 import benchmarks
from gtp.stage1.benchmarks import counts, notes_of, outputs, scores
from gtp.stage1.inference import PianoTranscription


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='tolerance_curve')
def main(cfg):
    out = HydraConfig.get().runtime.output_dir
    model = PianoTranscription(str(REPO_ROOT / cfg.checkpoint), device=cfg.device or auto_device())
    model.batch_size = cfg.batch
    cache = REPO_ROOT / cfg.cache_dir / cfg.checkpoint.replace('/', '__')
    rows = []
    for name in cfg.benchmarks:
        for r in benchmarks.build(name, cfg):
            output = outputs(model, cache / f'{r.rec.replace("/", "__")}.npz', partial(benchmarks.load, r.audio))
            est = notes_of(output, cfg.postprocess)
            for tol in cfg.tolerances:
                tp, n_est, n_ref = counts(r.notes, est, tol)
                rows.append({'benchmark': name, 'tolerance': tol, 'rec': r.rec, 'tp': tp, 'est': n_est, 'ref': n_ref})
        info(f'{name} done')
    per_rec, summary = scores(pd.DataFrame(rows), ['benchmark', 'tolerance'])
    per_rec.to_csv(f'{out}/per_recording.csv', index=False)
    summary.insert(0, 'checkpoint', cfg.checkpoint)
    summary.to_csv(f'{out}/summary.csv', index=False)
    table = summary.pivot(index='benchmark', columns='tolerance', values='f1').loc[list(cfg.benchmarks)]
    info(f'\nF1 (mean over recordings) by onset tolerance (s)\n{table.round(4).to_string()}')


if __name__ == '__main__':
    main()

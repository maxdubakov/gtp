"""Note P/R/F1 of Stage 1 checkpoints on the published benchmarks (src/gtp/stage1/benchmarks.py).

Onset-only matching as in mir_eval (onsets within `onset_tolerance`, pitch within 50 cents, offsets ignored), the
measure of Riley et al., GAPS, TART and YourMT3+. `f1` is the mean of the recordings' F1 (mir_eval per file, as those
papers report); `micro_f1` pools the notes. Model outputs are cached per checkpoint and recording, so every
post-processing setting is scored from one pass of the model. A checkpoint entry joining paths with `+` scores the
ensemble of those checkpoints: their output rolls are averaged before notes are picked.

Usage:
  uv run scripts/stage1/benchmark.py
  uv run scripts/stage1/benchmark.py "checkpoints=[runs/stage1/<run>/checkpoints/step_0030000.pth]" "benchmarks=[egdb]"
"""

from functools import partial

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1 import benchmarks
from gtp.stage1.benchmarks import KEYS, counts, notes_of, outputs, scores
from gtp.stage1.inference import PianoTranscription


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='benchmark')
def main(cfg):
    device = cfg.device or auto_device()
    out = HydraConfig.get().runtime.output_dir
    suites = {name: benchmarks.build(name, cfg) for name in cfg.benchmarks}
    info(' '.join(f'{n}: {len(s)}' for n, s in suites.items()))
    rows = []
    for ckpt in cfg.checkpoints:
        members = []
        for path in str(ckpt).split('+'):
            model = PianoTranscription(str(REPO_ROOT / path), device=device)
            model.batch_size = cfg.batch
            members.append((model, REPO_ROOT / cfg.cache_dir / path.replace('/', '__')))
        for name, suite in suites.items():
            for r in suite:
                audio = partial(benchmarks.load, r.audio)
                runs = [outputs(m, cache / f'{r.rec.replace("/", "__")}.npz', audio) for m, cache in members]
                output = {k: np.mean([o[k] for o in runs], axis=0) for k in KEYS}
                for pp in cfg.postprocess:
                    tp, n_est, n_ref = counts(r.notes, notes_of(output, pp), cfg.onset_tolerance)
                    rows.append({'checkpoint': ckpt, 'benchmark': name, 'postprocess': pp.name, 'rec': r.rec,
                                 'tp': tp, 'est': n_est, 'ref': n_ref})  # fmt: skip
            info(f'{ckpt} {name} done')
    per_rec, summary = scores(pd.DataFrame(rows), ['checkpoint', 'benchmark', 'postprocess'])
    per_rec.to_csv(f'{out}/per_recording.csv', index=False)
    summary.to_csv(f'{out}/summary.csv', index=False)
    info(f'\n{summary.drop(columns=["tp", "est", "ref"]).round(4).to_string(index=False)}')


if __name__ == '__main__':
    main()

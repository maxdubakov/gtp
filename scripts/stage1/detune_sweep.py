"""Note P/R/F1 of one Stage 1 checkpoint on benchmark recordings detuned by a few cents, to see how fragile it is to
a guitar that is not at A440.

A detune of c cents plays the recording back r = 2^(c/1200) times faster: the audio at its native rate is treated as
sampled at r times that rate and resampled to 16 kHz with soxr (librosa's soxr_hq, as librosa.load uses), so every
frequency is multiplied by r and every time divided by r. The reference onsets and offsets are divided by r too; the
reference pitch stays the nominal MIDI note, since a detuned string still plays that note. At 0 cents the audio is
exactly what benchmark.py feeds the model. Scoring is benchmark.py's at `onset_tolerance`. Model outputs are cached
per detune under `cache_dir`, never in the benchmark cache.

Writes per_recording.csv and summary.csv (one row per benchmark and detune) to the Hydra run dir.

Usage:
  uv run scripts/stage1/detune_sweep.py
  uv run scripts/stage1/detune_sweep.py subset=120 "cents=[-25,0,25]"
"""

from functools import partial

import hydra
import librosa
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1 import benchmarks
from gtp.stage1.benchmarks import counts, notes_of, outputs, scores
from gtp.stage1.inference import PianoTranscription


def detuned(path, cents):
    """The recording at `path` played back 2^(cents/1200) times faster, as 16 kHz mono."""
    y, sr = librosa.load(path, sr=None, mono=True)
    return librosa.resample(y, orig_sr=sr * 2 ** (cents / 1200), target_sr=benchmarks.SR, res_type='soxr_hq')


def subset(suite, n, seed):
    if n is None or n >= len(suite):
        return suite
    return [suite[i] for i in sorted(np.random.default_rng(seed).choice(len(suite), n, replace=False))]


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='detune_sweep')
def main(cfg):
    out = HydraConfig.get().runtime.output_dir
    model = PianoTranscription(str(REPO_ROOT / cfg.checkpoint), device=cfg.device or auto_device())
    model.batch_size = cfg.batch
    root = REPO_ROOT / cfg.cache_dir / cfg.checkpoint.replace('/', '__')
    suites = {name: subset(benchmarks.build(name, cfg), cfg.subset, cfg.seed) for name in cfg.benchmarks}
    info(' '.join(f'{n}: {len(s)}' for n, s in suites.items()))
    rows = []
    for cents in cfg.cents:
        r = 2 ** (cents / 1200)
        for name, suite in suites.items():
            for rec in suite:
                path = root / f'{cents:+d}' / f'{rec.rec.replace("/", "__")}.npz'
                output = outputs(model, path, partial(detuned, rec.audio, cents))
                ref = rec.notes * np.array([1 / r, 1 / r, 1.0])
                tp, n_est, n_ref = counts(ref, notes_of(output, cfg.postprocess), cfg.onset_tolerance)
                rows.append({'benchmark': name, 'cents': cents, 'rec': rec.rec, 'tp': tp, 'est': n_est, 'ref': n_ref})
            info(f'{cents:+d} cents {name} done')
    per_rec, summary = scores(pd.DataFrame(rows), ['benchmark', 'cents'])
    per_rec.to_csv(f'{out}/per_recording.csv', index=False)
    summary.insert(0, 'checkpoint', cfg.checkpoint)
    summary.to_csv(f'{out}/summary.csv', index=False)
    for metric in ('f1', 'p', 'r'):
        table = summary.pivot(index='benchmark', columns='cents', values=metric).loc[list(cfg.benchmarks)]
        info(f'\n{metric} by detune (cents)\n{table.round(4).to_string()}')


if __name__ == '__main__':
    main()

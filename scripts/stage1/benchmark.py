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

import hydra
import librosa
import mir_eval
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1 import benchmarks
from gtp.stage1.inference import SAMPLE_RATE, PianoTranscription
from gtp.stage1.postprocess import RegressionPostProcessor

KEYS = ['reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output']


def outputs(model, recording, path):
    if not path.exists():
        audio, _ = librosa.load(recording.audio, sr=SAMPLE_RATE, mono=True)
        out = model._run_model(audio)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: out[k].astype(np.float16) for k in KEYS})
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in KEYS}


def notes_of(output, pp):
    processor = RegressionPostProcessor(100, 88, onset_threshold=pp.onset, offset_threshold=pp.offset,
                                        frame_threshold=pp.frame)  # fmt: skip
    events, _ = processor.output_dict_to_note_events(output)
    notes = np.array([[e['onset_time'], e['offset_time'], e['midi_note']] for e in events], float).reshape(-1, 3)
    return notes[notes[:, 1] - notes[:, 0] >= pp.min_duration_s]


def counts(ref, est, tolerance):
    if len(ref) and len(est):

        def arrays(n):
            return np.c_[n[:, 0], np.maximum(n[:, 1], n[:, 0] + 1e-3)], librosa.midi_to_hz(n[:, 2])

        tp = len(mir_eval.transcription.match_notes(*arrays(ref), *arrays(est), onset_tolerance=tolerance,
                                                     offset_ratio=None))  # fmt: skip
    else:
        tp = 0
    return tp, len(est), len(ref)


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
                runs = [outputs(m, r, cache / f'{r.rec.replace("/", "__")}.npz') for m, cache in members]
                output = {k: np.mean([o[k] for o in runs], axis=0) for k in KEYS}
                for pp in cfg.postprocess:
                    tp, n_est, n_ref = counts(r.notes, notes_of(output, pp), cfg.onset_tolerance)
                    rows.append({'checkpoint': ckpt, 'benchmark': name, 'postprocess': pp.name, 'rec': r.rec,
                                 'tp': tp, 'est': n_est, 'ref': n_ref})  # fmt: skip
            info(f'{ckpt} {name} done')
    per_rec = pd.DataFrame(rows)
    per_rec['p'] = per_rec.tp / per_rec.est.clip(lower=1)
    per_rec['r'] = per_rec.tp / per_rec.ref.clip(lower=1)
    per_rec['f1'] = 2 * per_rec.p * per_rec.r / (per_rec.p + per_rec.r).clip(lower=1e-12)
    per_rec.to_csv(f'{out}/per_recording.csv', index=False)
    g = per_rec.groupby(['checkpoint', 'benchmark', 'postprocess'], sort=False)
    summary = g[['p', 'r', 'f1']].mean().join(g[['tp', 'est', 'ref']].sum()).reset_index()
    summary['micro_f1'] = 2 * summary.tp / (summary.est + summary.ref)
    summary['recordings'] = g.size().to_numpy()
    summary.to_csv(f'{out}/summary.csv', index=False)
    info(f'\n{summary.drop(columns=["tp", "est", "ref"]).round(4).to_string(index=False)}')


if __name__ == '__main__':
    main()

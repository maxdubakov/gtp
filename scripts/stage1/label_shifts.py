"""One constant onset shift per benchmark recording, for scoring on corrected test labels.

A benchmark's labels can sit off the audio by a constant per recording (GOAT's test labels: up to 42 ms). This script
estimates that constant with a checkpoint that is not under test (Riley et al.'s released FL model by default): the
median offset of its detected onsets from the labelled ones, over notes of the same pitch matched within `window_s`.
Only the constant is corrected, never single notes, so no model's opinion of a note enters the labels. A render of a
DI recording (GOAT's amp renders, `renders`) takes its DI's shift plus the render's latency, measured without a model
by cross-correlating the two onset-strength envelopes; a model hears distorted onsets too unreliably. benchmarks.py
applies the shifts to the `<benchmark>_corrected` suites; the original suites stay as published.

Usage:
  uv run scripts/stage1/label_shifts.py
"""

import hydra
import librosa
import mir_eval
import numpy as np
import pandas as pd
import scipy.signal

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1 import benchmarks
from gtp.stage1.inference import SAMPLE_RATE, PianoTranscription
from gtp.stage1.postprocess import RegressionPostProcessor


def latency(render, source, max_s, hop=32):
    """Seconds by which `render` lags `source`, from their onset-strength envelopes."""
    env = [librosa.onset.onset_strength(y=librosa.load(p, sr=SAMPLE_RATE, mono=True)[0], sr=SAMPLE_RATE, hop_length=hop)
           for p in (render, source)]  # fmt: skip
    n = min(len(e) for e in env)
    a, b = (e[:n] - e[:n].mean() for e in env)
    corr = scipy.signal.correlate(a, b, mode='full')
    lags = np.arange(-n + 1, n)
    keep = np.abs(lags) <= max_s * SAMPLE_RATE / hop
    return lags[keep][np.argmax(corr[keep])] * hop / SAMPLE_RATE


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='label_shifts')
def main(cfg):
    model = PianoTranscription(str(REPO_ROOT / cfg.checkpoint), device=cfg.device or auto_device())
    model.batch_size = cfg.batch
    processor = RegressionPostProcessor(100, 88)
    rows = []
    for name in cfg.benchmarks:
        if name in cfg.renders:
            continue
        for r in benchmarks.build(name, cfg):
            audio, _ = librosa.load(r.audio, sr=SAMPLE_RATE, mono=True)
            events, _ = processor.output_dict_to_note_events(model._run_model(audio))
            est = np.array([[e['onset_time'], e['midi_note']] for e in events], float).reshape(-1, 2)

            def arrays(onsets, pitch):
                return np.c_[onsets, onsets + 0.1], librosa.midi_to_hz(pitch)

            pairs = mir_eval.transcription.match_notes(*arrays(r.notes[:, 0], r.notes[:, 2]), *arrays(est[:, 0], est[:, 1]),
                                                       onset_tolerance=cfg.window_s, offset_ratio=None)  # fmt: skip
            lag = np.median([est[j, 0] - r.notes[i, 0] for i, j in pairs]) if len(pairs) >= cfg.min_matched else 0.0
            rows.append({'rec': r.rec, 'shift_ms': round(1000 * float(lag), 1), 'matched': len(pairs)})
    by_rec = {row['rec']: row for row in rows}
    for name, source in cfg.renders.items():
        sources = {r.rec.rsplit('/', 1)[0]: r for r in benchmarks.build(source, cfg)}
        for r in benchmarks.build(name, cfg):
            di = sources[r.rec.rsplit('/', 1)[0]]
            lag = latency(r.audio, di.audio, cfg.max_latency_s)
            rows.append({'rec': r.rec, 'shift_ms': round(by_rec[di.rec]['shift_ms'] + 1000 * lag, 1),
                         'matched': by_rec[di.rec]['matched'], 'render_latency_ms': round(1000 * lag, 1)})  # fmt: skip
    shifts = pd.DataFrame(rows)
    shifts.to_csv(REPO_ROOT / cfg.out, index=False)
    info(f'{cfg.checkpoint} -> {cfg.out}\n{shifts.to_string(index=False)}')


if __name__ == '__main__':
    main()

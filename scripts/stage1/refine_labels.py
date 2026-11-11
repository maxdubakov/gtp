"""Re-time a notes manifest's labels to the audio with a Stage 1 checkpoint, as NoteEM's alignment step does.

Per recording: the labels shift by the median offset of their onsets from the onsets the checkpoint detects (same
pitch, within `lag_window_s`); then each labelled onset moves to a detected onset of its pitch within `max_shift_s`,
one detection per label (mir_eval's matching). Offsets shift with the recording, not per note. Pitches, the notes kept
and the splits are unchanged, so the refined manifest trains on the same notes, better timed.

Usage:
  uv run scripts/stage1/refine_labels.py checkpoint=runs/stage1/<run>/checkpoints/step_0040000.pth
"""

import json

import hydra
import librosa
import mir_eval
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.inference import SAMPLE_RATE, PianoTranscription
from gtp.stage1.postprocess import RegressionPostProcessor


def pairs(ref_onsets, ref_pitch, est_onsets, est_pitch, tolerance):
    def arrays(onsets, pitch):
        return np.c_[onsets, onsets + 0.1], librosa.midi_to_hz(pitch)

    return mir_eval.transcription.match_notes(*arrays(ref_onsets, ref_pitch), *arrays(est_onsets, est_pitch),
                                              onset_tolerance=tolerance, offset_ratio=None)  # fmt: skip


def refine(entry, est, cfg):
    on = np.array([n['onset_time'] for n in entry['notes']])
    pitch = np.array([n['midi_note'] for n in entry['notes']], float)
    est_on, est_pitch = est[:, 0], est[:, 1]
    matched = pairs(on, pitch, est_on, est_pitch, cfg.lag_window_s)
    lag = float(np.median([est_on[j] - on[i] for i, j in matched])) if matched else 0.0
    new_on = on + lag
    moved = pairs(new_on, pitch, est_on, est_pitch, cfg.max_shift_s)
    for i, j in moved:
        new_on[i] = est_on[j]
    notes = [
        {**n, 'onset_time': float(o), 'offset_time': float(max(n['offset_time'] + lag, o + 0.01))}
        for n, o in zip(entry['notes'], new_on, strict=True)
    ]
    return {**entry, 'notes': notes}, {'id': entry['id'], 'lag_ms': 1000 * lag, 'moved': len(moved) / max(len(on), 1)}


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='refine_labels')
def main(cfg):
    src, out = REPO_ROOT / cfg.src_dir, REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'audio').exists():
        (out / 'audio').symlink_to(src / 'audio')
    with open(src / 'manifest.jsonl') as fh:
        entries = [json.loads(line) for line in fh]
    model = PianoTranscription(str(REPO_ROOT / cfg.checkpoint), device=cfg.device or auto_device())
    model.batch_size = cfg.batch
    processor = RegressionPostProcessor(100, 88, onset_threshold=cfg.onset_threshold)
    refined, report = [], []
    for e in entries:
        if e['dataset'] not in cfg.datasets or not e['notes']:
            refined.append(e)
            continue
        audio, _ = librosa.load(src / e['audio'], sr=SAMPLE_RATE, mono=True)
        events, _ = processor.output_dict_to_note_events(model._run_model(audio))
        est = np.array([[v['onset_time'], v['midi_note']] for v in events], float).reshape(-1, 2)
        new, row = refine(e, est, cfg)
        refined.append(new)
        report.append(row)
    with open(out / 'manifest.jsonl', 'w') as fh:
        fh.writelines(json.dumps(e) + '\n' for e in refined)
    report = pd.DataFrame(report)
    report.to_csv(out / 'refine_report.csv', index=False)
    info(f'{len(report)} recordings refined -> {out}\n{report[["lag_ms", "moved"]].describe().round(3).to_string()}')


if __name__ == '__main__':
    main()

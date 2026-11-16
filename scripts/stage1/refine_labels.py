"""Re-time a notes manifest's labels to the audio with a Stage 1 checkpoint (src/gtp/stage1/retiming.py).

Pitches, the notes kept and the splits are unchanged, so the refined manifest trains on the same notes, better timed.

Usage:
  uv run scripts/stage1/refine_labels.py checkpoint=runs/stage1/<run>/checkpoints/step_0040000.pth
"""

import json

import hydra
import librosa
import pandas as pd

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.benchmarks import note_array
from gtp.stage1.inference import SAMPLE_RATE, PianoTranscription
from gtp.stage1.retiming import detected, retime


def refine(entry, est, cfg):
    new, lag, moved = retime(note_array(entry['notes']), est, cfg.lag_window_s, cfg.max_shift_s)
    notes = [
        {**n, 'onset_time': float(on), 'offset_time': float(off)}
        for n, (on, off, _) in zip(entry['notes'], new, strict=True)
    ]
    return {**entry, 'notes': notes}, {'id': entry['id'], 'lag_ms': 1000 * lag, 'moved': moved}


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
    refined, report = [], []
    for e in entries:
        if e['dataset'] not in cfg.datasets or not e['notes']:
            refined.append(e)
            continue
        audio, _ = librosa.load(src / e['audio'], sr=SAMPLE_RATE, mono=True)
        new, row = refine(e, detected(model, audio, cfg.onset_threshold), cfg)
        refined.append(new)
        report.append(row)
    with open(out / 'manifest.jsonl', 'w') as fh:
        fh.writelines(json.dumps(e) + '\n' for e in refined)
    report = pd.DataFrame(report)
    report.to_csv(out / 'refine_report.csv', index=False)
    info(f'{len(report)} recordings refined -> {out}\n{report[["lag_ms", "moved"]].describe().round(3).to_string()}')


if __name__ == '__main__':
    main()

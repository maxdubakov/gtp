"""Note-level precision, recall and F1 of Stage 1 checkpoints on the notes dataset, per split, dataset and chain.

Each recording is transcribed and its notes matched to the reference with mir_eval (same pitch, onsets within
`onset_tolerance`, offsets ignored), the measure the end-to-end runs use. `macro_f1` averages the groups of a split,
so the largest dataset doesn't decide alone; pick checkpoints on it for the val split.

Usage:
  uv run scripts/stage1/eval_notes.py "checkpoints=[models/finetuned/step_0070000_final.pth]"
"""

import json

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
from gtp.stage1.inference import SAMPLE_RATE, PianoTranscription


def matched(ref, est, tolerance):
    if not ref or not est:
        return 0

    def arrays(notes, on, off, pitch):
        intervals = np.array([[n[on], max(n[off], n[on] + 1e-3)] for n in notes])
        return intervals, librosa.midi_to_hz(np.array([n[pitch] for n in notes], float))

    pairs = mir_eval.transcription.match_notes(
        *arrays(ref, 'onset_time', 'offset_time', 'midi_note'), *arrays(est, 'onset_time', 'offset_time', 'midi_note'),
        onset_tolerance=tolerance, offset_ratio=None,
    )  # fmt: skip
    return len(pairs)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='eval_notes')
def main(cfg):
    device = cfg.device or auto_device()
    root = REPO_ROOT / cfg.data_dir
    with open(root / 'manifest.jsonl') as fh:
        entries = [e for e in map(json.loads, fh) if e['split'] in list(cfg.splits)]
    if cfg.max_per_group:
        frame = pd.DataFrame(
            [{'i': i, 'g': (e['split'], e['dataset'], e['chain'], e['content'])} for i, e in enumerate(entries)]
        )
        keep = frame.groupby('g').head(cfg.max_per_group)['i']
        entries = [entries[i] for i in keep]
    info(f'{len(entries)} recordings, {sum(e["duration"] for e in entries) / 3600:.1f} hours, on {device}')
    rows = []
    for ckpt in cfg.checkpoints:
        model = PianoTranscription(str(REPO_ROOT / ckpt), device=device)
        model.batch_size = cfg.batch
        for e in entries:
            audio, _ = librosa.load(root / e['audio'], sr=SAMPLE_RATE, mono=True)
            est = model.transcribe(audio)['note_events']
            rows.append({'checkpoint': ckpt, 'split': e['split'], 'dataset': e['dataset'], 'chain': e['chain'],
                         'ref': len(e['notes']), 'est': len(est), 'hit': matched(e['notes'], est, cfg.onset_tolerance)})  # fmt: skip
        info(f'{ckpt} done')
    keys = ['checkpoint', 'split', 'dataset', 'chain']
    table = pd.DataFrame(rows).groupby(keys)[['ref', 'est', 'hit']].sum().reset_index()
    table['precision'], table['recall'] = (
        table['hit'] / table['est'].clip(lower=1),
        table['hit'] / table['ref'].clip(lower=1),
    )
    table['f1'] = 2 * table['precision'] * table['recall'] / (table['precision'] + table['recall']).clip(lower=1e-12)
    macro = table.groupby(['checkpoint', 'split'])['f1'].mean().rename('macro_f1').reset_index()
    info('\n' + table.round(3).to_string(index=False))
    info('\n' + macro.round(4).to_string(index=False))
    out = HydraConfig.get().runtime.output_dir
    table.to_csv(f'{out}/groups.csv', index=False)
    macro.to_csv(f'{out}/macro.csv', index=False)
    info(f'Saved {out}')


if __name__ == '__main__':
    main()

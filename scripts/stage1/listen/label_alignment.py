"""Listening clips for judging GOAT's label timing: the recording with its labels played back as plucks.

Each clip is `clip_s` of an item's DI recording, in three label versions shown blind:
  shipped    GOAT's fine-aligned MIDI as distributed;
  ours       for test items, the shipped labels moved by one constant (label_shifts.py); for training items, our
             re-timed labels (refine_labels.py);
  unaligned  GOAT's score-time MIDI, a control that should sound clearly off.
Test items are all of GOAT's test split; training items are the ones re-timing moved most plus a random few. The
window is the stretch of the item's middle half with the most labelled notes. Writes <id>_<version>.mp3 and
clips.json (versions in a random blind order per clip).

Usage:
  uv run scripts/stage1/listen/label_alignment.py
"""

import json

import hydra
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from hydra.core.hydra_config import HydraConfig

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.benchmarks import midi_notes


def manifest_notes(path, rec):
    with open(REPO_ROOT / path / 'manifest.jsonl') as fh:
        for line in fh:
            e = json.loads(line)
            if e['id'] == rec:
                return np.array([[n['onset_time'], n['offset_time'], n['midi_note']] for n in e['notes']])
    raise KeyError(rec)


def window(notes, duration, length):
    starts = np.arange(duration * 0.25, max(duration * 0.75 - length, duration * 0.25) + 0.01, 0.5)
    counts = [((notes[:, 0] >= s) & (notes[:, 0] < s + length)).sum() for s in starts]
    return float(starts[int(np.argmax(counts))])


def plucks(notes, start, length, sr):
    out = np.zeros(int(length * sr))
    for on, off, pitch in notes:
        if not start <= on < start + length:
            continue
        f0 = librosa.midi_to_hz(pitch)
        dur = min(max(off - on, 0.08), 0.5)
        t = np.arange(int(dur * sr)) / sr
        tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in (1, 2, 3)) * np.exp(-t / 0.12)
        tone *= np.minimum(1, t / 0.002)
        i = int((on - start) * sr)
        out[i : i + len(tone)] += tone[: len(out) - i]
    return out


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='label_alignment')
def main(cfg):
    out = HydraConfig.get().runtime.output_dir
    root = REPO_ROOT / cfg.goat_dir
    meta = pd.read_csv(root / 'metadata.csv').set_index('item')
    shifts = pd.read_csv(REPO_ROOT / cfg.shifts).set_index('rec').shift_ms / 1000
    report = pd.read_csv(REPO_ROOT / cfg.refine_report)
    report = report[~report.id.str.contains('_amp_')].assign(item=lambda d: d.id.str.split('/').str[1])
    rng = np.random.default_rng(cfg.seed)
    largest = report.reindex(report.lag_ms.abs().sort_values(ascending=False).index).item[: cfg.train_largest].tolist()
    rest = report[~report.item.isin(largest)].item.tolist()
    train = largest + list(rng.choice(rest, cfg.train_random, replace=False))

    def path(column, item):
        return root / 'data' / str(meta.loc[item, column]).split('/', 1)[1]

    items = [(item, 'test') for item in meta[meta.split == 'test'].index] + [(item, 'train') for item in train]
    clips = []
    for n, (item, split) in enumerate(items):
        shipped = midi_notes(path('finealigned_midi_path', item))
        if split == 'test':
            ours = shipped + np.array([1.0, 1.0, 0.0]) * shifts.get(f'goat/{item}/{item}', 0.0)
        else:
            rec = f'goat/{item}/{item}'
            shipped = manifest_notes(cfg.shipped_manifest, rec)
            ours = manifest_notes(cfg.refined_manifest, rec)
        versions = {'shipped': shipped, 'ours': ours, 'unaligned': midi_notes(path('unaligned_midi_path', item))}
        y, _ = librosa.load(path('di_audio_path', item), sr=cfg.sr, mono=True)
        start = window(shipped, len(y) / cfg.sr, cfg.clip_s)
        seg = y[int(start * cfg.sr) : int((start + cfg.clip_s) * cfg.sr)]
        seg = 0.7 * seg / max(np.abs(seg).max(), 1e-6)
        cid = f'a{n + 1:02d}'
        for version, notes in versions.items():
            lab = plucks(notes, start, cfg.clip_s, cfg.sr)[: len(seg)]
            lab = cfg.label_gain * 0.7 * lab / max(np.abs(lab).max(), 1e-6)
            left, right = seg + (1 - cfg.label_pan) * lab, seg + lab
            stereo = np.stack([left, right], axis=1)
            stereo /= max(np.abs(stereo).max() / 0.95, 1)
            sf.write(f'{out}/{cid}_{version}.mp3', stereo.astype(np.float32), cfg.sr, format='MP3')
        order = list(rng.permutation(list(versions)))
        lag = shifts.get(f'goat/{item}/{item}', 0.0) if split == 'test' else float(
            report.set_index('item').loc[item, 'lag_ms']) / 1000  # fmt: skip
        clips.append({'id': cid, 'item': item, 'split': split, 'start': round(start, 2), 'order': order,
                      'guitar': str(meta.loc[item, 'guitar_brand']), 'our_shift_ms': round(1000 * lag, 1),
                      'alignment_f': round(float(meta.loc[item, 'alignment_f_measure_fine']), 3)})  # fmt: skip
    (REPO_ROOT / out / 'clips.json').write_text(json.dumps(clips, indent=1))
    info(f'{len(clips)} clips in {out}')


if __name__ == '__main__':
    main()

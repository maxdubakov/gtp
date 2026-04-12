"""Evaluate the pretrained Kong piano model on GuitarSet.

Metrics: onset-only precision / recall / F1 using mir_eval with 50 ms tolerance,
matching the protocol from Riley et al.

Usage:
    python scripts/eval_guitarset.py              # all 360 files
    python scripts/eval_guitarset.py -n 10        # first 10 files (quick test)
    python scripts/eval_guitarset.py -j 4         # 4 parallel workers
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import csv
import numpy as np
import librosa
import jams
import mir_eval

from gtp.inference import PianoTranscription
from gtp.log import set_verbose, trace

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
CHECKPOINT_PATH = os.path.join(REPO_ROOT, 'models', 'pretrained',
                                'CRNN_note_F1=0.9677_pedal_F1=0.9186.pth')
AUDIO_DIR = os.path.join(REPO_ROOT, 'data', 'guitarset', 'audio_mono-mic')
ANNOTATION_DIR = os.path.join(REPO_ROOT, 'data', 'guitarset', 'annotation')
RESULTS_PATH = os.path.join(REPO_ROOT, 'results', 'baseline_guitarset.csv')

MODEL_SAMPLE_RATE = 16000
ONSET_TOLERANCE = 0.05   # 50 ms, matching Riley et al.

# GuitarSet note_midi annotations are interleaved with pitch_contour annotations.
# Indices 1, 3, 5, 7, 9, 11 are the note_midi annotations for strings 1–6.
NOTE_MIDI_ANNOTATION_INDICES = [1, 3, 5, 7, 9, 11]


def load_guitarset_notes(jams_path):
    """Parse a GuitarSet JAMS file and return flat sorted note arrays.

    Returns:
      ref_intervals: (N, 2) float64 array of [onset, offset] in seconds
      ref_pitches:   (N,) float64 array of MIDI pitch (rounded to nearest semitone)
    """
    STRING_NAMES = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']

    score = jams.load(jams_path)

    trace("JAMS file", path=jams_path, total_annotations=len(score.annotations))
    trace("annotation layout:")
    for i, ann in enumerate(score.annotations):
        marker = " <-- note_midi" if i in NOTE_MIDI_ANNOTATION_INDICES else ""
        trace(f"  [{i:2d}] {ann.namespace}", observations=len(ann.data), suffix=marker)

    onsets, offsets, pitches = [], [], []

    for string_idx, ann_idx in enumerate(NOTE_MIDI_ANNOTATION_INDICES):
        ann = score.annotations[ann_idx]
        string_name = STRING_NAMES[string_idx]

        if ann.data:
            raw_example = ann.data[0]
            trace(f"string {string_name} (ann[{ann_idx}]) raw example:",
                  time=float(raw_example.time),
                  duration=float(raw_example.duration),
                  value=float(raw_example.value),
                  confidence=raw_example.confidence)

        string_onsets, string_pitches_raw, string_pitches_rounded = [], [], []
        for obs in ann.data:
            onset = float(obs.time)
            duration = float(obs.duration)
            raw_pitch = float(obs.value)
            rounded_pitch = round(raw_pitch)

            string_onsets.append(onset)
            string_pitches_raw.append(raw_pitch)
            string_pitches_rounded.append(rounded_pitch)
            onsets.append(onset)
            offsets.append(onset + duration)
            pitches.append(float(rounded_pitch))

        if string_pitches_raw:
            trace(f"string {string_name}: {len(ann.data)} notes",
                  pitch_raw_range=f"{min(string_pitches_raw):.2f}-{max(string_pitches_raw):.2f}",
                  pitch_rounded_range=f"{min(string_pitches_rounded)}-{max(string_pitches_rounded)}",
                  time_span=f"{min(string_onsets):.2f}-{max(string_onsets):.2f}s")

    sort_order = np.argsort(onsets)
    ref_intervals = np.column_stack([
        np.array(onsets)[sort_order],
        np.array(offsets)[sort_order],
    ])
    ref_pitches = np.array(pitches)[sort_order]

    trace("flattened & sorted ground truth:")
    trace("  ref_intervals", ref_intervals)
    trace("  ref_pitches", ref_pitches)
    if len(ref_intervals) > 0:
        trace("  first 3 notes:")
        for i in range(min(3, len(ref_intervals))):
            trace(f"    note {i}", onset=f"{ref_intervals[i,0]:.3f}s",
                  offset=f"{ref_intervals[i,1]:.3f}s", midi_pitch=int(ref_pitches[i]))
        trace(f"  mir_eval expects: ref_intervals=(N,2) float64, ref_pitches=(N,) float64")

    return ref_intervals, ref_pitches


def note_events_to_arrays(note_events):
    """Convert inference note_events list to mir_eval-compatible arrays.

    Returns:
      est_intervals: (N, 2) float64 array of [onset, offset]
      est_pitches:   (N,) float64 array of MIDI pitch
    """
    if not note_events:
        trace("no predicted notes")
        return np.zeros((0, 2)), np.zeros(0)

    trace("raw note_events example:", note_events[0])

    onsets = np.array([e['onset_time'] for e in note_events])
    offsets = np.array([e['offset_time'] for e in note_events])
    pitches = np.array([float(e['midi_note']) for e in note_events])

    sort_order = np.argsort(onsets)
    est_intervals = np.column_stack([onsets[sort_order], offsets[sort_order]])
    est_pitches = pitches[sort_order]

    trace("sorted predictions:")
    trace("  est_intervals", est_intervals)
    trace("  est_pitches", est_pitches)
    if len(est_intervals) > 0:
        trace("  first 3 predicted notes:")
        for i in range(min(3, len(est_intervals))):
            trace(f"    note {i}", onset=f"{est_intervals[i,0]:.3f}s",
                  offset=f"{est_intervals[i,1]:.3f}s", midi_pitch=int(est_pitches[i]))

    return est_intervals, est_pitches


def evaluate_file(transcriptor, audio_path, jams_path):
    """Run inference on one file and return (precision, recall, f1).

    The model requires 16 kHz mono audio; GuitarSet is 44.1 kHz so we resample.
    """
    trace("loading audio", path=audio_path, target_sr=MODEL_SAMPLE_RATE)
    audio, _ = librosa.load(audio_path, sr=MODEL_SAMPLE_RATE, mono=True)
    trace("resampled audio", audio)

    result = transcriptor.transcribe(audio)
    note_events = result['note_events']

    ref_intervals, ref_pitches = load_guitarset_notes(jams_path)
    est_intervals, est_pitches = note_events_to_arrays(note_events)

    trace("ground truth", ref_intervals, notes=len(ref_pitches),
          pitch_range=f"{ref_pitches.min():.0f}-{ref_pitches.max():.0f}" if len(ref_pitches) > 0 else "empty")
    trace("predictions", est_intervals, notes=len(est_pitches),
          pitch_range=f"{est_pitches.min():.0f}-{est_pitches.max():.0f}" if len(est_pitches) > 0 else "empty")

    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=None,
    )
    trace("mir_eval result", P=f"{precision:.3f}", R=f"{recall:.3f}", F1=f"{f1:.3f}")
    return precision, recall, f1


def process_one_file(args_tuple):
    """Worker: load model, run inference, evaluate one file. Returns result dict."""
    wav_name, checkpoint_path, device_str = args_tuple
    stem = wav_name.replace('_mic.wav', '')
    audio_path = os.path.join(AUDIO_DIR, wav_name)
    jams_path = os.path.join(ANNOTATION_DIR, stem + '.jams')

    if not os.path.exists(jams_path):
        return None

    # Each worker loads its own model (MPS/CUDA can't share across processes)
    transcriptor = PianoTranscription(checkpoint_path=checkpoint_path, device=device_str)
    p, r, f1 = evaluate_file(transcriptor, audio_path, jams_path)
    return {'file': stem, 'precision': p, 'recall': r, 'f1': f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=None, help='Evaluate only first N files')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='Parallel workers (each loads own model on CPU)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Trace data shapes and values through pipeline')
    parser.add_argument('--device', default=None, help='Force device (cpu/mps/cuda). Default: auto')
    args = parser.parse_args()

    if args.verbose:
        set_verbose(True)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    audio_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav'))
    if args.n:
        audio_files = audio_files[:args.n]

    # For parallel: force CPU since MPS/CUDA don't multiprocess well
    device_str = args.device or ('cpu' if args.jobs > 1 else None)
    if args.jobs > 1 and device_str != 'cpu':
        print(f'Warning: parallel mode forces CPU (MPS/CUDA cannot share across processes)')
        device_str = 'cpu'

    print(f'Files: {len(audio_files)}, Workers: {args.jobs}, Device: {device_str or "auto"}')
    t0 = time.time()

    rows = []

    if args.jobs <= 1:
        transcriptor = PianoTranscription(checkpoint_path=CHECKPOINT_PATH, device=device_str)
        print(f'Device: {transcriptor.device}\n')

        for i, wav_name in enumerate(audio_files):
            stem = wav_name.replace('_mic.wav', '')
            audio_path = os.path.join(AUDIO_DIR, wav_name)
            jams_path = os.path.join(ANNOTATION_DIR, stem + '.jams')

            if not os.path.exists(jams_path):
                continue

            p, r, f1 = evaluate_file(transcriptor, audio_path, jams_path)
            rows.append({'file': stem, 'precision': p, 'recall': r, 'f1': f1})

            elapsed = time.time() - t0
            per_file = elapsed / (i + 1)
            remaining = per_file * (len(audio_files) - i - 1)
            running_f1 = np.mean([r['f1'] for r in rows])
            print(f'  [{i+1:3d}/{len(audio_files)}] F1={f1:.3f}  avg={running_f1:.3f}  '
                  f'({per_file:.1f}s/file, ~{remaining:.0f}s left)')
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        work = [(f, CHECKPOINT_PATH, device_str) for f in audio_files]
        done = 0

        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_one_file, w): w for w in work}
            for future in as_completed(futures):
                done += 1
                result = future.result()
                if result is None:
                    continue
                rows.append(result)

                elapsed = time.time() - t0
                per_file = elapsed / done
                remaining = per_file * (len(audio_files) - done)
                running_f1 = np.mean([r['f1'] for r in rows])
                print(f'  [{done:3d}/{len(audio_files)}] {result["file"]}: '
                      f'F1={result["f1"]:.3f}  avg={running_f1:.3f}  '
                      f'({per_file:.1f}s/file, ~{remaining:.0f}s left)')

    precisions = np.array([row['precision'] for row in rows])
    recalls = np.array([row['recall'] for row in rows])
    f1s = np.array([row['f1'] for row in rows])

    print(f'\n--- Aggregate ({len(rows)} files, {time.time() - t0:.0f}s) ---')
    print(f'Precision : {precisions.mean():.4f}  (std {precisions.std():.4f})')
    print(f'Recall    : {recalls.mean():.4f}  (std {recalls.std():.4f})')
    print(f'F1        : {f1s.mean():.4f}  (std {f1s.std():.4f})')

    with open(RESULTS_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'precision', 'recall', 'f1'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['file']))
        writer.writerow({'file': 'AGGREGATE_MEAN',
                         'precision': precisions.mean(),
                         'recall': recalls.mean(),
                         'f1': f1s.mean()})

    print(f'Results saved to {RESULTS_PATH}')


if __name__ == '__main__':
    main()

"""Thorough verification that the full pipeline works end-to-end.

Checks:
  1. Directory structure and file counts for all datasets
  2. Model checkpoint loads and forward pass succeeds
  3. Inference produces note events on a GAPS file
  4. Inference produces note events on a GuitarSet file
  5. GuitarSet evaluation (2 files) produces valid P/R/F1
  6. Training loop runs 2 steps without crashing

Usage:
    python scripts/verify_setup.py           # run all checks
    python scripts/verify_setup.py --device cuda   # force device
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import torch
import librosa

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
CHECKPOINT = os.path.join(REPO_ROOT, 'models', 'pretrained',
                          'CRNN_note_F1=0.9677_pedal_F1=0.9186.pth')
GAPS_DIR = os.path.join(REPO_ROOT, 'data', 'gaps_hf')
GUITARSET_DIR = os.path.join(REPO_ROOT, 'data', 'guitarset')

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}  -- {detail}")
    return condition


def main():
    global passed, failed

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("GTP Setup Verification")
    print("=" * 60)

    # --- 1. Directory structure & file counts ---
    print("\n[1/6] Checking directory structure and datasets...")

    check("GAPS audio dir exists", os.path.isdir(os.path.join(GAPS_DIR, 'audio')))
    check("GAPS midi dir exists", os.path.isdir(os.path.join(GAPS_DIR, 'midi')))
    check("GAPS metadata exists", os.path.isfile(os.path.join(GAPS_DIR, 'gaps_metadata_with_splits.csv')))

    gaps_audio_count = len([f for f in os.listdir(os.path.join(GAPS_DIR, 'audio')) if f.endswith('.wav')])
    gaps_midi_count = len([f for f in os.listdir(os.path.join(GAPS_DIR, 'midi')) if f.endswith('.mid')])
    check(f"GAPS audio files: {gaps_audio_count}", gaps_audio_count >= 400, f"expected ~404, got {gaps_audio_count}")
    check(f"GAPS midi files: {gaps_midi_count}", gaps_midi_count >= 400, f"expected ~404, got {gaps_midi_count}")

    check("GuitarSet audio dir exists", os.path.isdir(os.path.join(GUITARSET_DIR, 'audio_mono-mic')))
    check("GuitarSet annotation dir exists", os.path.isdir(os.path.join(GUITARSET_DIR, 'annotation')))

    gs_audio_count = len([f for f in os.listdir(os.path.join(GUITARSET_DIR, 'audio_mono-mic')) if f.endswith('.wav')])
    gs_annot_count = len([f for f in os.listdir(os.path.join(GUITARSET_DIR, 'annotation')) if f.endswith('.jams')])
    check(f"GuitarSet audio files: {gs_audio_count}", gs_audio_count == 360, f"expected 360, got {gs_audio_count}")
    check(f"GuitarSet annotation files: {gs_annot_count}", gs_annot_count == 360, f"expected 360, got {gs_annot_count}")

    check("Pretrained checkpoint exists", os.path.isfile(CHECKPOINT))
    ckpt_size_mb = os.path.getsize(CHECKPOINT) / (1024 * 1024)
    check(f"Checkpoint size: {ckpt_size_mb:.0f}MB", 160 < ckpt_size_mb < 170, f"expected ~164MB, got {ckpt_size_mb:.0f}MB")

    # --- 2. Model load & forward pass ---
    print("\n[2/6] Loading model and running forward pass...")

    device_str = args.device
    if device_str is None:
        if torch.cuda.is_available():
            device_str = 'cuda'
        elif torch.backends.mps.is_available():
            device_str = 'mps'
        else:
            device_str = 'cpu'
    print(f"  Device: {device_str}")

    from gtp.inference import PianoTranscription

    t0 = time.time()
    transcriptor = PianoTranscription(checkpoint_path=CHECKPOINT, device=device_str)
    load_time = time.time() - t0
    check(f"Model loaded in {load_time:.1f}s", True)

    dummy_audio = np.zeros(16000 * 5, dtype=np.float32)
    result = transcriptor.transcribe(dummy_audio)
    check("Forward pass on silent audio", 'note_events' in result)
    check("Silent audio produces 0 notes", len(result['note_events']) == 0,
          f"got {len(result['note_events'])} notes on silence")

    # --- 3. Inference on GAPS file ---
    print("\n[3/6] Running inference on a GAPS file...")

    gaps_audio_files = sorted(os.listdir(os.path.join(GAPS_DIR, 'audio')))
    gaps_test_file = os.path.join(GAPS_DIR, 'audio', gaps_audio_files[0])
    print(f"  File: {gaps_audio_files[0]}")

    t0 = time.time()
    audio, _ = librosa.load(gaps_test_file, sr=16000, mono=True)
    duration = len(audio) / 16000
    result = transcriptor.transcribe(audio)
    inference_time = time.time() - t0

    n_notes = len(result['note_events'])
    check(f"GAPS inference: {n_notes} notes in {duration:.0f}s audio ({inference_time:.1f}s)", n_notes > 0)

    if n_notes > 0:
        pitches = [e['midi_note'] for e in result['note_events']]
        times = [e['onset_time'] for e in result['note_events']]
        check(f"  Pitch range: {min(pitches)}-{max(pitches)} (MIDI)", 20 < min(pitches) and max(pitches) < 110)
        check(f"  Time range: {min(times):.1f}-{max(times):.1f}s", max(times) <= duration + 1)

    # --- 4. Inference on GuitarSet file ---
    print("\n[4/6] Running inference on a GuitarSet file...")

    gs_audio_files = sorted(os.listdir(os.path.join(GUITARSET_DIR, 'audio_mono-mic')))
    gs_test_file = os.path.join(GUITARSET_DIR, 'audio_mono-mic', gs_audio_files[0])
    print(f"  File: {gs_audio_files[0]}")

    t0 = time.time()
    audio, _ = librosa.load(gs_test_file, sr=16000, mono=True)
    duration = len(audio) / 16000
    result = transcriptor.transcribe(audio)
    inference_time = time.time() - t0

    n_notes = len(result['note_events'])
    check(f"GuitarSet inference: {n_notes} notes in {duration:.0f}s audio ({inference_time:.1f}s)", n_notes > 0)

    if n_notes > 0:
        pitches = [e['midi_note'] for e in result['note_events']]
        check(f"  Pitch range: {min(pitches)}-{max(pitches)} (MIDI)", 20 < min(pitches) and max(pitches) < 110)

    # --- 5. GuitarSet evaluation (2 files) ---
    print("\n[5/6] Running mir_eval on 2 GuitarSet files...")

    import jams
    import mir_eval

    NOTE_MIDI_INDICES = [1, 3, 5, 7, 9, 11]
    f1_scores = []

    for wav_name in gs_audio_files[:2]:
        stem = wav_name.replace('_mic.wav', '')
        audio_path = os.path.join(GUITARSET_DIR, 'audio_mono-mic', wav_name)
        jams_path = os.path.join(GUITARSET_DIR, 'annotation', stem + '.jams')

        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        result = transcriptor.transcribe(audio)

        score = jams.load(jams_path)
        ref_onsets, ref_offsets, ref_pitches = [], [], []
        for idx in NOTE_MIDI_INDICES:
            for obs in score.annotations[idx].data:
                ref_onsets.append(float(obs.time))
                ref_offsets.append(float(obs.time) + float(obs.duration))
                ref_pitches.append(float(round(float(obs.value))))

        order = np.argsort(ref_onsets)
        ref_intervals = np.column_stack([np.array(ref_onsets)[order], np.array(ref_offsets)[order]])
        ref_pitches = np.array(ref_pitches)[order]

        events = result['note_events']
        if events:
            est_onsets = np.array([e['onset_time'] for e in events])
            est_offsets = np.array([e['offset_time'] for e in events])
            est_pitches = np.array([float(e['midi_note']) for e in events])
            order = np.argsort(est_onsets)
            est_intervals = np.column_stack([est_onsets[order], est_offsets[order]])
            est_pitches_sorted = est_pitches[order]
        else:
            est_intervals = np.zeros((0, 2))
            est_pitches_sorted = np.zeros(0)

        p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, est_intervals, est_pitches_sorted,
            onset_tolerance=0.05, offset_ratio=None,
        )
        f1_scores.append(f1)
        print(f"  {stem}: P={p:.3f} R={r:.3f} F1={f1:.3f} (ref={len(ref_pitches)} est={len(events)})")

    avg_f1 = np.mean(f1_scores)
    check(f"Average F1: {avg_f1:.3f}", avg_f1 > 0.1, "F1 suspiciously low")
    check(f"F1 in expected baseline range", 0.2 < avg_f1 < 0.8,
          f"expected 0.2-0.8 for pretrained piano on guitar, got {avg_f1:.3f}")

    # --- 6. Training loop (2 steps) ---
    print("\n[6/6] Running 2 training steps...")

    from gtp.model.kong import Regress_onset_offset_frame_velocity_CRNN
    from gtp.model.losses import regress_onset_offset_frame_velocity_bce
    from gtp.model.utils import move_data_to_device
    from gtp.data import build_dataset
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        keys = batch[0].keys()
        return {k: torch.from_numpy(np.stack([item[k] for item in batch])) for k in keys}

    model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=100, classes_num=88)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model']['note_model'])
    model.to(device_str)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_ds = build_dataset(GAPS_DIR, GUITARSET_DIR, split='train')
    check(f"Training dataset: {len(train_ds)} segments", len(train_ds) > 50000)

    loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0,
                        collate_fn=collate_fn, drop_last=True)

    losses = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        for key in batch:
            batch[key] = move_data_to_device(batch[key], device_str)
        output = model(batch['waveform'])
        loss = regress_onset_offset_frame_velocity_bce(model, output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {i+1}: loss={loss.item():.4f}")

    train_time = time.time() - t0
    check(f"2 training steps completed in {train_time:.1f}s", len(losses) == 2)
    check(f"Loss is finite", all(np.isfinite(l) for l in losses), f"losses={losses}")

    # --- Summary ---
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All checks passed! Ready to train.")
    else:
        print("Some checks failed — review output above.")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()

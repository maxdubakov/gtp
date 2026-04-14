"""Generate listenable verification WAVs for interesting GuitarSet predictions.

For each picked file, produces two outputs:
  - stereo WAV: LEFT = original audio, RIGHT = onset blips at predicted onsets
  - guitar WAV: MIDI synthesized with guitar voice (MuseScore soundfont)

Usage:
    python scripts/listen_to_predictions.py --checkpoint models/finetuned/step_0070000_final.pth
"""

import sys
import os
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import pretty_midi
import librosa
import soundfile as sf

from gtp.inference import PianoTranscription

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
AUDIO_DIR = os.path.join(REPO_ROOT, 'data', 'guitarset', 'audio_mono-mic')
ANNOTATION_DIR = os.path.join(REPO_ROOT, 'data', 'guitarset', 'annotation')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'listen')
SOUNDFONT_PATH = os.path.join(REPO_ROOT, 'models', 'soundfonts', 'ms_basic.sf3')
GUITAR_PROGRAM = 24  # General MIDI: 24 = Acoustic Guitar (nylon)
MODEL_SR = 16000
LISTEN_SR = 22050


def synth_midi_onsets(note_events, duration, sr=LISTEN_SR):
    """Synthesize short sine blips at each predicted note onset."""
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    for event in note_events:
        onset_sample = int(event['onset_time'] * sr)
        blip_samples = int(0.05 * sr)
        freq = pretty_midi.note_number_to_hz(event['midi_note'])
        t = np.arange(blip_samples) / sr
        envelope = np.exp(-t * 40)
        blip = 0.3 * np.sin(2 * np.pi * freq * t) * envelope

        end = min(onset_sample + blip_samples, n_samples)
        if onset_sample < 0 or end <= onset_sample:
            continue
        actual_len = end - onset_sample
        audio[onset_sample:end] += blip[:actual_len]

    return np.clip(audio, -1.0, 1.0)


def write_stereo_verification(audio_path, note_events, output_path):
    """Write a stereo WAV: left=audio, right=predicted onset blips."""
    audio, _ = librosa.load(audio_path, sr=LISTEN_SR, mono=True)
    duration = len(audio) / LISTEN_SR
    blips = synth_midi_onsets(note_events, duration, sr=LISTEN_SR)

    min_len = min(len(audio), len(blips))
    audio = audio[:min_len]
    blips = blips[:min_len]
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8

    stereo = np.stack([audio, blips], axis=-1)
    sf.write(output_path, stereo, LISTEN_SR)


def write_guitar_synth(note_events, output_path, sr=LISTEN_SR):
    """Synthesize note events with a guitar voice using fluidsynth."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=GUITAR_PROGRAM)
    for event in note_events:
        instrument.notes.append(pretty_midi.Note(
            velocity=max(30, min(127, event.get('velocity', 80))),
            pitch=int(event['midi_note']),
            start=float(event['onset_time']),
            end=max(float(event['onset_time']) + 0.05, float(event['offset_time'])),
        ))
    midi.instruments.append(instrument)

    audio = midi.fluidsynth(fs=sr, sf2_path=SOUNDFONT_PATH)
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
    sf.write(output_path, audio, sr)


def pick_files(csv_path, player_filter=None):
    """Read eval CSV, return sorted list of (filename, p, r, f1). Skip aggregate row."""
    rows = []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            if row['file'].startswith('AGGREGATE'):
                continue
            if player_filter and not row['file'].startswith(player_filter):
                continue
            rows.append({
                'file': row['file'],
                'precision': float(row['precision']),
                'recall': float(row['recall']),
                'f1': float(row['f1']),
            })
    rows.sort(key=lambda r: r['f1'])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Fine-tuned checkpoint path')
    parser.add_argument('--all-csv', default=os.path.join(REPO_ROOT, 'results', 'finetuned_guitarset.csv'),
                        help='Eval CSV for all 360 files')
    parser.add_argument('--player5-csv', default=os.path.join(REPO_ROOT, 'results', 'finetuned_player05.csv'),
                        help='Eval CSV for player 5 only')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    transcriptor = PianoTranscription(checkpoint_path=args.checkpoint, device=args.device)
    print(f"Device: {transcriptor.device}\n")

    # --- Pick files ---
    all_rows = pick_files(args.all_csv)
    player5_rows = pick_files(args.player5_csv)

    targets = []
    # 2 worst from all files
    for row in all_rows[:2]:
        targets.append(('worst', row))
    # 2 best from all files
    for row in all_rows[-2:]:
        targets.append(('best', row))
    # 2 from player 5 (held-out)
    for row in player5_rows[:1] + player5_rows[-1:]:
        targets.append(('player5', row))

    print("Files to process:")
    for category, row in targets:
        print(f"  [{category}] {row['file']:<40} F1={row['f1']:.3f}")
    print()

    for category, row in targets:
        stem = row['file']
        audio_path = os.path.join(AUDIO_DIR, f'{stem}_mic.wav')

        if not os.path.exists(audio_path):
            print(f"  Skipping {stem}: audio not found")
            continue

        audio, _ = librosa.load(audio_path, sr=MODEL_SR, mono=True)
        result = transcriptor.transcribe(audio)
        note_events = result['note_events']

        base_name = f'{category}__{stem}__f1={row["f1"]:.3f}'
        stereo_path = os.path.join(OUTPUT_DIR, f'{base_name}__overlay.wav')
        guitar_path = os.path.join(OUTPUT_DIR, f'{base_name}__guitar.wav')

        write_stereo_verification(audio_path, note_events, stereo_path)
        write_guitar_synth(note_events, guitar_path)

        print(f"  {category:<8} {stem:<40} F1={row['f1']:.3f} -> {base_name}__(overlay|guitar).wav")

    print(f"\nDone. Listen with headphones:")
    print(f"  LEFT = guitar audio, RIGHT = MIDI blips at predicted onsets")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

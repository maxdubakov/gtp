"""Transcribe any audio file (wav, m4a, mp3) with the fine-tuned model.

Outputs:
  - <stem>_predicted.mid     -- MIDI file of the transcription
  - <stem>_overlay.wav       -- stereo: left=original, right=onset blips
  - <stem>_guitar.wav        -- MIDI synthesized as acoustic guitar
"""

import argparse
import os

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from gtp import REPO_ROOT
from gtp.stage1.inference import PianoTranscription

DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, 'models', 'finetuned', 'step_0070000_final.pth')
SOUNDFONT_PATH = os.path.join(REPO_ROOT, 'models', 'soundfonts', 'ms_basic.sf3')
GUITAR_PROGRAM = 24  # General MIDI: Acoustic Guitar (nylon)
MODEL_SR = 16000
OUTPUT_SR = 22050


def synth_onset_blips(note_events, duration, sr):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', help='Path to audio file (wav, m4a, mp3, ...)')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: next to input file)')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = args.output_dir or os.path.dirname(audio_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    transcriptor = PianoTranscription(checkpoint_path=args.checkpoint, device=args.device)
    print(f"Device: {transcriptor.device}")

    print(f"\nLoading audio: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=MODEL_SR, mono=True)
    duration = len(audio) / MODEL_SR
    print(f"Duration: {duration:.1f}s")

    print("Transcribing...")
    midi_path = os.path.join(output_dir, f'{stem}_predicted.mid')
    result = transcriptor.transcribe(audio, midi_path=midi_path)
    note_events = result['note_events']
    print(f"Detected {len(note_events)} notes")
    print(f"  MIDI written: {midi_path}")

    # Overlay: original + blips
    audio_listen, _ = librosa.load(audio_path, sr=OUTPUT_SR, mono=True)
    blips = synth_onset_blips(note_events, len(audio_listen) / OUTPUT_SR, sr=OUTPUT_SR)
    min_len = min(len(audio_listen), len(blips))
    audio_listen = audio_listen[:min_len] / (np.max(np.abs(audio_listen)) + 1e-8) * 0.8
    stereo = np.stack([audio_listen, blips[:min_len]], axis=-1)
    overlay_path = os.path.join(output_dir, f'{stem}_overlay.wav')
    sf.write(overlay_path, stereo, OUTPUT_SR)
    print(f"  Overlay written: {overlay_path}")

    # Guitar-synthesized MIDI
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
    guitar_audio = midi.fluidsynth(fs=OUTPUT_SR, sf2_path=SOUNDFONT_PATH)
    guitar_audio = guitar_audio / (np.max(np.abs(guitar_audio)) + 1e-8) * 0.8
    guitar_path = os.path.join(output_dir, f'{stem}_guitar.wav')
    sf.write(guitar_path, guitar_audio, OUTPUT_SR)
    print(f"  Guitar synth written: {guitar_path}")


if __name__ == '__main__':
    main()

"""Produce a listenable verification artifact for one transcribed audio file.

Output: stereo WAV where left channel = original audio, right channel = synthesized
onset blips at predicted note onsets. Also saves the predicted MIDI file.
"""

import os


import argparse
import numpy as np
import librosa
import soundfile as sf

from gtp.inference import PianoTranscription

from gtp import REPO_ROOT
CHECKPOINT_PATH = os.path.join(REPO_ROOT, 'models', 'pretrained',
                                'CRNN_note_F1=0.9677_pedal_F1=0.9186.pth')
MODEL_SAMPLE_RATE = 16000

# Short 440 Hz sine blip placed at each predicted onset in the right channel
BLIP_DURATION_SEC = 0.02
BLIP_FREQ_HZ = 440.0


def make_blip(sample_rate, duration=BLIP_DURATION_SEC, freq=BLIP_FREQ_HZ):
    """Return a short sine-wave blip, windowed to avoid clicks."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    blip = np.sin(2 * np.pi * freq * t)
    window = np.hanning(len(blip))
    return (blip * window).astype(np.float32)


def build_onset_track(note_events, total_samples, sample_rate):
    """Build a mono audio track with a blip at each predicted onset."""
    track = np.zeros(total_samples, dtype=np.float32)
    blip = make_blip(sample_rate)
    blip_len = len(blip)

    for event in note_events:
        onset_sample = int(event['onset_time'] * sample_rate)
        end_sample = min(onset_sample + blip_len, total_samples)
        copy_len = end_sample - onset_sample
        if copy_len > 0:
            track[onset_sample:end_sample] += blip[:copy_len]

    # Normalize to prevent clipping
    peak = np.max(np.abs(track))
    if peak > 1.0:
        track /= peak
    return track


def verify_file(audio_path, out_dir, transcriptor):
    """Run transcription on one file and write verification WAV + MIDI.

    Args:
      audio_path: path to input WAV (any sample rate, mono or stereo)
      out_dir: directory to write outputs
      transcriptor: PianoTranscription instance (shared across calls)

    Returns:
      note_events: list of predicted note event dicts
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(audio_path))[0]

    # Load original audio at native sample rate for the left channel
    original_audio, native_sr = librosa.load(audio_path, sr=None, mono=True)

    # Resample to model rate for inference
    model_audio = librosa.resample(original_audio, orig_sr=native_sr, target_sr=MODEL_SAMPLE_RATE)

    result = transcriptor.transcribe(model_audio)
    note_events = result['note_events']
    pedal_events = result['pedal_events']

    print(f'  {stem}: {len(note_events)} notes detected')

    # Save MIDI
    midi_path = os.path.join(out_dir, f'{stem}_predicted.mid')
    from gtp.postprocess import write_events_to_midi
    write_events_to_midi(note_events, midi_path, pedal_events=pedal_events)
    print(f'  MIDI -> {midi_path}')

    # Build onset-blip track at native sample rate for temporal alignment
    # (resample model-rate onsets to native-sr sample positions)
    onset_track = build_onset_track(note_events, len(original_audio), native_sr)

    # Stereo: left = original, right = onset blips
    stereo = np.stack([original_audio, onset_track], axis=1)
    wav_path = os.path.join(out_dir, f'{stem}_verification.wav')
    sf.write(wav_path, stereo, native_sr)
    print(f'  WAV  -> {wav_path}')

    return note_events


def main():
    parser = argparse.ArgumentParser(description='Transcription verification artifact')
    parser.add_argument('--audio', type=str, default=None,
                        help='Path to audio file. If omitted, runs on default files.')
    parser.add_argument('--out', type=str, default=os.path.join(REPO_ROOT, 'results', 'verify'),
                        help='Output directory')
    args = parser.parse_args()

    transcriptor = PianoTranscription(checkpoint_path=CHECKPOINT_PATH)
    print(f'Device: {transcriptor.device}\n')

    if args.audio:
        verify_file(args.audio, args.out, transcriptor)
    else:
        # Default: one GuitarSet file + one GAPS file
        default_files = [
            os.path.join(REPO_ROOT, 'data', 'guitarset', 'audio_mono-mic',
                         '00_BN1-129-Eb_comp_mic.wav'),
            os.path.join(REPO_ROOT, 'data', 'gaps_hf', 'audio', '001_mvswc.wav'),
        ]
        for audio_path in default_files:
            if os.path.exists(audio_path):
                print(f'Processing: {audio_path}')
                verify_file(audio_path, args.out, transcriptor)
                print()
            else:
                print(f'[SKIP] not found: {audio_path}')


if __name__ == '__main__':
    main()

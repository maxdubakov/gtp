"""Finalize alphaTab-extracted Leduc piece JSON: estimate real tempo from embedded MP3, rescale, write MIDI.

Args: <json_path> <midi_path> <gp_path>

JSON in: alphaTab-emitted dict with note times at the score's effective tempo.
JSON out (overwrites): same fields, but:
  - if the GP zip has an embedded MP3 backing track, run librosa beat_track →
    treat that as the real tempo. Rescale all note times so they're at the real tempo,
    set `tempo` field to the real tempo (rounded).
  - otherwise: leave note times at 120 BPM density (rescaling from the score's
    effective tempo if it differs), set `tempo` to None.

The "tempo unknown" case is signaled by `tempo: null` in JSON. Downstream
(tokenizer.notes_to_encoder_tokens) omits the TEMPO conditioning token in that case.
"""

import json
import os
import sys
import tempfile
import zipfile

import librosa
import numpy as np
import pretty_midi

DEFAULT_TUNING = [64, 59, 55, 50, 45, 40]
DEFAULT_TIMING_BPM = 120  # what we normalize to when real tempo is unknown
TEMPO_MIN = 40
TEMPO_MAX = 240
GUITAR_PROGRAM = 24


def extract_mp3_tempo(gp_path):
    """Try to extract the first embedded MP3 from a GP zip and run librosa beat_track.
    Returns (tempo_bpm, None) on success, (None, reason_str) otherwise.
    """
    try:
        with zipfile.ZipFile(gp_path) as zf:
            mp3_names = [n for n in zf.namelist() if n.lower().endswith('.mp3')]
            if not mp3_names:
                return None, 'no_mp3'
            with tempfile.TemporaryDirectory() as td:
                mp3_path = os.path.join(td, 'audio.mp3')
                with open(mp3_path, 'wb') as fh:
                    fh.write(zf.read(mp3_names[0]))
                y, sr = librosa.load(mp3_path, sr=22050, mono=True)
                if len(y) < sr * 2:
                    return None, 'mp3_too_short'
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(np.atleast_1d(tempo)[0])
                if not (TEMPO_MIN <= tempo <= TEMPO_MAX):
                    return None, f'tempo_out_of_range_{tempo:.1f}'
                return tempo, None
    except Exception as e:
        return None, f'err_{type(e).__name__}'


def main():
    if len(sys.argv) != 4:
        print('Usage: postprocess.py <json> <midi> <gp_file>', file=sys.stderr)
        sys.exit(2)

    json_path, midi_path, gp_path = sys.argv[1:]

    with open(json_path) as f:
        data = json.load(f)

    if 'tuning' not in data and data.get('tracks'):
        data['tuning'] = data['tracks'][0].get('tuning', DEFAULT_TUNING)
    tuning = data.get('tuning', DEFAULT_TUNING)

    # Sanity check: pieces where every note's pitch is inconsistent with (string, fret)
    # are corrupt — drop entirely (matches the existing behavior).
    bad = sum(
        1
        for n in data['notes']
        if n['string'] < 1 or n['string'] > len(tuning) or tuning[n['string'] - 1] + n['fret'] != n['pitch']
    )
    if bad == len(data['notes']) and len(data['notes']) > 0:
        os.remove(json_path)
        sys.exit(1)

    alphatab_tempo = data.get('tempo', DEFAULT_TIMING_BPM) or DEFAULT_TIMING_BPM
    real_tempo, reason = extract_mp3_tempo(gp_path)
    target_tempo = real_tempo if real_tempo is not None else DEFAULT_TIMING_BPM

    if alphatab_tempo != target_tempo and data['notes']:
        scale = alphatab_tempo / target_tempo
        for n in data['notes']:
            n['start'] = round(n['start'] * scale, 4)
            n['end'] = round(n['end'] * scale, 4)

    data['tempo'] = round(real_tempo, 2) if real_tempo is not None else None

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    midi_tempo = real_tempo if real_tempo is not None else DEFAULT_TIMING_BPM
    midi = pretty_midi.PrettyMIDI(initial_tempo=midi_tempo)
    inst = pretty_midi.Instrument(program=GUITAR_PROGRAM)
    for n in data['notes']:
        inst.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=int(n['pitch']),
                start=float(n['start']),
                end=float(max(n['start'] + 0.01, n['end'])),
            )
        )
    midi.instruments.append(inst)
    midi.write(midi_path)

    if real_tempo is None and reason:
        # informational; not an error
        print(f'  [tempo unknown] {os.path.basename(gp_path)} ({reason})', file=sys.stderr)


if __name__ == '__main__':
    main()

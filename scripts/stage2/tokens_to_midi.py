"""Round-trip a piece through encoder/decoder tokens and write MIDI files.

Verifies tokenizer correctness by ear: enc.mid should sound nearly identical to orig.mid
(modulo tick quantization). dec.mid uses a fixed sustain since decoder tokens carry no duration —
expect it to sound rhythmically right but more 'plucky'.

Usage:
  python tokens_to_midi.py
  python tokens_to_midi.py --source dadagp
  python tokens_to_midi.py --file 00_BN3-154-E_comp.json
  python tokens_to_midi.py --output-dir /tmp/midi_check
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from gtp import REPO_ROOT
from gtp.stage2.data import PROCESSED_DIRS, filter_notes
from gtp.stage2.tokenizer import (
    decoder_tokens_to_notes,
    encoder_tokens_to_notes,
    notes_to_decoder_tokens,
    notes_to_encoder_tokens,
)

GUITAR_PROGRAM = 24
DEFAULT_OUTPUT = REPO_ROOT / 'results' / 'tokenizer_roundtrip'
SOUNDFONT_PATH = REPO_ROOT / 'models' / 'soundfonts' / 'ms_basic.sf3'
WAV_SR = 22050


def notes_to_midi(notes, path, tempo):
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    inst = pretty_midi.Instrument(program=GUITAR_PROGRAM)
    for n in notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=int(n['pitch']),
                start=float(n['start']),
                end=float(max(n['end'], n['start'] + 0.05)),
            )
        )
    midi.instruments.append(inst)
    midi.write(str(path))
    return midi


def synth_wav(midi, path):
    audio = midi.fluidsynth(fs=WAV_SR, sf2_path=str(SOUNDFONT_PATH))
    peak = np.max(np.abs(audio)) + 1e-8
    audio = audio / peak * 0.8
    sf.write(str(path), audio, WAV_SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='guitarset', choices=list(PROCESSED_DIRS.keys()))
    ap.add_argument('--file', default=None)
    ap.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    ap.add_argument('--default-dur', type=float, default=0.3, help='Decoder reconstruction note length (seconds)')
    ap.add_argument('--no-wav', action='store_true', help='Skip WAV synthesis (MIDI only)')
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    src_dir = PROCESSED_DIRS[args.source]
    if args.file:
        path = src_dir / args.file
    else:
        candidates = sorted(src_dir.glob('*.json'))
        if not candidates:
            raise SystemExit(f'No JSON files in {src_dir}')
        path = random.choice(candidates)

    with open(path) as fh:
        data = json.load(fh)

    tuning = data.get('tuning', [64, 59, 55, 50, 45, 40])
    raw_notes = data.get('notes', [])
    notes, _ = filter_notes(raw_notes, tuning)
    tempo = data.get('tempo', 120)
    capo = data.get('capo', 0)
    notes = sorted(notes, key=lambda n: (n['start'], n['pitch']))

    enc_tokens = notes_to_encoder_tokens(notes, tempo, tuning, capo)
    dec_tokens = notes_to_decoder_tokens(notes, tempo)
    enc_strs = [str(t) for t in enc_tokens]
    dec_strs = [str(t) for t in dec_tokens]

    enc_notes, enc_tempo, _enc_capo, _enc_tuning = encoder_tokens_to_notes(enc_strs)
    dec_notes = decoder_tokens_to_notes(dec_strs, tempo, tuning)
    # decoder output has no `end` — synthesize a fixed sustain just for MIDI rendering
    dec_notes_for_midi = [{**n, 'end': n['start'] + args.default_dur} for n in dec_notes]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    paths = {
        'orig': out_dir / f'{stem}_orig.mid',
        'enc': out_dir / f'{stem}_enc.mid',
        'dec': out_dir / f'{stem}_dec.mid',
    }

    midis = {
        'orig': notes_to_midi(notes, paths['orig'], tempo),
        'enc': notes_to_midi(enc_notes, paths['enc'], enc_tempo),
        'dec': notes_to_midi(dec_notes_for_midi, paths['dec'], tempo),
    }

    if not args.no_wav:
        if not SOUNDFONT_PATH.exists():
            print(f'Warning: soundfont missing at {SOUNDFONT_PATH}, skipping WAV')
        else:
            for tag, midi in midis.items():
                wav_path = paths[tag].with_suffix('.wav')
                synth_wav(midi, wav_path)

    print(f'Piece: {args.source}/{path.name}')
    print(f'  notes: {len(notes)}  tempo={tempo}  capo={capo}  tuning={tuning}')
    print(f'  encoder: {len(enc_tokens)} tokens → {len(enc_notes)} notes back')
    print(f'  decoder: {len(dec_tokens)} tokens → {len(dec_notes)} notes back  (default_dur={args.default_dur}s)')
    print()
    print(f'Output dir: {out_dir}')
    for tag, p in paths.items():
        wav_p = p.with_suffix('.wav')
        line = f'  {tag}: {os.path.relpath(p, REPO_ROOT)}'
        if not args.no_wav and wav_p.exists():
            line += f'  +  {os.path.relpath(wav_p, REPO_ROOT)}'
        print(line)


if __name__ == '__main__':
    main()

"""Listen to tokenized piece to assess by ear that tokenizer works"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from gtp import REPO_ROOT
from gtp.stage2.data import filter_notes
from gtp.stage2.paths import PROCESSED_DIRS, SOUNDFONT_PATH
from gtp.stage2.tokenizer import (
    Vocabulary,
    decoder_tokens_to_notes,
    encoder_tokens_to_notes,
    notes_to_decoder_tokens,
    notes_to_encoder_tokens,
)

GUITAR_PROGRAM = 24
DEFAULT_OUTPUT = REPO_ROOT / 'results' / 'tokenizer'
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
    ap.add_argument('--source', required=True, choices=list(PROCESSED_DIRS.keys()))
    ap.add_argument(
        '--file',
        required=True,
    )
    ap.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    path = PROCESSED_DIRS[args.source] / args.file
    with open(path) as fh:
        data = json.load(fh)

    tuning = data.get('tuning', [64, 59, 55, 50, 45, 40])
    raw_notes = data.get('notes', [])
    notes, _ = filter_notes(raw_notes, tuning)
    tempo = data.get('tempo', 120)
    tempo_fallback = tempo if tempo is not None else 120
    capo = data.get('capo', 0)
    notes = sorted(notes, key=lambda n: (n['start'], n['pitch']))

    vocab = Vocabulary(include_genre=True)
    enc_tokens = notes_to_encoder_tokens(notes, tempo, tuning, capo)
    dec_tokens = notes_to_decoder_tokens(notes, tempo_fallback)
    enc_ids = [vocab.encode(t) for t in enc_tokens]
    dec_ids = [vocab.encode(t) for t in dec_tokens]

    enc_notes, enc_tempo, _enc_capo, _enc_tuning = encoder_tokens_to_notes(enc_ids, vocab)
    dec_notes = decoder_tokens_to_notes(dec_ids, vocab, tempo_fallback, tuning)
    # decoder output has no `end`, assume it's always 0.4s
    dec_notes_for_midi = [{**n, 'end': n['start'] + 0.4} for n in dec_notes]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    paths = {
        'orig': out_dir / f'{stem}_original.mid',
        'enc': out_dir / f'{stem}_encoder.mid',
        'dec': out_dir / f'{stem}_decoder.mid',
    }

    midis = {
        'orig': notes_to_midi(notes, paths['orig'], tempo_fallback),
        'enc': notes_to_midi(enc_notes, paths['enc'], enc_tempo),
        'dec': notes_to_midi(dec_notes_for_midi, paths['dec'], tempo_fallback),
    }

    for tag, midi in midis.items():
        wav_path = paths[tag].with_suffix('.wav')
        synth_wav(midi, wav_path)

    fields = {
        'Piece': f'{args.source}/{path.name}',
        'Tempo': tempo,
        'Capo': capo,
        'Tuning': tuning,
        'Notes': len(notes),
        'Encoder': f'{len(enc_tokens)} tokens, got {len(enc_notes)} notes back',
        'Decoder': f'{len(dec_tokens)} tokens, got {len(dec_notes)} notes back',
    }
    w = max(len(k) for k in fields)
    for k, v in fields.items():
        print(f'  {k:<{w}}  {v}')

    print()
    print(f'Output dir: {out_dir}')
    tw = max(len(t) for t in paths)
    for tag, p in paths.items():
        wav_p = p.with_suffix('.wav')
        line = f'  {tag:<{tw}}  {os.path.relpath(p, REPO_ROOT)}'
        if wav_p.exists():
            line += f'  +  {os.path.relpath(wav_p, REPO_ROOT)}'
        print(line)


if __name__ == '__main__':
    main()

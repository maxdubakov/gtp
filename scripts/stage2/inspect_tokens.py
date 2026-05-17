"""Print tokenized encoder/decoder sequences for a piece, alongside the source notes"""

import argparse
import json
import textwrap

from gtp.stage2.data import filter_notes
from gtp.stage2.paths import PROCESSED_DIRS
from gtp.stage2.tokenizer import Vocabulary, tokenize_piece


def _render_tokens(ids, vocab):
    shown = list(ids)
    suffix = ''
    text = ' '.join(vocab.decode(t) for t in shown) + suffix
    return textwrap.indent(textwrap.fill(text, width=110), '  ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, choices=list(PROCESSED_DIRS.keys()))
    ap.add_argument('--file', required=True, help='Specific JSON filename from processed directory')
    ap.add_argument('--genre-conditioning', action='store_true', help='Enable genre conditioning')
    args = ap.parse_args()

    vocab = Vocabulary(include_genre=args.genre_conditioning)
    path = PROCESSED_DIRS[args.source] / args.file
    with open(path) as fh:
        data = json.load(fh)

    tuning = data.get('tuning', [64, 59, 55, 50, 45, 40])
    raw_notes = data.get('notes', [])
    notes, _reasons = filter_notes(raw_notes, tuning)
    tempo = data.get('tempo', 120)
    capo = data.get('capo', 0)

    piece = {'tuning': tuning, 'tempo': tempo, 'capo': capo, 'genre': data.get('genre', 'unknown'), 'notes': notes}
    sequences = tokenize_piece(piece, vocab)

    fields = {
        'Piece': f'{args.source}/{path.name}',
        'Tempo': tempo,
        'Capo': capo,
        'Tuning': tuning,
        'Notes (before filter)': len(raw_notes),
        'Notes (after filter)': len(notes),
        'Sequences': len(sequences),
    }
    w = max(len(k) for k in fields)
    for k, v in fields.items():
        print(f'  {k:<{w}}  {v}')

    for i in range(len(sequences)):
        enc_ids, dec_ids = sequences[i]
        print()
        print(f'--- Sequence {i}  enc_len={len(enc_ids)}  dec_len={len(dec_ids)} ---')
        print('ENCODER:')
        print(_render_tokens(enc_ids, vocab))
        print('DECODER:')
        print(_render_tokens(dec_ids, vocab))


if __name__ == '__main__':
    main()

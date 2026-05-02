"""Print tokenized encoder/decoder sequences for a piece, alongside the source notes.

Usage:
  python inspect_tokens.py (random guitarset piece, first sequence)
  python inspect_tokens.py --source dadagp
  python inspect_tokens.py --source leduc --file foo.json
  python inspect_tokens.py --all-seqs --head 60
"""

import argparse
import json
import random
import textwrap

from gtp.stage2.data import filter_notes
from gtp.stage2.paths import PROCESSED_DIRS
from gtp.stage2.tokenizer import VOCAB, tokenize_piece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='guitarset', choices=list(PROCESSED_DIRS.keys()))
    ap.add_argument('--file', default=None, help='Specific JSON filename (default: random)')
    ap.add_argument('--seq', type=int, default=0, help='Sequence index to print (default: 0)')
    ap.add_argument('--all-seqs', action='store_true', help='Print every sequence in the piece')
    ap.add_argument('--head', type=int, default=0, help='Truncate token list to first N (0 = no truncation)')
    ap.add_argument('--n-notes', type=int, default=8, help='How many raw notes to show for context')
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
    notes, reasons = filter_notes(raw_notes, tuning)
    tempo = data.get('tempo', 120)
    capo = data.get('capo', 0)

    piece = {'tuning': tuning, 'tempo': tempo, 'capo': capo, 'notes': notes}
    sequences = tokenize_piece(piece)

    print(f'Piece: {args.source}/{path.name}')
    print(f'  tempo={tempo}  capo={capo}  tuning={tuning}')
    print(f'  notes raw→clean: {len(raw_notes)} → {len(notes)}' + (f'  ({dict(reasons)})' if reasons else ''))
    print(f'  sequences: {len(sequences)}')

    if args.n_notes > 0 and notes:
        print()
        print(f'First {min(args.n_notes, len(notes))} notes (after filter, sorted by start, pitch):')
        sorted_notes = sorted(notes, key=lambda n: (n['start'], n['pitch']))
        for n in sorted_notes[: args.n_notes]:
            print(
                f'  start={n["start"]:7.3f}  end={n["end"]:7.3f}  '
                f'pitch={n["pitch"]:3d}  string={n["string"]}  fret={n["fret"]:2d}'
            )

    indices = range(len(sequences)) if args.all_seqs else [args.seq]
    for i in indices:
        if i < 0 or i >= len(sequences):
            print(f'\n(sequence {i} out of range; piece has {len(sequences)})')
            continue
        enc_ids, dec_ids = sequences[i]
        print()
        print(f'--- Sequence {i}  enc_len={len(enc_ids)}  dec_len={len(dec_ids)} ---')
        print('ENCODER:')
        print(_render_tokens(enc_ids, args.head))
        print('DECODER:')
        print(_render_tokens(dec_ids, args.head))


def _render_tokens(ids, head):
    if 0 < head < len(ids):
        shown = list(ids[:head])
        suffix = f' … ({len(ids) - head} more)'
    else:
        shown = list(ids)
        suffix = ''
    text = ' '.join(VOCAB.decode(t) for t in shown) + suffix
    return textwrap.indent(textwrap.fill(text, width=110), '  ')


if __name__ == '__main__':
    main()

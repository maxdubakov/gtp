"""Build the offline-augmented Stage 2 dataset.

Loads all pieces from data/<source>/processed/, runs the same stratified split
as the runtime path (by source + tuning + capo), then:
  - train + val: emits all valid capo variants per piece (capo 0-7, skipping
    variants where pitches fall outside MIDI [0, 127]).
  - test:        rotates one capo per piece across 0-7, paper-style.

Writes data/stage2_aug/{train,val,test}.jsonl. Run once before training.

Tuning augmentation is NOT applied here — it happens online in TabDataset.

Usage:
  python scripts/stage2/setup/build_aug_dataset.py
  python scripts/stage2/setup/build_aug_dataset.py --output-dir /tmp/aug --seed 7
"""

import argparse
import json
from pathlib import Path

from gtp.stage2.data import AUG_DATA_DIR, CAPO_RANGE, expand_capo, load_all_pieces, stratified_split
from gtp.stage2.tokenizer import tokenize_piece


def expand_all(pieces):
    for piece in pieces:
        yield from expand_capo(piece, capos=CAPO_RANGE)


def rotate_capo(pieces):
    """Yield one capo variant per piece, cycling through CAPO_RANGE."""
    rotation = list(CAPO_RANGE)
    for i, piece in enumerate(pieces):
        target = rotation[i % len(rotation)]
        for variant in expand_capo(piece, capos=[target]):
            yield variant
            break  # one variant per piece


def annotate_with_subseqs(pieces, max_seq_len=512):
    """Tokenize each piece once and stash sub-sequence count.

    Tuning augmentation (online) preserves token count, so num_subseqs is invariant
    under runtime augmentation. TabDataset uses this to build its flat index in O(1)
    per piece instead of re-tokenizing 49K+ pieces at __init__.
    """
    for piece in pieces:
        seqs = tokenize_piece(piece, max_seq_len=max_seq_len)
        yield {**piece, 'num_subseqs': len(seqs)}


def write_jsonl(path, items, progress_every=5000):
    count = 0
    with open(path, 'w') as fh:
        for item in items:
            fh.write(json.dumps(item) + '\n')
            count += 1
            if count % progress_every == 0:
                print(f'    {path.name}: {count} pieces written...', flush=True)
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=str(AUG_DATA_DIR))
    ap.add_argument('--train-ratio', type=float, default=0.90)
    ap.add_argument('--val-ratio', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading pieces...')
    pieces, _summary = load_all_pieces()
    print(f'  total pieces kept: {len(pieces)}')

    print('Stratified split...')
    train_pieces, val_pieces, test_pieces = stratified_split(
        pieces,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f'  pre-aug counts: train={len(train_pieces)}  val={len(val_pieces)}  test={len(test_pieces)}')

    print('Expanding capo variants and annotating sub-sequence counts...')
    train_path = out_dir / 'train.jsonl'
    val_path = out_dir / 'val.jsonl'
    test_path = out_dir / 'test.jsonl'

    n_train = write_jsonl(train_path, annotate_with_subseqs(expand_all(train_pieces)))
    n_val = write_jsonl(val_path, annotate_with_subseqs(expand_all(val_pieces)))
    n_test = write_jsonl(test_path, annotate_with_subseqs(rotate_capo(test_pieces)))

    print('\nWrote (post-capo-aug):')
    for path, n in [(train_path, n_train), (val_path, n_val), (test_path, n_test)]:
        size_mb = path.stat().st_size / 1e6
        print(f'  {path.relative_to(out_dir.parent.parent)}  pieces={n}  size={size_mb:.1f} MB')

    avg_per_piece_train = n_train / max(1, len(train_pieces))
    avg_per_piece_val = n_val / max(1, len(val_pieces))
    print(f'\navg capo variants/piece: train={avg_per_piece_train:.2f}  val={avg_per_piece_val:.2f}')


if __name__ == '__main__':
    main()

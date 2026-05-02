"""Build the offline-augmented Stage 2 dataset.

Loads all pieces from data/<source>/processed/, runs a stratified split (by source +
tuning + capo), then:
  - train + val: emits all valid capo variants per piece (capo 0-7, skipping variants
    that fall outside MIDI range or push the relative fret past MAX_PLAYABLE_FRET when
    a capo is added).
  - test:        rotates one capo per piece across 0-7, paper-style (rotate_capo).

Writes data/stage2_aug/{train,val,test}.jsonl. Run once before training.

Tuning augmentation is NOT applied here — it happens online in TabDataset.

Usage:
  python scripts/stage2/build_aug_dataset.py
  python scripts/stage2/build_aug_dataset.py --output-dir /tmp/aug --seed 7
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from gtp.stage2.data import (
    MAX_PLAYABLE_FRET,
    MIN_NOTES_PER_PIECE,
    filter_notes,
)
from gtp.stage2.paths import AUG_DATA_DIR, PROCESSED_DIRS
from gtp.stage2.tokenizer import tokenize_piece

CAPO_RANGE = range(0, 8)
MIDI_MIN = 0
MIDI_MAX = 127


# ---------------------------------------------------------------------------
# Loading processed JSONs
# ---------------------------------------------------------------------------


def _print_summary(summary):
    """Print a compact per-source data quality table."""
    if not summary:
        return
    print('\n[data load] per-source summary:')
    print(f'  {"source":<12} {"seen":>6} {"kept":>6} {"skip<10":>8} {"notes":>10}  filtered')
    for name, s in summary.items():
        reason_str = ', '.join(f'{k}={v}' for k, v in s['filter_reasons'].items()) or '-'
        print(
            f'  {name:<12} {s["pieces_seen"]:>6} {s["pieces_kept"]:>6} '
            f'{s["pieces_skipped_few_notes"]:>8} {s["notes_kept"]:>10}  {reason_str}'
        )


def load_all_pieces(datasets=None):
    """Load processed JSON files. Returns (pieces, summary) and prints per-source stats."""
    if datasets is None:
        datasets = list(PROCESSED_DIRS.keys())

    pieces = []
    summary = {}
    for name in datasets:
        path = PROCESSED_DIRS[name]
        if not path.exists():
            continue

        s = {
            'pieces_seen': 0,
            'pieces_kept': 0,
            'pieces_skipped_few_notes': 0,
            'notes_kept': 0,
            'filter_reasons': Counter(),
        }

        for f in sorted(os.listdir(path)):
            if not f.endswith('.json'):
                continue
            if f.startswith('._'):
                # macOS AppleDouble metadata sidecars (from BSD tar without
                # COPYFILE_DISABLE=1) — not real JSON, skip silently.
                continue
            with open(path / f, encoding='utf-8', errors='replace') as fh:
                data = json.load(fh)

            tuning = data.get('tuning', [64, 59, 55, 50, 45, 40])
            notes, reasons = filter_notes(data.get('notes', []), tuning)
            s['pieces_seen'] += 1
            s['filter_reasons'] += reasons

            if len(notes) < MIN_NOTES_PER_PIECE:
                s['pieces_skipped_few_notes'] += 1
                continue

            s['pieces_kept'] += 1
            s['notes_kept'] += len(notes)
            pieces.append(
                {
                    'source': name,
                    'filename': f,
                    'tuning': tuning,
                    'tempo': data.get('tempo', 120),
                    'capo': data.get('capo', 0),
                    'notes': notes,
                }
            )

        summary[name] = s

    _print_summary(summary)
    return pieces, summary


# ---------------------------------------------------------------------------
# Stratified split + capo augmentation
# ---------------------------------------------------------------------------


def _tuning_key(tuning):
    """Normalize tuning to a hashable key for stratification."""
    return tuple(tuning)


def _capo_key(capo):
    """Bucket capo values: 0 vs non-zero."""
    return 'capo' if capo > 0 else 'no_capo'


def stratified_split(pieces, train_ratio=0.90, val_ratio=0.05, seed=42):
    """Split pieces into train/val/test, stratified by source + tuning + capo."""
    rng = random.Random(seed)

    groups = defaultdict(list)
    for piece in pieces:
        key = (piece['source'], _tuning_key(piece['tuning']), _capo_key(piece['capo']))
        groups[key].append(piece)

    train, val, test = [], [], []

    for _key, group in groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val = max(0, round(n * val_ratio))

        # Tiny groups: keep them all in train.
        if n <= 2:
            train.extend(group)
            continue

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def expand_capo(piece, capos=CAPO_RANGE):
    """Generate capo variants of a piece. Yields new piece dicts.

    Tuning convention: tuning already includes capo (pitch = tuning[s-1] + fret),
    where `fret` is relative to capo. A new variant with capo=N has pitches/tuning
    shifted by (N - piece['capo']); the relative `fret` of each note is unchanged.

    Skips variants that would be:
      - musically invalid (any pitch outside MIDI [0, 127]), or
      - physically unplayable (any relative fret + new_capo > MAX_PLAYABLE_FRET).
    """
    old_capo = piece['capo']
    max_fret = max((n['fret'] for n in piece['notes']), default=0)
    for new_capo in capos:
        if max_fret + new_capo > MAX_PLAYABLE_FRET:
            continue
        delta = new_capo - old_capo
        new_pitches = [n['pitch'] + delta for n in piece['notes']]
        if any(p < MIDI_MIN or p > MIDI_MAX for p in new_pitches):
            continue
        new_tuning = [t + delta for t in piece['tuning']]
        new_notes = [{**n, 'pitch': n['pitch'] + delta} for n in piece['notes']]
        yield {**piece, 'tuning': new_tuning, 'capo': new_capo, 'notes': new_notes}


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


# ---------------------------------------------------------------------------
# Tokenization annotation + JSONL writing
# ---------------------------------------------------------------------------


def annotate_with_subseqs(pieces, stats, max_seq_len=512):
    """Tokenize each piece once, stash sub-sequence count, and accumulate stats.

    Tuning augmentation (online) preserves token count, so num_subseqs is invariant
    under runtime augmentation. TabDataset uses this to build its flat index in O(1)
    per piece instead of re-tokenizing 49K+ pieces at __init__.

    `stats` is a dict mutated in place: stats[source] = {pieces, subseqs, enc_tokens, dec_tokens}.
    """
    for piece in pieces:
        seqs = tokenize_piece(piece, max_seq_len=max_seq_len)
        enc_tokens = sum(len(enc) for enc, _ in seqs)
        dec_tokens = sum(len(dec) for _, dec in seqs)
        s = stats.setdefault(piece['source'], {'pieces': 0, 'subseqs': 0, 'enc_tokens': 0, 'dec_tokens': 0})
        s['pieces'] += 1
        s['subseqs'] += len(seqs)
        s['enc_tokens'] += enc_tokens
        s['dec_tokens'] += dec_tokens
        yield {**piece, 'num_subseqs': len(seqs)}


def print_split_stats(label, stats):
    """Print a per-source table + totals for one split."""
    if not stats:
        print(f'  {label}: (empty)')
        return
    print(f'  {label}:')
    print(f'    {"source":<12} {"pieces":>7} {"sub-seqs":>9} {"enc tokens":>13} {"dec tokens":>13}')
    tot_p = tot_s = tot_e = tot_d = 0
    for src, s in sorted(stats.items()):
        print(
            f'    {src:<12} {s["pieces"]:>7} {s["subseqs"]:>9} '
            f'{s["enc_tokens"]:>13,} {s["dec_tokens"]:>13,}'
        )
        tot_p += s['pieces']
        tot_s += s['subseqs']
        tot_e += s['enc_tokens']
        tot_d += s['dec_tokens']
    print(f'    {"TOTAL":<12} {tot_p:>7} {tot_s:>9} {tot_e:>13,} {tot_d:>13,}')


def write_jsonl(path, items, progress_every=5000):
    count = 0
    with open(path, 'w') as fh:
        for item in items:
            fh.write(json.dumps(item) + '\n')
            count += 1
            if count % progress_every == 0:
                print(f'    {path.name}: {count} pieces written...', flush=True)
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    train_stats, val_stats, test_stats = {}, {}, {}
    n_train = write_jsonl(train_path, annotate_with_subseqs(expand_all(train_pieces), train_stats))
    n_val = write_jsonl(val_path, annotate_with_subseqs(expand_all(val_pieces), val_stats))
    n_test = write_jsonl(test_path, annotate_with_subseqs(rotate_capo(test_pieces), test_stats))

    print('\nWrote (post-capo-aug):')
    for path, n in [(train_path, n_train), (val_path, n_val), (test_path, n_test)]:
        size_mb = path.stat().st_size / 1e6
        print(f'  {path.relative_to(out_dir.parent.parent)}  pieces={n}  size={size_mb:.1f} MB')

    avg_per_piece_train = n_train / max(1, len(train_pieces))
    avg_per_piece_val = n_val / max(1, len(val_pieces))
    print(f'\navg capo variants/piece: train={avg_per_piece_train:.2f}  val={avg_per_piece_val:.2f}')

    print('\nPer-source stats (post-aug, post-tokenization):')
    print_split_stats('train', train_stats)
    print_split_stats('val', val_stats)
    print_split_stats('test', test_stats)


if __name__ == '__main__':
    main()

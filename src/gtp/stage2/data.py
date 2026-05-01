"""Stage 2 dataset: loads processed tab JSONs, tokenizes, and serves padded batches.

Handles all four data sources (DadaGP, GuitarToday, GuitarSet, Leduc) uniformly.
Filters bad notes at load time. Stratified train/val/test split by source, tuning, and capo.

Augmentation:
  - Capo augmentation: applied offline by scripts/stage2/setup/build_aug_dataset.py.
    Reads originals, splits stratified, expands each train/val piece into capo 0-7
    variants (skipping any that push pitches outside MIDI range), and rotates one
    capo per test piece. Writes JSONLs to AUG_DATA_DIR.
  - Tuning augmentation: applied online in TabDataset.__getitem__ when augment=True.
    Picks a random tuning from STANDARD_TUNINGS and re-tokenizes; preserves capo,
    string, and fret. Pieces with non-6-string tunings are passed through unchanged.

build_datasets() prefers AUG_DATA_DIR JSONLs when present; falls back to processed/.
"""

import json
import os
import random
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset

from gtp import REPO_ROOT
from gtp.stage2.tokenizer import VOCAB, tokenize_piece

PROCESSED_DIRS = {
    'dadagp': REPO_ROOT / 'data' / 'dadagp' / 'processed',
    'guitartoday': REPO_ROOT / 'data' / 'guitartoday' / 'processed',
    'guitarset': REPO_ROOT / 'data' / 'guitarset' / 'processed',
    'leduc': REPO_ROOT / 'data' / 'leduc' / 'processed',
}

AUG_DATA_DIR = REPO_ROOT / 'data' / 'stage2_aug'

MIN_NOTES_PER_PIECE = 10
MAX_FRET = 24  # vocabulary / filter_notes upper bound (some 24-fret guitars exist in the data)
MAX_PLAYABLE_FRET = 22  # most production guitars stop at 21-22; used for capo-aug playability check
MIN_NOTE_DURATION = 0.001  # seconds

# 6-string standard tunings used for online tuning augmentation.
# Each entry is open-string pitches in our convention: string 1 = high E (index 0).
STANDARD_TUNINGS = [
    [64, 59, 55, 50, 45, 40],  # standard E A D G B E
    [63, 58, 54, 49, 44, 39],  # half-step down
    [62, 57, 53, 48, 43, 38],  # full-step down (D)
    [64, 59, 55, 50, 45, 38],  # drop-D
]
CAPO_RANGE = range(0, 8)
MIDI_MIN = 0
MIDI_MAX = 127


def filter_notes(notes, tuning):
    """Drop invalid notes; return (clean_notes, reason_counts)."""
    clean = []
    reasons = Counter()
    for n in notes:
        if n['fret'] < 0 or n['fret'] > MAX_FRET:
            reasons['bad_fret'] += 1
            continue
        if n['end'] - n['start'] < MIN_NOTE_DURATION:
            reasons['zero_dur'] += 1
            continue
        if n['string'] < 1 or n['string'] > len(tuning):
            reasons['bad_string'] += 1
            continue
        expected_pitch = tuning[n['string'] - 1] + n['fret']
        if expected_pitch != n['pitch']:
            reasons['pitch_mismatch'] += 1
            continue
        clean.append(n)
    return clean, reasons


def _tuning_key(tuning):
    """Normalize tuning to a hashable key for stratification."""
    return tuple(tuning)


def _capo_key(capo):
    """Bucket capo values: 0 vs non-zero."""
    return 'capo' if capo > 0 else 'no_capo'


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
            with open(path / f) as fh:
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


def random_tuning(piece, rng):
    """Apply random tuning augmentation. Preserves capo, string, fret; replaces tuning.

    Pieces with non-6-string tunings are returned unchanged (no augmentation applied).
    """
    if len(piece['tuning']) != 6:
        return piece
    base = rng.choice(STANDARD_TUNINGS)
    capo = piece['capo']
    new_tuning = [t + capo for t in base]
    new_notes = [
        {**n, 'pitch': new_tuning[n['string'] - 1] + n['fret']} for n in piece['notes']
    ]
    return {**piece, 'tuning': new_tuning, 'notes': new_notes}


def load_jsonl_pieces(path):
    """Load pieces from a JSONL file (one piece dict per line)."""
    pieces = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                pieces.append(json.loads(line))
    return pieces


def stratified_split(pieces, train_ratio=0.90, val_ratio=0.05, seed=42):
    """Split pieces into train/val/test, stratified by source + tuning + capo.

    Returns (train_pieces, val_pieces, test_pieces).
    """
    rng = random.Random(seed)

    # Group pieces by stratification key
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

        # For very small groups, put everything in train
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


class TabDataset(Dataset):
    """PyTorch Dataset for Stage 2 (MIDI → Tab) training.

    Each item is (encoder_ids, decoder_ids, source), padded to max_seq_len.
    Source travels with the sample so per-source eval works regardless of shuffling.

    When augment=True, applies random tuning augmentation per __getitem__ and
    re-tokenizes from the underlying piece. Capo is preserved (already varied
    offline by build_aug_dataset.py for the train/val splits).
    """

    def __init__(self, pieces, max_seq_len=512, augment=False):
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.vocab = VOCAB
        self._rng = random.Random()  # un-seeded → independent per DataLoader worker

        if augment:
            # Keep the originals; tokenize on demand. Build flat index from cached
            # num_subseqs annotations (written by build_aug_dataset.py). Falls back to
            # tokenizing if the field is missing.
            self.pieces = pieces
            self._index = []
            self._sources = []
            for pi, piece in enumerate(pieces):
                n = piece.get('num_subseqs')
                if n is None:
                    n = len(tokenize_piece(piece, max_seq_len=max_seq_len))
                for si in range(n):
                    self._index.append((pi, si))
                    self._sources.append(piece['source'])
        else:
            # Pre-tokenize and cache; pieces no longer needed.
            self.sequences = []
            self._sources = []
            for piece in pieces:
                seqs = tokenize_piece(piece, max_seq_len=max_seq_len)
                self.sequences.extend(seqs)
                self._sources.extend([piece['source']] * len(seqs))

    def __len__(self):
        return len(self._sources)

    def __getitem__(self, idx):
        if self.augment:
            pi, si = self._index[idx]
            piece = random_tuning(self.pieces[pi], self._rng)
            seqs = tokenize_piece(piece, max_seq_len=self.max_seq_len)
            enc_ids, dec_ids = seqs[min(si, len(seqs) - 1)]
        else:
            enc_ids, dec_ids = self.sequences[idx]
        return self._pad(enc_ids), self._pad(dec_ids), self._sources[idx]

    def _pad(self, ids):
        padded = ids + [self.vocab.pad_id] * (self.max_seq_len - len(ids))
        return torch.tensor(padded[: self.max_seq_len], dtype=torch.long)


def build_datasets(
    datasets=None,
    train_ratio=0.90,
    val_ratio=0.05,
    max_seq_len=512,
    seed=42,
    augment_train=True,
):
    """Build train, validation, and test TabDatasets.

    If AUG_DATA_DIR/{train,val,test}.jsonl exist, loads those (capo augmentation already
    applied offline). Otherwise builds from processed/ with on-the-fly stratification.

    augment_train controls whether train applies online tuning augmentation per __getitem__.

    Returns (train_dataset, val_dataset, test_dataset, stats_dict).
    """
    aug_train_path = AUG_DATA_DIR / 'train.jsonl'
    if aug_train_path.exists():
        print(f'Loading augmented JSONLs from {AUG_DATA_DIR}')
        train_pieces = load_jsonl_pieces(AUG_DATA_DIR / 'train.jsonl')
        val_pieces = load_jsonl_pieces(AUG_DATA_DIR / 'val.jsonl')
        test_pieces = load_jsonl_pieces(AUG_DATA_DIR / 'test.jsonl')
        if datasets:
            wanted = set(datasets)
            train_pieces = [p for p in train_pieces if p['source'] in wanted]
            val_pieces = [p for p in val_pieces if p['source'] in wanted]
            test_pieces = [p for p in test_pieces if p['source'] in wanted]
        data_quality = None
        total_pieces = len(train_pieces) + len(val_pieces) + len(test_pieces)
    else:
        pieces, data_quality = load_all_pieces(datasets)
        train_pieces, val_pieces, test_pieces = stratified_split(
            pieces, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
        total_pieces = len(pieces)

    train_ds = TabDataset(train_pieces, max_seq_len=max_seq_len, augment=augment_train)
    val_ds = TabDataset(val_pieces, max_seq_len=max_seq_len, augment=False)
    test_ds = TabDataset(test_pieces, max_seq_len=max_seq_len, augment=False)

    def count_by_source(piece_list):
        counts = defaultdict(int)
        for p in piece_list:
            counts[p['source']] += 1
        return dict(counts)

    stats = {
        'total_pieces': total_pieces,
        'train_pieces': len(train_pieces),
        'val_pieces': len(val_pieces),
        'test_pieces': len(test_pieces),
        'train_sequences': len(train_ds),
        'val_sequences': len(val_ds),
        'test_sequences': len(test_ds),
        'vocab_size': len(VOCAB),
        'max_seq_len': max_seq_len,
        'sources_total': count_by_source(train_pieces + val_pieces + test_pieces),
        'sources_train': count_by_source(train_pieces),
        'sources_val': count_by_source(val_pieces),
        'sources_test': count_by_source(test_pieces),
        'augment_train': augment_train,
        'data_quality': data_quality,
    }

    return train_ds, val_ds, test_ds, stats

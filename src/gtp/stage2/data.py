"""Stage 2 dataset: loads processed tab JSONs, tokenizes, and serves padded batches.

Handles all four data sources (DadaGP, GuitarToday, GuitarSet, Leduc) uniformly.
Filters bad notes at load time. Stratified train/val/test split by source, tuning, and capo.
"""

import json
import os
import random
from collections import defaultdict

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

MIN_NOTES_PER_PIECE = 10
MAX_FRET = 24
MIN_NOTE_DURATION = 0.001  # seconds


def filter_notes(notes, tuning):
    """Remove invalid notes: negative frets, fret>24, zero duration, pitch mismatch."""
    clean = []
    for n in notes:
        if n['fret'] < 0 or n['fret'] > MAX_FRET:
            continue
        if n['end'] - n['start'] < MIN_NOTE_DURATION:
            continue
        if n['string'] < 1 or n['string'] > len(tuning):
            continue
        expected_pitch = tuning[n['string'] - 1] + n['fret']
        if expected_pitch != n['pitch']:
            continue
        clean.append(n)
    return clean


def _tuning_key(tuning):
    """Normalize tuning to a hashable key for stratification."""
    return tuple(tuning)


def _capo_key(capo):
    """Bucket capo values: 0 vs non-zero."""
    return 'capo' if capo > 0 else 'no_capo'


def load_all_pieces(datasets=None):
    """Load all processed JSON files and return a list of clean piece dicts."""
    if datasets is None:
        datasets = list(PROCESSED_DIRS.keys())

    pieces = []
    for name in datasets:
        path = PROCESSED_DIRS[name]
        if not path.exists():
            continue
        for f in sorted(os.listdir(path)):
            if not f.endswith('.json'):
                continue
            with open(path / f) as fh:
                data = json.load(fh)

            tuning = data.get('tuning', [64, 59, 55, 50, 45, 40])
            notes = filter_notes(data.get('notes', []), tuning)

            if len(notes) < MIN_NOTES_PER_PIECE:
                continue

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

    Each item is a (encoder_ids, decoder_ids) pair, padded to max_seq_len.
    Tracks source dataset per sequence for per-dataset evaluation.
    """

    def __init__(self, pieces, max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.vocab = VOCAB
        self.sequences = []
        self.sources = []

        for piece in pieces:
            seqs = tokenize_piece(piece, max_seq_len=max_seq_len)
            self.sequences.extend(seqs)
            self.sources.extend([piece['source']] * len(seqs))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        enc_ids, dec_ids = self.sequences[idx]
        return self._pad(enc_ids), self._pad(dec_ids)

    def _pad(self, ids):
        padded = ids + [self.vocab.pad_id] * (self.max_seq_len - len(ids))
        return torch.tensor(padded[: self.max_seq_len], dtype=torch.long)


def build_datasets(datasets=None, train_ratio=0.90, val_ratio=0.05, max_seq_len=512, seed=42):
    """Build train, validation, and test TabDatasets from all processed data.

    Returns (train_dataset, val_dataset, test_dataset, stats_dict)
    """
    pieces = load_all_pieces(datasets)
    train_pieces, val_pieces, test_pieces = stratified_split(
        pieces, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    train_ds = TabDataset(train_pieces, max_seq_len=max_seq_len)
    val_ds = TabDataset(val_pieces, max_seq_len=max_seq_len)
    test_ds = TabDataset(test_pieces, max_seq_len=max_seq_len)

    def count_by_source(piece_list):
        counts = defaultdict(int)
        for p in piece_list:
            counts[p['source']] += 1
        return dict(counts)

    stats = {
        'total_pieces': len(pieces),
        'train_pieces': len(train_pieces),
        'val_pieces': len(val_pieces),
        'test_pieces': len(test_pieces),
        'train_sequences': len(train_ds),
        'val_sequences': len(val_ds),
        'test_sequences': len(test_ds),
        'vocab_size': len(VOCAB),
        'max_seq_len': max_seq_len,
        'sources_total': count_by_source(pieces),
        'sources_train': count_by_source(train_pieces),
        'sources_val': count_by_source(val_pieces),
        'sources_test': count_by_source(test_pieces),
    }

    return train_ds, val_ds, test_ds, stats

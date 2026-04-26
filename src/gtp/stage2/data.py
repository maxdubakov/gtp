"""Stage 2 dataset: loads processed tab JSONs, tokenizes, and serves padded batches.

Handles all four data sources (DadaGP, GuitarToday, GuitarSet, Leduc) uniformly.
Filters bad notes at load time. Splits into train/val by file.
"""

import json
import os
import random
from pathlib import Path

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


def load_all_pieces(datasets=None):
    """Load all processed JSON files and return a list of clean piece dicts.

    Each piece: {source, tuning, tempo, capo, notes}
    """
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

            pieces.append({
                'source': name,
                'filename': f,
                'tuning': tuning,
                'tempo': data.get('tempo', 120),
                'capo': data.get('capo', 0),
                'notes': notes,
            })

    return pieces


def split_pieces(pieces, val_ratio=0.1, seed=42):
    """Split pieces into train/val by file (not by sequence)."""
    rng = random.Random(seed)
    shuffled = list(pieces)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


class TabDataset(Dataset):
    """PyTorch Dataset for Stage 2 (MIDI → Tab) training.

    Each item is a (encoder_ids, decoder_ids) pair, padded to max_seq_len.
    """

    def __init__(self, pieces, max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.vocab = VOCAB
        self.sequences = []

        for piece in pieces:
            seqs = tokenize_piece(piece, max_seq_len=max_seq_len)
            self.sequences.extend(seqs)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        enc_ids, dec_ids = self.sequences[idx]
        return self._pad(enc_ids), self._pad(dec_ids)

    def _pad(self, ids):
        padded = ids + [self.vocab.pad_id] * (self.max_seq_len - len(ids))
        return torch.tensor(padded[:self.max_seq_len], dtype=torch.long)


def build_datasets(datasets=None, val_ratio=0.1, max_seq_len=512, seed=42):
    """Build train and validation TabDatasets from all processed data.

    Returns (train_dataset, val_dataset, stats_dict)
    """
    pieces = load_all_pieces(datasets)
    train_pieces, val_pieces = split_pieces(pieces, val_ratio=val_ratio, seed=seed)

    train_ds = TabDataset(train_pieces, max_seq_len=max_seq_len)
    val_ds = TabDataset(val_pieces, max_seq_len=max_seq_len)

    # Count pieces per source
    source_counts = {}
    for p in pieces:
        source_counts[p['source']] = source_counts.get(p['source'], 0) + 1

    stats = {
        'total_pieces': len(pieces),
        'train_pieces': len(train_pieces),
        'val_pieces': len(val_pieces),
        'train_sequences': len(train_ds),
        'val_sequences': len(val_ds),
        'vocab_size': len(VOCAB),
        'max_seq_len': max_seq_len,
        'sources': source_counts,
    }

    return train_ds, val_ds, stats

"""Stage 2 runtime data layer: loads pre-augmented JSONLs, vends padded batches.

The offline build pipeline (scripts/stage2/build_aug_dataset.py) is responsible for
producing the JSONL files this module reads. That script does the stratified split
and capo augmentation; this module only handles:
  - reading pre-built JSONLs
  - online tuning augmentation in TabDataset.__getitem__ (when augment=True)
  - assembling padded torch tensors
"""

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gtp.log import info
from gtp.stage2.evidence import Perturbation, StringEvidence, with_evidence
from gtp.stage2.genres import GENRES
from gtp.stage2.metrics import pitch_of
from gtp.stage2.paths import AUG_DATA_DIR
from gtp.stage2.tokenizer import (
    EVIDENCE_DIM,
    MAX_FRET,
    MAX_SEQ_LEN,
    MAX_TIME_SHIFT,
    TEMPO_MAX,
    TEMPO_MIN,
    TEMPO_STEP,
    TIME_SHIFT_BINS,
    Vocabulary,
    tokenize_piece,
)

MIN_NOTES_PER_PIECE = 10
MIN_NOTE_DURATION = 0.001  # seconds

# 6-string standard tunings used for online tuning augmentation in TabDataset.
# Each entry is open-string pitches in convention: string 1 = high E (index 0).
STANDARD_TUNINGS = [
    [64, 59, 55, 50, 45, 40],  # standard E A D G B E
    [63, 58, 54, 49, 44, 39],  # half-step down
    [62, 57, 53, 48, 43, 38],  # full-step down (D)
    [64, 59, 55, 50, 45, 38],  # drop-D
]

DEFAULT_REBALANCE_SOURCE = {
    'dadagp': 1.0,
    'guitarset': 8.0,
    'leduc': 5.0,
    'guitartoday': 1.0,
}

DEFAULT_REBALANCE_GENRE = {
    'classical': 1.5,
    'jazz': 1.5,
    'blues': 1.5,
}


def _piece_id(piece):
    """Synthesize a stable piece identifier from source/filename/capo"""
    src = piece.get('source', '?')
    fname = piece.get('filename', piece.get('file', '?'))
    capo = piece.get('capo', 0)
    return f'{src}:{fname}:capo{capo}'


class TabDataset(Dataset):
    """PyTorch Dataset for Stage 2 (MIDI → Tab) training.

    Each item is (encoder_ids, decoder_ids, encoder_evidence, source, piece_id), padded to max_seq_len;
    encoder_evidence is all zeros without `evidence`. With augment=True, tuning (and evidence perturbation)
    is resampled per item; otherwise evidence is sampled once with a fixed seed.
    """

    def __init__(
        self,
        pieces,
        vocab: Vocabulary,
        max_seq_len: int = MAX_SEQ_LEN,
        augment=False,
        genre_dropout: float = 0.0,
        evidence: StringEvidence | None = None,
        perturbation: Perturbation | None = None,
    ):
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.genre_dropout = genre_dropout
        self.vocab = vocab
        self.evidence = evidence
        self.perturbation = perturbation
        self._rng_pid = None

        if augment:
            self.pieces = pieces
            self._index = []
            self._sources = []
            self._genres = []
            self._piece_ids = []
            for piece_idx, piece in enumerate(pieces):
                n_seqs = piece.get('num_subseqs')
                if n_seqs is None:
                    n_seqs = len(tokenize_piece(piece, vocab, max_seq_len=max_seq_len))
                piece_id = _piece_id(piece)
                genre = piece.get('genre', 'unknown')
                for seq_idx in range(n_seqs):
                    self._index.append((piece_idx, seq_idx))
                    self._sources.append(piece['source'])
                    self._genres.append(genre)
                    self._piece_ids.append(piece_id)
        else:
            self.sequences = []
            self._evidence = []
            self._sources = []
            self._genres = []
            self._piece_ids = []
            rng = np.random.default_rng(0)
            for piece in pieces:
                if evidence is not None:
                    seqs = tokenize_piece(
                        with_evidence(piece, evidence, rng), vocab, max_seq_len=max_seq_len, with_evidence=True
                    )
                    self._evidence.extend(ev.astype(np.float16) for _, _, ev in seqs)
                    seqs = [(enc, dec) for enc, dec, _ in seqs]
                else:
                    seqs = tokenize_piece(piece, vocab, max_seq_len=max_seq_len)
                piece_id = _piece_id(piece)
                genre = piece.get('genre', 'unknown')
                self.sequences.extend(seqs)
                self._sources.extend([piece['source']] * len(seqs))
                self._genres.extend([genre] * len(seqs))
                self._piece_ids.extend([piece_id] * len(seqs))

    def __len__(self):
        return len(self._sources)

    def _rngs(self):
        """(random.Random, np.random.Generator) private to this process, so DataLoader workers differ."""
        if self._rng_pid != os.getpid():
            self._rng_pid = os.getpid()
            self._rng, self._np_rng = random.Random(), np.random.default_rng()
        return self._rng, self._np_rng

    def __getitem__(self, idx):
        ev = None
        if self.augment:
            rng, np_rng = self._rngs()
            piece_id, seq_id = self._index[idx]
            piece = self._random_tuning(piece_id)
            if self.evidence is not None:
                piece = with_evidence(piece, self.evidence, np_rng, self.perturbation)
            genre_override = 'unknown' if self.genre_dropout > 0 and rng.random() < self.genre_dropout else None
            seqs = tokenize_piece(
                piece,
                self.vocab,
                max_seq_len=self.max_seq_len,
                genre_override=genre_override,
                with_evidence=self.evidence is not None,
            )
            enc_ids, dec_ids, *rest = seqs[seq_id]
            ev = rest[0] if rest else None
        else:
            enc_ids, dec_ids = self.sequences[idx]
            ev = self._evidence[idx] if self._evidence else None
        return (
            self._pad(enc_ids),
            self._pad(dec_ids),
            self._pad_evidence(ev),
            self._sources[idx],
            self._piece_ids[idx],
        )

    def _random_tuning(self, piece_id):
        """Apply random tuning augmentation. Fixes capo, string, fret, and changes played pitch instead"""
        piece = self.pieces[piece_id]
        if len(piece['tuning']) != 6:
            return piece
        base = self._rngs()[0].choice(STANDARD_TUNINGS)
        capo = piece['capo']
        new_tuning = [t + capo for t in base]
        new_notes = [{**n, 'pitch': pitch_of((n['string'], n['fret']), new_tuning)} for n in piece['notes']]
        return {**piece, 'tuning': new_tuning, 'notes': new_notes}

    def _pad(self, ids):
        padded = ids + [self.vocab.pad_id] * (self.max_seq_len - len(ids))
        return torch.tensor(padded[: self.max_seq_len], dtype=torch.long)

    def _pad_evidence(self, ev):
        out = torch.zeros(self.max_seq_len, EVIDENCE_DIM)
        if ev is not None:
            n = min(len(ev), self.max_seq_len)
            out[:n] = torch.from_numpy(np.asarray(ev[:n], dtype=np.float32))
        return out

    @property
    def sources(self):
        return self._sources

    @property
    def genres(self):
        return self._genres


def filter_notes(notes, tuning):
    """Drop invalid notes; return (clean_notes, reason_counts)"""
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
        if pitch_of((n['string'], n['fret']), tuning) != n['pitch']:
            reasons['pitch_mismatch'] += 1
            continue
        clean.append(n)
    return clean, reasons


def load_jsonl_pieces(path):
    """Load pieces from a JSONL file (one piece dict per line)."""
    pieces = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                pieces.append(json.loads(line))
    return pieces


def validate_aug_data_config(vocab: Vocabulary, max_seq_len: int, data_dir=AUG_DATA_DIR):
    """Check that script which generated augmented data has the same config"""
    config_path = Path(data_dir) / 'data_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'No configuration file found at {config_path}')
    cfg = json.loads(config_path.read_text())

    expected = {
        'include_genre': vocab.include_genre,
        'max_seq_len': max_seq_len,
        'genres': list(GENRES),
        'tempo_min': TEMPO_MIN,
        'tempo_max': TEMPO_MAX,
        'tempo_step': TEMPO_STEP,
        'max_fret': MAX_FRET,
        'time_shift_step': TIME_SHIFT_BINS[0],
        'max_time_shift': MAX_TIME_SHIFT,
    }
    mismatches = []
    for key, runtime_val in expected.items():
        build_val = cfg.get(key)
        if build_val != runtime_val:
            mismatches.append(f'{key}: build={build_val!r}, runtime={runtime_val!r}')

    if mismatches:
        raise RuntimeError('Augmented dataset built differently. Mismatches:' + '\n  '.join(mismatches))


def build_datasets(
    vocab: Vocabulary,
    splits: tuple[str, ...] = ('train', 'val', 'test'),
    max_seq_len: int = MAX_SEQ_LEN,
    augment_train: bool = True,
    genre_dropout: float = 0.0,
    data_dir=AUG_DATA_DIR,
    evidence: StringEvidence | None = None,
    perturbation: Perturbation | None = None,
) -> tuple[dict[str, 'TabDataset'], dict]:
    """Build TabDatasets from the augmented JSONLs. Returns ({split: TabDataset}, stats_dict)."""
    valid = {'train', 'val', 'test'}
    bad = set(splits) - valid
    if bad:
        raise ValueError(f'Unknown split(s): {sorted(bad)}. Allowed: {sorted(valid)}')

    data_dir = Path(data_dir)
    validate_aug_data_config(vocab, max_seq_len, data_dir)
    info(f'Loading augmented JSONLs from {data_dir}: {list(splits)}')
    pieces: dict[str, list[dict]] = {}
    for split in splits:
        path = data_dir / f'{split}.jsonl'
        if not path.exists():
            raise FileNotFoundError(f'Augmented JSONL not found: {path}')
        pieces[split] = load_jsonl_pieces(path)

    datasets: dict[str, TabDataset] = {}
    for split, ps in pieces.items():
        augment = augment_train and split == 'train'
        dropout = genre_dropout if split == 'train' else 0.0
        datasets[split] = TabDataset(
            ps,
            vocab,
            max_seq_len=max_seq_len,
            augment=augment,
            genre_dropout=dropout,
            evidence=evidence,
            perturbation=perturbation if augment else None,
        )

    def count_by_source(piece_list):
        counts: dict = defaultdict(int)
        for p in piece_list:
            counts[p['source']] += 1
        return dict(counts)

    all_pieces = [p for ps in pieces.values() for p in ps]
    stats = {
        'vocab_size': len(vocab),
        'max_seq_len': max_seq_len,
        'augment_train': augment_train,
        'total_pieces': len(all_pieces),
        'sources_total': count_by_source(all_pieces),
        **{f'{split}_pieces': len(pieces[split]) for split in splits},
        **{f'{split}_sequences': len(datasets[split]) for split in splits},
        **{f'sources_{split}': count_by_source(pieces[split]) for split in splits},
    }
    return datasets, stats


def compute_sampling_weights(
    sources: list[str],
    genres: list[str],
) -> list[float]:
    """Per-sample weight = source_weights[src] * genre_weights[genre].

    Both weight dicts default to 1.0 for any key not present. Used to feed
    `torch.utils.data.WeightedRandomSampler` for per-source / per-genre
    upsampling of the training mix.
    """
    if len(sources) != len(genres):
        raise ValueError(f'sources/genres length mismatch: {len(sources)} vs {len(genres)}')
    return [
        DEFAULT_REBALANCE_SOURCE.get(src, 1.0) * DEFAULT_REBALANCE_GENRE.get(g, 1.0)
        for src, g in zip(sources, genres, strict=True)
    ]

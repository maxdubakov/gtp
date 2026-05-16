"""Stage 2 runtime data layer: loads pre-augmented JSONLs, vends padded batches.

The offline build pipeline (scripts/stage2/build_aug_dataset.py) is responsible for
producing the JSONL files this module reads. That script does the stratified split
and capo augmentation; this module only handles:
  - reading pre-built JSONLs
  - online tuning augmentation in TabDataset.__getitem__ (when augment=True)
  - assembling padded torch tensors
"""

import json
import random
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset

from gtp.log import info
from gtp.stage2.paths import AUG_DATA_DIR
from gtp.stage2.tokenizer import MAX_FRET, Vocabulary, tokenize_piece

MIN_NOTES_PER_PIECE = 10
MIN_NOTE_DURATION = 0.001  # seconds

# 6-string standard tunings used for online tuning augmentation in TabDataset.
# Each entry is open-string pitches in convention: string 1 = high E (index 0).
STANDARD_TUNINGS = [
    [64, 59, 55, 50, 45, 40],  # standard E A D G B E
    [63, 58, 54, 49, 44, 39],  # half-step down
    [62, 57, 53, 48, 43, 38],  # full-step down (D)
    [64, 59, 55, 50, 45, 40],  # drop-D
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


def random_tuning(piece, rng):
    """Apply random tuning augmentation. Preserves capo, string, fret; replaces tuning.

    Pieces with non-6-string tunings are returned unchanged.
    """
    if len(piece['tuning']) != 6:
        return piece
    base = rng.choice(STANDARD_TUNINGS)
    capo = piece['capo']
    new_tuning = [t + capo for t in base]
    new_notes = [{**n, 'pitch': new_tuning[n['string'] - 1] + n['fret']} for n in piece['notes']]
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


def _piece_id(piece):
    """Synthesize a stable piece identifier from source/filename/capo.

    Mirrors the convention used by dump_eval_predictions.py so per-piece
    metrics line up across the train-time and post-hoc analysis pipelines.
    """
    src = piece.get('source', '?')
    fname = piece.get('filename', piece.get('file', '?'))
    capo = piece.get('capo', 0)
    return f'{src}:{fname}:capo{capo}'


class TabDataset(Dataset):
    """PyTorch Dataset for Stage 2 (MIDI → Tab) training.

    Each item is (encoder_ids, decoder_ids, source, piece_id), padded to max_seq_len.
    `source` and `piece_id` travel with the sample so per-source / per-piece eval
    work regardless of shuffling.

    When augment=True, applies random tuning augmentation per __getitem__ and
    re-tokenizes from the underlying piece. Capo is preserved (already varied
    offline by build_aug_dataset.py for the train/val splits).

    `genre_dropout` (training-side classifier-free guidance): when augment=True,
    each sample's GENRE token is replaced with `GENRE<unknown>` with this
    probability. 0.0 = always use ground-truth genre. Has no effect when
    augment=False (val/test always use ground-truth genre).
    """

    def __init__(self, pieces, vocab: Vocabulary, max_seq_len=512, augment=False, genre_dropout: float = 0.0):
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.genre_dropout = genre_dropout
        self.vocab = vocab
        self._rng = random.Random()  # un-seeded → independent per DataLoader worker

        if augment:
            # Keep originals; tokenize on demand. Build flat index from cached
            # num_subseqs annotations (written by build_aug_dataset.py). Fall back to
            # tokenizing if the field is missing.
            self.pieces = pieces
            self._index = []
            self._sources = []
            self._genres = []
            self._piece_ids = []
            for pi, piece in enumerate(pieces):
                n = piece.get('num_subseqs')
                if n is None:
                    n = len(tokenize_piece(piece, vocab, max_seq_len=max_seq_len))
                pid = _piece_id(piece)
                genre = piece.get('genre', 'unknown')
                for si in range(n):
                    self._index.append((pi, si))
                    self._sources.append(piece['source'])
                    self._genres.append(genre)
                    self._piece_ids.append(pid)
        else:
            # Pre-tokenize and cache; pieces no longer needed.
            self.sequences = []
            self._sources = []
            self._genres = []
            self._piece_ids = []
            for piece in pieces:
                seqs = tokenize_piece(piece, vocab, max_seq_len=max_seq_len)
                pid = _piece_id(piece)
                genre = piece.get('genre', 'unknown')
                self.sequences.extend(seqs)
                self._sources.extend([piece['source']] * len(seqs))
                self._genres.extend([genre] * len(seqs))
                self._piece_ids.extend([pid] * len(seqs))

    def __len__(self):
        return len(self._sources)

    def __getitem__(self, idx):
        if self.augment:
            pi, si = self._index[idx]
            piece = random_tuning(self.pieces[pi], self._rng)
            genre_override = 'unknown' if self.genre_dropout > 0 and self._rng.random() < self.genre_dropout else None
            seqs = tokenize_piece(
                piece,
                self.vocab,
                max_seq_len=self.max_seq_len,
                genre_override=genre_override,
            )
            enc_ids, dec_ids = seqs[min(si, len(seqs) - 1)]
        else:
            enc_ids, dec_ids = self.sequences[idx]
        return (
            self._pad(enc_ids),
            self._pad(dec_ids),
            self._sources[idx],
            self._piece_ids[idx],
        )

    def _pad(self, ids):
        padded = ids + [self.vocab.pad_id] * (self.max_seq_len - len(ids))
        return torch.tensor(padded[: self.max_seq_len], dtype=torch.long)

    @property
    def sources(self):
        return self._sources

    @property
    def genres(self):
        return self._genres


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


def build_datasets(
    vocab: Vocabulary,
    splits: tuple[str, ...] = ('train', 'val', 'test'),
    max_seq_len: int = 512,
    augment_train: bool = True,
    genre_dropout: float = 0.0,
) -> tuple[dict[str, 'TabDataset'], dict]:
    """Build TabDatasets for the requested splits from the augmented JSONLs.

    Requires that scripts/stage2/build_aug_dataset.py has been run. Only the
    splits in `splits` are loaded — pass e.g. `splits=('val', 'test')` from
    eval.py to skip the 3 GB train.jsonl.

    `augment_train` and `genre_dropout` only apply to the 'train' split.

    Returns ({split: TabDataset}, stats_dict).
    """
    valid = {'train', 'val', 'test'}
    bad = set(splits) - valid
    if bad:
        raise ValueError(f'Unknown split(s): {sorted(bad)}. Allowed: {sorted(valid)}')

    info(f'Loading augmented JSONLs from {AUG_DATA_DIR}: {list(splits)}')
    pieces: dict[str, list[dict]] = {}
    for split in splits:
        path = AUG_DATA_DIR / f'{split}.jsonl'
        if not path.exists():
            raise FileNotFoundError(
                f'Augmented JSONL not found: {path}. Run scripts/stage2/build_aug_dataset.py first.'
            )
        pieces[split] = load_jsonl_pieces(path)

    datasets: dict[str, TabDataset] = {}
    for split, ps in pieces.items():
        augment = augment_train and split == 'train'
        dropout = genre_dropout if split == 'train' else 0.0
        datasets[split] = TabDataset(ps, vocab, max_seq_len=max_seq_len, augment=augment, genre_dropout=dropout)

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

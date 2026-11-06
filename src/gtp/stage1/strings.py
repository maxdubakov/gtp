"""Per-note string classifier on GuitarSet (stage 1 string detection, v0).

A note is classified from a spectrogram window around its ground-truth onset.
Features are cached once per (audio source, feature type) for the widest window;
shorter windows are cropped from the cache at train time.

Conventions: string 1 = high E (MIDI 64) ... string 6 = low E (MIDI 40), as in
the processed GuitarSet JSON. Class index = string - 1.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn

from gtp import REPO_ROOT

GUITARSET_DIR = Path(REPO_ROOT) / 'data' / 'guitarset'
CACHE_DIR = GUITARSET_DIR / 'string_cache'

OPEN_STRINGS = np.array([64, 59, 55, 50, 45, 40])
MAX_FRET = 19
N_STRINGS = 6

SAMPLE_RATE = 22050
HOP = 256
PRE_FRAMES = 4
MAX_POST_FRAMES = 35
WINDOW_FRAMES = PRE_FRAMES + MAX_POST_FRAMES

# The hex file has one channel per string, low E first, so it gives the string away:
# it is a label/alignment sanity check, never a real input.
SOURCES = {
    'mic': ('audio_mono-mic', '_mic.wav'),
    'mix': ('audio_mono-pickup_mix', '_mix.wav'),
    'hex': ('audio_hex-pickup_debleeded', '_hex_cln.wav'),
}
FEATURES = ('logmel64', 'logmel229', 'cqt')
SPLITS = {'train': ('00', '01', '02', '03'), 'val': ('04',), 'test': ('05',)}


@dataclass
class Notes:
    """All annotated GuitarSet notes, one row per note, in a fixed order shared by every cache."""

    track: np.ndarray
    player: np.ndarray
    is_solo: np.ndarray
    onset: np.ndarray
    pitch: np.ndarray
    string: np.ndarray

    @property
    def label(self):
        return self.string - 1

    @property
    def candidates(self):
        fret = self.pitch[:, None] - OPEN_STRINGS[None, :]
        return (fret >= 0) & (fret <= MAX_FRET)

    def split_mask(self, split):
        return np.isin(self.player, SPLITS[split])


def load_notes():
    rows = []
    for path in sorted((GUITARSET_DIR / 'processed').glob('*.json')):
        for n in json.loads(path.read_text())['notes']:
            rows.append((path.stem, path.stem[:2], path.stem.endswith('_solo'), n['start'], n['pitch'], n['string']))
    track, player, is_solo, onset, pitch, string = zip(*rows, strict=True)
    return Notes(
        track=np.array(track),
        player=np.array(player),
        is_solo=np.array(is_solo),
        onset=np.array(onset, dtype=np.float64),
        pitch=np.array(pitch, dtype=np.int64),
        string=np.array(string, dtype=np.int64),
    )


def spectrogram(audio, feature):
    """(channels, samples) at SAMPLE_RATE -> (channels, freq, frames) log-magnitude."""
    if feature.startswith('logmel'):
        n_mels = int(feature.removeprefix('logmel'))
        spec = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=HOP, n_mels=n_mels, fmin=30.0, fmax=SAMPLE_RATE / 2
        )
    elif feature == 'cqt':
        spec = (
            np.abs(
                librosa.cqt(
                    audio, sr=SAMPLE_RATE, hop_length=HOP, fmin=librosa.midi_to_hz(36), n_bins=168, bins_per_octave=24
                )
            )
            ** 2
        )
    else:
        raise ValueError(f'Unknown feature {feature!r}, expected one of {FEATURES}')
    return np.log(spec + 1e-8).astype(np.float32)


def load_audio(track, source):
    subdir, suffix = SOURCES[source]
    audio, sr = sf.read(GUITARSET_DIR / subdir / f'{track}{suffix}', dtype='float32', always_2d=True)
    return librosa.resample(audio.T, orig_sr=sr, target_sr=SAMPLE_RATE)


def note_windows(spec, onsets):
    """Crop (channels, freq, PRE+MAX_POST) windows starting PRE_FRAMES before each onset."""
    starts = np.round(onsets * SAMPLE_RATE / HOP).astype(int) - PRE_FRAMES
    pad = WINDOW_FRAMES + PRE_FRAMES
    padded = np.pad(spec, ((0, 0), (0, 0), (pad, pad)), constant_values=spec.min())
    return np.stack([padded[:, :, s + pad : s + pad + WINDOW_FRAMES] for s in starts])


def cache_path(source, feature):
    return CACHE_DIR / f'{source}_{feature}.npy'


class StringNet(nn.Module):
    """TART-style per-note encoder (3 conv blocks) + pitch embedding -> logits over the 6 strings."""

    def __init__(self, in_channels, n_freq, n_frames, width=1.0, use_pitch=True):
        super().__init__()
        chans = [max(4, round(c * width)) for c in (32, 64, 128)]
        layers, prev = [], in_channels
        for c in chans:
            layers += [nn.Conv2d(prev, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            prev = c
        self.encoder = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            enc_dim = self.encoder(torch.zeros(1, in_channels, n_freq, n_frames)).shape[1]
        self.proj = nn.Sequential(nn.Linear(enc_dim, 256), nn.LayerNorm(256), nn.ReLU())
        self.pitch_emb = nn.Embedding(128, 16) if use_pitch else None
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256 + (16 if use_pitch else 0), 128), nn.ReLU(), nn.Linear(128, N_STRINGS)
        )

    def forward(self, x, pitch, candidates):
        h = self.proj(self.encoder(x))
        if self.pitch_emb is not None:
            h = torch.cat([h, self.pitch_emb(pitch)], dim=1)
        return self.head(h).masked_fill(~candidates, float('-inf'))


def pitch_prior(notes, train_mask, alpha=1.0):
    """P(string | pitch) from train counts, add-alpha smoothed over each note's valid strings."""
    counts = np.zeros((128, N_STRINGS))
    np.add.at(counts, (notes.pitch[train_mask], notes.label[train_mask]), 1)
    cand = notes.candidates
    p = (counts[notes.pitch] + alpha) * cand
    return p / p.sum(axis=1, keepdims=True)


def summarize(probs, labels):
    """Accuracy, NLL in bits and 15-bin ECE for (N, 6) probabilities."""
    p_true = probs[np.arange(len(labels)), labels]
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == labels
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in np.unique(bins))
    return {
        'n': len(labels),
        'acc': float(correct.mean()),
        'nll_bits': float(-np.log2(np.clip(p_true, 1e-12, None)).mean()),
        'ece': float(ece),
    }

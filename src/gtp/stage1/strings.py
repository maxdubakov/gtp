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
from hydra.utils import instantiate
from omegaconf import OmegaConf
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
PLAYERS = ('00', '01', '02', '03', '04', '05')


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
        return candidates(self.pitch)

    def split_mask(self, players):
        return np.isin(self.player, players)


def candidates(pitch):
    """(N,) MIDI pitches -> (N, 6) bool, True where the string can play the pitch."""
    fret = np.asarray(pitch)[:, None] - OPEN_STRINGS[None, :]
    return (fret >= 0) & (fret <= MAX_FRET)


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


def split_players(test_player, exclude=()):
    """Leave-one-player-out fold: `test_player` is test, the previous remaining player (cyclically) is val.

    `exclude` players are left out of every split (e.g. a downstream model's test player).
    """
    test = PLAYERS[test_player]
    kept = [p for i, p in enumerate(PLAYERS) if i not in exclude]
    assert test in kept, (test_player, exclude)
    val = kept[kept.index(test) - 1]
    return {'train': tuple(p for p in kept if p not in (test, val)), 'val': (val,), 'test': (test,)}


def post_frames(window_ms):
    frames = round(window_ms / 1000 * SAMPLE_RATE / HOP)
    if not 1 <= frames <= MAX_POST_FRAMES:
        raise ValueError(f'window_ms={window_ms} gives {frames} frames, cache holds 1..{MAX_POST_FRAMES}')
    return frames


def freq_stats(x, idx, chunk=4096):
    total, total_sq, n = 0.0, 0.0, 0
    for i in range(0, len(idx), chunk):
        b = torch.from_numpy(x[idx[i : i + chunk]]).float()
        total += b.sum(dim=(0, 3))
        total_sq += (b**2).sum(dim=(0, 3))
        n += b.shape[0] * b.shape[3]
    mean = total / n
    std = (total_sq / n - mean**2).clamp_min(1e-6).sqrt()
    return mean[None, :, :, None], std[None, :, :, None]


def cache_path(source, feature):
    return CACHE_DIR / f'{source}_{feature}.npy'


def cache_layout():
    """Constants a cache was built with; a cache built under different ones is stale."""
    return {'sample_rate': SAMPLE_RATE, 'hop': HOP, 'pre_frames': PRE_FRAMES, 'max_post_frames': MAX_POST_FRAMES}


def load_cache(source, feature, n_notes):
    """Memmap of cached windows plus its build metadata, checked against the current notes and layout."""
    path = cache_path(source, feature)
    meta_path = path.with_suffix('.json')
    if not meta_path.exists():
        raise FileNotFoundError(f'{meta_path} missing: rebuild with scripts/stage1/strings/build_cache.py')
    meta = json.loads(meta_path.read_text())
    x = np.load(path, mmap_mode='r')
    if len(x) != n_notes or meta['layout'] != cache_layout():
        raise ValueError(f'{path} is stale ({len(x)} rows, layout {meta["layout"]}): rebuild it')
    return x, meta


class StringNet(nn.Module):
    """TART-style per-note encoder (3 conv blocks) + pitch embedding -> logits over the 6 strings.

    `pool_first` pools before BatchNorm/ReLU: about 1.4x faster on MPS, where BN/ReLU on the
    full-resolution first block dominates step time; BN then normalizes pooled activations.
    """

    def __init__(self, in_channels, n_freq, n_frames, width=1.0, use_pitch=True, pool_first=False):
        super().__init__()
        chans = [max(4, round(c * width)) for c in (32, 64, 128)]
        layers, prev = [], in_channels
        for c in chans:
            conv, bn, relu, pool = nn.Conv2d(prev, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)
            layers += [conv, pool, bn, relu] if pool_first else [conv, bn, relu, pool]
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


def pitch_prior(notes, train_mask, pitches=None, alpha=1.0):
    """P(string | pitch) from train counts, add-alpha smoothed over valid strings, for `pitches` (default: every note)."""
    counts = np.zeros((128, N_STRINGS))
    np.add.at(counts, (notes.pitch[train_mask], notes.label[train_mask]), 1)
    pitches = notes.pitch if pitches is None else np.asarray(pitches)
    p = (counts[pitches] + alpha) * candidates(pitches)
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


@dataclass
class StringClassifier:
    """A trained string-classifier run, ready to score arbitrary (onset, pitch) notes."""

    model: nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    temperature: float
    n_frames: int
    source: str
    feature: str
    prior_notes: Notes
    prior_mask: np.ndarray
    device: str

    @classmethod
    def from_run(cls, run_dir, device):
        """Rebuild a run from its saved config; input normalization is recomputed from its training notes."""
        cfg = OmegaConf.load(Path(run_dir) / '.hydra' / 'config.yaml')
        notes = load_notes()
        n_frames = PRE_FRAMES + post_frames(cfg.data.window_ms)
        cache, _ = load_cache(cfg.data.source, cfg.data.feature, len(notes.pitch))
        train_mask = notes.split_mask(split_players(cfg.test_player, cfg.get('exclude_players', ()))['train'])
        mean, std = freq_stats(cache[..., :n_frames], np.flatnonzero(train_mask))
        model = instantiate(cfg.model, in_channels=cache.shape[1], n_freq=cache.shape[2], n_frames=n_frames)
        model.load_state_dict(torch.load(Path(run_dir) / 'model.pt', map_location='cpu'))
        temperature = json.loads((Path(run_dir) / 'metrics.json').read_text())['temperature']
        return cls(
            model.to(device).eval(),
            mean.to(device),
            std.to(device),
            temperature,
            n_frames,
            cfg.data.source,
            cfg.data.feature,
            notes,
            train_mask,
            device,
        )

    def prior(self, pitches):
        return pitch_prior(self.prior_notes, self.prior_mask, pitches)

    @torch.no_grad()
    def predict(self, spec, onsets, pitches, batch=512):
        """Calibrated (N, 6) string probabilities for notes in one track's spectrogram."""
        windows = note_windows(spec, np.asarray(onsets))[..., : self.n_frames]
        pitch = torch.as_tensor(np.asarray(pitches), dtype=torch.long)
        cand = torch.as_tensor(candidates(pitches))
        out = []
        for i in range(0, len(windows), batch):
            x = (torch.from_numpy(windows[i : i + batch]).to(self.device).float() - self.mean) / self.std
            logits = self.model(x, pitch[i : i + batch].to(self.device), cand[i : i + batch].to(self.device))
            out.append(torch.softmax(logits / self.temperature, dim=1).cpu())
        return torch.cat(out).numpy()

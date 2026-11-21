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
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from gtp import REPO_ROOT

GUITARSET_DIR = Path(REPO_ROOT) / 'data' / 'guitarset'
CACHE_DIR = GUITARSET_DIR / 'string_cache'

OPEN_STRINGS = np.array([64, 59, 55, 50, 45, 40])
# String-set families, told apart by the G string (inharmonicity.STRING_SETS has their typical constants).
FAMILIES = ('plain_g', 'wound_g')
MAX_FRET = 24
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
FEATURES = ('logmel64', 'logmel229', 'cqt', 'cqt36')
CQT36_LOWEST = 33
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
    elif feature == 'cqt36':
        # A third of a semitone per bin from A1, with filters half the usual length so fast notes stay apart.
        spec = (
            np.abs(
                librosa.cqt(
                    audio,
                    sr=SAMPLE_RATE,
                    hop_length=HOP,
                    fmin=librosa.midi_to_hz(CQT36_LOWEST),
                    n_bins=252,
                    bins_per_octave=36,
                    filter_scale=0.5,
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


def note_windows(spec, onsets, pre=PRE_FRAMES, post=MAX_POST_FRAMES):
    """Crop (channels, freq, pre+post) windows starting `pre` frames before each onset."""
    starts = np.round(onsets * SAMPLE_RATE / HOP).astype(int) - pre
    pad = 2 * pre + post
    padded = np.pad(spec, ((0, 0), (0, 0), (pad, pad)), constant_values=spec.min())
    return np.stack([padded[:, :, s + pad : s + pad + pre + post] for s in starts])


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


def cache_layout(pre=PRE_FRAMES, post=MAX_POST_FRAMES):
    """Constants a cache was built with; a cache built under different ones is stale."""
    return {'sample_rate': SAMPLE_RATE, 'hop': HOP, 'pre_frames': pre, 'max_post_frames': post}


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


POOLS = {
    'flatten': lambda h: h.flatten(1),
    'mean': lambda h: h.mean(dim=3).flatten(1),
    'meanmax': lambda h: torch.cat([h.mean(dim=3), h.amax(dim=3)], dim=1).flatten(1),
}


class StringNet(nn.Module):
    """TART-style per-note encoder (3 conv blocks) + pitch embedding -> logits over the 6 strings.

    `pool_first` pools before BatchNorm/ReLU: about 1.4x faster on MPS, where BN/ReLU on the
    full-resolution first block dominates step time; BN then normalizes pooled activations.
    `pool` collapses the encoder's time axis before the projection: 'flatten' keeps every time cell (TART),
    'mean' averages them, 'meanmax' concatenates their mean and max (as in PANNs).
    `use_family` adds the guitar's string-set family (FAMILIES) as an input: a plain G string sounds unlike a wound one,
    so which timbre means "G string" depends on the family.
    `n_classes` other than the six strings makes it a classifier of something else heard in the note (its playing
    technique); `candidates` then is an all-true mask.
    """

    def __init__(
        self,
        in_channels,
        n_freq,
        n_frames,
        width=1.0,
        use_pitch=True,
        pool_first=False,
        pool='flatten',
        use_family=False,
        n_classes=N_STRINGS,
    ):
        super().__init__()
        chans = [max(4, round(c * width)) for c in (32, 64, 128)]
        layers, prev = [], in_channels
        for c in chans:
            conv, bn, relu, halve = nn.Conv2d(prev, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)
            layers += [conv, halve, bn, relu] if pool_first else [conv, bn, relu, halve]
            prev = c
        self.encoder = nn.Sequential(*layers)
        self.pool = POOLS[pool]
        with torch.no_grad():
            enc_dim = self.pool(self.encoder(torch.zeros(1, in_channels, n_freq, n_frames))).shape[1]
        self.proj = nn.Sequential(nn.Linear(enc_dim, 256), nn.LayerNorm(256), nn.ReLU())
        self.pitch_emb = nn.Embedding(128, 16) if use_pitch else None
        self.family_emb = nn.Embedding(len(FAMILIES), 8) if use_family else None
        width_in = 256 + (16 if use_pitch else 0) + (8 if use_family else 0)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(width_in, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x, pitch, candidates, family=None):
        h = self.proj(self.pool(self.encoder(x)))
        if self.pitch_emb is not None:
            h = torch.cat([h, self.pitch_emb(pitch)], dim=1)
        if self.family_emb is not None:
            h = torch.cat([h, self.family_emb(family)], dim=1)
        return self.head(h).masked_fill(~candidates, float('-inf'))


def string_counts(pitch, label):
    """(128, 6) training counts of each string per MIDI pitch."""
    counts = np.zeros((128, N_STRINGS))
    np.add.at(counts, (np.asarray(pitch), np.asarray(label)), 1)
    return counts


def prior_from_counts(counts, pitches, alpha=1.0):
    """P(string | pitch), add-alpha smoothed over the strings that can play it."""
    pitches = np.asarray(pitches)
    p = (counts[pitches] + alpha) * candidates(pitches)
    return p / p.sum(axis=1, keepdims=True)


def pitch_prior(notes, train_mask, pitches=None, alpha=1.0):
    """P(string | pitch) from train counts, for `pitches` (default: every note)."""
    counts = string_counts(notes.pitch[train_mask], notes.label[train_mask])
    return prior_from_counts(counts, notes.pitch if pitches is None else pitches, alpha)


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


def fit_temperature(logits, labels):
    """Single temperature minimizing NLL (Guo et al. 2017); masked strings stay at -inf."""
    valid = torch.isfinite(logits)
    finite = logits.masked_fill(~valid, 0.0)
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy((finite / log_t.exp()).masked_fill(~valid, float('-inf')), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def mean_entropy(logits):
    """Mean entropy (nats) of the untempered predictions on the notes with more than one candidate string; 0 if none."""
    multi = torch.isfinite(logits).sum(dim=1) > 1
    p = torch.softmax(logits[multi], dim=1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum(dim=1).mean()) if multi.any() else 0.0


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
    prior_counts: np.ndarray
    device: str
    # Trained on recording-centred windows (train_index.py `centre`): predict_windows must get a whole recording.
    centre: bool = False
    # adapt_bn's weight on the training statistics, in windows: n windows get n / (n + bn_prior) of their own
    # (Schneider et al. 2020), so a short recording stays close to the trained model.
    bn_prior: float = 0.0
    # (a, b): each predict_windows call (one recording) is tempered by exp(a + b * mean_entropy of its logits) instead
    # of `temperature`, so a recording the model is unsure of gets softened more.
    reliability: tuple | None = None

    @classmethod
    def from_run(cls, run_dir, device):
        """Rebuild a run from its saved config. Index runs (train_index.py) saved their input normalization and
        prior in classifier.npz; GuitarSet runs recompute them from their training notes."""
        run_dir = Path(run_dir)
        cfg = OmegaConf.load(run_dir / '.hydra' / 'config.yaml')
        n_frames = PRE_FRAMES + post_frames(cfg.data.window_ms)
        saved = run_dir / 'classifier.npz'
        if saved.exists():
            z = np.load(saved)
            mean, std, counts = torch.from_numpy(z['mean']), torch.from_numpy(z['std']), z['prior_counts']
        else:
            notes = load_notes()
            cache, _ = load_cache(cfg.data.source, cfg.data.feature, len(notes.pitch))
            train_mask = notes.split_mask(split_players(cfg.test_player, cfg.get('exclude_players', ()))['train'])
            mean, std = freq_stats(cache[..., :n_frames], np.flatnonzero(train_mask))
            counts = string_counts(notes.pitch[train_mask], notes.label[train_mask])
        model = instantiate(cfg.model, in_channels=mean.shape[1], n_freq=mean.shape[2], n_frames=n_frames)
        model.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cpu'))
        temperature = json.loads((run_dir / 'metrics.json').read_text())['temperature']
        return cls(
            model.to(device).eval(),
            mean.to(device),
            std.to(device),
            temperature,
            n_frames,
            cfg.data.source,
            cfg.data.feature,
            counts,
            device,
            cfg.get('centre', False),
        )

    def prior(self, pitches):
        return prior_from_counts(self.prior_counts, pitches)

    @property
    def uses_family(self):
        return self.model.family_emb is not None

    def predict(self, spec, onsets, pitches, batch=512, family=None, adapt=None):
        """Calibrated (N, 6) string probabilities for notes in one track's spectrogram."""
        return self.predict_windows(note_windows(spec, np.asarray(onsets)), pitches, batch, family, adapt)

    def inputs(self, windows, batch):
        """Normalized network inputs, batch by batch; a `centre` model first has the windows' mean spectrum removed."""
        shift = self.mean
        if self.centre:
            spectrum = windows[..., : self.n_frames].mean(axis=(0, 1, 3), dtype=np.float64)
            shift = shift + torch.from_numpy(spectrum).float().to(self.device)[None, None, :, None]
        for i in range(0, len(windows), batch):
            w = np.ascontiguousarray(windows[i : i + batch, ..., : self.n_frames])
            yield (torch.from_numpy(w).to(self.device).float() - shift) / self.std

    @torch.no_grad()
    def adapt_bn(self, windows, batch=512):
        """Replace every BatchNorm's running statistics with those of `windows` (AdaBN, Li et al. 2016), layer by
        layer so each layer sees the inputs of the already adapted ones. Returns the replaced statistics."""
        saved = {k: v.clone() for k, v in self.model.state_dict().items() if 'running_' in k}
        own = len(windows) / (len(windows) + self.bn_prior)
        encoder = self.model.encoder
        for i, bn in enumerate(encoder):
            if not isinstance(bn, nn.BatchNorm2d):
                continue
            total, total_sq, n = 0.0, 0.0, 0
            for x in self.inputs(windows, batch):
                h = encoder[:i](x)
                total = total + h.sum(dim=(0, 2, 3)).cpu().double()
                total_sq = total_sq + (h**2).sum(dim=(0, 2, 3)).cpu().double()
                n += h.numel() // h.shape[1]
            mean, var = total / n, (total_sq / n - (total / n) ** 2).clamp_min(0)
            bn.running_mean.lerp_(mean.to(bn.running_mean), own)
            bn.running_var.lerp_(var.to(bn.running_var), own)
        return saved

    @torch.no_grad()
    def predict_windows(self, windows, pitches, batch=512, family=None, adapt=None):
        """Calibrated (N, 6) string probabilities for cached note windows (N, 1, freq, >= n_frames).

        `family` (a name in FAMILIES) is required by a model trained with use_family. With `adapt` (windows, e.g. the
        whole recording's), the BatchNorm statistics are re-estimated on those windows for this call (adapt_bn).
        """
        saved = None if adapt is None else self.adapt_bn(adapt, batch)
        pitch = torch.as_tensor(np.asarray(pitches), dtype=torch.long)
        cand = torch.as_tensor(candidates(pitches))
        fam = torch.full((batch,), FAMILIES.index(family) if self.uses_family else 0, device=self.device)
        out = []
        for i, x in zip(range(0, len(windows), batch), self.inputs(windows, batch), strict=True):
            p, c = pitch[i : i + batch].to(self.device), cand[i : i + batch].to(self.device)
            out.append(self.model(x, p, c, fam[: len(p)]).cpu())
        if saved is not None:
            self.model.load_state_dict(saved, strict=False)
        logits = torch.cat(out)
        t = self.temperature
        if self.reliability is not None:
            t = float(np.exp(self.reliability[0] + self.reliability[1] * mean_entropy(logits)))
        return torch.softmax(logits / t, dim=1).numpy()

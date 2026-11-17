"""Harmonic branch for Kong's note CRNN, after Harmonica (Ou et al. 2026, arXiv 2609.04640).

A log-frequency spectrogram at three bins per semitone (MIDI 21-108) goes through a small frontend, is reduced to one
bin per pitch, and then through `blocks` trunk blocks, each a harmonic convolution (shift-and-aggregate over the
harmonics 2..`harmonics` and their subharmonics, plus the bin itself) and a 3x3 residual block. Its output is added to
the GRU input of each of the four sub-models through a zero-initialised projection, so a model with a new branch
computes exactly what it computed without one.
"""

import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gtp.stage1.model.kong import Regress_onset_offset_frame_velocity_CRNN

SAMPLE_RATE = 16000
HOP = 160
LOWEST_PITCH = 21
PITCHES = 88
BINS_PER_PITCH = 3
# Each log bin reads from the shortest window whose bin spacing is at most half a semitone at its frequency; Kong's
# own 2048-sample STFT covers the bins from about 263 Hz up.
N_FFTS = (2048, 4096, 8192)
MOMENTUM = 0.01


def harmonic_offsets(harmonics):
    """Semitone distances to harmonics 2..`harmonics`, their subharmonics, and 0."""
    up = [round(12 * math.log2(k)) for k in range(2, harmonics + 1)]
    return [-o for o in reversed(up)] + [0] + up


def log_filterbanks():
    """[(n_fft, first STFT bin, weights (STFT bins, log bins))] for the log-frequency axis.

    Triangular filters between neighbouring log bins, widened to the STFT bin spacing where they are narrower, each
    summing to 1; a log bin has weights in exactly one of the filterbanks.
    """
    pitch = LOWEST_PITCH + (np.arange(PITCHES * BINS_PER_PITCH) - 1) / BINS_PER_PITCH
    centre = 440 * 2 ** ((pitch - 69) / 12)
    step = 2 ** (1 / (12 * BINS_PER_PITCH))
    semitone = centre * (2 ** (1 / 12) - 1)
    choice = np.full(len(centre), len(N_FFTS) - 1)
    for i, n_fft in reversed(list(enumerate(N_FFTS))):
        choice[SAMPLE_RATE / n_fft <= semitone / 2] = i
    banks = []
    for i, n_fft in enumerate(N_FFTS):
        df = SAMPLE_RATE / n_fft
        freqs = np.arange(n_fft // 2 + 1) * df
        lo = np.maximum(centre - centre / step, df)
        hi = np.maximum(centre * step - centre, df)
        d = freqs[:, None] - centre[None, :]
        w = np.clip(np.minimum(1 + d / lo, 1 - d / hi), 0, None)
        w[:, choice != i] = 0
        w /= np.maximum(w.sum(0, keepdims=True), 1e-12)
        rows = np.flatnonzero(w.sum(1))
        if len(rows):
            banks.append((n_fft, int(rows[0]), torch.tensor(w[rows[0] : rows[-1] + 1], dtype=torch.float32)))
    return banks


def bn(channels):
    return nn.BatchNorm2d(channels, momentum=MOMENTUM)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, bias=False)
        self.bn1 = bn(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, bias=False)
        self.bn2 = bn(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(x + self.bn2(self.conv2(y)))


class HarmonicConv(nn.Module):
    """Harmonica's shift-and-aggregate harmonic convolution, with its normalisation, residual and ReLU."""

    def __init__(self, channels, offsets):
        super().__init__()
        self.offsets = offsets
        self.conv = nn.Conv2d(channels * len(offsets), channels, 1, bias=False)
        self.bn = bn(channels)

    def forward(self, x):
        pad = max(abs(o) for o in self.offsets)
        xp = F.pad(x, (pad, pad))
        bins = x.shape[-1]
        shifted = torch.cat([xp[..., pad + o : pad + o + bins] for o in self.offsets], dim=1)
        return F.relu(x + self.bn(self.conv(shifted)))


class HarmonicBranch(nn.Module):
    def __init__(self, channels=32, blocks=4, harmonics=8, out_channels=8, heads=4, gru_input=768):
        super().__init__()
        self.banks = log_filterbanks()
        for n_fft, _, w in self.banks:
            self.register_buffer(f'fb_{n_fft}', w, persistent=False)
            if n_fft != N_FFTS[0]:
                self.register_buffer(f'window_{n_fft}', torch.hann_window(n_fft), persistent=False)
        front = channels // 4
        self.bn0 = bn(PITCHES * BINS_PER_PITCH)
        self.stem = nn.Sequential(nn.Conv2d(1, front, 5, padding=2, bias=False), bn(front), nn.ReLU())
        self.frontend = nn.Sequential(ResBlock(front, 5), ResBlock(front, 5))
        self.aggregate = nn.Conv2d(front, channels, (1, BINS_PER_PITCH), stride=(1, BINS_PER_PITCH), bias=False)
        self.bn_aggregate = bn(channels)
        offsets = harmonic_offsets(harmonics)
        self.blocks = nn.ModuleList(
            nn.Sequential(HarmonicConv(channels, offsets), ResBlock(channels, 3)) for _ in range(blocks)
        )
        self.out = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn_out = bn(out_channels)
        self.proj = nn.ModuleList(nn.Linear(out_channels * PITCHES, gru_input, bias=False) for _ in range(heads))
        for p in self.proj:
            nn.init.zeros_(p.weight)

    def log_spectrogram(self, audio, spec):
        """(batch, 1, frames, log bins) in dB, from the audio and Kong's (batch, 1, frames, 1025) power spectrogram."""
        x = 0
        for n_fft, first, _ in self.banks:
            w = getattr(self, f'fb_{n_fft}')
            if n_fft == N_FFTS[0]:
                band = spec[:, 0, :, first : first + len(w)]
            else:
                stft = torch.stft(
                    audio, n_fft, HOP, window=getattr(self, f'window_{n_fft}'), center=True, pad_mode='reflect',
                    return_complex=True,
                )  # fmt: skip
                band = stft[:, first : first + len(w)].abs().pow(2).transpose(1, 2)
            x = x + band @ w
        return 10 * torch.log10(x.clamp(min=1e-10))[:, None]

    def forward(self, audio, spec):
        """[(batch, frames, gru_input)] per head, from the audio and Kong's power spectrogram."""
        x = self.bn0(self.log_spectrogram(audio, spec).transpose(1, 3)).transpose(1, 3)
        x = self.frontend(self.stem(x))
        x = F.relu(self.bn_aggregate(self.aggregate(x)))
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.bn_out(self.out(x)))
        x = x.transpose(1, 2).flatten(2)
        return [p(x) for p in self.proj]


def branch_settings(state):
    """HarmonicBranch settings of a state dict that has a branch."""
    channels = state['harmonic.aggregate.weight'].shape[0]
    offsets = state['harmonic.blocks.0.0.conv.weight'].shape[1] // channels
    return dict(
        channels=channels,
        blocks=len({m.group(1) for k in state if (m := re.match(r'harmonic\.blocks\.(\d+)\.', k))}),
        harmonics=(offsets - 1) // 2 + 1,
        out_channels=state['harmonic.out.weight'].shape[0],
    )


def note_model(state, harmonic=None):
    """Kong's note model with `state` loaded: with its harmonic branch if `state` has one, else with a new
    zero-initialised branch of settings `harmonic` (a dict, see HarmonicBranch) if given."""
    model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=100, classes_num=88)
    if any(k.startswith('harmonic.') for k in state):
        model.harmonic = HarmonicBranch(**branch_settings(state))
        model.load_state_dict(state)
    else:
        model.load_state_dict(state)
        if harmonic:
            model.harmonic = HarmonicBranch(**harmonic)
    return model

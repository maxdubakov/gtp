"""Playing-technique labels of notes."""

import librosa
import numpy as np
import pandas as pd
import torch
from torch import nn

from gtp.stage1.strings import CQT36_LOWEST, HOP, POOLS, SAMPLE_RATE


def class_of(techniques, classes, other_marks):
    """Index into `classes` per note, or -1 for a note to leave out."""
    out = np.full(len(techniques), -1)
    for i, marks in enumerate(techniques.fillna('').str.split(',')):
        named = [classes.index(m) for m in marks if m in classes]
        if named:
            out[i] = max(named)
        elif not set(marks) & set(other_marks):
            out[i] = classes.index('plain')
    return out


def head_labels(notes, heads, unlabelled):
    """{head: class index per note}, -1 where the head has no label for the note.

    A head's class is the last of its `classes` one of whose marks the note carries, the first class for a note
    with none; a note with a mark in the head's `unknown` has no label. Each rule of `unlabelled` removes the
    labels of its `heads` from the notes of a `dataset` (and of its `content`, of the recordings `recs`, and
    carrying one of `marks`, where given).
    """
    marks = [set(t.split(',')) - {''} if isinstance(t, str) else set() for t in notes['techniques']]

    def carrying(names):
        return np.array([bool(m & set(names)) for m in marks])

    out = {}
    for name, head in heads.items():
        label = np.zeros(len(notes), dtype=np.int64)
        for k, of in enumerate(head.classes.values()):
            label[carrying(of)] = k
        label[carrying(head.unknown)] = -1
        out[name] = label
    for rule in unlabelled:
        hit = (notes['dataset'] == rule.dataset).to_numpy()
        if rule.get('content'):
            hit = hit & (notes['content'] == rule.content).to_numpy()
        if rule.get('recs'):
            hit = hit & notes['rec'].isin(list(rule.recs)).to_numpy()
        if rule.get('marks'):
            hit = hit & carrying(rule.marks)
        for name in rule.heads:
            out[name][hit] = -1
    return out


def pitch_comb(n_mels, harmonics):
    """(128, n_mels) table: for each MIDI pitch, ones at the log-mel bins (gtp.stage1.strings.spectrogram) of its
    first `harmonics` partials. As an input channel it shows a network which of the sounding notes is meant."""
    centres = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=30.0, fmax=SAMPLE_RATE / 2)[1:-1]
    table = np.zeros((128, n_mels), dtype=np.float32)
    for pitch in range(128):
        partials = librosa.midi_to_hz(pitch) * np.arange(1, harmonics + 1)
        partials = partials[(partials > centres[0]) & (partials < centres[-1])]
        table[pitch, np.abs(centres[None, :] - partials[:, None]).argmin(axis=1)] = 1
    return table


def pitch_windows(spec, onsets, pitches, pre, post, semitones, harmonics):
    """Windows of a 'cqt36' spectrogram that follow each note's pitch: (notes, harmonics, bins, pre + post), the
    bins reaching `semitones` below and above the pitch (channel 0) and above its next partials (the others), from
    `pre` frames before the onset. A bend, a slide or the note before then looks the same at every pitch."""
    reach = 3 * semitones
    partial = np.round(36 * np.log2(np.arange(1, harmonics + 1))).astype(int)
    pad_t, pad_f = 2 * pre + post, reach + 36
    padded = np.pad(spec[0], ((pad_f, pad_f + partial[-1]), (pad_t, pad_t)), constant_values=spec.min())
    starts = np.round(onsets * SAMPLE_RATE / HOP).astype(int) - pre + pad_t
    rows = (np.asarray(pitches) - CQT36_LOWEST) * 3 + pad_f
    return np.stack(
        [
            np.stack([padded[r + k - reach : r + k + reach + 1, s : s + pre + post] for k in partial])
            for r, s in zip(rows, starts, strict=True)
        ]
    )


def string_context(notes):
    """(notes, 4) numbers about the note before each note on its string, zeros where there is none: a one, that
    note's fret minus this one's in octaves, and the seconds since it started and since it ended (negative while it
    still sounds), each clipped. A hammer-on, a pull-off or a slide is a way of getting from that note to this one."""
    ordered = notes.sort_values(['rec', 'string', 'onset'], kind='stable')
    before = ordered.groupby(['rec', 'string'], sort=False)[['onset', 'offset', 'fret']].shift()
    out = np.stack(
        [
            np.ones(len(ordered)),
            ((before['fret'] - ordered['fret']) / 12).clip(-1, 1),
            (ordered['onset'] - before['onset']).clip(0, 2),
            (ordered['onset'] - before['offset']).clip(-1, 1),
        ],
        axis=1,
    )
    out[before['onset'].isna().to_numpy()] = 0
    return pd.DataFrame(out, index=ordered.index).reindex(notes.index).to_numpy(dtype=np.float32)


def attack_burst(windows, pre, reach=12, lowest_bin=24):
    """How far the level of the upper bins jumps near each note's onset, in the spectrogram's units: the largest
    rise of a three-frame peak over the seven frames before it, searched `reach` frames either side of frame `pre`.
    `windows` are (notes, bins, frames) of the 'attack48' feature. A pick stroke is a burst there; a hammer-on, a
    pull-off or a legato slide has next to none."""
    level = windows[:, lowest_bin:].mean(axis=1)
    rises = [
        level[:, t : t + 3].max(axis=1) - level[:, t - 9 : t - 2].mean(axis=1)
        for t in range(pre - reach, pre + reach + 2)
    ]
    return np.max(rises, axis=0)


class TechniqueNet(nn.Module):
    """Per-note classifier of playing techniques. Each view of the note (a spectrogram window of some kind, given as
    its (channels, bins, frames)) goes through three conv blocks of its own; the views' features, the note's pitch
    and, with `context`, that many numbers about its surroundings (string_context) go to one head. `pool` is how a
    view's time axis is collapsed (gtp.stage1.strings.POOLS)."""

    def __init__(self, views, n_classes, width=1.0, pool='flatten', dropout=0.3, conv_dropout=0.0, context=0):
        super().__init__()
        self.pool = POOLS[pool]
        self.encoders, self.projs = nn.ModuleList(), nn.ModuleList()
        for view in views:
            layers, prev = [], view[0]
            for c in (max(4, round(c * width)) for c in (32, 64, 128)):
                layers += [nn.Conv2d(prev, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
                layers += [nn.Dropout2d(conv_dropout)]
                prev = c
            encoder = nn.Sequential(*layers)
            with torch.no_grad():
                dim = self.pool(encoder(torch.zeros(1, *view))).shape[1]
            self.encoders.append(encoder)
            self.projs.append(nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256), nn.ReLU()))
        self.pitch_emb = nn.Embedding(128, 16)
        width_in = 256 * len(views) + 16 + context
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width_in, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, views, pitch, context=None):
        h = [proj(self.pool(enc(x))) for enc, proj, x in zip(self.encoders, self.projs, views, strict=True)]
        h += [self.pitch_emb(pitch)] + ([] if context is None else [context])
        return self.head(torch.cat(h, dim=1))

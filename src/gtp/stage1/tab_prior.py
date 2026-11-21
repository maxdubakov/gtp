"""What a tab alone says about each note's technique: the notes around it, read by a small recurrent network.

The technique classifier hears a note; this reads where the note sits among the others (strings, frets, timing),
which is what tells a slide from a hammer-on when the two sound alike. Its probabilities multiply the classifier's.
"""

import json
from itertools import pairwise

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

FRETS = 25
# Fret steps to the neighbours on the string are clipped to this many frets either way.
STEP = 12
FAR_S = 10.0


def tab_arrays(track, start, end, string, fret, tie):
    """A tab as the arrays tab_features reads. Notes come track after track, each track in order of start.

    A tied note is no new note: it is dropped and the note it continues on its string lasts until its end. Returns
    the indices of the notes kept and, for those, {track, start, end, string (from 0), fret, before, after}; `before`
    and `after` are the positions of the note's neighbours on its string within its track, -1 where there is none.
    """
    n, tie = len(track), np.asarray(tie, dtype=bool)
    order = np.lexsort((start, string, track))
    opens = np.r_[True, (track[order][1:] != track[order][:-1]) | (string[order][1:] != string[order][:-1])]
    group_start = np.maximum.accumulate(np.where(opens, np.arange(n), 0))
    owner = np.maximum.accumulate(np.where(~tie[order], np.arange(n), -1))
    continued = tie[order] & (owner >= group_start)
    end = np.array(end, dtype=np.float32)
    np.maximum.at(end, order[owner[continued]], end[order[continued]])

    kept = np.flatnonzero(~tie)
    out = {
        'track': np.asarray(track)[kept].astype(np.int32),
        'start': np.asarray(start, dtype=np.float32)[kept],
        'end': end[kept],
        'string': (np.asarray(string)[kept] - 1).astype(np.uint8),
        'fret': np.clip(np.asarray(fret)[kept], 0, FRETS - 1).astype(np.uint8),
    }
    order = np.lexsort((out['start'], out['string'], out['track']))
    same = (out['track'][order][1:] == out['track'][order][:-1]) & (
        out['string'][order][1:] == out['string'][order][:-1]
    )
    out['before'] = np.full(len(kept), -1, dtype=np.int32)
    out['after'] = np.full(len(kept), -1, dtype=np.int32)
    out['before'][order[1:][same]] = order[:-1][same]
    out['after'][order[:-1][same]] = order[1:][same]
    return kept, out


def windows_of(track, length):
    """Starts and lengths of windows of up to `length` notes that cover every note of every track once."""
    edges = np.flatnonzero(np.r_[True, track[1:] != track[:-1], True])
    starts = np.concatenate([np.arange(a, b, length) for a, b in pairwise(edges)])
    ends = edges[np.searchsorted(edges, starts, 'right')]
    return starts, np.minimum(ends - starts, length)


def tab_features(tab, idx, speed=1.0):
    """Categories (..., 4) and numbers (..., 7) of the notes at positions `idx` of a tab (tab_arrays, as tensors).

    Categories: string, fret, and the fret steps from the note before and to the note after on the string. Numbers:
    seconds, as logs, since the previous note of the tab and until the next, the note's length, since the note
    before on its string started and until the note after starts; and, plainly, the seconds since the note before
    ended and from this note's end to the note after (negative where they overlap). All times are divided by
    `speed`, one per window, as if the piece were played that much faster.
    """
    start, end, track, fret = tab['start'], tab['end'], tab['track'], tab['fret']
    last = len(start) - 1
    prev, nxt = (idx - 1).clamp(0, last), (idx + 1).clamp(0, last)
    before, after = tab['before'][idx].long(), tab['after'][idx].long()
    known = {
        'prev': (idx > 0) & (track[prev] == track[idx]),
        'next': (idx < last) & (track[nxt] == track[idx]),
        'before': before >= 0,
        'after': after >= 0,
    }
    before, after = before.clamp(min=0), after.clamp(min=0)
    s, e = start[idx], end[idx]

    def log_gap(seconds, where=None):
        gap = (seconds / speed).clamp(0.005, FAR_S).log()
        return gap if where is None else torch.where(known[where], gap, torch.full_like(gap, np.log(FAR_S)))

    def overlap(seconds, where):
        return torch.where(known[where], (seconds / speed).clamp(-1, 1), torch.zeros_like(seconds))

    numbers = torch.stack(
        [
            log_gap(s - start[prev], 'prev'),
            log_gap(start[nxt] - s, 'next'),
            log_gap(e - s),
            log_gap(s - start[before], 'before'),
            log_gap(start[after] - s, 'after'),
            overlap(s - end[before], 'before'),
            overlap(start[after] - e, 'after'),
        ],
        dim=-1,
    )

    def step(a, b, where):
        frets = (fret[a].long() - fret[b].long()).clamp(-STEP, STEP) + STEP
        return torch.where(known[where], frets, torch.full_like(frets, 2 * STEP + 1))

    cats = [tab['string'][idx].long(), fret[idx].long(), step(idx, before, 'before'), step(after, idx, 'after')]
    return torch.stack(cats, dim=-1), numbers


class TabPrior(nn.Module):
    """Technique logits per note from a tab's notes in order (tab_features): the classes of all heads side by side,
    as TechniqueNet gives them. `lengths` are the notes that count in each row of a padded batch."""

    def __init__(self, n_classes, width=128, layers=2, dropout=0.1):
        super().__init__()
        sizes = [(6, 8), (FRETS, 16), (2 * STEP + 2, 16), (2 * STEP + 2, 16)]
        self.embeddings = nn.ModuleList(nn.Embedding(n, dim) for n, dim in sizes)
        self.inp = nn.Sequential(nn.Linear(sum(dim for _, dim in sizes) + 7, width), nn.ReLU())
        self.gru = nn.GRU(width, width, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(2 * width, n_classes)

    def forward(self, cats, numbers, lengths=None):
        h = [emb(cats[..., i]) for i, emb in enumerate(self.embeddings)]
        h = self.inp(torch.cat([*h, numbers], dim=-1))
        if lengths is not None:
            h = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.gru(h)
        if lengths is not None:
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=cats.shape[1])
        return self.head(h)


def read_tab(model, tab, device, length=4096, batch=64):
    """The model's logits (notes, classes) for every note of a tab (tab_arrays, as numpy), each track read in
    windows of `length` notes."""
    on_device = {k: torch.from_numpy(v).to(device) for k, v in tab.items()}
    starts, lengths = windows_of(tab['track'], length)
    out = np.zeros((len(tab['track']), model.head.out_features), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            s = torch.from_numpy(starts[i : i + batch]).to(device)
            n = torch.from_numpy(lengths[i : i + batch]).to(device)
            idx = (s[:, None] + torch.arange(int(n.max()), device=device)[None]).clamp(max=len(out) - 1)
            logits = model(*tab_features(on_device, idx), n)
            inside = torch.arange(idx.shape[1], device=device)[None] < n[:, None]
            out[idx[inside].cpu().numpy()] = logits[inside].cpu().numpy()
    return out


def log_softmax(x):
    return torch.log_softmax(torch.from_numpy(x), dim=1).numpy()


def index_prior(prior_dir, notes, heads, device):
    """A trained prior (train_prior.py's run directory) read over a technique index's notes, each recording as one
    tab. Per head, (notes, classes): what the prior adds to each class's log share in its training tabs, and its
    log-probabilities alone; zeros and NaN for tied notes. `heads` are the classifier's {head: classes}."""
    trained, metrics = (
        OmegaConf.load(prior_dir / '.hydra' / 'config.yaml'),
        json.loads((prior_dir / 'metrics.json').read_text()),
    )
    if metrics['heads'] != heads:
        raise ValueError(f'the prior answers {metrics["heads"]}, the classifier {heads}')
    sizes_of = [len(c) for c in heads.values()]
    model = instantiate(trained.model, n_classes=sum(sizes_of)).to(device)
    model.load_state_dict(torch.load(prior_dir / 'model.pt', map_location=device, weights_only=True))
    ordered = notes.sort_values(['rec', 'onset', 'string'], kind='stable')
    tie = ordered['techniques'].fillna('').str.split(',').map(lambda marks: 'tie' in marks).to_numpy()
    columns = (ordered[c].to_numpy() for c in ('onset', 'offset', 'string', 'fret'))
    kept, tab = tab_arrays(ordered['rec'].factorize()[0], *columns, tie)
    at = notes.index.get_indexer(ordered.index[kept])
    gains, alone = {}, {}
    parts = np.split(read_tab(model, tab, device), np.cumsum(sizes_of)[:-1], axis=1)
    for (name, classes), logits, share in zip(heads.items(), parts, metrics['shares'], strict=True):
        gains[name] = np.zeros((len(notes), len(classes)), dtype=np.float32)
        alone[name] = np.full((len(notes), len(classes)), np.nan, dtype=np.float32)
        alone[name][at] = log_softmax(logits)
        gains[name][at] = alone[name][at] - np.log(share, dtype=np.float32)
    return gains, alone

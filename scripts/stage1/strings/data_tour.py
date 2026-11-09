"""A static, scrollable tour of the string classifier's training data, built from a string index.

Writes <out_dir>/<timestamp>/index.html plus assets/ (PNG figures, 22050 Hz mono WAV clips): datasets,
splits, label distributions, listenable samples with their labels drawn on a spectrogram, label
alignment, label cleaning and each preprocessing step down to the per-note windows the classifier sees.
The page has no external dependencies, so it opens from file:// or any static server.

Usage:
  uv run scripts/stage1/strings/data_tour.py
  uv run scripts/stage1/strings/data_tour.py index_dir=/path/to/index out_dir=/tmp/tour samples_per_dataset=2
"""

import html
import itertools
import json
import time
from datetime import datetime

import hydra
import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import OmegaConf

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1 import chains
from gtp.stage1 import string_data as sd
from gtp.stage1 import strings as st

SR = st.SAMPLE_RATE
STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E']
# The first six slots of the dataviz reference palette, in an order validated for adjacent pairs.
STRING_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
DROP_COLOR = '#e34948'
INK, MUTED, GRID = '#2b2b29', '#77766f', '#d9d8d3'
SPLITS = ['train', 'val', 'test']
CHAINS = {
    'di': 'direct input from the pickup',
    'fx': 'the pickup through real effect pedals or reverbs, no amp',
    'amp-mic': 'a microphone in front of an amp',
    'acoustic-mic': 'a microphone in front of an acoustic guitar',
}
# Short single-note files share their labels by chance, so only longer ones count as a take.
MIN_TAKE_NOTES = 8
UNRESOLVED = 'Unaligned recordings'
PAIR_TOLERANCE_MS = 2 * 1000 * sd.ALIGN_HOP / sd.ALIGN_SR
CQT_LO, CQT_HI, CQT_BPS = 38, 92, 3
PITCH_TICKS = [40, 45, 50, 55, 59, 64, 76, 88]
FEATURE = 'logmel229'
MEL_FMIN = 30.0

plt.rcParams.update(
    {
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.edgecolor': MUTED,
        'axes.labelcolor': INK,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'text.color': INK,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
    }
)


class Assets:
    def __init__(self, page_dir, dpi):
        self.dir = page_dir / 'assets'
        self.dir.mkdir(parents=True)
        self.dpi = dpi

    def fig(self, fig, name):
        fig.savefig(self.dir / f'{name}.png', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return f'assets/{name}.png'

    def wav(self, y, name):
        # Peak-normalised so quiet DI files are audible; levels are not comparable across clips.
        y = np.asarray(y, dtype=np.float32)
        sf.write(self.dir / f'{name}.wav', y / (np.abs(y).max() + 1e-9) * 0.9, SR, subtype='PCM_16')
        return f'assets/{name}.wav'


def esc(x):
    return html.escape(str(x))


def figure(src, alt, caption):
    return f'<figure><img src="{src}" alt="{esc(alt)}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def audio(src):
    return f'<audio controls preload="none" src="{src}"></audio>'


def table(rows, cols=None, cls=''):
    cols = cols or list(rows[0])
    head = ''.join(f'<th>{esc(c)}</th>' for c in cols)
    body = ''.join('<tr>' + ''.join(f'<td>{r.get(c, "")}</td>' for c in cols) + '</tr>' for r in rows)
    return (
        f'<div class="tablewrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def section(sid, title, body):
    return f'<section id="{sid}"><h2>{esc(title)}</h2>{body}</section>'


def string_legend():
    items = ''.join(
        f'<span><i style="background:{c}"></i>{n} <small>(string {i + 1})</small></span>'
        for i, (n, c) in enumerate(zip(STRING_NAMES, STRING_COLORS, strict=True))
    )
    return f'<div class="legend">{items}</div>'


def num(x, digits=0):
    return f'{x:,.{digits}f}'


def pct(x, digits=1):
    return f'{100 * x:.{digits}f}%'


def counts(series, order=None):
    c = series.value_counts()
    keys = [k for k in (order or []) if k in c] + sorted(k for k in c.index if k not in (order or []))
    return ' · '.join(f'{esc(k)} {c[k]:,}' for k in keys)


def note_name(pitch):
    return librosa.midi_to_note(int(pitch), unicode=False)


def cqt_db(y):
    c = np.abs(
        librosa.cqt(
            y,
            sr=SR,
            hop_length=st.HOP,
            fmin=librosa.midi_to_hz(CQT_LO),
            n_bins=(CQT_HI - CQT_LO) * CQT_BPS,
            bins_per_octave=12 * CQT_BPS,
        )
    )
    return librosa.amplitude_to_db(c, ref=np.max)


def draw_cqt(ax, y, t0):
    c = cqt_db(y)
    dt, half = st.HOP / SR, 0.5 / CQT_BPS
    extent = [t0 - dt / 2, t0 + (c.shape[1] - 0.5) * dt, CQT_LO - half, CQT_LO + (c.shape[0] - 0.5) / CQT_BPS]
    ax.imshow(c, origin='lower', aspect='auto', cmap='gray', vmin=-75, vmax=0, extent=extent, interpolation='nearest')
    ax.set_yticks(PITCH_TICKS, [note_name(p) for p in PITCH_TICKS])
    ax.set_ylim(CQT_LO, CQT_HI - 1)
    ax.set_ylabel('pitch')


def draw_boxes(ax, notes, color=None, lw=1.3, ls='-'):
    for n in notes.itertuples():
        c = color or STRING_COLORS[n.string - 1]
        ax.add_patch(
            Rectangle((n.onset, n.pitch - 0.5), n.offset - n.onset, 1, fill=False, ec=c, lw=lw, ls=ls, zorder=3)
        )


def string_handles():
    return [Line2D([], [], color=c, lw=3, label=n) for n, c in zip(STRING_NAMES, STRING_COLORS, strict=True)]


def draw_tab(ax, notes, t0, t1):
    for s in range(1, 7):
        ax.axhline(s, color=GRID, lw=1, zorder=1)
    for n in notes.itertuples():
        if t0 <= n.onset <= t1:
            ax.text(
                n.onset,
                n.string,
                str(n.fret),
                ha='center',
                va='center',
                fontsize=7.5,
                fontweight='bold',
                color=INK,
                zorder=3,
                bbox={'boxstyle': 'round,pad=0.15', 'fc': 'white', 'ec': STRING_COLORS[n.string - 1], 'lw': 1.2},
            )
    ax.set_ylim(6.7, 0.3)
    ax.set_yticks(range(1, 7), STRING_NAMES)
    ax.set_xlim(t0, t1)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('time in recording (s)')


def mel_ticks(ax, n_mels):
    centers = librosa.mel_frequencies(n_mels + 2, fmin=MEL_FMIN, fmax=SR / 2)[1:-1]
    hz = [f for f in (100, 300, 1000, 2000, 5000, 10000) if centers[0] <= f <= centers[-1]]
    ax.set_yticks(np.interp(hz, centers, np.arange(n_mels)), [f'{f // 1000}k' if f >= 1000 else str(f) for f in hz])
    ax.set_ylabel('Hz (mel scale)')


def draw_logmel(ax, spec, t0, vmax):
    dt = st.HOP / SR
    ax.imshow(
        spec,
        origin='lower',
        aspect='auto',
        cmap='magma',
        vmin=vmax - 14,
        vmax=vmax,
        extent=[t0 - dt / 2, t0 + (spec.shape[1] - 0.5) * dt, -0.5, spec.shape[0] - 0.5],
        interpolation='nearest',
    )
    mel_ticks(ax, spec.shape[0])


def load_clip(audio_path, start, length, sr=SR):
    y, _ = librosa.load(REPO_ROOT / audio_path, sr=sr, mono=True, offset=max(start, 0.0), duration=length)
    return y


def in_window(notes, t0, t1):
    return notes[(notes['offset'] > t0) & (notes['onset'] < t1)]


def densest_start(onsets, length, lead=0.2, weights=None):
    """Window start (lead s before some onset) that holds the most onsets (or the largest weight sum)."""
    onsets = np.sort(np.asarray(onsets))
    w = np.ones(len(onsets)) if weights is None else np.asarray(weights)[np.argsort(np.asarray(onsets))]
    cum = np.concatenate([[0], np.cumsum(w)])
    starts = onsets - lead
    ends = np.searchsorted(onsets, starts + length)
    return float(starts[int(np.argmax(cum[ends] - cum[np.arange(len(onsets))]))])


def load_index(index_dir):
    recs = pd.read_csv(index_dir / 'recordings.csv')
    notes = pd.read_csv(index_dir / 'notes.csv.gz')
    report = json.loads((index_dir / 'report.json').read_text())
    meta = recs.set_index('rec')
    for col in ('dataset', 'split', 'player', 'guitar', 'chain'):
        notes[col] = notes['rec'].map(meta[col])
    notes['polyphony'] = polyphony(notes)
    notes['n_candidates'] = st.candidates(notes['pitch'].to_numpy()).sum(axis=1)
    notes['true_is_candidate'] = st.candidates(notes['pitch'].to_numpy())[np.arange(len(notes)), notes['string'] - 1]
    recs['take'] = recs['rec'].map(label_signatures(notes))
    known = [d for d in sd.DATASET_INFO if d in set(recs['dataset'])]
    datasets = known + sorted(set(recs['dataset']) - set(known))
    return recs, notes, report, datasets


def polyphony(notes):
    """Notes sounding at each note's onset, itself included."""
    out = np.zeros(len(notes), dtype=int)
    for _, g in notes.groupby('rec'):
        on, off, t = np.sort(g['onset'].to_numpy()), np.sort(g['offset'].to_numpy()), g['onset'].to_numpy()
        out[g.index.to_numpy()] = np.searchsorted(on, t, 'right') - np.searchsorted(off, t, 'right')
    return out


def info_of(d):
    return sd.DATASET_INFO.get(d, {'name': d, 'url': '', 'license': 'unknown', 'about': ''})


def take_signature(g):
    rel = (g['onset'] - g['onset'].min()).round(2)
    return str(hash(tuple(zip(g['pitch'], g['string'], rel, strict=True))))


def label_signatures(notes):
    """A hash of each recording's (pitch, string, onset after its first) sequence, so equal values mean the same
    take with the same labels; recordings with fewer than MIN_TAKE_NOTES notes get their own name instead."""
    return (
        notes.sort_values(['rec', 'onset', 'string'])
        .groupby('rec')
        .apply(lambda g: take_signature(g) if len(g) >= MIN_TAKE_NOTES else g.name, include_groups=False)
    )


def overview_section(recs, notes, report, datasets, index_dir):
    rows, notes_dup = [], []
    for d in datasets:
        r, n, i = recs[recs['dataset'] == d], notes[notes['dataset'] == d], info_of(d)
        name = f'<a href="{esc(i["url"])}">{esc(i["name"])}</a>' if i['url'] else esc(i['name'])
        raw = report['raw_notes'].get(d)
        rows.append(
            {
                'Dataset': f'<b>{name}</b>',
                'License': esc(i['license']),
                'About': esc(i['about']),
                'Recordings': num(len(r)),
                'Hours': num(r['duration'].sum() / 3600, 1),
                'Notes raw → kept': f'{num(raw) if raw is not None else "?"} → {num(len(n))}',
                'Players': f'{r["player"].nunique()}',
                'Guitars': f'{r["guitar"].nunique()}',
                'Chains': counts(r['chain'], list(CHAINS)),
                'Splits': counts(r['split'], SPLITS),
            }
        )
        s = r['take']
        shared = int(s.duplicated(keep=False).sum())
        if shared:
            notes_dup.append(
                f'<li><b>{esc(i["name"])}</b>: {shared} of {len(r)} recordings share their exact label sequence with '
                f'another recording (the same take through different chains), so {num(len(r) - s.nunique())} '
                'takes are counted twice in the recording and note totals.</li>'
            )
    for d in pending(report, datasets):
        i = info_of(d)
        name = f'<a href="{esc(i["url"])}">{esc(i["name"])}</a>' if i['url'] else esc(i['name'])
        rows.append(
            {
                'Dataset': f'<b>{name}</b>',
                'License': esc(i['license']),
                'About': esc(i['about']),
                'Recordings': '<span class="muted">pending: no labelled recordings in this index yet</span>',
            }
        )
    rows.append(
        {
            'Dataset': '<b>All</b>',
            'Recordings': num(len(recs)),
            'Hours': num(recs['duration'].sum() / 3600, 1),
            'Notes raw → kept': f'{num(sum(report["raw_notes"].values()))} → {num(len(notes))}',
            'Players': f'{recs["player"].nunique()}',
            'Guitars': f'{recs["guitar"].nunique()}',
            'Chains': counts(recs['chain'], list(CHAINS)),
            'Splits': counts(recs['split'], SPLITS),
        }
    )
    body = (
        '<p>Every note in the index is one training example: the classifier hears a short window around the '
        "note's onset, is told its pitch, and picks the string. Chains: "
        + ', '.join(f'<b>{c}</b> = {t}' for c, t in CHAINS.items() if c in set(recs['chain']))
        + '. Hours and notes are after label cleaning and alignment.</p>'
        + table(rows, cls='overview')
        + (f'<ul class="note">{"".join(notes_dup)}</ul>' if notes_dup else '')
        + checks_box(recs, notes, report, datasets)
        + f'<p class="meta">Index: <code>{esc(index_dir)}</code> · built at git '
        f'<code>{esc(report.get("git_sha", "?"))[:8]}</code>{" (dirty)" if report.get("git_dirty") else ""}.</p>'
    )
    return section('overview', '1. Overview', body)


def pending(report, datasets):
    return [d for d in report.get('datasets', []) if d not in datasets]


def checks_box(recs, notes, report, datasets):
    items = []
    for kind in ('player', 'guitar'):
        splits = recs.groupby(kind)['split'].unique()
        bad = splits[splits.map(len) > 1]
        if len(bad):
            where = '; '.join(
                f'{esc(k)} ({", ".join(s for s in SPLITS if s in v)}; '
                f'{esc(", ".join(info_of(d)["name"] for d in recs.loc[recs[kind] == k, "dataset"].unique()))})'
                for k, v in bad.items()
            )
            items.append(('warn', f'{len(bad)} {kind}(s) appear in more than one split: {where}.'))
        else:
            items.append(('ok', f'No {kind} appears in more than one split (across all datasets).'))
    bad = notes[~notes['true_is_candidate']]
    if len(bad):
        per = ', '.join(f'{info_of(d)["name"]} {c}' for d, c in bad.groupby('dataset').size().items())
        items.append(
            (
                'warn',
                f"{num(len(bad))} notes have a fret above the classifier's <code>MAX_FRET</code> = {st.MAX_FRET} "
                f'(cleaning allows up to {sd.MAX_FRET}): {esc(per)}. <code>candidates()</code> excludes their true '
                'string, so the masked logit for the label is &minus;&infin; and its cross-entropy is infinite.',
            )
        )
    low = recs[recs['matched'] < 0.8]
    if len(low):
        per = ', '.join(f'{info_of(d)["name"]} {c}' for d, c in low.groupby('dataset').size().items())
        items.append(
            (
                'warn',
                f'{len(low)} recordings have under 80% of labels near a detected onset ({esc(per)}); see Alignment.',
            )
        )
    gaps = paired_lags(recs)
    if len(gaps):
        items.append(
            (
                'warn',
                f'{gaps["take"].nunique()} takes recorded through two chains at once got lags that differ by more '
                f'than {PAIR_TOLERANCE_MS:.1f} ms. Both files carry the same performance, so at least one lag '
                'is wrong, typically a match locked onto the neighbouring note of a strum; see Alignment.',
            )
        )
    end = notes['rec'].map(recs.set_index('rec')['duration'])
    outside = notes[(notes['onset'] < 0) | (notes['onset'] > end)]
    if len(outside):
        per = ', '.join(f'{info_of(d)["name"]} {c}' for d, c in outside.groupby('dataset').size().items())
        items.append(
            (
                'warn',
                f'{len(outside)} notes in {outside["rec"].nunique()} recordings start before 0 s or after the end '
                f'of the audio ({esc(per)}), mostly labels at 0 s moved earlier by the alignment shift; their '
                'windows are partly padding.',
            )
        )
    unresolved = report.get('unresolved_recordings', {})
    if sum(unresolved.values()):
        per = ', '.join(f'{info_of(d)["name"]} {c}' for d, c in unresolved.items())
        items.append(
            (
                'info',
                f'{sum(unresolved.values())} recordings were left out because their lag hit the '
                f'±{1000 * sd.MAX_LAG:.0f} ms edge of the search ({esc(per)}); see Alignment.',
            )
        )
    shifted = sum(report.get('shifted_recordings', {}).values())
    items.append(('info', f'{shifted} recordings had their labels shifted in time to match the audio; see Alignment.'))
    lis = ''.join(f'<li class="{k}">{t}</li>' for k, t in items)
    return f'<div class="checks"><h3>Automatic checks</h3><ul>{lis}</ul></div>'


def splits_section(recs, notes, datasets):
    rows = []
    for d in datasets:
        for s in SPLITS:
            r = recs[(recs['dataset'] == d) & (recs['split'] == s)]
            if r.empty:
                continue
            n = notes[notes['rec'].isin(r['rec'])]
            rows.append(
                {
                    'Dataset': esc(info_of(d)['name']),
                    'Split': f'<span class="split {s}">{s}</span>',
                    'Players': esc(', '.join(sorted(r['player'].unique()))),
                    'Guitars': esc(', '.join(sorted(r['guitar'].unique()))),
                    'Chains': counts(r['chain']),
                    'Recordings': num(len(r)),
                    'Notes': num(len(n)),
                    'Hours': num(r['duration'].sum() / 3600, 2),
                }
            )
    for s in SPLITS:
        r = recs[recs['split'] == s]
        rows.append(
            {
                'Dataset': '<b>All</b>',
                'Split': f'<span class="split {s}">{s}</span>',
                'Recordings': num(len(r)),
                'Notes': num((notes['split'] == s).sum()),
                'Hours': num(r['duration'].sum() / 3600, 2),
            }
        )
    body = (
        '<p>Splits go by player or guitar, never by clip: every recording of a val or test player (or guitar) is '
        'held out, so the model is scored on hands and instruments it never trained on.</p>' + table(rows)
    )
    return section('splits', '2. Splits', body)


def labels_section(notes, datasets, assets):
    n = len(datasets)
    fig, axes = plt.subplots(
        n,
        5,
        figsize=(15, 1.9 * n + 0.7),
        squeeze=False,
        gridspec_kw={'width_ratios': [0.9, 1.6, 2.2, 1, 1.9], 'wspace': 0.3},
    )
    heat = []
    for d in datasets:
        x = notes[notes['dataset'] == d]
        h = np.zeros((6, sd.MAX_FRET + 1))
        np.add.at(h, (x['string'] - 1, x['fret'].clip(0, sd.MAX_FRET)), 1)
        heat.append(h / len(x))
    vmax = max(h.max() for h in heat)
    tops = np.zeros(4)
    for row, d in enumerate(datasets):
        x = notes[notes['dataset'] == d]
        ax = axes[row]
        frac = x['string'].value_counts(normalize=True).reindex(range(1, 7), fill_value=0)
        ax[0].bar(range(1, 7), 100 * frac, color=STRING_COLORS, width=0.75)
        ax[0].set_xticks(range(1, 7), STRING_NAMES)
        fret = x['fret'].value_counts(normalize=True).reindex(range(sd.MAX_FRET + 1), fill_value=0)
        ax[1].bar(fret.index, 100 * fret, color=INK, width=0.8)
        ax[1].axvline(st.MAX_FRET + 0.5, color=DROP_COLOR, lw=1, ls='--')
        pitch = x['pitch'].value_counts(normalize=True).reindex(range(38, 90), fill_value=0)
        ax[2].bar(pitch.index, 100 * pitch, color=INK, width=0.8)
        ax[2].set_xticks(PITCH_TICKS, [note_name(p) for p in PITCH_TICKS])
        poly = x['polyphony'].clip(upper=8).value_counts(normalize=True).reindex(range(1, 9), fill_value=0)
        ax[3].bar(poly.index, 100 * poly, color=INK, width=0.75)
        ax[3].set_xticks(range(1, 9), [*map(str, range(1, 8)), '8+'])
        im = ax[4].imshow(100 * heat[row], aspect='auto', cmap='Blues', vmin=0, vmax=100 * vmax)
        ax[4].set_yticks(range(6), STRING_NAMES)
        ax[4].set_xticks(range(0, sd.MAX_FRET + 1, 3))
        for col, v in enumerate((frac, fret, pitch, poly)):
            tops[col] = max(tops[col], 100 * v.max())
        ax[0].set_ylabel(f'{info_of(d)["name"]}\n% of notes', fontsize=9)
        if row == 0:
            for a, t in zip(
                ax,
                [
                    'string',
                    f'fret (red line: classifier MAX_FRET {st.MAX_FRET})',
                    'pitch',
                    'polyphony at onset',
                    'where on the neck (% of notes)',
                ],
                strict=True,
            ):
                a.set_title(t, loc='left')
        if row == n - 1:
            ax[1].set_xlabel('fret')
            ax[2].set_xlabel('MIDI pitch')
            ax[3].set_xlabel('notes')
            ax[4].set_xlabel('fret')
    for row in range(n):
        for col in range(4):
            axes[row][col].set_ylim(0, tops[col] * 1.08)
            axes[row][col].grid(axis='y', color=GRID, lw=0.6)
            axes[row][col].set_axisbelow(True)
    fig.colorbar(im, ax=axes[:, 4], shrink=0.8, label='% of notes')
    src = assets.fig(fig, 'labels')

    rows = []
    for d in datasets:
        x = notes[notes['dataset'] == d]
        rows.append(
            {
                'Dataset': esc(info_of(d)['name']),
                'Notes': num(len(x)),
                'Open strings': pct((x['fret'] == 0).mean()),
                'Median fret': f'{x["fret"].median():.0f}',
                'Pitch range': f'{note_name(x["pitch"].min())}&ndash;{note_name(x["pitch"].max())}',
                'Single notes (polyphony 1)': pct((x['polyphony'] == 1).mean()),
                'Pitch playable on ≥2 strings': pct((x['n_candidates'] >= 2).mean()),
                'Mean candidate strings': f'{x["n_candidates"].mean():.2f}',
                f'Fret > {st.MAX_FRET}': num((~x['true_is_candidate']).sum()),
            }
        )
    body = (
        string_legend()
        + figure(
            src,
            'Label distributions per dataset',
            'One row per dataset, same axes in each column so rows compare directly; bars are % of that '
            "dataset's notes. Look for strings or neck regions a dataset barely covers, and for how chordal it "
            'is (the polyphony column counts notes still ringing at each onset, the note itself included).',
        )
        + '<p>Only notes whose pitch can be played on two or more strings give the classifier a real choice; the '
        'rest are decided by the pitch alone. <i>Candidate strings</i> are those that reach the pitch within '
        f'frets 0&ndash;{st.MAX_FRET}.</p>' + table(rows)
    )
    return section('labels', '3. Labels per dataset', body)


def pick_recordings(rng, recs, k):
    """k recordings in random order, each preferring player/chain/content values not picked yet."""
    pool = recs.iloc[rng.permutation(len(recs))]
    picked, seen = [], {c: set() for c in ('player', 'chain', 'content', 'split')}
    for _ in range(min(k, len(pool))):
        score = sum((~pool[c].isin(s)).astype(int) for c, s in seen.items())
        row = pool.loc[score.idxmax()]
        picked.append(row)
        pool = pool.drop(row.name)
        for c in seen:
            seen[c].add(row[c])
    return picked


def excerpt_start(rng, rec_notes, duration, length):
    t = float(rng.choice(rec_notes['onset'].to_numpy()))
    return float(np.clip(t - 0.5, 0, max(duration - length, 0)))


def excerpt_figure(y, start, notes, title):
    end = start + len(y) / SR
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True, gridspec_kw={'height_ratios': [3, 1.1]})
    draw_cqt(a, y, start)
    draw_boxes(a, notes)
    a.legend(handles=string_handles(), loc='lower left', bbox_to_anchor=(0, 1.0), ncol=6, frameon=False, fontsize=8)
    a.set_title(title, loc='right', fontsize=8.5, color=MUTED)
    draw_tab(b, notes, start, end)
    fig.tight_layout(h_pad=0.4)
    return fig


def rec_label(r, start=None, length=None):
    parts = [
        f'<code>{esc(r["rec"])}</code>',
        f'player <b>{esc(r["player"])}</b>',
        f'guitar <b>{esc(r["guitar"])}</b>',
        f'chain <b>{esc(r["chain"])}</b>',
        f'content <b>{esc(r["content"])}</b>',
        f'<span class="split {r["split"]}">{r["split"]}</span>',
    ]
    if start is not None:
        parts.append(f'{start:.1f}&ndash;{start + length:.1f} s of {r["duration"]:.0f} s')
    return ' · '.join(parts)


def samples_section(recs, notes, datasets, cfg, rng, assets):
    by_rec = dict(tuple(notes.groupby('rec')))
    blocks = []
    for d in datasets:
        cards = []
        for i, r in enumerate(pick_recordings(rng, recs[recs['dataset'] == d], cfg.samples_per_dataset)):
            rn = by_rec[r['rec']]
            start = excerpt_start(rng, rn, r['duration'], cfg.clip_seconds)
            y = load_clip(r['audio'], start, cfg.clip_seconds)
            shown = in_window(rn, start, start + len(y) / SR)
            name = f'sample_{d}_{i}'
            fig = excerpt_figure(y, start, shown, f'{r["rec"]}  ({len(shown)} notes)')
            cards.append(
                f'<div class="card"><p class="reclabel">{rec_label(r, start, len(y) / SR)}</p>'
                f'{audio(assets.wav(y, name))}'
                + figure(
                    assets.fig(fig, name),
                    f'Excerpt of {r["rec"]}',
                    'Each box is one labelled note: it spans onset to offset at the labelled pitch, colored by '
                    "string. A good label sits on the lowest line of its note's harmonic stack (the fundamental, "
                    'often faint on a mic recording) and starts where that line starts. Below: the same notes as tab.',
                )
                + '</div>'
            )
        blocks.append(f'<h3>{esc(info_of(d)["name"])}</h3>' + ''.join(cards))
    body = (
        f'<p>{cfg.samples_per_dataset} random excerpts per dataset (seed {cfg.seed}), each starting just before a '
        f'random labelled note. Spectrograms are a constant-Q transform with one row per third of a semitone, so '
        'the y axis is pitch. Labels are drawn as they are in the index, i.e. after the alignment shift. Audio '
        'clips are 22,050 Hz mono, peak-normalised for listening.</p>' + string_legend() + ''.join(blocks)
    )
    return section('samples', '4. Samples', body)


def zoom_figure(y, w0, w1, after, before=None, unmatched=None):
    """CQT and onset envelope of [w0, w1]; `y` is the whole recording at ALIGN_SR, so the envelope and detected
    onsets are exactly those alignment() used."""
    env = sd.onset_envelope(y)
    t_env = np.arange(len(env)) * sd.ALIGN_HOP / sd.ALIGN_SR
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sd.ALIGN_SR, hop_length=sd.ALIGN_HOP, units='time')
    view = (t_env >= w0) & (t_env <= w1)
    scale = env[view].max() + 1e-9
    t_seg = max(w0 - 0.6, 0.0)
    seg = y[int(t_seg * sd.ALIGN_SR) : int((w1 + 0.6) * sd.ALIGN_SR)]
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True, gridspec_kw={'height_ratios': [2, 1.3]})
    draw_cqt(a, seg, t_seg)
    draw_boxes(a, after)
    b.fill_between(t_env, env / scale, color='#c3c2b7', lw=0)
    b.plot(t_env, env / scale, color=MUTED, lw=0.8)
    b.plot(peaks, np.full(len(peaks), -0.06), '^', color=INK, ms=5)
    if before is not None:
        for t in before['onset'].unique():
            b.axvline(t, color='#eb6834', lw=1.4, ls=(0, (3, 2)), ymax=0.95)
    for t in after['onset'].unique():
        b.axvline(t, color='#2a78d6', lw=1.4)
    if unmatched is not None and len(unmatched):
        b.plot(unmatched['onset'], np.full(len(unmatched), 1.05), 'x', color=DROP_COLOR, ms=7, mew=2)
    handles = [Line2D([], [], color='#2a78d6', lw=1.4, label='label onset (in the index)')]
    if before is not None:
        handles.insert(0, Line2D([], [], color='#eb6834', lw=1.4, ls=(0, (3, 2)), label='label onset before shift'))
    if unmatched is not None:
        handles.append(
            Line2D([], [], color=DROP_COLOR, marker='x', lw=0, mew=2, label=f'no detected onset within {sd.NEAR:g} s')
        )
    handles.append(Line2D([], [], color=INK, marker='^', lw=0, label='detected onset'))
    b.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    a.legend(handles=string_handles(), loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    b.set_ylim(-0.12, 1.12)
    b.set_ylabel('onset strength')
    b.set_xlabel('time in recording (s)')
    b.set_xlim(w0, w1)
    fig.tight_layout(h_pad=0.4)
    return fig


def paired_lags(recs):
    """Recordings of the same take (identical labels) whose measured lags disagree by more than PAIR_TOLERANCE_MS."""
    multi = recs[recs.groupby('take')['rec'].transform('size') > 1]
    spread = multi.groupby('take')['lag_ms'].transform(lambda x: x.max() - x.min())
    return multi[spread > PAIR_TOLERANCE_MS + 0.1].sort_values(['dataset', 'take', 'chain'])


def onset_offsets(recs, notes, k, seed):
    """Per dataset: nearest detected onset minus label onset (ms), labels as in the index, for up to k recordings;
    NaN where no detected onset is within NEAR."""
    rng = np.random.default_rng(seed)
    out = {}
    for d, r in recs.groupby('dataset'):
        vals = []
        for row in r.iloc[rng.permutation(len(r))[:k]].itertuples():
            on = notes.loc[notes['rec'] == row.rec, 'onset'].to_numpy()
            env = sd.onset_envelope(load_clip(row.audio, 0, None, sr=sd.ALIGN_SR))
            peaks = librosa.onset.onset_detect(
                onset_envelope=env, sr=sd.ALIGN_SR, hop_length=sd.ALIGN_HOP, units='time'
            )
            if len(peaks) == 0:
                vals.append(np.full(len(on), np.nan))
                continue
            dist = peaks[np.abs(peaks[None, :] - on[:, None]).argmin(axis=1)] - on
            vals.append(np.where(np.abs(dist) <= sd.NEAR, 1000 * dist, np.nan))
        out[d] = (np.concatenate(vals), min(k, len(r)))
    return out


def offsets_reference(offsets):
    """GuitarSet's median offset: peak-picked onsets sit at a different point of the attack than the envelope
    sum the lag search uses, so this measure needs its own reference."""
    return float(np.nanmedian(offsets['guitarset'][0])) if 'guitarset' in offsets else 0.0


def offsets_figure(offsets, datasets, ref):
    n = len(datasets)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.5 * rows), squeeze=False)
    bins = np.arange(-1000 * sd.NEAR, 1000 * sd.NEAR + 2.5, 2.5)
    for ax, d in zip(axes.flat, datasets, strict=False):
        v, k = offsets[d]
        ok = v[~np.isnan(v)]
        ax.hist(ok, bins=bins, color=INK, weights=np.full(len(ok), 100 / max(len(v), 1)))
        ax.axvline(ref, color='#2a78d6', lw=1.2, ls='--')
        late, early = np.mean(v - ref < -25), np.mean(v - ref > 25)
        ax.set_title(f'{info_of(d)["name"]} ({k} recordings)', loc='left')
        ax.text(
            0.02,
            0.95,
            f'median {np.nanmedian(v):+.0f} ms\nlabel >25 ms late: {100 * late:.0f}%\n'
            f'label >25 ms early: {100 * early:.0f}%\nno onset within {1000 * sd.NEAR:.0f} ms: '
            f'{100 * np.isnan(v).mean():.0f}%',
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=8,
            color=MUTED,
        )
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('detected onset - label onset (ms)')
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylabel('% of notes')
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def shift_examples(recs, notes):
    """A shifted recording with a typical shift and the one with the largest, each the densest of its ties."""
    shifted = recs[recs['shift_ms'] != 0].copy()
    if shifted.empty:
        return []
    size = abs(shifted['shift_ms'])
    shifted['dev'] = (size - size.median()).abs()
    shifted['size'] = -size
    shifted['density'] = -shifted['rec'].map(notes.groupby('rec').size()) / shifted['duration']
    typical = shifted.sort_values(['dev', 'density']).iloc[0]
    largest = shifted.sort_values(['size', 'density']).iloc[0]
    picks = [('A typical shift', typical)]
    if largest['rec'] != typical['rec']:
        picks.append(('The largest shift', largest))
    return picks


def lag_scores(env, onsets):
    """The score alignment() maximises, for every candidate lag in its search window (lags in ms)."""
    fps = sd.ALIGN_SR / sd.ALIGN_HOP
    frames = np.round(np.asarray(onsets) * fps).astype(int)
    frames = frames[(frames >= 0) & (frames < len(env))]
    lags = np.arange(-int(sd.MAX_LAG * fps), int(sd.MAX_LAG * fps) + 1)
    score = np.array([env[np.clip(frames - lag, 0, len(env) - 1)].sum() for lag in lags])
    return 1000 * lags / fps, score


def walkthrough_section(recs, notes, report, cfg, assets):
    bias, thr = report['detector_bias_ms'], report['shift_threshold_ms']
    r = recs[recs['rec'] == cfg.walkthrough_rec].iloc[0]
    rn = notes[notes['rec'] == cfg.walkthrough_rec]
    shipped = rn.assign(onset=rn['onset'] - r['shift_ms'] / 1000, offset=rn['offset'] - r['shift_ms'] / 1000)
    y = load_clip(r['audio'], 0, None, sr=sd.ALIGN_SR)
    env = sd.onset_envelope(y)
    t_env = np.arange(len(env)) * sd.ALIGN_HOP / sd.ALIGN_SR
    w0 = densest_start(shipped['onset'], cfg.align_seconds)
    w1 = w0 + cfg.align_seconds
    view = in_window(shipped, w0, w1)

    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True, gridspec_kw={'height_ratios': [2, 1.3]})
    seg = y[int(max(w0 - 0.6, 0) * sd.ALIGN_SR) : int((w1 + 0.6) * sd.ALIGN_SR)]
    draw_cqt(a, seg, max(w0 - 0.6, 0))
    draw_boxes(a, view)
    for n in view.itertuples():
        a.text(
            n.onset,
            n.pitch + 1.2,
            f'{STRING_NAMES[n.string - 1]}{n.fret}',
            fontsize=7,
            color=INK,
            bbox={'boxstyle': 'round,pad=0.1', 'fc': 'white', 'ec': 'none', 'alpha': 0.8},
        )
    b.fill_between(t_env, env, color='#c3c2b7', lw=0)
    b.plot(t_env, env, color=MUTED, lw=0.8)
    for t in view['onset']:
        b.axvline(t, color='#eb6834', lw=1.4, ls=(0, (3, 2)))
    b.set_ylabel('onset strength')
    b.set_xlabel('time in recording (s)')
    b.set_xlim(w0, w1)
    a.legend(handles=string_handles(), loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    b.legend(
        handles=[Line2D([], [], color='#eb6834', lw=1.4, ls=(0, (3, 2)), label='label onset as shipped')],
        loc='upper left',
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(h_pad=0.4)
    step1 = assets.fig(fig, 'walk_1_input')

    lags, score = lag_scores(env, shipped['onset'])
    best = lags[np.argmax(score)]
    first = view['onset'].iloc[min(2, len(view) - 1)]
    z0, z1 = first - 0.12, first + 0.38
    tries = [0.0, -best / 2, -best, -best + 40]
    fig, axes = plt.subplots(1, len(tries), figsize=(15, 2.7), sharey=True)
    for ax, move in zip(axes, tries, strict=True):
        moved = shipped['onset'].to_numpy() + move / 1000
        idx = np.clip(np.round(moved * sd.ALIGN_SR / sd.ALIGN_HOP).astype(int), 0, len(env) - 1)
        total = score[np.argmin(np.abs(lags + move))]
        ax.fill_between(t_env, env, color='#c3c2b7', lw=0)
        ax.plot(t_env, env, color=MUTED, lw=0.8)
        inside = (moved >= z0) & (moved <= z1)
        for t in moved[inside]:
            ax.axvline(t, color='#2a78d6' if move == -best else '#eb6834', lw=1.3)
        ax.plot(moved[inside], env[idx[inside]], 'o', color=INK, ms=5, zorder=4)
        ax.set_xlim(z0, z1)
        ax.set_title(
            f'labels moved {move:+.0f} ms\nsum over all {len(moved)} labels = {total:.2f}', loc='left', fontsize=9
        )
        ax.set_xlabel('time (s)')
    axes[0].set_ylabel('onset strength')
    fig.tight_layout()
    step3 = assets.fig(fig, 'walk_3_tries')

    gs = recs.loc[recs['dataset'] == 'guitarset', 'lag_ms']
    keys = ['dataset', 'player', 'chain', 'content']
    grp = recs[(recs[keys] == r[keys]).all(axis=1)]
    q = 1000 * sd.ALIGN_HOP / sd.ALIGN_SR
    edges = np.arange(-1000 * sd.MAX_LAG - q / 2, 1000 * sd.MAX_LAG + q, q)
    fig, (s, g, h) = plt.subplots(1, 3, figsize=(15, 3.2))
    s.plot(-lags, score, color=INK, lw=1.2)
    s.axvline(-best, color='#2a78d6', lw=1.2)
    s.axvline(0, color='#eb6834', lw=1, ls=(0, (3, 2)))
    s.set_title(f'This recording: best move {-best:+.1f} ms', loc='left', fontsize=10)
    s.set_xlabel('labels moved by (ms, + = later)')
    s.set_ylabel('sum of onset strength\nunder the labels')
    g.hist(-gs, bins=-edges[::-1], color=MUTED, edgecolor='white', lw=0.4)
    g.axvline(-bias, color=INK, lw=1.2, ls='--')
    g.set_title(f'GuitarSet ({len(gs)} recordings): median best move {-bias:+.1f} ms', loc='left', fontsize=10)
    g.set_xlabel('best move (ms)')
    g.set_ylabel('recordings')
    h.hist(-grp['lag_ms'], bins=-edges[::-1], color=MUTED, edgecolor='white', lw=0.4)
    h.axvline(-r['lag_used_ms'], color='#2a78d6', lw=1.2)
    h.axvline(-bias, color=INK, lw=1.2, ls='--')
    h.set_title(f'Its group ({len(grp)} recordings): median {-r["lag_used_ms"]:+.1f} ms', loc='left', fontsize=10)
    h.set_xlabel('best move (ms)')
    for ax in (s, g, h):
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(-1000 * sd.MAX_LAG, 1000 * sd.MAX_LAG)
    fig.tight_layout()
    step4 = assets.fig(fig, 'walk_4_scores')

    after = in_window(rn, w0, w1)
    before = after.assign(onset=after['onset'] - r['shift_ms'] / 1000, offset=after['offset'] - r['shift_ms'] / 1000)
    step6 = assets.fig(zoom_figure(y, w0, w1, after, before=before), 'walk_6_after')

    group_name = f'{info_of(r["dataset"])["name"]} / {r["player"]} / {r["chain"]} / {r["content"]}'
    body = (
        '<p>One recording, followed through every step. It is '
        f'<code>{esc(cfg.walkthrough_rec)}</code>, {len(rn)} notes, picked because its labels are visibly off.</p>'
        + f'<p class="reclabel">{rec_label(r)}</p>'
        '<h3>Step 1. What a dataset gives us</h3>'
        '<p>An audio file and, for every note, an onset time, an offset time, a pitch and a string (the fret '
        'follows from pitch and string). Nothing in the labels says how they were timed: a hexaphonic pickup '
        '(GuitarSet), a MIDI pickup (Guitar-TECHS), or a person clicking in an editor (IDMT). Top: the '
        "spectrogram with every label drawn as a box in its string's colour, at the time the dataset says. "
        'The boxes start before the notes visibly do.</p>'
        + figure(
            step1,
            'Audio and labels as shipped',
            'Top: spectrogram, label boxes as the dataset ships them, '
            'tagged string and fret. Bottom: the onset-strength curve of the same audio (step 2), with '
            'the label onsets as dashed lines.',
        )
        + '<h3>Step 2. Turn the audio into an "a note starts here" curve</h3>'
        f'<p>The tabs carry only times, so the audio must be turned into something comparable. '
        f'<code>librosa.onset.onset_strength</code> computes, every {sd.ALIGN_HOP} samples ({q:.1f} ms), how much '
        'the spectrum got louder than a moment ago, summed over frequencies. A pluck makes a sharp jump, '
        'a sustained note makes almost nothing, so the curve is a row of spikes, one per attack (bottom panel '
        'above). Pitch and string play no part here: only <i>when</i> things start.</p>'
        '<h3>Step 3. Slide the labels and score each position</h3>'
        f"<p>Move every label onset of the recording by the same amount, read the curve's height at each moved "
        'onset (the dots), and add them up. Labels sitting on the spikes give a large sum; labels sitting in the '
        f'quiet gaps give a small one. Do that for every move from {-1000 * sd.MAX_LAG:.0f} to '
        f'+{1000 * sd.MAX_LAG:.0f} ms in {q:.1f} ms steps. One move for the whole recording, not one per note: '
        "a dataset's timing error is a constant offset, while a per-note move would just snap each label to "
        'whatever is nearby.</p>'
        + figure(
            step3,
            'Trying moves',
            'The same half second with the labels moved by four amounts. Blue is the '
            'best move. The title gives the sum over every label in the recording, not just those in view.',
        )
        + '<h3>Step 4. Take the best move, minus what perfect labels get</h3>'
        f'<p>Left: the sum for every move. It peaks at <b>{-best:+.1f} ms</b>, the labels are that much early. '
        "But even perfect labels do not score best at 0, because the curve's spike comes a little after the "
        "string is struck (the spectrum needs a few milliseconds of the attack to rise). GuitarSet's labels "
        'come from a pickup under each string and are taken as correct; their best move has a median of '
        f"<b>{-bias:+.1f} ms</b> (middle). That is the detector's own delay, so the correction is the "
        f'difference: {-best:+.1f} &minus; ({-bias:+.1f}) = <b>{-best + bias:+.1f} ms</b>. Corrections smaller '
        f'than {thr} ms are left alone, as noise.</p>'
        '<h3>Step 5. Use the group, not the single recording</h3>'
        '<p>With chords and fast runs the curve has several spikes within a few tens of ms, and one recording '
        'can score best one note off. So every recording with more than one note takes the median best move '
        f'of its group, same dataset, player, recording chain and kind of material (right; here '
        f'<i>{esc(group_name)}</i>). The whole group was labelled the same way, so it shares one offset. '
        f'This recording measured {-r["lag_ms"]:+.1f} ms itself, the group median is '
        f'{-r["lag_used_ms"]:+.1f} ms, so its labels move by <b>{r["shift_ms"]:+.1f} ms</b>. '
        'Single-note recordings (EGFxSet) keep their own value, as there is nothing to confuse.</p>'
        + figure(
            step4,
            'Scores and references',
            'Left: sum of onset strength under the labels for every move. '
            "Middle: GuitarSet's best moves, the reference (dashed). Right: best moves of every recording "
            "in this recording's group, the median (blue) is what is applied.",
        )
        + '<h3>Step 6. Result</h3>'
        '<p>Every label onset and offset of the recording is moved by the same amount and written to the '
        'index. Pitch and string are never touched. Labels moved before 0 s are dropped; recordings whose best '
        f'move sits at the edge of the ±{1000 * sd.MAX_LAG:.0f} ms search are dropped too, since their real '
        'offset is outside it.</p>'
        + figure(
            step6,
            'Labels before and after',
            'Dashed orange: label onsets as shipped. Solid blue: as stored '
            'in the index. Triangles: onsets picked from the curve, used only to measure what is left over '
            '(the per-note offsets below).',
        )
    )
    return section('walkthrough', '5. Alignment, step by step', body)


def alignment_section(recs, notes, report, datasets, cfg, assets):
    bias, thr = report['detector_bias_ms'], report['shift_threshold_ms']
    q = 1000 * sd.ALIGN_HOP / sd.ALIGN_SR
    lim = 1000 * sd.MAX_LAG
    bins = {
        'lag_ms': np.arange(-lim - q / 2, lim + q, q),
        'shift_ms': np.arange(-lim - q / 2, lim + q, q),
        'residual_ms': np.arange(0, 1000 * sd.NEAR + 2.5, 2.5),
        'matched': np.linspace(0, 1, 41),
    }
    titles = {
        'lag_ms': 'measured label lag (ms, + = labels late)',
        'shift_ms': 'applied shift (ms)',
        'residual_ms': 'median |label - onset| after lag (ms)',
        'matched': 'fraction of labels near an onset',
    }
    chain_order = [c for c in CHAINS if c in set(recs['chain'])]
    chain_order += sorted(set(recs['chain']) - set(chain_order))
    # Kept apart from the string colors so a chain is never read as a string.
    chain_colors = dict(zip(chain_order, ['#6250d6', '#c3a35a', '#52514e', '#a8a79f'], strict=False))
    n = len(datasets)
    fig, axes = plt.subplots(n, 4, figsize=(15, 1.9 * n + 0.8), squeeze=False, sharex='col')
    for row, d in enumerate(datasets):
        r = recs[recs['dataset'] == d]
        present = [c for c in chain_order if c in set(r['chain'])]
        for col, key in enumerate(bins):
            ax = axes[row][col]
            ax.hist(
                [r.loc[r['chain'] == c, key].clip(bins[key][0], bins[key][-1]) for c in present],
                bins=bins[key],
                stacked=True,
                color=[chain_colors[c] for c in present],
                label=present,
                edgecolor='white',
                linewidth=0.4,
            )
            ax.grid(axis='y', color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(titles[key], loc='left')
        axes[row][0].axvspan(bias - thr, bias + thr, color='#e8e7e2', zorder=0)
        axes[row][0].axvline(bias, color=INK, lw=1, ls='--')
        axes[row][0].set_ylabel(f'{info_of(d)["name"]}\nrecordings', fontsize=9)
        axes[row][3].legend(loc='upper left', frameon=False, fontsize=8)
    fig.tight_layout()
    hist_src = assets.fig(fig, 'alignment_hist')

    rows = []
    for (d, chain, content), r in recs.groupby(['dataset', 'chain', 'content'], sort=False):
        rows.append(
            {
                'Dataset': esc(info_of(d)['name']),
                'Chain': esc(chain),
                'Content': esc(content),
                'Recordings': num(len(r)),
                'Median lag': f'{r["lag_ms"].median():+.1f} ms',
                'Shifted': pct((r['shift_ms'] != 0).mean(), 0),
                'Median shift': f'{r["shift_ms"].median():+.1f} ms',
                'Median residual': f'{r["residual_ms"].median():.1f} ms',
                'Median matched': pct(r['matched'].median(), 0),
                'Worst matched': pct(r['matched'].min(), 0),
            }
        )
    rows.sort(key=lambda x: (x['Dataset'], x['Chain'], x['Content']))

    body = (
        "<p>For each recording the labels are slid against the audio's onset envelope (librosa onset strength, "
        f'{sd.ALIGN_HOP}-sample hop at {sd.ALIGN_SR} Hz, so lags come in steps of {q:.1f} ms) within '
        f'±{lim:.0f} ms, and the lag with the most onset strength under the labels wins. '
        '<i>Matched</i> is the share of labels with a detected onset within '
        f'{1000 * sd.NEAR:.0f} ms once that lag is removed; <i>residual</i> is the median distance to it.</p>'
        f'<p>The detector itself is biased: on GuitarSet, whose labels come from a hexaphonic pickup and are '
        f'taken as correct, the median measured lag is <b>{bias:+.1f} ms</b>. That number is the reference: a '
        f"recording's labels are moved only when its lag differs from {bias:+.1f} ms by more than "
        f'{thr} ms (the grey band), and then by exactly that difference, so after the shift every recording '
        'measures like GuitarSet does.</p>'
        + unresolved_note(report)
        + figure(
            hist_src,
            'Alignment histograms',
            'Stacked by chain. Left: the grey band is the no-shift zone around the GuitarSet bias (dashed). '
            'Recordings outside it were shifted (second column). The right two columns are measured with the '
            'labels as stored before shifting, minus the best lag, so they describe label quality, not the shift.',
        )
        + table(rows)
    )

    offsets = onset_offsets(recs, notes, cfg.offset_recordings_per_dataset, cfg.seed)
    ref = offsets_reference(offsets)
    body += '<h3>Per-note offsets</h3>' + figure(
        assets.fig(offsets_figure(offsets, datasets, ref), 'alignment_offsets'),
        'Per-note label offsets',
        'For every note of a random sample of recordings: the nearest detected (peak-picked) onset minus the '
        f"label onset, labels as stored in the index. The dashed line is GuitarSet's median ({ref:+.1f} ms); "
        'peak picking lands at a different point of the attack than the lag search, so its reference differs '
        f'from the {bias:+.1f} ms above. A narrow peak on the dashed line means labels sit where the audio says; '
        'mass far to the left means labels that come after the audible onset. Strums put several onsets within '
        "a few tens of ms, so some notes pick a neighbour's onset.",
    )
    gaps = paired_lags(recs)
    if len(gaps):
        rows = [
            {
                'Dataset': esc(info_of(g['dataset'])['name']),
                'Recording': f'<code>{esc(g["rec"])}</code>',
                'Chain': esc(g['chain']),
                'Lag': f'{g["lag_ms"]:+.1f} ms',
                'Shift applied': f'{g["shift_ms"]:+.1f} ms',
                'Matched': pct(g['matched'], 0),
                'Residual': f'{g["residual_ms"]:.0f} ms',
            }
            for _, g in gaps.iterrows()
        ]
        body += (
            '<h3>Same take, different lag</h3>'
            f'<p>These recordings share their labels with another file of the same take (recorded through another '
            f'chain at the same time), yet their measured lags differ by more than {PAIR_TOLERANCE_MS:.1f} ms (two lag steps; one or two '
            'steps apart is quantization noise). The audio of both '
            'files starts together, so the true lag is nearly the same for both. A lag near the edge of the search '
            'is likely a match one strum note off (see the largest shift below); the matched fraction cannot tell, '
            'since a one-note-off match still lands near some onset. Their shifts are suspect.</p>' + table(rows)
        )

    for title, r in shift_examples(recs, notes):
        rn = notes[notes['rec'] == r['rec']]
        y = load_clip(r['audio'], 0, None, sr=sd.ALIGN_SR)
        w0 = densest_start(rn['onset'].round(2).drop_duplicates(), cfg.align_seconds)
        after = in_window(rn, w0, w0 + cfg.align_seconds)
        dt = r['shift_ms'] / 1000
        before = after.assign(onset=after['onset'] - dt, offset=after['offset'] - dt)
        name = 'alignment_shift_' + title.split()[1]
        src = assets.fig(zoom_figure(y, w0, w0 + cfg.align_seconds, after, before=before), name)
        body += f'<h3>{title}</h3><p class="reclabel">{rec_label(r)}</p>' + figure(
            src,
            f'{title}: labels before and after',
            f'Measured lag {r["lag_ms"]:+.1f} ms against the GuitarSet reference of {bias:+.1f} ms, so the labels '
            f'moved by {r["shift_ms"]:+.1f} ms (matched {pct(r["matched"], 0)}, residual '
            f'{r["residual_ms"]:.0f} ms). Dashed orange: label onsets as the dataset ships them; solid blue: in '
            f'the index. On GuitarSet the nearest triangle sits a median {ref:+.0f} ms from the label, so '
            'well-placed blue lines sit on or just before the triangles, at the foot of each rise.',
        )

    worst = []
    for d in datasets:
        r = recs[recs['dataset'] == d].sort_values('matched').iloc[0]
        rn = notes[notes['rec'] == r['rec']].reset_index(drop=True)
        y = load_clip(r['audio'], 0, None, sr=sd.ALIGN_SR)
        _, dist = sd.alignment(y, rn['onset'].to_numpy())
        miss = np.isnan(dist)
        w0 = max(densest_start(rn['onset'], cfg.align_seconds, weights=miss.astype(float) + 1e-3), -0.1)
        after = in_window(rn, w0, w0 + cfg.align_seconds)
        unmatched = after[miss[after.index.to_numpy()]]
        src = assets.fig(zoom_figure(y, w0, w0 + cfg.align_seconds, after, unmatched=unmatched), f'alignment_worst_{d}')
        worst.append(
            f'<h4>{esc(info_of(d)["name"])}: matched {pct(r["matched"], 0)}, residual {r["residual_ms"]:.0f} ms</h4>'
            f'<p class="reclabel">{rec_label(r)}</p>'
            + figure(
                src,
                f'Worst-aligned recording of {info_of(d)["name"]}',
                f'The {cfg.align_seconds:g} s with the most unmatched labels (red &times;) in the worst-matched '
                'recording of this dataset; the onset curve is rescaled to this view. Check whether the boxes miss '
                'real notes (a label problem) or the notes are soft, fast or masked by ringing strings (a detector '
                'problem).',
            )
        )
    body += '<h3>Worst-aligned recording per dataset</h3>' + ''.join(worst)
    return section('alignment', '5. Alignment', body)


def drop_reasons(raw, kept_ids, rule_names):
    """Rule that removed each dropped note, in clean()'s order; the last rule (re-plucks) is the remainder."""
    dropped = raw[~raw['nid'].isin(kept_ids)].copy()
    conditions = [
        dropped['rec'].str.contains(sd.UNFRETTED),
        (dropped['fret'] < 0) | (dropped['fret'] > sd.MAX_FRET),
        (dropped['offset'] - dropped['onset']) < sd.MIN_DURATION,
    ]
    dropped['rule'] = np.select(conditions, rule_names[:3], default=rule_names[3])
    return dropped


def unresolved_note(report):
    unresolved = report.get('unresolved_recordings', {})
    if not sum(unresolved.values()):
        return ''
    per = ', '.join(f'{esc(info_of(d)["name"])} {c:,}' for d, c in unresolved.items())
    return (
        f'<p><b>Left out:</b> {sum(unresolved.values()):,} recordings ({per}) whose best lag landed on the '
        f'±{1000 * sd.MAX_LAG:.0f} ms edge of the search. There the true lag may lie outside the search, so their '
        'labels cannot be trusted to line up and the whole recording is dropped. They are not in any figure below.</p>'
    )


def cleaning_section(recs, notes, report, datasets, cfg, assets):
    rules = list(report['dropped'])
    rows = []
    for d in datasets:
        raw = report['raw_notes'].get(d, 0)
        row = {'Dataset': esc(info_of(d)['name']), 'Raw notes': num(raw)}
        for rule in rules:
            k = report['dropped'][rule].get(d, 0)
            row[rule] = f'{num(k)} <small>({pct(k / max(raw, 1))})</small>' if k else '0'
        kept = int((notes['dataset'] == d).sum())
        lost = raw - sum(report['dropped'][rule].get(d, 0) for rule in rules) - kept
        n_unres = report.get('unresolved_recordings', {}).get(d, 0)
        row[UNRESOLVED] = f'{num(lost)} <small>({n_unres} recordings)</small>' if n_unres else '0'
        row['Kept'] = num(kept)
        rows.append(row)
    body = (
        '<p>Labels a real pluck cannot produce are dropped before anything else. In order: '
        '<b>unfretted technique</b> — whole recordings of bends or harmonics, whose sounding pitch is not the '
        f'fretted one; <b>impossible fret</b> — fret below 0 or above {sd.MAX_FRET}; <b>shorter than '
        f'{1000 * sd.MIN_DURATION:.0f} ms</b>; <b>same string re-plucked within {1000 * sd.RETRIGGER:.0f} ms</b> '
        '— the second of two onsets on one string closer than that, typical of a MIDI pickup re-triggering. A '
        "note counts under the first rule that hits it. <i>Raw</i> is what each dataset's loader returns; a "
        'loader may already skip notes its annotations mark as unpitched (see <code>string_data.py</code>). '
        f'<i>{UNRESOLVED}</i> counts the notes of recordings left out at the alignment step.</p>'
        + table(rows, ['Dataset', 'Raw notes', *rules, UNRESOLVED, 'Kept'])
    )

    real = {d: sum(report['dropped'][r].get(d, 0) for r in rules[1:]) for d in datasets}
    d = max(real, key=real.get) if real else None
    if d is None or real[d] == 0 or d not in sd.LOADERS:
        return section('cleaning', '7. Cleaning', body + '<p>No dropped labels to show.</p>')

    raw_recs, raw = sd.LOADERS[d]()
    raw['nid'] = np.arange(len(raw))
    kept, _ = sd.clean(raw_recs, raw)
    dropped = drop_reasons(raw, set(kept['nid']), rules)
    gone = sorted(set(raw_recs['rec']) - set(kept['rec']))
    fixable = dropped[dropped['rule'] != rules[0]]
    rec = fixable['rec'].value_counts().index[0]
    r = raw_recs.set_index('rec').loc[rec]
    meta = recs.set_index('rec')
    shift = meta.loc[rec, 'shift_ms'] / 1000 if rec in meta.index else 0.0
    rd = fixable[fixable['rec'] == rec]
    w0 = max(densest_start(rd['onset'], cfg.clip_seconds, lead=0.5), 0.0)
    w1 = w0 + cfg.clip_seconds
    moved = {'onset': lambda x: x['onset'] + shift, 'offset': lambda x: x['offset'] + shift}
    shown_kept = in_window(kept[kept['rec'] == rec].assign(**moved), w0, w1)
    shown_drop = in_window(rd.assign(**moved), w0, w1)
    y = load_clip(r['audio'], w0, cfg.clip_seconds)

    fig, ax = plt.subplots(figsize=(12, 4.2))
    draw_cqt(ax, y, w0)
    draw_boxes(ax, shown_kept, lw=1.0)
    styles = dict(zip(rules[1:], itertools.cycle(['-', (0, (1, 1)), (0, (4, 2)), (0, (6, 2, 1, 2))])))
    for rule, ls in styles.items():
        draw_boxes(ax, shown_drop[shown_drop['rule'] == rule], color=DROP_COLOR, lw=2.2, ls=ls)
    ax.set_xlim(w0, w0 + len(y) / SR)
    ax.set_xlabel('time in recording (s)')
    handles = string_handles() + [
        Line2D([], [], color=DROP_COLOR, lw=2.2, ls=ls, label=f'dropped: {rule}') for rule, ls in styles.items()
    ]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, frameon=False, fontsize=8)
    fig.tight_layout()
    src = assets.fig(fig, 'cleaning_example')
    name = info_of(d)['name']
    rule_counts = ', '.join(f'{k} {v}' for k, v in shown_drop['rule'].value_counts().items())
    body += (
        f'<h3>Dropped labels in {esc(name)}</h3>'
        f'<p class="reclabel"><code>{esc(rec)}</code> · player <b>{esc(r["player"])}</b> · chain '
        f'<b>{esc(r["chain"])}</b> · {w0:.1f}&ndash;{w1:.1f} s</p>'
        + audio(assets.wav(y, 'cleaning_example'))
        + figure(
            src,
            'Dropped labels over the spectrogram',
            f'The {cfg.clip_seconds} s of the recording with the most dropped labels ({esc(rule_counts)}). Red boxes '
            "were removed, colored boxes kept; all are drawn with the recording's alignment shift applied. A "
            'dropped box sitting inside or right after a kept one on the same pitch is a duplicate trigger.',
        )
    )
    if gone:
        examples = ', '.join(f'<code>{esc(g)}</code>' for g in gone[:6])
        body += (
            f'<p>{len(gone)} {esc(name)} recordings lost every label and are not in the index '
            f'(e.g. {examples}{" …" if len(gone) > 6 else ""}).</p>'
        )
    return section('cleaning', '7. Cleaning', body)


def model_window_frames():
    """Frames the default training config uses, read from conf/strings so the page follows it."""
    conf = REPO_ROOT / 'conf' / 'strings'
    data = next(d['data'] for d in OmegaConf.load(conf / 'train.yaml').defaults if 'data' in d)
    return st.PRE_FRAMES + st.post_frames(OmegaConf.load(conf / 'data' / f'{data}.yaml').window_ms), data


def fmt_param(key, v):
    label = key.removesuffix('_hz').removesuffix('_db').replace('_', ' ')
    if isinstance(v, bool):
        return label, 'on' if v else 'off'
    if key == 'snr_db':
        return 'SNR', f'{v:.0f} dB'
    if key.endswith('_hz'):
        return label, f'{v:,.0f} Hz'
    if key.endswith('_db'):
        return label, f'{v:+.1f} dB'
    return label, f'{v:.2f}'


def chain_table(params, stages):
    rows = []
    for s in stages:
        if s in params:
            cells = ' · '.join(f'{a} <b>{b}</b>' for a, b in (fmt_param(k, v) for k, v in params[s].items()))
        else:
            cells = '<span class="muted">skipped in this draw</span>'
        rows.append({'Stage': s, 'Parameters': cells})
    return table(rows, cls='compact')


def pick_window_notes(rng, notes, start, length, k):
    """Up to k notes away from the excerpt edges, one per string first, onsets at least a window apart if possible."""
    ok = notes[(notes['onset'] >= start + 0.1) & (notes['onset'] <= start + length - 0.6)]
    order = ok.iloc[rng.permutation(len(ok))]
    order = pd.concat([order.drop_duplicates('string'), order[order.duplicated('string')]])
    gap = st.WINDOW_FRAMES * st.HOP / SR
    picked = []
    for spacing in (gap, 0.0):
        for i, n in order.iterrows():
            if (
                len(picked) < k
                and i not in picked
                and all(abs(n['onset'] - order.loc[j, 'onset']) >= spacing for j in picked)
            ):
                picked.append(i)
    return order.loc[picked].sort_values('onset')


def preprocessing_section(recs, notes, cfg, rng, assets):
    pool = recs[recs['split'] == 'train']
    di = pool[pool['chain'] == 'di']
    pool = di if len(di) else pool
    # A dense excerpt shows more windows and more of what the chains change than a single ringing note.
    by_rec = dict(tuple(notes[notes['rec'].isin(pool['rec'])].groupby('rec')))
    draws = []
    for i in rng.permutation(len(pool))[:20]:
        r = pool.iloc[i]
        t = excerpt_start(rng, by_rec[r['rec']], r['duration'], cfg.clip_seconds)
        draws.append((in_window(by_rec[r['rec']], t, t + cfg.clip_seconds)['onset'].round(2).nunique(), t, r))
    _, start, r = max(draws, key=lambda x: x[0])
    rn = by_rec[r['rec']]
    length = cfg.clip_seconds
    shown = in_window(rn, start, start + length)
    fi = sf.info(REPO_ROOT / r['audio'])

    raw, sr0 = sf.read(
        REPO_ROOT / r['audio'], start=int(start * fi.samplerate), frames=int(length * fi.samplerate), always_2d=True
    )
    y = load_clip(r['audio'], start, length)
    fig, axes = plt.subplots(2, 1, figsize=(12, 2.8), sharex=True)
    t_raw = start + np.arange(len(raw)) / sr0
    for c in range(raw.shape[1]):
        axes[0].plot(t_raw, raw[:, c] + 2 * np.abs(raw).max() * c, color=INK, lw=0.4)
    axes[0].set_title(f'(a) original: {sr0:,} Hz, {fi.channels} channel(s), {fi.subtype}', loc='left')
    axes[1].plot(start + np.arange(len(y)) / SR, y, color=INK, lw=0.4)
    axes[1].set_title(f'(b) resampled: {SR:,} Hz mono', loc='left')
    axes[1].set_xlabel('time in recording (s)')
    for a in axes:
        a.set_yticks([])
    fig.tight_layout()
    wave_src = assets.fig(fig, 'prep_waveforms')

    spec = st.spectrogram(y[None], FEATURE)
    vmax = float(spec.max())
    stages = chains.stages_for(r['chain'])
    chain_rng = np.random.default_rng(cfg.seed)
    chain_blocks, aug_specs = [], []
    for i in range(cfg.n_chains):
        params = chains.draw(chain_rng, stages)
        y_aug = chains.render(y, SR, params, chain_rng)
        s_aug = st.spectrogram(y_aug[None], FEATURE)
        aug_specs.append(s_aug)
        fig, ax = plt.subplots(figsize=(12, 2.6))
        draw_logmel(ax, s_aug[0], start, float(s_aug.max()))
        ax.set_xlabel('time in recording (s)')
        fig.tight_layout()
        chain_blocks.append(
            f'<div class="card"><h4>Chain {i + 1}</h4>{chain_table(params, stages)}'
            f'{audio(assets.wav(y_aug, f"prep_chain_{i}"))}'
            + figure(
                assets.fig(fig, f'prep_chain_{i}'),
                f'Augmentation chain {i + 1}',
                f"{FEATURE} of this chain's output, same display as (d). Compare the upper harmonics and the "
                'noise floor with (d): that is what the augmentation changes.',
            )
            + '</div>'
        )

    n_frames, data_cfg = model_window_frames()
    picked = pick_window_notes(rng, rn, start, length, cfg.n_windows)
    rel_onsets = picked['onset'].to_numpy() - start
    windows = st.note_windows(spec, rel_onsets)

    fig, ax = plt.subplots(figsize=(12, 3.4))
    draw_logmel(ax, spec[0], start, vmax)
    for t in shown.loc[shown['onset'] >= start, 'onset'].unique():
        ax.plot([t, t], [spec.shape[1] - 4, spec.shape[1] - 0.5], color='white', lw=1)
    dt = st.HOP / SR
    for k, n in enumerate(picked.itertuples()):
        f0 = round((n.onset - start) * SR / st.HOP) - st.PRE_FRAMES
        x0 = start + (f0 - 0.5) * dt
        ax.add_patch(
            Rectangle((x0, -0.5), st.WINDOW_FRAMES * dt, spec.shape[1], fill=False, ec='white', lw=1.2, ls='--')
        )
        ax.text(x0 + 0.01, 3 + 18 * (k % 3), str(k + 1), color='white', fontsize=8, fontweight='bold')
    ax.set_xlabel('time in recording (s)')
    ax.set_xlim(start, start + len(y) / SR)
    fig.tight_layout()
    spec_src = assets.fig(fig, 'prep_logmel')

    fig, axes = plt.subplots(1, len(picked), figsize=(2.05 * len(picked), 3.6), squeeze=False)
    for k, (n, w, ax) in enumerate(zip(picked.itertuples(), windows, axes[0], strict=True)):
        ax.imshow(w[0], origin='lower', aspect='auto', cmap='magma', vmin=vmax - 14, vmax=vmax, interpolation='nearest')
        ax.axvline(st.PRE_FRAMES - 0.5, color='white', lw=1)
        ax.axvline(n_frames - 0.5, color='white', lw=1, ls=':')
        cand = [STRING_NAMES[s] for s in np.flatnonzero(st.candidates(np.array([n.pitch]))[0])]
        true = STRING_NAMES[n.string - 1]
        ax.set_title(
            f'#{k + 1}  {note_name(n.pitch)} (MIDI {n.pitch})\ntrue: {true} string, fret {n.fret}\n'
            f'candidates: {" ".join(cand) or "none"}' + ('' if true in cand else '\n(true string excluded!)'),
            fontsize=8,
            loc='left',
        )
        ax.set_xticks([0, st.PRE_FRAMES, st.WINDOW_FRAMES - 1])
        ax.set_xlabel('frame')
        if k == 0:
            mel_ticks(ax, w.shape[1])
        else:
            ax.set_yticks([])
    fig.tight_layout()
    win_src = assets.fig(fig, 'prep_windows')

    one = picked.iloc[0]
    rows = [('clean', spec), *((f'chain {i + 1}', s) for i, s in enumerate(aug_specs))]
    fig, axes = plt.subplots(1, len(rows), figsize=(2.05 * len(rows), 3.0), squeeze=False)
    for k, ((label, s), ax) in enumerate(zip(rows, axes[0], strict=True)):
        w = st.note_windows(s, np.array([one['onset'] - start]))[0, 0]
        ax.imshow(
            w, origin='lower', aspect='auto', cmap='magma', vmin=s.max() - 14, vmax=s.max(), interpolation='nearest'
        )
        ax.axvline(st.PRE_FRAMES - 0.5, color='white', lw=1)
        ax.set_title(label, fontsize=8.5, loc='left')
        ax.set_xticks([0, st.PRE_FRAMES, st.WINDOW_FRAMES - 1])
        if k == 0:
            mel_ticks(ax, w.shape[0])
        else:
            ax.set_yticks([])
    fig.tight_layout()
    same_src = assets.fig(fig, 'prep_same_note')

    ms = 1000 * st.HOP / SR
    body = (
        '<p>One training excerpt, followed from the file on disk to the tensors the classifier sees. '
        f'<b>Augmentation is train-time only</b>: val and test notes always come from the real recordings, steps '
        '(a), (b), (d) and (e) without (c).</p>'
        f'<p class="reclabel">{rec_label(r, start, length)}</p>'
        f'<h3>(a) Original file → (b) {SR:,} Hz mono</h3>'
        f'<p>The file is {fi.samplerate:,} Hz, {fi.channels} channel(s), {esc(fi.subtype)}, '
        f'{fi.duration:.1f} s. It is loaded with <code>librosa.load(sr={SR}, mono=True)</code>, as '
        '<code>build_index.py</code> does: channels are averaged and the signal is resampled (soxr).</p>'
        + audio(assets.wav(y, 'prep_clean'))
        + figure(wave_src, 'Waveforms before and after resampling', 'Same excerpt; the shape must not change.')
        + f'<h3>(c) Random recording chains</h3><p>A chain is drawn from a seeded generator and rendered with '
        f'pedalboard. This is a <b>{esc(r["chain"])}</b> recording, so it gets the stages '
        f'<b>{esc(" → ".join(stages))}</b>: DI recordings get the whole pickup → amp → cabinet → room → mic chain, '
        'recordings already made through an amp or a microphone only get room and mic. Input is peak-normalised '
        'before the chain and output after it, so loudness does not leak through. Three draws:</p>'
        + ''.join(chain_blocks)
        + f'<h3>(d) Classifier input: {FEATURE}</h3>'
        f'<p><code>strings.spectrogram(audio, "{FEATURE}")</code>: {spec.shape[1]} mel bands from {MEL_FMIN:.0f} Hz '
        f'to {SR // 2:,} Hz, 2048-point FFT, hop {st.HOP} samples ({ms:.1f} ms per frame), natural log of power. '
        f'This excerpt is {spec.shape[2]} frames. White ticks on top: label onsets; dashed boxes: the windows '
        'shown in (e).</p>'
        + figure(spec_src, f'{FEATURE} of the clean excerpt', 'Brightness is log power; the color scale spans 14 nats.')
        + '<h3>(e) Per-note windows</h3>'
        f'<p><code>strings.note_windows</code> crops {st.PRE_FRAMES} frames before each onset and '
        f'{st.MAX_POST_FRAMES} after ({st.WINDOW_FRAMES} frames, {st.WINDOW_FRAMES * ms:.0f} ms). The default config '
        f'(<code>data={esc(data_cfg)}</code>) keeps the first {n_frames} (dotted line). The model gets the window, '
        'standardised per mel band with train-set mean and std, plus the MIDI pitch, and its logits outside the '
        f'candidate strings (those reaching the pitch within frets 0&ndash;{st.MAX_FRET}) are masked out.</p>'
        + figure(
            win_src,
            'Per-note windows',
            'Solid line: the labelled onset. Each window shows its note together with whatever else is ringing; '
            'the classifier has to read the string from the timbre of the note that starts at the line.',
        )
        + figure(
            same_src,
            'One note through each chain',
            f"Window #1 cropped from the clean audio and from each chain's output: the same label, "
            f'{cfg.n_chains + 1} different inputs.',
        )
    )
    return section('preprocessing', '8. Preprocessing, step by step', body)


CSS = """
:root{--bg:#fcfcfb;--fg:#1d1d1b;--muted:#6b6a64;--line:#e3e2dd;--card:#f4f3ef;--accent:#2a78d6;
--ok:#1a7f4b;--warn:#b5521b;--train:#1c5cab;--val:#8a5a00;--test:#9b2c63}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a8a79f;--line:#34332f;--card:#1f1f1d;
--accent:#6da7ec;--ok:#5cc28d;--warn:#f0a070;--train:#86b6ef;--val:#e8b94a;--test:#e98bb6}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",
Roboto,Helvetica,Arial,sans-serif;display:grid;grid-template-columns:230px minmax(0,1fr)}
nav{position:sticky;top:0;height:100vh;overflow:auto;padding:28px 18px;border-right:1px solid var(--line)}
nav b{display:block;font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
nav a{display:block;padding:5px 8px;border-radius:6px;color:var(--fg);text-decoration:none;font-size:14.5px}
nav a:hover{background:var(--card)}
main{max-width:1240px;padding:28px 36px 120px}
h1{font-size:28px;margin:0 0 6px}
h2{font-size:22px;margin:56px 0 12px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:18px;margin:32px 0 8px}
h4{font-size:15.5px;margin:18px 0 6px}
p{max-width:78ch}
a{color:var(--accent)}
code{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--card);padding:1px 5px;border-radius:4px;
overflow-wrap:anywhere}
.meta,.muted,small{color:var(--muted)}
.tablewrap{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;font-size:14px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:13px;white-space:nowrap}
td{font-variant-numeric:tabular-nums}
table.overview td:nth-child(3){min-width:260px;max-width:340px}
table.compact td{padding:3px 10px;font-size:13.5px}
figure{margin:14px 0 22px}
figure img{max-width:100%;height:auto;display:block;background:#fff;border-radius:6px;border:1px solid var(--line)}
figcaption{font-size:14px;color:var(--muted);margin-top:6px;max-width:90ch}
audio{display:block;width:100%;max-width:520px;margin:6px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:14px 0}
.reclabel{font-size:14px;margin:4px 0}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:10px 0;font-size:14px}
.legend i{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:6px;vertical-align:-2px}
.split{font-weight:600;font-size:13px;padding:1px 7px;border-radius:9px;border:1px solid currentColor}
.split.train{color:var(--train)}.split.val{color:var(--val)}.split.test{color:var(--test)}
.checks{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:6px 18px;margin:18px 0}
.checks h3{margin:10px 0 4px;font-size:16px}
.checks li{margin:6px 0}.checks li.ok::marker{content:"✓  ";color:var(--ok)}
.checks li.warn::marker{content:"!  ";color:var(--warn);font-weight:700}.checks li.info::marker{content:"i  ";color:var(--accent)}
ul.note{font-size:14.5px}
@media (max-width:900px){body{display:block}nav{position:sticky;height:auto;padding:8px 12px;display:flex;
gap:4px;overflow-x:auto;border-right:0;border-bottom:1px solid var(--line);background:var(--bg);z-index:2}
nav b{display:none}nav a{white-space:nowrap;font-size:13.5px}main{padding:16px}}
"""

TOC = [
    ('overview', 'Overview'),
    ('splits', 'Splits'),
    ('labels', 'Labels'),
    ('samples', 'Samples'),
    ('walkthrough', 'How alignment works'),
    ('alignment', 'Alignment results'),
    ('cleaning', 'Cleaning'),
    ('preprocessing', 'Preprocessing'),
]


def page(sections, recs, datasets):
    toc = ''.join(f'<a href="#{sid}">{i + 1}. {t}</a>' for i, (sid, t) in enumerate(TOC))
    names = ', '.join(info_of(d)['name'] for d in datasets)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f'<title>String data tour</title><style>{CSS}</style></head><body>'
        f'<nav><b>String data tour</b>{toc}</nav><main>'
        '<h1>String classifier training data</h1>'
        f'<p class="meta">{len(recs):,} recordings from {esc(names)} · generated '
        f'{datetime.now():%Y-%m-%d %H:%M}</p>' + ''.join(sections) + '</main></body></html>'
    )


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='data_tour')
def main(cfg):
    t0 = time.time()
    index_dir = REPO_ROOT / cfg.index_dir
    out = REPO_ROOT / cfg.out_dir / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    assets = Assets(out, cfg.dpi)
    recs, notes, report, datasets = load_index(index_dir)
    info(f'{len(recs)} recordings, {len(notes)} notes from {datasets}')
    rng = np.random.default_rng(cfg.seed)
    steps = [
        ('overview', lambda: overview_section(recs, notes, report, datasets, index_dir)),
        ('splits', lambda: splits_section(recs, notes, datasets)),
        ('labels', lambda: labels_section(notes, datasets, assets)),
        ('samples', lambda: samples_section(recs, notes, datasets, cfg, rng, assets)),
        ('walkthrough', lambda: walkthrough_section(recs, notes, report, cfg, assets)),
        ('alignment', lambda: alignment_section(recs, notes, report, datasets, cfg, assets)),
        ('cleaning', lambda: cleaning_section(recs, notes, report, datasets, cfg, assets)),
        ('preprocessing', lambda: preprocessing_section(recs, notes, cfg, rng, assets)),
    ]
    sections = []
    for name, build in steps:
        sections.append(build())
        info(f'{name} done ({time.time() - t0:.0f} s)')
    (out / 'index.html').write_text(page(sections, recs, datasets))
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6
    info(f'Wrote {out}/index.html ({size:.1f} MB, {time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()

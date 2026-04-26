"""Comprehensive data quality analysis for stage-2 guitar tablature datasets.

Reads data/{dadagp,guitarset,guitartoday,leduc}/processed/*.json, accumulates
statistics without holding all notes in memory at once, generates publication-
quality charts to results/analysis/, and prints a summary report to stdout.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from gtp import REPO_ROOT

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATASETS = ['dadagp', 'guitarset', 'guitartoday', 'leduc']
DATA_ROOT = REPO_ROOT / 'data'
OUTPUT_DIR = REPO_ROOT / 'results' / 'analysis'

STANDARD_TUNING = [64, 59, 55, 50, 45, 40]
MIDI_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Seaborn palette - one colour per dataset, consistent across all charts
PALETTE = dict(zip(DATASETS, sns.color_palette('tab10', n_colors=len(DATASETS)), strict=True))

# Histogram bin edges used consistently
PITCH_BINS = np.arange(30, 100, 1)  # MIDI pitch 30-99
FRET_BINS = np.arange(-2, 30, 1)  # frets -2 to 29
DUR_BINS = np.logspace(-2, 2, 80)  # 0.01 s - 100 s (log)
ONSET_GAP_BINS = np.logspace(-3, 2, 80)  # 0.001 s - 100 s (log)
NOTES_PER_PIECE_BINS = np.arange(0, 2000, 20)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def midi_to_name(midi: int) -> str:
    octave = (midi // 12) - 1
    name = MIDI_NOTE_NAMES[midi % 12]
    return f'{name}{octave}'


def tuning_label(tuning: list) -> str:
    return '/'.join(midi_to_name(m) for m in tuning)


def json_files(dataset: str) -> list[Path]:
    return sorted((DATA_ROOT / dataset / 'processed').glob('*.json'))


# ---------------------------------------------------------------------------
# Per-dataset accumulation
# ---------------------------------------------------------------------------


def analyse_dataset(dataset: str) -> dict:
    """Stream through every JSON file in *dataset* and accumulate statistics.

    Returns a stats dict suitable for reporting and plotting.
    """
    files = json_files(dataset)
    n_files = len(files)
    print(f'  {dataset}: {n_files} files', flush=True)

    # Accumulators
    total_notes = 0
    total_inconsistent = 0
    pitch_counts = Counter()  # MIDI pitch -> count
    string_counts = Counter()  # string number -> count
    fret_counts = Counter()  # fret -> count
    notes_per_piece: list[int] = []  # one entry per file
    piece_durations: list[float] = []  # last-note-end per file, seconds
    tuning_counts = Counter()  # tuple(tuning) -> count
    std_tuning_notes = 0  # notes from files with standard tuning

    # Pre-binned histograms (avoids storing millions of raw floats)
    dur_hist = np.zeros(len(DUR_BINS) - 1, dtype=np.int64)
    dur_min = np.inf
    dur_max = -np.inf
    onset_gap_hist = np.zeros(len(ONSET_GAP_BINS) - 1, dtype=np.int64)
    onset_gap_large = 0  # count of onset gaps > 30 s

    # For consistency checking
    capo_files: list[dict] = []  # {file, offset}
    harmonic_files: list[dict] = []  # {file, n_inconsistent}
    bug_files: list[dict] = []  # {file, n_inconsistent, diffs}

    # Red-flag counters
    neg_fret_notes: list[dict] = []  # {file, fret}
    high_fret_notes: list[dict] = []  # {file, fret}
    zero_dur_notes: list[dict] = []  # {file}
    long_notes: list[dict] = []  # {file, duration}
    short_pieces: list[str] = []  # filenames with < 10 notes

    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f'    WARNING: cannot read {path.name}: {exc}', file=sys.stderr)
            continue

        tuning = data.get('tuning', STANDARD_TUNING)
        notes = data.get('notes', [])
        n = len(notes)

        tuning_counts[tuple(tuning)] += 1
        total_notes += n
        notes_per_piece.append(n)
        if tuple(tuning) == tuple(STANDARD_TUNING):
            std_tuning_notes += n

        if n < 10:
            short_pieces.append(path.name)

        # --- per-note pass ---
        starts: list[float] = []
        piece_end = 0.0
        n_incon = 0
        diffs: list[int] = []

        for note in notes:
            pitch = note['pitch']
            string = note['string']
            fret = note['fret']
            start = note['start']
            end = note['end']

            pitch_counts[pitch] += 1
            string_counts[string] += 1
            fret_counts[fret] += 1
            starts.append(start)

            dur = end - start
            if dur <= 0:
                zero_dur_notes.append({'file': path.name})
            if dur > 10:
                long_notes.append({'file': path.name, 'duration': round(dur, 2)})
            if dur < dur_min:
                dur_min = dur
            if dur > dur_max:
                dur_max = dur

            if fret < 0:
                neg_fret_notes.append({'file': path.name, 'fret': fret})
            if fret > 24:
                high_fret_notes.append({'file': path.name, 'fret': fret})

            if end > piece_end:
                piece_end = end

            # consistency: pitch should equal tuning[string-1] + fret
            if 1 <= string <= len(tuning):
                expected = tuning[string - 1] + fret
                if pitch != expected:
                    n_incon += 1
                    diffs.append(pitch - expected)

        piece_durations.append(piece_end)
        total_inconsistent += n_incon

        if n_incon > 0:
            unique_diffs = set(diffs)
            if len(unique_diffs) == 1:
                d = diffs[0]
                if n_incon == n:
                    capo_files.append({'file': path.name, 'offset': d, 'n_notes': n})
                else:
                    # constant diff but not all notes - could be partial capo or harmonics
                    harmonic_files.append(
                        {'file': path.name, 'n_inconsistent': n_incon, 'n_notes': n, 'diffs': sorted(unique_diffs)}
                    )
            elif all(d in (12, 19, 24) for d in unique_diffs):
                harmonic_files.append(
                    {'file': path.name, 'n_inconsistent': n_incon, 'n_notes': n, 'diffs': sorted(unique_diffs)}
                )
            else:
                bug_files.append(
                    {'file': path.name, 'n_inconsistent': n_incon, 'n_notes': n, 'diffs': sorted(unique_diffs)}
                )

        # Accumulate duration histogram for this file's notes
        if notes:
            file_durs = np.array([note['end'] - note['start'] for note in notes])
            file_durs_pos = file_durs[file_durs > 0]
            if len(file_durs_pos):
                dur_hist += np.histogram(file_durs_pos, bins=DUR_BINS)[0]

        # Accumulate onset gap histogram for this file
        if len(starts) > 1:
            s = np.sort(np.array(starts))
            gaps = np.diff(s)
            gaps_pos = gaps[gaps > 0]
            if len(gaps_pos):
                onset_gap_hist += np.histogram(gaps_pos, bins=ONSET_GAP_BINS)[0]
                onset_gap_large += int((gaps_pos > 30).sum())

    # Compact summary
    all_pitches = sorted(pitch_counts.keys())
    all_frets = sorted(fret_counts.keys())

    return {
        'dataset': dataset,
        'n_files': n_files,
        'total_notes': total_notes,
        'total_inconsistent': total_inconsistent,
        'pitch_counts': pitch_counts,
        'string_counts': string_counts,
        'fret_counts': fret_counts,
        'notes_per_piece': notes_per_piece,
        'piece_durations': piece_durations,
        'tuning_counts': tuning_counts,
        'std_tuning_notes': std_tuning_notes,
        'dur_hist': dur_hist,
        'dur_min': dur_min if np.isfinite(dur_min) else 0.0,
        'dur_max': dur_max if np.isfinite(dur_max) else 0.0,
        'onset_gap_hist': onset_gap_hist,
        'onset_gap_large': onset_gap_large,
        'pitch_range': (min(all_pitches), max(all_pitches)) if all_pitches else (None, None),
        'fret_range': (min(all_frets), max(all_frets)) if all_frets else (None, None),
        'capo_files': capo_files,
        'harmonic_files': harmonic_files,
        'bug_files': bug_files,
        'neg_fret_notes': neg_fret_notes,
        'high_fret_notes': high_fret_notes,
        'zero_dur_notes': zero_dur_notes,
        'long_notes': long_notes,
        'short_pieces': short_pieces,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def apply_style():
    sns.set_theme(style='whitegrid', palette='tab10', font_scale=1.1)


def save(fig: plt.Figure, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path.relative_to(REPO_ROOT)}')


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------


def plot_pitch_distribution(stats_list: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig.suptitle('Pitch Distribution (MIDI)', fontsize=14, fontweight='bold')

    for ax, stats in zip(axes.flat, stats_list, strict=True):
        ds = stats['dataset']
        counts = stats['pitch_counts']
        pitches = np.array(sorted(counts.keys()))
        vals = np.array([counts[p] for p in pitches])
        ax.bar(pitches, vals, color=PALETTE[ds], alpha=0.8, width=1.0)
        ax.axvspan(40, 84, alpha=0.08, color='green', label='Standard guitar range (40-84)')
        ax.set_title(ds, fontweight='bold')
        ax.set_xlabel('MIDI pitch')
        ax.set_ylabel('Note count')
        ax.set_xlim(28, 100)
        # tick labels with note names every 12 semitones
        tick_vals = list(range(36, 97, 12))
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([midi_to_name(v) for v in tick_vals])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        if ax == axes.flat[0]:
            ax.legend(fontsize=9)

    fig.tight_layout()
    save(fig, '1_pitch_distribution.png')


def plot_string_usage(stats_list: list[dict]):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    fig.suptitle('String Usage (1 = high E, 6 = low E)', fontsize=14, fontweight='bold')

    for ax, stats in zip(axes.flat, stats_list, strict=True):
        ds = stats['dataset']
        counts = stats['string_counts']
        strings = sorted(counts.keys())
        vals = [counts[s] for s in strings]
        colors = ['red' if s > 6 else PALETTE[ds] for s in strings]
        ax.bar(strings, vals, color=colors, alpha=0.85)
        ax.set_title(ds, fontweight='bold')
        ax.set_xlabel('String number')
        ax.set_ylabel('Note count')
        ax.set_xticks(range(1, max(strings) + 1))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        if any(s > 6 for s in strings):
            ax.bar([], [], color='red', label='String > 6 (flag)')
            ax.legend(fontsize=9)

    fig.tight_layout()
    save(fig, '2_string_usage.png')


def plot_fret_distribution(stats_list: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig.suptitle('Fret Distribution', fontsize=14, fontweight='bold')

    for ax, stats in zip(axes.flat, stats_list, strict=True):
        ds = stats['dataset']
        counts = stats['fret_counts']
        frets = np.array(sorted(counts.keys()))
        vals = np.array([counts[f] for f in frets])
        ax.bar(frets, vals, color=PALETTE[ds], alpha=0.85, width=1.0)
        ax.axvspan(0, 12, alpha=0.08, color='green', label='Frets 0-12 (common range)')
        ax.set_title(ds, fontweight='bold')
        ax.set_xlabel('Fret number')
        ax.set_ylabel('Note count')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        if ax == axes.flat[0]:
            ax.legend(fontsize=9)

    fig.tight_layout()
    save(fig, '3_fret_distribution.png')


def plot_duration_distribution(stats_list: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig.suptitle('Note Duration Distribution (log scale)', fontsize=14, fontweight='bold')

    bin_centers = np.sqrt(DUR_BINS[:-1] * DUR_BINS[1:])  # geometric midpoints
    bin_widths = DUR_BINS[1:] - DUR_BINS[:-1]

    for ax, stats in zip(axes.flat, stats_list, strict=True):
        ds = stats['dataset']
        counts = stats['dur_hist']
        ax.bar(bin_centers, counts, width=bin_widths, color=PALETTE[ds], alpha=0.8, align='center')
        ax.set_xscale('log')
        ax.set_title(ds, fontweight='bold')
        ax.set_xlabel('Duration (s)')
        ax.set_ylabel('Note count')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        n_zero = len(stats['zero_dur_notes'])
        if n_zero:
            ax.text(
                0.97,
                0.95,
                f'zero/neg dur: {n_zero:,}',
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=9,
                color='red',
            )

    fig.tight_layout()
    save(fig, '4_note_duration.png')


def plot_notes_per_piece(stats_list: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig.suptitle('Notes per Piece', fontsize=14, fontweight='bold')

    for ax, stats in zip(axes.flat, stats_list, strict=True):
        ds = stats['dataset']
        npp = np.array(stats['notes_per_piece'])
        ax.hist(npp, bins=NOTES_PER_PIECE_BINS, color=PALETTE[ds], alpha=0.85)
        ax.set_title(ds, fontweight='bold')
        ax.set_xlabel('Notes per piece')
        ax.set_ylabel('Number of pieces')
        med = float(np.median(npp))
        ax.axvline(med, color='black', linestyle='--', linewidth=1.2, label=f'Median {med:.0f}')
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    fig.tight_layout()
    save(fig, '5_notes_per_piece.png')


def plot_onset_gaps(stats_list: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig.suptitle('Inter-note Onset Gap Distribution (log scale)', fontsize=14, fontweight='bold')

    bin_centers = np.sqrt(ONSET_GAP_BINS[:-1] * ONSET_GAP_BINS[1:])  # geometric midpoints
    bin_widths = ONSET_GAP_BINS[1:] - ONSET_GAP_BINS[:-1]

    for ax, stats in zip(axes.flat, stats_list, strict=True):
        ds = stats['dataset']
        counts = stats['onset_gap_hist']
        if counts.sum() == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(ds, fontweight='bold')
            continue
        ax.bar(bin_centers, counts, width=bin_widths, color=PALETTE[ds], alpha=0.8, align='center')
        ax.set_xscale('log')
        ax.axvline(30, color='red', linestyle='--', linewidth=1.2, label='30 s threshold')
        ax.set_title(ds, fontweight='bold')
        ax.set_xlabel('Onset gap (s)')
        ax.set_ylabel('Count')
        n_large = stats['onset_gap_large']
        if n_large:
            ax.text(
                0.97,
                0.95,
                f'gaps > 30 s: {n_large:,}',
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=9,
                color='red',
            )
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    fig.tight_layout()
    save(fig, '6_onset_gaps.png')


def plot_tuning_distribution(stats_list: list[dict]):
    """Bar chart of unique tuning counts across all datasets."""
    # Aggregate across datasets
    global_counts: Counter = Counter()
    for stats in stats_list:
        global_counts.update(stats['tuning_counts'])

    # Sort by frequency descending, keep top 20
    top = global_counts.most_common(20)
    tunings = [t[0] for t in top]
    counts = [t[1] for t in top]
    labels = [tuning_label(list(t)) for t in tunings]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tunings))
    bars = ax.bar(x, counts, color=sns.color_palette('tab10', n_colors=len(tunings)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of pieces')
    ax.set_title('Tuning Distribution (top 20, all datasets)', fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Annotate standard tuning
    std_tup = tuple(STANDARD_TUNING)
    for i, t in enumerate(tunings):
        if t == std_tup:
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2.0)
            ax.text(
                i, counts[i] + counts[0] * 0.01, 'Standard', ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

    fig.tight_layout()
    save(fig, '7_tuning_distribution.png')


def plot_summary_overview(stats_list: list[dict]):
    """4-panel summary: pitch, fret, duration, notes-per-piece (overlaid by dataset)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Overview — All Datasets', fontsize=15, fontweight='bold')

    ax_pitch, ax_fret, ax_dur, ax_npp = axes.flat

    # Pitch: one line per dataset
    for stats in stats_list:
        ds = stats['dataset']
        counts = stats['pitch_counts']
        pitches = sorted(counts.keys())
        vals = [counts[p] for p in pitches]
        ax_pitch.bar(pitches, vals, color=PALETTE[ds], alpha=0.55, width=1.0, label=ds)
    ax_pitch.axvspan(40, 84, alpha=0.08, color='green')
    ax_pitch.set_xlabel('MIDI pitch')
    ax_pitch.set_ylabel('Note count')
    ax_pitch.set_title('Pitch Distribution', fontweight='bold')
    tick_vals = list(range(36, 97, 12))
    ax_pitch.set_xticks(tick_vals)
    ax_pitch.set_xticklabels([midi_to_name(v) for v in tick_vals])
    ax_pitch.legend(fontsize=9)
    ax_pitch.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Fret
    for stats in stats_list:
        ds = stats['dataset']
        counts = stats['fret_counts']
        frets = sorted(counts.keys())
        vals = [counts[f] for f in frets]
        ax_fret.bar(frets, vals, color=PALETTE[ds], alpha=0.55, width=1.0, label=ds)
    ax_fret.axvspan(0, 12, alpha=0.08, color='green')
    ax_fret.set_xlabel('Fret number')
    ax_fret.set_ylabel('Note count')
    ax_fret.set_title('Fret Distribution', fontweight='bold')
    ax_fret.legend(fontsize=9)
    ax_fret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Duration (log) — use pre-binned histograms
    dur_bin_centers = np.sqrt(DUR_BINS[:-1] * DUR_BINS[1:])
    dur_bin_widths = DUR_BINS[1:] - DUR_BINS[:-1]
    for stats in stats_list:
        ds = stats['dataset']
        ax_dur.bar(
            dur_bin_centers,
            stats['dur_hist'],
            width=dur_bin_widths,
            color=PALETTE[ds],
            alpha=0.55,
            label=ds,
            align='center',
        )
    ax_dur.set_xscale('log')
    ax_dur.set_xlabel('Duration (s)')
    ax_dur.set_ylabel('Note count')
    ax_dur.set_title('Note Duration (log x)', fontweight='bold')
    ax_dur.legend(fontsize=9)
    ax_dur.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Notes per piece
    for stats in stats_list:
        ds = stats['dataset']
        npp = np.array(stats['notes_per_piece'])
        ax_npp.hist(npp, bins=NOTES_PER_PIECE_BINS, color=PALETTE[ds], alpha=0.55, label=ds)
    ax_npp.set_xlabel('Notes per piece')
    ax_npp.set_ylabel('Number of pieces')
    ax_npp.set_title('Notes per Piece', fontweight='bold')
    ax_npp.legend(fontsize=9)
    ax_npp.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    fig.tight_layout()
    save(fig, '0_summary_overview.png')


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def print_summary_table(stats_list: list[dict]):
    header = (
        f'{"Dataset":<14} {"Files":>6} {"Notes":>9} {"Pitch":>10} {"Fret":>8} '
        f'{"Dur (s)":>14} {"Piece (s)":>14} {"% Std":>7} {"% Incon":>8}'
    )
    sep = '-' * len(header)
    print('\n' + '=' * len(header))
    print('SUMMARY STATISTICS TABLE')
    print('=' * len(header))
    print(header)
    print(sep)

    agg = defaultdict(int)
    agg_pitch_min, agg_pitch_max = 999, -999
    agg_fret_min, agg_fret_max = 999, -999
    agg_dur_min, agg_dur_max = 1e9, -1e9
    agg_piece_min, agg_piece_max = 1e9, -1e9
    agg_std_notes = 0
    agg_notes = 0

    for stats in stats_list:
        ds = stats['dataset']
        nf = stats['n_files']
        nt = stats['total_notes']
        ni = stats['total_inconsistent']
        pmin, pmax = stats['pitch_range']
        fmin, fmax = stats['fret_range']

        dmin = stats['dur_min']
        dmax = stats['dur_max']

        pds = stats['piece_durations']
        if pds:
            pdmin = min(pds)
            pdmax = max(pds)
        else:
            pdmin = pdmax = 0.0

        # % notes from files using standard tuning
        pct_std = 100.0 * stats['std_tuning_notes'] / nt if nt else 0.0
        pct_incon = 100.0 * ni / nt if nt else 0.0

        print(
            f'{ds:<14} {nf:>6,} {nt:>9,} '
            f'{f"{pmin}-{pmax}":>10} {f"{fmin}-{fmax}":>8} '
            f'{f"{dmin:.2f}-{dmax:.1f}":>14} {f"{pdmin:.0f}-{pdmax:.0f}":>14} '
            f'{pct_std:>6.1f}% {pct_incon:>7.2f}%'
        )

        agg['n_files'] += nf
        agg['n_notes'] += nt
        agg['n_incon'] += ni
        agg_pitch_min = min(agg_pitch_min, pmin) if pmin is not None else agg_pitch_min
        agg_pitch_max = max(agg_pitch_max, pmax) if pmax is not None else agg_pitch_max
        agg_fret_min = min(agg_fret_min, fmin) if fmin is not None else agg_fret_min
        agg_fret_max = max(agg_fret_max, fmax) if fmax is not None else agg_fret_max
        agg_dur_min = min(agg_dur_min, dmin)
        agg_dur_max = max(agg_dur_max, dmax)
        agg_piece_min = min(agg_piece_min, pdmin)
        agg_piece_max = max(agg_piece_max, pdmax)
        agg_std_notes += stats['std_tuning_notes']
        agg_notes += nt

    print(sep)
    pct_std_all = 100.0 * agg_std_notes / agg_notes if agg_notes else 0
    pct_incon_all = 100.0 * agg['n_incon'] / agg['n_notes'] if agg['n_notes'] else 0
    print(
        f'{"Combined":<14} {agg["n_files"]:>6,} {agg["n_notes"]:>9,} '
        f'{f"{agg_pitch_min}-{agg_pitch_max}":>10} {f"{agg_fret_min}-{agg_fret_max}":>8} '
        f'{f"{agg_dur_min:.2f}-{agg_dur_max:.1f}":>14} '
        f'{f"{agg_piece_min:.0f}-{agg_piece_max:.0f}":>14} '
        f'{pct_std_all:>6.1f}% {pct_incon_all:>7.2f}%'
    )
    print('=' * len(header))
    print('  Note: "% Std" = notes from files using standard tuning [E2/A2/D3/G3/B3/E4]')
    print('  Note: "% Incon" = notes where pitch != tuning[string-1] + fret')


def print_consistency_report(stats_list: list[dict]):
    print('\n' + '=' * 70)
    print('PITCH CONSISTENCY REPORT')
    print('=' * 70)

    for stats in stats_list:
        ds = stats['dataset']
        nt = stats['total_notes']
        ni = stats['total_inconsistent']
        capo = stats['capo_files']
        harm = stats['harmonic_files']
        bugs = stats['bug_files']

        print(f'\n[{ds}]')
        print(f'  Inconsistent notes: {ni:,} / {nt:,} ({100 * ni / nt:.2f}%)' if nt else '  No notes.')

        if capo:
            print(f'  Capo files ({len(capo)} total — 100% inconsistent, constant diff):')
            for c in capo[:20]:
                print(f'    offset={c["offset"]:+d}  {c["file"]}')
            if len(capo) > 20:
                print(f'    ... and {len(capo) - 20} more')

        if harm:
            print(f'  Likely-harmonic files ({len(harm)} total — partial inconsistency):')
            for h in harm[:10]:
                print(f'    n_incon={h["n_inconsistent"]}  diffs={h["diffs"]}  {h["file"]}')
            if len(harm) > 10:
                print(f'    ... and {len(harm) - 10} more')

        if bugs:
            print(f'  Unclear/possible-bug files ({len(bugs)} total):')
            for b in bugs[:10]:
                print(f'    n_incon={b["n_inconsistent"]}  diffs={b["diffs"]}  {b["file"]}')
            if len(bugs) > 10:
                print(f'    ... and {len(bugs) - 10} more')

        if not capo and not harm and not bugs:
            print('  All notes consistent.')

    # Recommendations
    print('\n--- Recommendations ---')
    for stats in stats_list:
        ds = stats['dataset']
        capo = stats['capo_files']
        harm = stats['harmonic_files']
        bugs = stats['bug_files']
        if not capo and not harm and not bugs:
            continue
        print(f'\n{ds}:')
        if capo:
            print(
                f'  FIXABLE (capo): {len(capo)} files — add track.offset to tuning or re-parse with offset correction.'
            )
        if harm:
            print(
                f'  FIXABLE (harmonics): {len(harm)} files — harmonic notes (diff=12/19/24)'
                ' are a known GP feature; strip or keep with a flag.'
            )
        if bugs:
            print(f'  INVESTIGATE: {len(bugs)} files have variable diffs — check source data.')

    # Per-file compact listing of all inconsistent files
    any_inconsistent = any(stats['capo_files'] or stats['harmonic_files'] or stats['bug_files'] for stats in stats_list)
    if any_inconsistent:
        print('\n--- Per-file listing ---')
        print(f'{"Dataset":<12} {"File":<50} {"Offset":>7} {"Notes":>7} {"Incon":>7} {"Class":<10}')
        print('-' * 100)
        for stats in stats_list:
            ds = stats['dataset']
            for c in stats['capo_files']:
                print(f'{ds:<12} {c["file"]:<50} {c["offset"]:>+7} {c["n_notes"]:>7,} {c["n_notes"]:>7,} {"capo":<10}')
            for h in stats['harmonic_files']:
                print(f'{ds:<12} {h["file"]:<50} {"":>7} {h["n_notes"]:>7,} {h["n_inconsistent"]:>7,} {"harmonic":<10}')
            for b in stats['bug_files']:
                print(f'{ds:<12} {b["file"]:<50} {"":>7} {b["n_notes"]:>7,} {b["n_inconsistent"]:>7,} {"mixed":<10}')


def print_red_flags(stats_list: list[dict]):
    print('\n' + '=' * 70)
    print('RED FLAGS')
    print('=' * 70)

    # Capo (100% inconsistency) — already in consistency report, summarise here
    all_capo = [(stats['dataset'], c) for stats in stats_list for c in stats['capo_files']]
    print(f'\n[1] Files with 100% pitch inconsistency (capo): {len(all_capo)}')
    for ds, c in all_capo[:15]:
        print(f'    [{ds}] offset={c["offset"]:+d}  {c["file"]}')
    if len(all_capo) > 15:
        print(f'    ... and {len(all_capo) - 15} more')

    # Negative frets
    all_neg = [(stats['dataset'], n) for stats in stats_list for n in stats['neg_fret_notes']]
    print(f'\n[2] Notes with negative fret: {len(all_neg)}')
    seen_files: set = set()
    for ds, n in all_neg[:20]:
        key = (ds, n['file'])
        if key not in seen_files:
            seen_files.add(key)
            print(f'    [{ds}] fret={n["fret"]}  {n["file"]}')

    # High frets (> 24)
    all_high = [(stats['dataset'], n) for stats in stats_list for n in stats['high_fret_notes']]
    print(f'\n[3] Notes with fret > 24: {len(all_high)}')
    seen_files = set()
    for ds, n in all_high[:20]:
        key = (ds, n['file'])
        if key not in seen_files:
            seen_files.add(key)
            print(f'    [{ds}] fret={n["fret"]}  {n["file"]}')
    if len(all_high) > 20:
        print(f'    ... ({len(all_high)} notes total)')

    # Zero/negative duration
    all_zero = [(stats['dataset'], n) for stats in stats_list for n in stats['zero_dur_notes']]
    # Group by file
    file_counts: Counter = Counter((ds, n['file']) for ds, n in all_zero)
    print(f'\n[4] Notes with zero or negative duration: {len(all_zero)} in {len(file_counts)} files')
    for (ds, fname), cnt in sorted(file_counts.most_common(15)):
        print(f'    [{ds}] {cnt} notes  {fname}')
    if len(file_counts) > 15:
        print('    ... and more files')

    # Short pieces (< 10 notes)
    all_short = [(stats['dataset'], f) for stats in stats_list for f in stats['short_pieces']]
    print(f'\n[5] Pieces with < 10 notes: {len(all_short)}')
    for ds, f in all_short[:15]:
        print(f'    [{ds}] {f}')

    # Long notes (> 10 s)
    all_long = [(stats['dataset'], n) for stats in stats_list for n in stats['long_notes']]
    print(f'\n[6] Notes with duration > 10 s: {len(all_long)}')
    for ds, n in all_long[:15]:
        print(f'    [{ds}] dur={n["duration"]} s  {n["file"]}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Stage-2 Data Quality Analysis')
    print('=' * 60)
    print(f'Output directory: {OUTPUT_DIR}')
    print()

    stats_list: list[dict] = []
    for ds in DATASETS:
        stats = analyse_dataset(ds)
        stats_list.append(stats)

    # Generate charts
    print('\nGenerating charts...')
    plot_summary_overview(stats_list)
    plot_pitch_distribution(stats_list)
    plot_string_usage(stats_list)
    plot_fret_distribution(stats_list)
    plot_duration_distribution(stats_list)
    plot_notes_per_piece(stats_list)
    plot_onset_gaps(stats_list)
    plot_tuning_distribution(stats_list)

    # Text reports
    print_summary_table(stats_list)
    print_consistency_report(stats_list)
    print_red_flags(stats_list)

    print('\nDone.')


if __name__ == '__main__':
    main()

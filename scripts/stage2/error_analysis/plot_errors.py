"""Build error-analysis plots from enriched.jsonl.

Reads the per-note records produced by enrich_errors.py and writes a set of
PNG plots to <input-dir>/plots/.

Plots produced:
  01_raw_pitch_delta.png             - histogram of pred_raw_pitch - true_pitch
                                       (theory test: are misses ±1-2 semitones?)
  02_pitch_mismatch_string_fret.png  - same-string fraction + on-same-string Δfret
  03_raw_error_heatmap.png           - 2D (Δstring, Δfret) heatmap of raw errors
  04_string_confusion.png            - 6×6 string confusion matrix (raw + pp)
  05_accuracy_vs_abs_fret.png        - raw/pp accuracy vs absolute_fret
  06_pp_source_breakdown.png         - which pp path produced correct vs error
  07_per_piece_error_rate.png        - histogram of pp error rate per piece
  08_top_confusion_patterns.png      - top 20 (true→pred) confusion patterns

Usage:
  python scripts/stage2/error_analysis/plot_errors.py \\
      --input-dir results/error_analysis/run_60k_orig
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Plot 01 - raw pitch delta histogram (theory test)
# ---------------------------------------------------------------------------


def plot_raw_pitch_delta(records, out):
    deltas = np.array([
        r['delta_pitch_raw']
        for r in records
        if r['error_type_raw'] == 'pitch_mismatch' and r['delta_pitch_raw'] is not None
    ])
    if len(deltas) == 0:
        return

    bins = np.arange(-13, 14)
    fig, ax = plt.subplots(figsize=(11, 5))
    _, _, patches = ax.hist(deltas, bins=bins, edgecolor='black', linewidth=0.5)
    for edge, patch in zip(bins[:-1], patches):
        if abs(edge) in (1, 2):
            patch.set_facecolor('tab:red')
        else:
            patch.set_facecolor('tab:blue')

    pct_pm1 = 100 * np.mean(np.abs(deltas) == 1)
    pct_pm2 = 100 * np.mean(np.abs(deltas) == 2)
    pct_le2 = 100 * np.mean(np.abs(deltas) <= 2)

    ax.set_title(
        f'Raw pitch mismatches: {len(deltas):,} notes\n'
        f'|Δ|=1: {pct_pm1:.1f}%   |Δ|=2: {pct_pm2:.1f}%   |Δ|≤2: {pct_le2:.1f}%'
    )
    ax.set_xlabel('pred_raw_pitch − true_pitch (semitones)')
    ax.set_ylabel('count')
    ax.set_xticks(np.arange(-12, 13, 2))
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 02 - pitch mismatch: same-string fraction + on-same-string Δfret
# ---------------------------------------------------------------------------


def plot_pitch_mismatch_string_fret(records, out):
    mm = [
        r for r in records
        if r['error_type_raw'] == 'pitch_mismatch'
        and r['delta_string_raw'] is not None
        and r['delta_fret_raw'] is not None
    ]
    if not mm:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: |Δstring| distribution among pitch mismatches
    abs_dstring = np.abs([r['delta_string_raw'] for r in mm])
    max_d = int(abs_dstring.max())
    bins = np.arange(0, max_d + 2) - 0.5
    counts, _ = np.histogram(abs_dstring, bins=bins)
    colors = ['tab:red'] + ['tab:blue'] * (len(counts) - 1)
    axes[0].bar(np.arange(len(counts)), counts, color=colors, edgecolor='black', linewidth=0.5)
    same_pct = 100 * counts[0] / counts.sum()
    axes[0].set_title(
        f'Pitch mismatches by |Δstring| (n={len(mm):,})\n'
        f'same-string fraction: {same_pct:.1f}%'
    )
    axes[0].set_xlabel('|pred_string − true_string|')
    axes[0].set_ylabel('count')
    axes[0].set_xticks(np.arange(len(counts)))
    axes[0].grid(axis='y', alpha=0.3)

    # Right: Δfret distribution restricted to same-string pitch mismatches
    same = [r for r in mm if r['delta_string_raw'] == 0]
    if same:
        dfret = np.array([r['delta_fret_raw'] for r in same])
        bins = np.arange(-13, 14)
        _, _, patches = axes[1].hist(dfret, bins=bins, edgecolor='black', linewidth=0.5)
        for edge, patch in zip(bins[:-1], patches):
            if abs(edge) in (1, 2):
                patch.set_facecolor('tab:red')
            else:
                patch.set_facecolor('tab:blue')
        pct_le2 = 100 * np.mean(np.abs(dfret) <= 2)
        axes[1].set_title(
            f'Same-string pitch mismatches: Δfret (n={len(same):,})\n'
            f'|Δ|≤2: {pct_le2:.1f}%'
        )
        axes[1].set_xlabel('pred_fret − true_fret')
        axes[1].set_ylabel('count')
        axes[1].set_xticks(np.arange(-12, 13, 2))
        axes[1].axvline(0, color='black', linewidth=0.8, alpha=0.4)
        axes[1].grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 03 - 2D heatmap of (Δstring, Δfret) for raw errors
# ---------------------------------------------------------------------------


def plot_raw_error_heatmap(records, out):
    errors = [r for r in records if r['error_type_raw'] != 'correct']
    if not errors:
        return

    s_range = 6
    f_range = 12
    grid = np.zeros((2 * s_range + 1, 2 * f_range + 1), dtype=np.int64)
    for r in errors:
        ds, df = r['delta_string_raw'], r['delta_fret_raw']
        if ds is None or df is None:
            continue
        if abs(ds) <= s_range and abs(df) <= f_range:
            grid[ds + s_range, df + f_range] += 1

    fig, ax = plt.subplots(figsize=(11, 6))
    log_grid = np.log10(grid + 1)
    im = ax.imshow(
        log_grid, cmap='viridis', aspect='auto', origin='lower',
        extent=[-f_range - 0.5, f_range + 0.5, -s_range - 0.5, s_range + 0.5],
    )
    ax.axhline(0, color='red', linewidth=0.8, alpha=0.6)
    ax.axvline(0, color='red', linewidth=0.8, alpha=0.6)
    ax.set_title(f'Raw errors: (Δstring, Δfret) heatmap, log10 scale (n={len(errors):,})')
    ax.set_xlabel('pred_fret − true_fret')
    ax.set_ylabel('pred_string − true_string')
    ax.set_xticks(np.arange(-f_range, f_range + 1, 2))
    ax.set_yticks(np.arange(-s_range, s_range + 1))
    fig.colorbar(im, ax=ax, label='log10(count + 1)')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 04 - string confusion matrix (raw + pp)
# ---------------------------------------------------------------------------


def plot_string_confusion(records, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, kind in zip(axes, ('raw', 'pp')):
        cm = np.zeros((6, 6), dtype=np.int64)
        for r in records:
            ts = r['true_string']
            ps = r[f'pred_{kind}_string']
            if ts is None or ps is None:
                continue
            if 1 <= ts <= 6 and 1 <= ps <= 6:
                cm[ts - 1, ps - 1] += 1
        row_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        im = ax.imshow(row_norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                val = row_norm[i, j]
                if val > 0.01:
                    ax.text(
                        j, i, f'{val:.2f}',
                        ha='center', va='center',
                        color='white' if val > 0.5 else 'black',
                        fontsize=9,
                    )
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels([f'{i + 1}' for i in range(6)])
        ax.set_yticklabels([f'{i + 1}' for i in range(6)])
        ax.set_xlabel(f'predicted string ({kind})')
        ax.set_ylabel('true string')
        ax.set_title(f'String confusion ({kind}, row-normalized)')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 05 - accuracy vs absolute fret
# ---------------------------------------------------------------------------


def plot_accuracy_vs_abs_fret(records, out):
    by_fret = defaultdict(lambda: {'n': 0, 'raw': 0, 'pp': 0})
    for r in records:
        af = r.get('absolute_fret')
        if af is None or af < 0 or af > 24:
            continue
        b = by_fret[af]
        b['n'] += 1
        b['raw'] += int(r['error_type_raw'] == 'correct')
        b['pp'] += int(r['error_type_pp'] == 'correct')

    frets = sorted(by_fret.keys())
    n = np.array([by_fret[f]['n'] for f in frets])
    raw_acc = np.array([by_fret[f]['raw'] / by_fret[f]['n'] for f in frets])
    pp_acc = np.array([by_fret[f]['pp'] / by_fret[f]['n'] for f in frets])

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(frets, raw_acc, 'o-', color='tab:orange', label='raw accuracy', linewidth=2)
    ax1.plot(frets, pp_acc, 's-', color='tab:blue', label='pp accuracy', linewidth=2)
    ax1.set_xlabel('absolute fret (true_fret + capo)')
    ax1.set_ylabel('tab accuracy')
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(frets)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='lower left')

    ax2 = ax1.twinx()
    ax2.bar(frets, n, alpha=0.15, color='gray', label='note count')
    ax2.set_ylabel('note count', color='gray')
    ax2.tick_params(axis='y', colors='gray')

    ax1.set_title('Tab accuracy vs absolute fret')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 06 - pp_source breakdown (correct vs error fractions per source)
# ---------------------------------------------------------------------------


def plot_pp_source_breakdown(records, out):
    by_source = defaultdict(lambda: {'correct': 0, 'error': 0})
    for r in records:
        src = r.get('pp_source', 'unknown')
        is_correct = r['error_type_pp'] == 'correct'
        by_source[src]['correct' if is_correct else 'error'] += 1

    sources = ['unchanged', 'window_swap', 'fallback']
    sources = [s for s in sources if s in by_source]
    correct = np.array([by_source[s]['correct'] for s in sources])
    error = np.array([by_source[s]['error'] for s in sources])
    totals = correct + error
    correct_pct = 100 * correct / np.clip(totals, 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: stacked absolute counts
    axes[0].bar(sources, correct, color='tab:green', label='correct')
    axes[0].bar(sources, error, bottom=correct, color='tab:red', label='error')
    for i, (s, c, e, t) in enumerate(zip(sources, correct, error, totals)):
        axes[0].text(i, t * 1.01, f'n={t:,}', ha='center', fontsize=9)
    axes[0].set_ylabel('note count')
    axes[0].set_title('pp prediction outcomes by source')
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)

    # Right: error contribution share — what fraction of all pp errors came from each path
    error_share = 100 * error / max(error.sum(), 1)
    axes[1].bar(sources, error_share, color='tab:red')
    for i, (s, e, share) in enumerate(zip(sources, error, error_share)):
        accuracy_in_path = correct_pct[i]
        axes[1].text(i, share + 0.5, f'{share:.1f}%\n(acc {accuracy_in_path:.1f}%)',
                     ha='center', fontsize=9)
    axes[1].set_ylabel('share of total pp errors (%)')
    axes[1].set_title(f'Where do pp errors come from? (total errors n={error.sum():,})')
    axes[1].set_ylim(0, max(error_share) * 1.2 + 5)
    axes[1].grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 07 - per-piece error rate distribution
# ---------------------------------------------------------------------------


def plot_per_piece_error_rate(records, out, min_notes=20):
    by_piece = defaultdict(lambda: {'n': 0, 'err': 0})
    for r in records:
        pid = r['piece_id']
        b = by_piece[pid]
        b['n'] += 1
        b['err'] += int(r['error_type_pp'] != 'correct')

    rates = np.array([
        b['err'] / b['n'] for b in by_piece.values() if b['n'] >= min_notes
    ])
    if not len(rates):
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rates, bins=np.linspace(0, 1, 41), edgecolor='black', linewidth=0.5,
            color='tab:purple')
    ax.axvline(np.median(rates), color='red', linestyle='--',
               label=f'median = {np.median(rates):.2f}')
    ax.axvline(np.mean(rates), color='black', linestyle='--',
               label=f'mean = {np.mean(rates):.2f}')
    ax.set_title(f'Per-piece pp error rate ({len(rates)} pieces with ≥{min_notes} notes)')
    ax.set_xlabel('pp error rate')
    ax.set_ylabel('piece count')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 08 - top confusion patterns (pp errors)
# ---------------------------------------------------------------------------


def plot_top_confusion_patterns(records, out, top_n=20):
    pairs = Counter()
    for r in records:
        if r['error_type_pp'] == 'correct':
            continue
        if r['pred_pp_string'] is None or r['pred_pp_fret'] is None:
            continue
        key = (
            (r['true_string'], r['true_fret']),
            (r['pred_pp_string'], r['pred_pp_fret']),
        )
        pairs[key] += 1

    top = pairs.most_common(top_n)
    if not top:
        return
    labels = [f'({t[0]},{t[1]}) → ({p[0]},{p[1]})' for (t, p), _ in top]
    counts = [c for _, c in top]
    total_err = sum(pairs.values())
    pct = [100 * c / total_err for c in counts]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, counts, color='tab:red')
    for i, (c, p) in enumerate(zip(counts, pct)):
        ax.text(c, y[i], f'  {c:,} ({p:.1f}%)', va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, family='monospace')
    ax.set_xlabel('count')
    ax.set_title(f'Top {top_n} pp confusion patterns (out of {total_err:,} pp errors)')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True,
                   help='Directory containing enriched.jsonl')
    p.add_argument('--enriched', type=Path,
                   help='Override path to enriched.jsonl')
    p.add_argument('--out-dir', type=Path,
                   help='Override output directory (default: <input-dir>/plots)')
    args = p.parse_args()

    enriched = args.enriched or (args.input_dir / 'enriched.jsonl')
    out_dir = args.out_dir or (args.input_dir / 'plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading {enriched}...')
    records = load_records(enriched)
    print(f'  {len(records):,} records')

    print('Plotting...')
    plot_raw_pitch_delta(records, out_dir / '01_raw_pitch_delta.png')
    plot_pitch_mismatch_string_fret(records, out_dir / '02_pitch_mismatch_string_fret.png')
    plot_raw_error_heatmap(records, out_dir / '03_raw_error_heatmap.png')
    plot_string_confusion(records, out_dir / '04_string_confusion.png')
    plot_accuracy_vs_abs_fret(records, out_dir / '05_accuracy_vs_abs_fret.png')
    plot_pp_source_breakdown(records, out_dir / '06_pp_source_breakdown.png')
    plot_per_piece_error_rate(records, out_dir / '07_per_piece_error_rate.png')
    plot_top_confusion_patterns(records, out_dir / '08_top_confusion_patterns.png')
    print(f'Wrote plots to {out_dir}/')


if __name__ == '__main__':
    main()

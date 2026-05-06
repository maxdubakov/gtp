"""Slice the enriched per-note records to find what correlates with errors.

Reads the enriched.jsonl produced by enrich_errors.py and reports:

  * Overall raw / pp accuracy and error-type breakdown.
  * Per-source accuracy.
  * Per-genre accuracy (DadaGP).
  * Per-style accuracy (GuitarSet).
  * Tempo buckets, polyphony buckets, fret-height buckets, density buckets.
  * Top (true -> pred) confusion patterns.
  * Per-piece error-rate distribution + outlier pieces.
  * Local-context correlations (interval_from_prev, prev_string_dist, ...)
  * Per-piece drift signature: bucket pieces by whether the model's errors
    are dominated by a single consistent (Δstring, Δfret) shift.
  * Tab-equivalent accuracy: tab_strict + notes whose error matches their
    piece's modal drift — captures "model picked a valid alternate fingering
    consistently throughout the piece".
  * Drift-pattern histogram across consistent-alternate pieces.

Output:
  Print a human-readable report and save aggregates to <output>/summary.json.

Usage:
  python scripts/stage2/error_analysis/analyze_errors.py \\
      --enriched results/error_analysis/run_60k/enriched.jsonl \\
      --output-dir results/error_analysis/run_60k
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gtp.stage2.metrics import piece_drift_signature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def accuracy(records: list[dict], field: str) -> tuple[float, int]:
    """(accuracy, n) where accuracy = correct fraction. `field` is 'raw' or 'pp'."""
    if not records:
        return 0.0, 0
    n_correct = sum(1 for r in records if r.get(f'error_type_{field}') == 'correct')
    return n_correct / len(records), len(records)


def slice_by(records: list[dict], key_fn, field: str = 'pp', min_n: int = 100) -> list[tuple]:
    """Return [(key, n, raw_acc, pp_acc), ...] sorted by descending n.

    `key_fn` extracts a slicing key from a record. Buckets with < min_n notes
    are dropped (statistics too noisy to interpret).
    """
    buckets = defaultdict(list)
    for r in records:
        k = key_fn(r)
        if k is None:
            continue
        buckets[k].append(r)
    rows = []
    for k, items in buckets.items():
        if len(items) < min_n:
            continue
        raw, _ = accuracy(items, 'raw')
        pp, n = accuracy(items, 'pp')
        rows.append((k, n, raw, pp))
    rows.sort(key=lambda x: -x[1])
    return rows


def print_table(title: str, rows: list[tuple], col0_name: str = 'bucket'):
    if not rows:
        print(f'\n{title}\n  (no buckets met min_n)')
        return
    print(f'\n{title}')
    print(f'  {col0_name:<24s} {"n":>10s} {"tab_raw":>9s} {"tab_pp":>9s}')
    for r in rows:
        key, n, raw, pp = r
        key_str = str(key)
        if len(key_str) > 24:
            key_str = key_str[:21] + '...'
        print(f'  {key_str:<24s} {n:>10,d} {raw:>8.1%}  {pp:>8.1%}')


def bucket_tempo(t):
    if t is None:
        return None
    if t < 80:
        return 'slow (<80)'
    if t < 110:
        return 'moderate (80-110)'
    if t < 140:
        return 'medium (110-140)'
    if t < 170:
        return 'fast (140-170)'
    return 'very_fast (170+)'


def bucket_density(d):
    if d is None:
        return None
    if d < 5:
        return '0-4 notes/2s'
    if d < 10:
        return '5-9 notes/2s'
    if d < 20:
        return '10-19 notes/2s'
    if d < 40:
        return '20-39 notes/2s'
    return '40+ notes/2s'


def bucket_fret(f):
    if f is None:
        return None
    if f == 0:
        return 'open (0)'
    if f <= 4:
        return 'low (1-4)'
    if f <= 9:
        return 'mid (5-9)'
    if f <= 14:
        return 'high (10-14)'
    return 'very_high (15+)'


# ---------------------------------------------------------------------------
# Per-piece outlier finder
# ---------------------------------------------------------------------------


def per_piece_drift(records: list[dict], min_notes: int = 20) -> list[dict]:
    """Per-piece drift signature, enriched with metadata for downstream slicing.

    Thin wrapper around `gtp.stage2.metrics.piece_drift_signature` that adds
    piece metadata (source, genre, gs_style, tempo, capo) so the
    analyze_errors tables can group by those fields.
    """
    by_piece: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_piece[r['piece_id']].append(r)

    out = []
    for pid, items in by_piece.items():
        if len(items) < min_notes:
            continue
        sig = piece_drift_signature(items)
        sample = items[0]
        out.append({
            'piece_id': pid,
            'source': sample.get('source'),
            'genre': sample.get('genre'),
            'gs_style': sample.get('gs_style'),
            'tempo': sample.get('tempo'),
            'capo': sample.get('capo'),
            **sig,
        })
    return out


def per_piece_stats(records: list[dict]) -> list[dict]:
    """Per-piece error rates + metadata. Useful to find quirky pieces."""
    by_piece: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_piece[r['piece_id']].append(r)
    out = []
    for pid, items in by_piece.items():
        raw, n = accuracy(items, 'raw')
        pp, _ = accuracy(items, 'pp')
        sample = items[0]
        out.append({
            'piece_id': pid,
            'source': sample.get('source'),
            'genre': sample.get('genre'),
            'gs_style': sample.get('gs_style'),
            'tempo': sample.get('tempo'),
            'capo': sample.get('capo'),
            'n': n,
            'raw_acc': raw,
            'pp_acc': pp,
            'pp_err_rate': 1 - pp,
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enriched', required=True, help='Path to enriched.jsonl')
    ap.add_argument('--output-dir', default=None,
                    help='Directory to write summary.json (default: alongside enriched.jsonl)')
    ap.add_argument('--min-bucket-n', type=int, default=100,
                    help='Minimum notes per bucket to include in slicing tables')
    args = ap.parse_args()

    enriched = Path(args.enriched)
    out_dir = Path(args.output_dir) if args.output_dir else enriched.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading {enriched}...')
    records = [json.loads(line) for line in enriched.open()]
    print(f'  {len(records):,} note records')

    # ---- Overall ----
    raw_acc, n = accuracy(records, 'raw')
    pp_acc, _ = accuracy(records, 'pp')
    print(f'\nOverall: n={n:,}  tab_raw={raw_acc:.3f}  tab_pp={pp_acc:.3f}')

    err_raw_dist = Counter(r['error_type_raw'] for r in records)
    err_pp_dist = Counter(r['error_type_pp'] for r in records)
    print('\nError-type distribution:')
    print(f'  {"type":<28s} {"raw":>10s} {"raw%":>7s} {"pp":>10s} {"pp%":>7s}')
    all_types = sorted(set(err_raw_dist) | set(err_pp_dist))
    for t in all_types:
        rc, pc = err_raw_dist.get(t, 0), err_pp_dist.get(t, 0)
        print(f'  {t:<28s} {rc:>10,d} {rc / n * 100:>6.2f}% '
              f'{pc:>10,d} {pc / n * 100:>6.2f}%')

    # Pitch accuracy (raw); pp has perfect pitch by construction
    n_pitch_correct_raw = sum(1 for r in records if r.get('pred_raw_pitch') == r['pitch'])
    n_pitch_correct_pp = sum(1 for r in records if r.get('pred_pp_pitch') == r['pitch'])
    print('\nPitch accuracy:')
    print(f'  raw: {n_pitch_correct_raw:,} / {n:,} = {n_pitch_correct_raw / n:.3f}')
    print(f'  pp:  {n_pitch_correct_pp:,} / {n:,} = {n_pitch_correct_pp / n:.3f} '
          '(should be ~1 by post-processing construction)')

    # ---- Original vs augmented ----
    print_table('Per is_augmented (False = original piece, True = capo-shifted variant):',
                slice_by(records, lambda r: r.get('is_augmented'), min_n=args.min_bucket_n),
                col0_name='is_augmented')

    # ---- Per-source ----
    print_table('Per source:', slice_by(records, lambda r: r['source'], min_n=args.min_bucket_n))

    # ---- Per genre (DadaGP only) ----
    dadagp_records = [r for r in records if r.get('source') == 'dadagp']
    print_table('Per coarse genre (DadaGP only):',
                slice_by(dadagp_records, lambda r: r.get('genre'), min_n=args.min_bucket_n),
                col0_name='genre')

    # ---- Per GuitarSet style ----
    gs_records = [r for r in records if r.get('source') == 'guitarset']
    print_table('Per GuitarSet style:',
                slice_by(gs_records, lambda r: r.get('gs_style'), min_n=args.min_bucket_n),
                col0_name='style')

    print_table('Per GuitarSet player:',
                slice_by(gs_records, lambda r: r.get('gs_player'), min_n=args.min_bucket_n),
                col0_name='player')

    # ---- Tempo / capo / density / polyphony / fret height ----
    print_table('Per tempo bucket:',
                slice_by(records, lambda r: bucket_tempo(r.get('tempo')),
                         min_n=args.min_bucket_n), col0_name='tempo')
    print_table('Per capo:',
                slice_by(records, lambda r: r.get('capo'), min_n=args.min_bucket_n),
                col0_name='capo')
    print_table('Per polyphony (chord size at onset):',
                slice_by(records, lambda r: r.get('polyphony'), min_n=args.min_bucket_n),
                col0_name='polyphony')
    print_table('Per density (notes/2s):',
                slice_by(records, lambda r: bucket_density(r.get('note_density_2s_window')),
                         min_n=args.min_bucket_n), col0_name='density')
    print_table('Per fret of true note:',
                slice_by(records, lambda r: bucket_fret(r.get('true_fret')),
                         min_n=args.min_bucket_n), col0_name='fret_bucket')
    print_table('Per absolute fret (true_fret + capo):',
                slice_by(records, lambda r: bucket_fret(r.get('absolute_fret')),
                         min_n=args.min_bucket_n), col0_name='abs_fret_bucket')
    print_table('Per true_string:',
                slice_by(records, lambda r: r.get('true_string'), min_n=args.min_bucket_n),
                col0_name='string')

    # ---- Local-context: relationship to previous note ----
    print_table('Per |Δstring from prev note|:',
                slice_by(records, lambda r: r.get('prev_string_dist'),
                         min_n=args.min_bucket_n), col0_name='prev_string_dist')
    print_table('Per |Δfret from prev note|:',
                slice_by(records, lambda r: bucket_fret(r.get('prev_fret_dist')),
                         min_n=args.min_bucket_n), col0_name='prev_fret_dist')

    # ---- Top (true -> pred) confusion patterns (errors only) ----
    error_recs = [r for r in records if r.get('error_type_pp') != 'correct']
    confusion = Counter(((r['true_string'], r['true_fret']),
                         (r.get('pred_pp_string'), r.get('pred_pp_fret')))
                        for r in error_recs)
    print(f'\nTop 20 (true -> pred) confusion patterns ({len(error_recs):,} pp errors):')
    print(f'  {"true(s,f)":>10s} {"->":>3s} {"pred(s,f)":>10s} {"count":>8s} {"pct of errors":>14s}')
    for (true_sf, pred_sf), c in confusion.most_common(20):
        pct = c / len(error_recs) * 100 if error_recs else 0
        print(f'  {true_sf!s:>10s} -> {pred_sf!s:>10s} {c:>8,d} {pct:>13.1f}%')

    # ---- Per-piece outliers ----
    piece_stats = per_piece_stats(records)
    piece_stats.sort(key=lambda r: -r['pp_err_rate'])
    print('\nTop 15 worst pieces by pp error rate (min n=20):')
    print(f'  {"err":>6s} {"n":>5s} {"src":>11s} {"genre/style":>16s} '
          f'{"tempo":>6s} {"capo":>4s}  piece_id')
    shown = 0
    for ps in piece_stats:
        if ps['n'] < 20:
            continue
        cat = ps.get('genre') or ps.get('gs_style') or '-'
        print(f'  {ps["pp_err_rate"]:>5.1%} {ps["n"]:>5d} {ps["source"] or "?":>11s} '
              f'{str(cat)[:16]:>16s} {str(ps["tempo"])[:6]:>6s} {ps["capo"]:>4} '
              f' {ps["piece_id"][:80]}')
        shown += 1
        if shown >= 15:
            break

    err_rates = [ps['pp_err_rate'] for ps in piece_stats if ps['n'] >= 20]
    if err_rates:
        print(f'\nPer-piece pp error-rate distribution (pieces with ≥20 notes, '
              f'n={len(err_rates)}):')
        print(f'  median={np.median(err_rates):.2%}  '
              f'p25={np.percentile(err_rates, 25):.2%}  '
              f'p75={np.percentile(err_rates, 75):.2%}  '
              f'p90={np.percentile(err_rates, 90):.2%}  '
              f'max={max(err_rates):.2%}')

    # ---- Per-piece drift signature ----
    drifts = per_piece_drift(records)
    bucket_order = ['perfect', 'consistent_alt', 'partial_alt', 'inconsistent']
    print(f'\nPer-piece drift signature (pieces with ≥20 notes, n={len(drifts)}):')
    print(f'  {"bucket":<18s} {"#pieces":>8s} {"%pieces":>8s} '
          f'{"#notes":>10s} {"%notes":>8s}')
    total_p = max(len(drifts), 1)
    total_n_drift_pieces = max(sum(d['n'] for d in drifts), 1)
    bucket_counts = {b: 0 for b in bucket_order}
    bucket_notes = {b: 0 for b in bucket_order}
    for d in drifts:
        bucket_counts[d['bucket']] += 1
        bucket_notes[d['bucket']] += d['n']
    for b in bucket_order:
        print(f'  {b:<18s} {bucket_counts[b]:>8d} '
              f'{100 * bucket_counts[b] / total_p:>7.1f}% '
              f'{bucket_notes[b]:>10,d} '
              f'{100 * bucket_notes[b] / total_n_drift_pieces:>7.1f}%')

    # ---- Tab-equivalent accuracy ----
    # A note is "equivalent-correct" if it's strictly correct OR if its drift
    # matches its piece's modal drift (i.e., the model committed to a single
    # consistent alternate fingering for the whole piece).
    piece_modal: dict[str, tuple[int, int] | None] = {
        d['piece_id']: d['modal_drift'] for d in drifts
    }
    n_strict = sum(1 for r in records if r['error_type_pp'] == 'correct')
    n_alt = 0
    for r in records:
        if r['error_type_pp'] == 'correct':
            continue
        md = piece_modal.get(r['piece_id'])
        if md is None:
            continue
        ds, df = r.get('delta_string_pp'), r.get('delta_fret_pp')
        if ds is None or df is None:
            continue
        if (ds, df) == md:
            n_alt += 1

    tab_equivalent = (n_strict + n_alt) / n if n else 0
    print('\nTab accuracy variants:')
    print(f'  tab_strict (exact (s,f) match):    {n_strict:>10,d} / {n:,} = {n_strict / n:.4f}')
    print(f'  tab_equivalent (+ piece-modal drift): '
          f'{n_strict + n_alt:>7,d} / {n:,} = {tab_equivalent:.4f}')
    print(f'  recovered by accepting consistent alt fingerings: '
          f'+{n_alt:,} notes ({100 * n_alt / n:+.2f}pp)')

    # ---- Drift-pattern histogram across consistent_alt pieces ----
    ca_drift_notes: Counter = Counter()
    ca_drift_pieces: Counter = Counter()
    for d in drifts:
        if d['bucket'] != 'consistent_alt' or d['modal_drift'] is None:
            continue
        ca_drift_notes[d['modal_drift']] += d['n_modal_drift']
        ca_drift_pieces[d['modal_drift']] += 1

    if ca_drift_pieces:
        print(f'\nDominant drifts in consistent_alt pieces '
              f'(n={bucket_counts["consistent_alt"]} pieces):')
        print(f'  {"(Δstring, Δfret)":>16s} {"# notes":>10s} {"# pieces":>9s}')
        for drift, notes in ca_drift_notes.most_common(15):
            print(f'  {str(drift):>16s} {notes:>10,d} {ca_drift_pieces[drift]:>9d}')

    # ---- Save summary JSON ----
    summary = {
        'overall': {'n': n, 'tab_raw': raw_acc, 'tab_pp': pp_acc,
                    'pitch_raw': n_pitch_correct_raw / n if n else 0,
                    'pitch_pp': n_pitch_correct_pp / n if n else 0,
                    'tab_strict': n_strict / n if n else 0,
                    'tab_equivalent': tab_equivalent,
                    'recovered_by_alt': n_alt},
        'error_type_distribution_raw': dict(err_raw_dist),
        'error_type_distribution_pp': dict(err_pp_dist),
        'top_confusions_pp': [
            {'true_sf': list(t), 'pred_sf': list(p), 'count': c}
            for (t, p), c in confusion.most_common(50)
        ],
        'piece_outliers': [ps for ps in piece_stats if ps['n'] >= 20][:30],
        'piece_drift_buckets': {
            b: {'n_pieces': bucket_counts[b], 'n_notes': bucket_notes[b]}
            for b in bucket_order
        },
        'consistent_alt_drift_histogram': [
            {'drift': list(drift), 'n_notes': n_notes,
             'n_pieces': ca_drift_pieces[drift]}
            for drift, n_notes in ca_drift_notes.most_common(50)
        ],
        'piece_drift_signatures': drifts,
    }
    out_path = out_dir / 'summary.json'
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nWrote summary to {out_path}')


if __name__ == '__main__':
    main()

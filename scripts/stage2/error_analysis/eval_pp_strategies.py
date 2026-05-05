"""Offline evaluation of post-processing fallback strategies.

Reads enriched.jsonl and replays the pp_source='fallback' events with
alternative tab-selection rules, keeping unchanged/window_swap fixed.
Reports tab accuracy on the fallback subset and overall, for each strategy.

We also run a bonus experiment: replace ALL pp predictions (including
unchanged-but-wrong) with a previous-note-anchored search. This previews
the position-tracking idea cheaply.

Usage:
  python scripts/stage2/error_analysis/eval_pp_strategies.py \\
      --enriched results/error_analysis/run_60k_orig/enriched.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


MAX_FRET = 24


# ---------------------------------------------------------------------------
# Candidate enumeration / search
# ---------------------------------------------------------------------------


def all_viable(pitch, tuning, max_fret=MAX_FRET):
    """All (string, fret) positions producing `pitch` on this tuning."""
    out = []
    for s_idx, open_pitch in enumerate(tuning):
        f = pitch - open_pitch
        if 0 <= f <= max_fret:
            out.append((s_idx + 1, f))
    return out


def first_viable(pitch, tuning, max_fret=MAX_FRET):
    """High-string-first search (mirrors src/gtp/stage2/postprocess.py)."""
    for s_idx, open_pitch in enumerate(tuning):
        f = pitch - open_pitch
        if 0 <= f <= max_fret:
            return (s_idx + 1, f)
    return None


def nearest_viable(
    pitch, anchor_string, anchor_fret, tuning,
    max_fret=MAX_FRET,
    max_string_dist=None,
    max_fret_dist=None,
    distance='chebyshev',
):
    """Closest viable to (anchor_string, anchor_fret); None if no candidate.

    distance: 'chebyshev' or 'manhattan'.
    Tie-break: smaller fret, then smaller string.
    """
    if anchor_string is None or anchor_fret is None:
        return None

    cands = all_viable(pitch, tuning, max_fret)
    if not cands:
        return None

    best = None
    best_key = None
    for s, f in cands:
        ds = abs(s - anchor_string)
        df = abs(f - anchor_fret)
        if max_string_dist is not None and ds > max_string_dist:
            continue
        if max_fret_dist is not None and df > max_fret_dist:
            continue
        d = max(ds, df) if distance == 'chebyshev' else ds + df
        key = (d, f, s)
        if best_key is None or key < best_key:
            best_key = key
            best = (s, f)
    return best


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
# Each strategy is a function (record) -> (string, fret) | None.
# It MUST emit a position whose pitch == record['pitch'].


def make_baseline_first_viable():
    def fn(r):
        return first_viable(r['pitch'], r['tuning'])
    return fn


def make_neighbor(max_string_dist, max_fret_dist, distance='chebyshev'):
    def fn(r):
        cand = nearest_viable(
            r['pitch'], r['pred_raw_string'], r['pred_raw_fret'], r['tuning'],
            max_string_dist=max_string_dist,
            max_fret_dist=max_fret_dist,
            distance=distance,
        )
        if cand is None:
            return first_viable(r['pitch'], r['tuning'])
        return cand
    return fn


def make_global_nearest(distance='chebyshev'):
    """No neighborhood cap — closest viable to raw output."""
    def fn(r):
        cand = nearest_viable(
            r['pitch'], r['pred_raw_string'], r['pred_raw_fret'], r['tuning'],
            distance=distance,
        )
        if cand is None:
            return first_viable(r['pitch'], r['tuning'])
        return cand
    return fn


def make_prev_anchored(distance='chebyshev'):
    """Anchor on previous note's (string, fret) instead of raw output.

    Previews the 'position-tracking' idea. For the very first note (no prev),
    falls back to first_viable.
    """
    def fn(r):
        ps, pf = r.get('prev_string'), r.get('prev_fret')
        if ps is None or pf is None:
            return first_viable(r['pitch'], r['tuning'])
        cand = nearest_viable(
            r['pitch'], ps, pf, r['tuning'], distance=distance,
        )
        if cand is None:
            return first_viable(r['pitch'], r['tuning'])
        return cand
    return fn


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_fallback_only(records, strategies):
    """For each strategy, replay pp_source='fallback' events, keep the rest fixed.

    Returns dict[strategy_name] -> (new_overall_correct, new_overall_total,
                                    fallback_correct, fallback_total).
    """
    fixed_correct = 0
    fixed_total = 0
    fallback_records = []
    for r in records:
        if r['pp_source'] == 'fallback':
            fallback_records.append(r)
        else:
            fixed_total += 1
            if r['error_type_pp'] == 'correct':
                fixed_correct += 1

    n_fb = len(fallback_records)
    results = {}
    for name, fn in strategies.items():
        fb_correct = 0
        for r in fallback_records:
            tab = fn(r)
            if tab is None:
                continue
            if tab == (r['true_string'], r['true_fret']):
                fb_correct += 1
        total = fixed_total + n_fb
        correct = fixed_correct + fb_correct
        results[name] = (correct, total, fb_correct, n_fb)
    return results


def evaluate_replace_all(records, strategy_fn):
    """Replay EVERY pp prediction with strategy_fn (ignoring window_swap path).

    This is the 'what if pp = strategy(raw or prev)?' upper bound check.
    """
    correct = 0
    for r in records:
        tab = strategy_fn(r)
        if tab is None:
            continue
        if tab == (r['true_string'], r['true_fret']):
            correct += 1
    return correct, len(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_fallback_report(records, results, baseline_name='baseline_first_viable'):
    bc, bt, bfc, bft = results[baseline_name]
    print(f'\n=== Fallback-path replay (n={bft:,} fallback events) ===\n')
    print(f'{"strategy":36s}  {"fallback acc":>13s}  {"Δ vs base":>10s}  {"overall pp":>11s}  {"Δ vs base":>10s}')
    print('-' * 90)
    base_overall = bc / bt
    base_fb = bfc / bft if bft else 0
    for name, (c, t, fc, ft) in results.items():
        fb_acc = fc / ft if ft else 0
        ov_acc = c / t
        marker = ' (baseline)' if name == baseline_name else ''
        print(f'{name + marker:36s}  {fb_acc * 100:11.2f}%  {(fb_acc - base_fb) * 100:+9.2f}pp  {ov_acc * 100:9.3f}%  {(ov_acc - base_overall) * 100:+9.3f}pp')


def print_replace_all_report(records, name, correct, total, baseline_overall):
    acc = correct / total
    print(f'  {name:40s}  {acc * 100:6.2f}%  ({correct:,} / {total:,})  '
          f'Δ vs base: {(acc - baseline_overall) * 100:+.2f}pp')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--enriched', type=Path, required=True)
    args = p.parse_args()

    print(f'Loading {args.enriched}...')
    records = []
    with open(args.enriched) as f:
        for line in f:
            records.append(json.loads(line))
    print(f'  {len(records):,} records')

    # Quick sanity: pp_source distribution
    src_counts = Counter(r['pp_source'] for r in records)
    print('\npp_source distribution:')
    for src, n in src_counts.most_common():
        print(f'  {src:14s}  {n:>10,}')

    # Build fallback-replay strategies.
    strategies = {
        'baseline_first_viable':         make_baseline_first_viable(),
        'A1_same_fret_pm1_string':       make_neighbor(1, 0),
        'A2_pm1_fret_pm1_string':        make_neighbor(1, 1),
        'A3_pm2_fret_pm1_string':        make_neighbor(1, 2),
        'A4_pm2_fret_pm2_string':        make_neighbor(2, 2),
        'B1_global_nearest_chebyshev':   make_global_nearest('chebyshev'),
        'B2_global_nearest_manhattan':   make_global_nearest('manhattan'),
    }

    results = evaluate_fallback_only(records, strategies)
    print_fallback_report(records, results)

    # Bonus: what if the prev-note-anchored search replaced ALL pp?
    print('\n=== Bonus: replace ALL pp with strategy(raw or prev) ===')
    overall_correct = sum(1 for r in records if r['error_type_pp'] == 'correct')
    base_acc = overall_correct / len(records)
    print(f'  current pp accuracy: {base_acc * 100:.3f}%')
    print()

    # Closest viable to RAW model output, applied everywhere.
    raw_anchor_correct, n = evaluate_replace_all(
        records, make_global_nearest('chebyshev'),
    )
    print_replace_all_report(records, 'closest-to-raw (chebyshev)',
                             raw_anchor_correct, n, base_acc)

    # Closest viable to PREV note position, applied everywhere.
    prev_anchor_correct, n = evaluate_replace_all(
        records, make_prev_anchored('chebyshev'),
    )
    print_replace_all_report(records, 'closest-to-prev-note (chebyshev)',
                             prev_anchor_correct, n, base_acc)


if __name__ == '__main__':
    main()

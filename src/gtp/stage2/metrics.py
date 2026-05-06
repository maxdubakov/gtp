"""All evaluation metrics for Stage 2.

Three layers of metrics:

  1. Paper-faithful per-pair atomic checks (Hamberger et al. §3.6):
       * `pitch_correct` / `pitch_accuracy`
       * `tab_correct` / `tab_accuracy`
       * `position_difficulty` / `difficulty_score`

  2. Per-piece drift signature: bucket each piece by whether the model's
     errors are dominated by a single consistent (Δstring, Δfret) shift.
     Used to distinguish "model picked a valid alternate fingering for the
     whole piece" from "model is genuinely confused".
       * `piece_drift_signature`
       * `aggregate_drift_buckets`

  3. Top-line summary across all per-note records:
       * `compute_eval_summary`  — tab_strict, tab_equivalent, pitch acc,
         error-type breakdown, drift-bucket counts. Used by `eval.py`,
         training-time eval, and `analyze_errors.py`.

Difficulty range per the paper: 0 (consecutive open strings on the same string)
to 18.5 (lowest open string → highest fret on highest string, 24-fret guitar).

NOTE on the difficulty formula: eq. (6) defines vertical_stretch as 0.25 whenever
Δstring ≤ 1, which includes Δstring = 0 (same string). That makes the per-pair
minimum 0.25, contradicting the paper's claim that the minimum is 0. We
implement the formula strictly as written; treat 0.25 as the practical floor.
"""

from collections import Counter, defaultdict
from itertools import pairwise


def pitch_of(tab, tuning) -> int | None:
    """Return MIDI pitch for (string, fret) on the given tuning, or None if invalid."""
    if tab is None:
        return None
    s, f = tab
    if 1 <= s <= len(tuning):
        return tuning[s - 1] + f
    return None


def pitch_correct(pred_tab, gt_pitch, tuning) -> bool:
    """True if pred_tab produces gt_pitch on the given tuning."""
    return pitch_of(pred_tab, tuning) == gt_pitch


def tab_correct(pred_tab, gt_tab) -> bool:
    """True if pred_tab matches gt_tab as (string, fret) tuples."""
    if pred_tab is None or gt_tab is None:
        return False
    return tuple(pred_tab) == tuple(gt_tab)


# ---------------------------------------------------------------------------
# Difficulty metric (Hamberger et al. §3.6, eqs. 1-6)
# ---------------------------------------------------------------------------


def _fret_stretch(prev_fret: int, curr_fret: int) -> float:
    """Eq. 4: horizontal movement penalty.

    Δfret = curr_fret - prev_fret.
    Moving up (Δ > 0) is easier — fret spacing is tighter higher on the neck —
    so it gets a smaller weight (0.50) than moving down (0.75).
    """
    delta = curr_fret - prev_fret
    if delta > 0:
        return 0.50 * abs(delta)
    return 0.75 * abs(delta)


def _locality(prev_fret: int, curr_fret: int) -> float:
    """Eq. 5: penalty for being high on the neck (strings are stiffer to press)."""
    return 0.25 * (prev_fret + curr_fret)


def _vertical_stretch(prev_string: int, curr_string: int) -> float:
    """Eq. 6: penalty for changing strings.

    Per the paper, 0.25 if Δstring ≤ 1 else 0.50 — interpreted on the absolute
    delta (the paper's notation is ambiguous; absolute makes physical sense).
    """
    if abs(curr_string - prev_string) <= 1:
        return 0.25
    return 0.50


def position_difficulty(prev_tab, curr_tab) -> float:
    """Eq. 1: total difficulty of moving prev_tab → curr_tab.

    difficulty = along + across
              = (fret_stretch + locality) + vertical_stretch.
    Both args are (string, fret); strings 1-indexed (1 = highest).
    """
    ps, pf = prev_tab
    cs, cf = curr_tab
    return _fret_stretch(pf, cf) + _locality(pf, cf) + _vertical_stretch(ps, cs)


def difficulty_score(tabs) -> float | None:
    """Mean position_difficulty across consecutive pairs in `tabs`.

    Skips entries that are None. Returns None if fewer than 2 valid tabs
    (difficulty is undefined for a 0- or 1-note sequence).
    """
    valid = [t for t in tabs if t is not None]
    if len(valid) < 2:
        return None
    diffs = [position_difficulty(p, q) for p, q in pairwise(valid)]
    return sum(diffs) / len(diffs)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def pitch_accuracy(pred_tabs, gt_pitches, tuning) -> tuple[int, int]:
    """Return (n_correct, n_total) over zip(pred_tabs, gt_pitches).

    Pred entries beyond the gt length are ignored. Gt entries beyond pred count
    as wrong (denominator = len(gt_pitches)).
    """
    n_total = len(gt_pitches)
    n_correct = 0
    for i, gp in enumerate(gt_pitches):
        if i < len(pred_tabs) and pitch_correct(pred_tabs[i], gp, tuning):
            n_correct += 1
    return n_correct, n_total


def tab_accuracy(pred_tabs, gt_tabs) -> tuple[int, int]:
    """Return (n_correct, n_total) over zip(pred_tabs, gt_tabs).

    Same denominator convention as pitch_accuracy.
    """
    n_total = len(gt_tabs)
    n_correct = 0
    for i, gt in enumerate(gt_tabs):
        if i < len(pred_tabs) and tab_correct(pred_tabs[i], gt):
            n_correct += 1
    return n_correct, n_total


# ---------------------------------------------------------------------------
# Per-note error categorization
# ---------------------------------------------------------------------------


ERROR_TYPES = (
    'correct',
    'no_prediction',
    'pitch_mismatch',
    'same_pitch_adj_string',
    'same_pitch_far_string',
)


def classify_error(
    true_s: int, true_f: int, true_pitch: int | None,
    pred_s: int | None, pred_f: int | None, pred_pitch: int | None,
) -> str:
    """Categorize a (predicted vs true) tab into one of `ERROR_TYPES`.

    Used by both the offline pipeline (`enrich_errors.py`) and the in-process
    eval (`eval.py`). Behavior:

      * pred is None              → 'no_prediction'
      * (pred_s, pred_f) == truth → 'correct'
      * pred_pitch != true_pitch  → 'pitch_mismatch'
      * |Δstring| == 1            → 'same_pitch_adj_string'
      * otherwise (same pitch, |Δstring| ≥ 2) → 'same_pitch_far_string'
    """
    if pred_s is None or pred_f is None:
        return 'no_prediction'
    if (pred_s, pred_f) == (true_s, true_f):
        return 'correct'
    if pred_pitch is not None and true_pitch is not None and pred_pitch != true_pitch:
        return 'pitch_mismatch'
    if abs(pred_s - true_s) == 1:
        return 'same_pitch_adj_string'
    return 'same_pitch_far_string'


# ---------------------------------------------------------------------------
# Per-piece drift signature
#
# Each per-note record (produced by enrich_errors / dump_eval_predictions / eval)
# is expected to carry at minimum:
#   piece_id, error_type_pp, delta_string_pp, delta_fret_pp,
#   pitch, pred_raw_pitch, pred_pp_pitch, error_type_raw.
# ---------------------------------------------------------------------------


DRIFT_BUCKETS = ('perfect', 'consistent_alt', 'partial_alt', 'inconsistent')


def piece_drift_signature(
    piece_records: list[dict],
    perfect_threshold: float = 0.95,
    consistent_threshold: float = 0.80,
    partial_threshold: float = 0.50,
) -> dict:
    """Drift signature for a single piece's per-note records.

    For all notes whose pp prediction differs from ground truth, take the
    (Δstring, Δfret) shift. The piece's "modal drift" is the most common
    non-zero shift among these error notes. error_consistency is the fraction
    of error notes that share that modal shift.

    Buckets:
      'perfect'        — correct_rate ≥ perfect_threshold (default 95%).
      'consistent_alt' — error_consistency ≥ consistent_threshold (default 80%).
      'partial_alt'    — partial_threshold ≤ error_consistency < consistent_threshold.
      'inconsistent'   — everything else (no dominant alt; genuinely confused).

    Returns dict with keys:
      n, n_correct, correct_rate, modal_drift, n_modal_drift, n_errors,
      error_consistency, bucket.
    """
    if not piece_records:
        return {
            'n': 0, 'n_correct': 0, 'correct_rate': 0.0,
            'modal_drift': None, 'n_modal_drift': 0, 'n_errors': 0,
            'error_consistency': 0.0, 'bucket': 'inconsistent',
        }

    n = len(piece_records)
    n_correct = sum(1 for r in piece_records if r.get('error_type_pp') == 'correct')
    correct_rate = n_correct / n

    drifts: Counter = Counter()
    for r in piece_records:
        ds, df = r.get('delta_string_pp'), r.get('delta_fret_pp')
        if ds is None or df is None or (ds, df) == (0, 0):
            continue
        drifts[(ds, df)] += 1

    n_errors = sum(drifts.values())
    if drifts:
        modal_drift, n_modal = drifts.most_common(1)[0]
        error_consistency = n_modal / n_errors
    else:
        modal_drift, n_modal, error_consistency = None, 0, 0.0

    if correct_rate >= perfect_threshold:
        bucket = 'perfect'
    elif error_consistency >= consistent_threshold:
        bucket = 'consistent_alt'
    elif error_consistency >= partial_threshold:
        bucket = 'partial_alt'
    else:
        bucket = 'inconsistent'

    return {
        'n': n,
        'n_correct': n_correct,
        'correct_rate': correct_rate,
        'modal_drift': modal_drift,
        'n_modal_drift': n_modal,
        'n_errors': n_errors,
        'error_consistency': error_consistency,
        'bucket': bucket,
    }


def aggregate_drift_buckets(
    records: list[dict],
    min_notes: int = 20,
    perfect_threshold: float = 0.95,
    consistent_threshold: float = 0.80,
    partial_threshold: float = 0.50,
) -> dict:
    """Per-piece drift signatures + aggregate bucket counts + tab_equivalent.

    `min_notes` skips tiny pieces where bucket assignments are too noisy.

    Returns dict with:
      n_pieces, n_notes (in qualifying pieces),
      bucket_counts: {bucket: int} — pieces per bucket.
      bucket_notes: {bucket: int} — notes per bucket.
      tab_strict_correct, tab_equivalent_correct (notes),
      tab_strict_acc, tab_equivalent_acc (rates over qualifying pieces),
      recovered_by_alt: notes added by accepting piece-modal drift,
      consistent_alt_drift_histogram: {(Δs, Δf): #pieces} restricted to consistent_alt,
      piece_drifts: list of per-piece drift signatures (with piece_id).
    """
    by_piece: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        pid = r.get('piece_id')
        if pid is None:
            continue
        by_piece[pid].append(r)

    piece_drifts = []
    for pid, items in by_piece.items():
        if len(items) < min_notes:
            continue
        sig = piece_drift_signature(
            items,
            perfect_threshold=perfect_threshold,
            consistent_threshold=consistent_threshold,
            partial_threshold=partial_threshold,
        )
        sig['piece_id'] = pid
        piece_drifts.append(sig)

    bucket_counts = {b: 0 for b in DRIFT_BUCKETS}
    bucket_notes = {b: 0 for b in DRIFT_BUCKETS}
    for d in piece_drifts:
        bucket_counts[d['bucket']] += 1
        bucket_notes[d['bucket']] += d['n']

    # Tab-equivalent: a note is "equivalent-correct" if it's strictly correct
    # OR if its drift matches its piece's modal drift (consistent alternate).
    piece_modal: dict[str, tuple[int, int] | None] = {
        d['piece_id']: d['modal_drift'] for d in piece_drifts
    }
    qualifying_pids = set(piece_modal)
    n_strict, n_alt, n_total = 0, 0, 0
    for r in records:
        pid = r.get('piece_id')
        if pid not in qualifying_pids:
            continue
        n_total += 1
        if r.get('error_type_pp') == 'correct':
            n_strict += 1
            continue
        md = piece_modal[pid]
        if md is None:
            continue
        ds, df = r.get('delta_string_pp'), r.get('delta_fret_pp')
        if ds is None or df is None:
            continue
        if (ds, df) == md:
            n_alt += 1

    consistent_alt_hist: Counter = Counter()
    for d in piece_drifts:
        if d['bucket'] == 'consistent_alt' and d['modal_drift'] is not None:
            consistent_alt_hist[d['modal_drift']] += 1

    return {
        'n_pieces': len(piece_drifts),
        'n_notes': sum(d['n'] for d in piece_drifts),
        'bucket_counts': bucket_counts,
        'bucket_notes': bucket_notes,
        'tab_strict_correct': n_strict,
        'tab_equivalent_correct': n_strict + n_alt,
        'tab_strict_acc': n_strict / n_total if n_total else 0.0,
        'tab_equivalent_acc': (n_strict + n_alt) / n_total if n_total else 0.0,
        'recovered_by_alt': n_alt,
        'consistent_alt_drift_histogram': {
            f'{ds},{df}': c for (ds, df), c in consistent_alt_hist.most_common()
        },
        'piece_drifts': piece_drifts,
    }


# ---------------------------------------------------------------------------
# Top-line summary across all per-note records
# ---------------------------------------------------------------------------


def compute_eval_summary(records: list[dict], min_notes_for_drift: int = 20) -> dict:
    """Top-line summary: all key Stage 2 metrics in one dict.

    Suitable for printing, JSON serialization, and downstream comparison
    across runs. Produces:

      n_notes, n_pieces,
      tab_strict_acc, tab_equivalent_acc (overall),
      pitch_raw_acc, pitch_pp_acc,
      error_type_raw / error_type_pp (count breakdowns),
      drift_buckets (piece counts), drift_buckets_notes (note counts),
      recovered_by_alt (notes recovered by piece-modal drift),
      consistent_alt_drift_histogram.

    Per-source / per-genre slicing is left to callers (eval.py / analyze_errors.py)
    since those slicing keys depend on which fields the caller has populated.
    """
    n = len(records)
    if n == 0:
        return {'n_notes': 0}

    n_pieces_unique = len({r.get('piece_id') for r in records if r.get('piece_id')})
    n_pitch_raw = sum(1 for r in records if r.get('pred_raw_pitch') == r.get('pitch'))
    n_pitch_pp = sum(1 for r in records if r.get('pred_pp_pitch') == r.get('pitch'))
    err_raw = Counter(r.get('error_type_raw') for r in records)
    err_pp = Counter(r.get('error_type_pp') for r in records)

    drift = aggregate_drift_buckets(records, min_notes=min_notes_for_drift)

    return {
        'n_notes': n,
        'n_pieces_total': n_pieces_unique,
        'n_pieces_qualified': drift['n_pieces'],
        'tab_strict_acc': drift['tab_strict_acc'],
        'tab_equivalent_acc': drift['tab_equivalent_acc'],
        'pitch_raw_acc': n_pitch_raw / n,
        'pitch_pp_acc': n_pitch_pp / n,
        'error_type_raw': dict(err_raw),
        'error_type_pp': dict(err_pp),
        'drift_buckets': drift['bucket_counts'],
        'drift_buckets_notes': drift['bucket_notes'],
        'recovered_by_alt': drift['recovered_by_alt'],
        'consistent_alt_drift_histogram': drift['consistent_alt_drift_histogram'],
    }

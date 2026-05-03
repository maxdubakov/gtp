"""Paper-faithful evaluation metrics for Stage 2.

Implements three metrics from Hamberger et al. §3.6:
  - pitch_correct / pitch_accuracy  — does the (string, fret) produce the right pitch?
  - tab_correct   / tab_accuracy    — does it match the ground-truth (string, fret)?
  - position_difficulty / difficulty_score — playability proxy from a modified
    version of the difficulty estimation framework (Heijink & Meulenbroeks, 2002),
    quantifying horizontal + vertical hand movement between consecutive positions.

Difficulty range per the paper: 0 (consecutive open strings on the same string)
to 18.5 (lowest open string → highest fret on highest string, 24-fret guitar).

NOTE on the difficulty formula: eq. (6) defines vertical_stretch as 0.25 whenever
Δstring ≤ 1, which includes Δstring = 0 (same string). That makes the per-pair
minimum 0.25, contradicting the paper's claim that the minimum is 0. We
implement the formula strictly as written; treat 0.25 as the practical floor.
"""

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

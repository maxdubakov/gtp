"""Post-processing for Stage 2 model output.

Implements the paper's correction algorithm (Hamberger et al., §3.5): for each
input note, find the closest predicted tab whose pitch matches within a ±window
neighborhood; fall back to a pitch-producing (string, fret) otherwise.
Guarantees that every input pitch is preserved in the output.

Two fallback strategies are available, selected by `correct_tabs(..., fallback=...)`:
  * 'first_viable'   — paper-faithful: high-string-first lowest-fret realization.
  * 'nearest_viable' — deviation from paper: Manhattan-nearest realization to the
                       model's raw output at this position. Motivated by error
                       analysis (raw pitch mismatches cluster at ±5/±7 semitones,
                       i.e. adjacent-string offsets at the same hand position).
                       See scripts/stage2/error_analysis/eval_pp_strategies.py.
"""

from gtp.stage2.data import MAX_FRET
from gtp.stage2.metrics import pitch_of


def first_viable_tab(pitch, tuning, max_fret=MAX_FRET):
    """Return the first (string, fret) on the given tuning that produces `pitch`.

    `string` is 1-indexed (string 1 = first entry of `tuning` = high E by our convention).
    Iterates strings from highest to lowest (i.e. index 0 → 5), returning the first
    string where (pitch - open_pitch) is a valid fret in [0, max_fret].
    Returns None if no playable position exists.
    """
    for s_idx, open_pitch in enumerate(tuning):
        fret = pitch - open_pitch
        if 0 <= fret <= max_fret:
            return (s_idx + 1, fret)
    return None


def nearest_viable_tab(pitch, predicted_tab, tuning, max_fret=MAX_FRET):
    """Manhattan-nearest playable (string, fret) to `predicted_tab` producing `pitch`.

    Deviation from the paper's `first_viable_tab`. Anchors the fallback on the
    model's own prediction at this position rather than a fixed canonical
    realization. Among all (string, fret) producing `pitch`, returns the one
    with minimum (|Δstring| + |Δfret|) from `predicted_tab`. Ties broken by
    smaller fret, then smaller string.

    Falls back to `first_viable_tab` if `predicted_tab` is None or contains a
    string outside [1, len(tuning)].

    Background: error analysis showed raw pitch mismatches cluster at ±5 and
    ±7 semitones — adjacent-string offsets at roughly the same hand position.
    Anchoring on the model's raw output recovers a meaningful fraction of
    those cases. Offline ablation: see eval_pp_strategies.py.
    """
    if predicted_tab is None:
        return first_viable_tab(pitch, tuning, max_fret)
    s_anchor, f_anchor = predicted_tab
    if s_anchor is None or f_anchor is None or not (1 <= s_anchor <= len(tuning)):
        return first_viable_tab(pitch, tuning, max_fret)

    best = None
    best_key = None
    for s_idx, open_pitch in enumerate(tuning):
        f = pitch - open_pitch
        if 0 <= f <= max_fret:
            s = s_idx + 1
            d = abs(s - s_anchor) + abs(f - f_anchor)
            key = (d, f, s)
            if best_key is None or key < best_key:
                best_key = key
                best = (s, f)
    return best


def correct_tabs(
    input_pitches, predicted_tabs, tuning,
    window=5, max_fret=MAX_FRET, return_sources=False,
    fallback='first_viable',
):
    """Paper's pitch-preserving post-processing (with optional fallback variant).

    For each input note at position i with pitch p:
      1. Search predicted positions in [i-window, i+window] (clipped to bounds).
      2. Among those whose pitch equals p, pick the position closest to i
         (ties broken by smallest index).
      3. If no position matches: emit `<fallback_fn>(p, ...)`.

    Args:
        input_pitches: list of N target pitches (from the encoder input).
        predicted_tabs: list of M (string, fret) tuples from the model.
            M may differ from N.
        tuning: open-string pitches (length 6 typically). By our convention,
            already includes capo (pitch = tuning[string-1] + fret).
        window: ±W index neighborhood to search. Default 5 matches the paper.
        max_fret: relative-fret upper bound for the fallback search.
        return_sources: if True, also return a parallel list of source tags
            (one per output entry):
              'unchanged'   — model's tab at position i already had the right pitch
              'window_swap' — replaced with a different model tab from within ±window
              'fallback'    — no model tab in window matched; used the fallback function
        fallback: which fallback to use when no model tab in window matches.
              'first_viable'   — paper-faithful (default).
              'nearest_viable' — Manhattan-nearest to predicted_tabs[i] (deviation from paper).

    Returns:
        list of N (string, fret) — one per input note. Every entry produces
        the corresponding `input_pitches[i]` exactly (assuming the pitch is
        playable on the given tuning + max_fret; otherwise None for that slot).
        If `return_sources=True`, returns `(corrected, sources)` instead.
    """
    if fallback not in ('first_viable', 'nearest_viable'):
        raise ValueError(f"fallback must be 'first_viable' or 'nearest_viable', got {fallback!r}")

    m = len(predicted_tabs)
    predicted_pitches = [pitch_of(tab, tuning) for tab in predicted_tabs]

    corrected = []
    sources = []
    for i, p_in in enumerate(input_pitches):
        lo = max(0, i - window)
        hi = min(m, i + window + 1)

        best_j = -1
        best_dist = window + 1
        for j in range(lo, hi):
            if predicted_pitches[j] == p_in:
                d = abs(j - i)
                if d < best_dist:
                    best_dist = d
                    best_j = j

        if best_j >= 0:
            corrected.append(predicted_tabs[best_j])
            sources.append('unchanged' if best_dist == 0 else 'window_swap')
        else:
            anchor = predicted_tabs[i] if i < m else None
            if fallback == 'nearest_viable':
                tab = nearest_viable_tab(p_in, anchor, tuning, max_fret=max_fret)
            else:
                tab = first_viable_tab(p_in, tuning, max_fret=max_fret)
            corrected.append(tab)
            sources.append('fallback')

    if return_sources:
        return corrected, sources
    return corrected

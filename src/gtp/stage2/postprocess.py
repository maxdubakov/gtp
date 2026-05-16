"""Post-processing (tab correction) for Stage 2 model output"""

from gtp.stage2.data import MAX_FRET
from gtp.stage2.metrics import pitch_of


def first_viable_tab(pitch, _predicted_tab, tuning, max_fret=MAX_FRET):
    """Implements the paper's correction algorithm (Hamberger et al., §3.5)"""
    for s_idx, open_pitch in enumerate(tuning):
        fret = pitch - open_pitch
        if 0 <= fret <= max_fret:
            return s_idx + 1, fret
    return None


def nearest_viable_tab(pitch, predicted_tab, tuning, max_fret=MAX_FRET):
    """Manhattan-nearest playable (string, fret) to `predicted_tab` producing `pitch`.
    Background: error analysis showed raw pitch mismatches cluster at +-5 and +-7 semitones.
    """
    if predicted_tab is None:
        return first_viable_tab(pitch, None, tuning, max_fret)
    s_anchor, f_anchor = predicted_tab
    if s_anchor is None or f_anchor is None or not (1 <= s_anchor <= len(tuning)):
        return first_viable_tab(pitch, predicted_tab, tuning, max_fret)

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


FALLBACK_STRATEGIES = {
    'first_viable': first_viable_tab,
    'nearest_viable': nearest_viable_tab,
}


def correct_tabs(
    input_pitches,
    predicted_tabs,
    tuning,
    window=5,
    max_fret=MAX_FRET,
    return_sources=False,
    fallback='first_viable',
):
    """Preserves pitches (with optional fallback).

    For each input note at position i with pitch p:
      1. Search predicted positions in [i-window, i+window]
      2. Among those whose pitch equals p, pick the position closest to i
      3. If no position matches, use selected fallback strategy

    Args:
        input_pitches: list of N target pitches (from the encoder input)
        predicted_tabs: list of M (string, fret) tuples from the model
        tuning: open-string pitches
        window: window size. Default 5 matches the paper
        max_fret: relative-fret upper bound for the fallback search.
        return_sources: if True, also return a list of fixes for each note/tab
        fallback: which fallback to use when no model tab in window matches

    Returns:
        list of N (string, fret) — one per input note + sources if selected
    """
    if fallback not in FALLBACK_STRATEGIES:
        raise ValueError(f'Unknown fallback strategy: {fallback!r}. Available: {list(FALLBACK_STRATEGIES)}')

    m = len(predicted_tabs)
    predicted_pitches = [pitch_of(tab, tuning) for tab in predicted_tabs]

    corrected = []
    sources = []
    for i, p_in in enumerate(input_pitches):
        if i < m and predicted_pitches[i] == p_in:
            corrected.append(predicted_tabs[i])
            sources.append('unchanged')
            continue

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
            sources.append('window_swap')
        else:
            anchor = predicted_tabs[i] if i < m else None
            tab = FALLBACK_STRATEGIES[fallback](p_in, anchor, tuning, max_fret=max_fret)
            corrected.append(tab)
            sources.append('fallback')

    if return_sources:
        return corrected, sources
    return corrected

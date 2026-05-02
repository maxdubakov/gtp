"""Post-processing for Stage 2 model output.

Implements the paper's correction algorithm (Hamberger et al., §3.5): for each
input note, find the closest predicted tab whose pitch matches within a ±window
neighborhood; fall back to the first viable (string, fret) producing that pitch
otherwise. Guarantees that every input pitch is preserved in the output.
"""

from gtp.stage2.data import MAX_FRET


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


def correct_tabs(input_pitches, predicted_tabs, tuning, window=5, max_fret=MAX_FRET):
    """Paper's pitch-preserving post-processing.

    For each input note at position i with pitch p:
      1. Search predicted positions in [i-window, i+window] (clipped to bounds).
      2. Among those whose pitch equals p, pick the position closest to i
         (ties broken by smallest index).
      3. If no position matches: emit `first_viable_tab(p, tuning)`.

    Args:
        input_pitches: list of N target pitches (from the encoder input).
        predicted_tabs: list of M (string, fret) tuples from the model.
            M may differ from N.
        tuning: open-string pitches (length 6 typically). By our convention,
            already includes capo (pitch = tuning[string-1] + fret).
        window: ±W index neighborhood to search. Default 5 matches the paper.
        max_fret: relative-fret upper bound for the fallback search.

    Returns:
        list of N (string, fret) — one per input note. Every entry produces
        the corresponding `input_pitches[i]` exactly (assuming the pitch is
        playable on the given tuning + max_fret; otherwise None for that slot).
    """
    m = len(predicted_tabs)
    predicted_pitches = [
        (tuning[s - 1] + f) if 1 <= s <= len(tuning) else None
        for s, f in predicted_tabs
    ]

    corrected = []
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
        else:
            corrected.append(first_viable_tab(p_in, tuning, max_fret=max_fret))

    return corrected

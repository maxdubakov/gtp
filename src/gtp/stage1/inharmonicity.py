"""String evidence from inharmonicity, self-calibrated per guitar.

A string's partials sit at f_k = k f0 sqrt(1 + B k^2). B grows as the vibrating length shortens: fretting f frets up
multiplies it by 2^(f/6), so log10(B) - f * log10(2)/6 is a constant of the open string. That constant depends on the
string's gauge and construction (a wound G sits far below a plain one), so it differs between guitars but is shared by
every note of one guitar. Each candidate string of a note implies a value of its string's constant; the string whose
implied value matches what the guitar's other notes say is the likely one.

`calibrate` finds the six constants without labels, by EM started from a classifier's string probabilities, and returns
those probabilities fused with the inharmonicity likelihood.
"""

import numpy as np

from gtp.stage1.strings import OPEN_STRINGS

LOG_B_PER_FRET = np.log10(2) / 6
MIN_SCALE = 0.05
# Constants and scale assumed for a string with too few notes to estimate them.
DEFAULT_CONSTANT, DEFAULT_SCALE, MIN_WEIGHT = -4.0, 0.3, 3.0


def implied_constants(log_b, pitch):
    """(N, 6) open-string constant each note's measured log10(B) implies if played on each string."""
    return np.asarray(log_b)[:, None] - (np.asarray(pitch)[:, None] - OPEN_STRINGS[None]) * LOG_B_PER_FRET


def weighted_median(x, w):
    order = np.argsort(x)
    cum = np.cumsum(w[order])
    return x[order][np.searchsorted(cum, cum[-1] / 2)]


def fit_strings(implied, resp):
    """Per-string constant (weighted median) and Laplace scale from notes weighted by their string responsibility."""
    const, scale = np.full(6, DEFAULT_CONSTANT), np.full(6, DEFAULT_SCALE)
    for s in range(6):
        w = resp[:, s]
        if w.sum() >= MIN_WEIGHT:
            const[s] = weighted_median(implied[:, s], w)
            scale[s] = max(MIN_SCALE, np.sum(w * np.abs(implied[:, s] - const[s])) / w.sum())
    return const, scale


def fuse(probs, implied, valid, const, scale, weight=1.0):
    """`probs` times the Laplace likelihood of each string's implied constant; notes without a valid B keep `probs`."""
    ll = -np.abs(implied - const) / scale - np.log(scale)
    logp = np.log(probs + 1e-9) + weight * np.where(valid[:, None], ll, 0.0)
    logp[probs <= 0] = -np.inf
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def calibrate(probs, implied, valid, iters=15, weight=1.0):
    """Label-free EM over one guitar's notes; returns (fused probabilities, string constants, scales)."""
    post = probs
    for _ in range(iters):
        const, scale = fit_strings(implied[valid], post[valid])
        post = fuse(probs, implied, valid, const, scale, weight)
    return post, const, scale


# Likelihood profiles (bprofile.py) instead of point estimates: each note carries log p(spectrum | B) on a grid.

CONSTANT_GRID = np.arange(-5.3, -3.2, 0.02)
# Weak prior on a string's constant, so a string without notes keeps a plausible value.
PRIOR_MEAN, PRIOR_SD = -4.1, 0.6


def string_tables(profile, log_b, pitch, spread):
    """(N, 6, len(CONSTANT_GRID)) log likelihood of each note on each string, for every value of that string's constant.

    `profile` (N, G) is the tempered log likelihood over the log10(B) grid. A string with constant c predicts
    log10(B) = c + fret * LOG_B_PER_FRET, blurred by a Laplace of scale `spread` (the within-guitar scatter).
    """
    kernel = -np.abs(log_b[None] - log_b[:, None]) / spread
    kernel -= np.log(np.exp(kernel).sum(axis=1, keepdims=True))
    peak = profile.max(axis=1, keepdims=True)
    smooth = np.log(np.exp(profile - peak) @ np.exp(kernel).T + 1e-300) + peak
    fret = np.asarray(pitch)[:, None] - OPEN_STRINGS[None]
    u = CONSTANT_GRID[None, None] + fret[:, :, None] * LOG_B_PER_FRET
    pos = np.clip((u - log_b[0]) / (log_b[1] - log_b[0]), 0, len(log_b) - 1.001)
    lo = pos.astype(int)
    frac = pos - lo
    rows = np.arange(len(profile))[:, None, None]
    return (1 - frac) * smooth[rows, lo] + frac * smooth[rows, lo + 1]


def fit_constants(tables, resp):
    """Index into CONSTANT_GRID of each string's most likely constant under the notes' string responsibilities."""
    objective = np.einsum('is,isc->sc', resp, tables) - 0.5 * ((CONSTANT_GRID - PRIOR_MEAN) / PRIOR_SD) ** 2
    return objective.argmax(axis=1)


def fuse_tables(probs, tables, const_idx):
    logp = np.log(probs + 1e-9) + tables[:, np.arange(6), const_idx]
    logp[probs <= 0] = -np.inf
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def calibrate_profiles(probs, tables, iters=15):
    """Label-free EM over one guitar's notes with profile likelihoods; returns (fused probabilities, constants)."""
    post = probs
    for _ in range(iters):
        const_idx = fit_constants(tables, post)
        post = fuse_tables(probs, tables, const_idx)
    return post, CONSTANT_GRID[const_idx]

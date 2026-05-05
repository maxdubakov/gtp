"""Inharmonicity-coefficient (β) estimator — port of Hjerrild WASPAA 2019.

Joint maximum-likelihood estimation of (f₀, β) via inharmonic summation.

Algorithm (per `recreate_plucking_experiment_WASPAA19.m` and the ICASSP 2019 utils):
  1) Take a short segment of audio (default 40 ms after the note onset).
  2) Apply a Gaussian window (a=2.5).
  3) Hilbert-transform → analytic signal (single-sided spectrum without
     wrap-around).
  4) Zero-padded FFT (default 2¹⁹ points → ~0.084 Hz/bin at 44.1 kHz).
  5) Coarse-to-fine harmonic summation gives an initial f₀ estimate (β=0,
     M_initial=5 partials).
  6) Inharmonic summation: 2D grid search over (f₀ around initial estimate,
     β over a plausible guitar range). The (f₀, β) pair maximizing the sum
     of spectral amplitudes at predicted partial positions
     `φₖ = k · f₀ · √(1 + β·k²)` for k=1..M=25 wins.

This file is a faithful port of the MATLAB reference implementation. We skip
the Bayesian (string, fret) classifier — we only need the β/f₀ extractor.

Reference values for validation: see `results/physical_models/beta-estimations.txt`
(36 segments from `recording_of_plucking_with_sudden_changes.wav`, 6 (string, fret)
positions, 6-7 segments each). Use `verify_matlab_match.py` to compare.
"""

import numpy as np
import scipy.signal

# ---------------------------------------------------------------------------
# Pre-processing primitives
# ---------------------------------------------------------------------------


def apply_gaussian_window(x: np.ndarray, a: float = 2.5) -> np.ndarray:
    """Mirror of icassp19_apply_gaussian_window. `a` controls window width
    (inversely; a=2.5 gives ~4% amplitude at the edges)."""
    n_win = len(x) - 1
    n = np.arange(n_win + 1) - n_win / 2
    w = np.exp(-0.5 * (a * n / (n_win / 2)) ** 2)
    return x * w


def hilbert_transform(sig: np.ndarray) -> np.ndarray:
    """Analytic signal via standard Hilbert transform.

    Mirror of icassp19_hilbert_transform (which manually constructs the
    multiplier h = [1, 2, ..., 2, 1, 0, ..., 0]). scipy.signal.hilbert does
    the same. If input is odd-length, MATLAB drops the last sample to make
    length even — we match that for exact reproducibility.
    """
    if len(sig) % 2 == 1:
        sig = sig[:-1]
    return scipy.signal.hilbert(sig)


def compute_fft(sig: np.ndarray, fs: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of icassp19_fft: returns (f_axis, X) where X is the
    single-sided power spectrum scaled by 2 (length n_fft/2)."""
    fft_long = np.fft.fft(sig, n_fft)
    X = 2 * np.abs(fft_long[: n_fft // 2]) ** 2
    f_axis = fs / 2 * np.linspace(0, 1, n_fft // 2)
    return f_axis, X


# ---------------------------------------------------------------------------
# Partial-position model + index lookup
# ---------------------------------------------------------------------------


def piano_model(f0: float, beta: float, M: int) -> np.ndarray:
    """Predicted partial frequencies under the stiff-string dispersion law:
    fₖ = k · f₀ · √(1 + β·k²) for k=1..M.

    Mirror of icassp19_piano_model (which special-cases k=1 as f₀, but the
    general formula gives the same result there since √(1+β·1²) ≈ 1 when
    β is small — we just write it uniformly).
    """
    k = np.arange(1, M + 1)
    return f0 * k * np.sqrt(1 + beta * k**2)


def inharmonic_index(spectrum_len: int, fs: int, M: int, f0: float, beta: float, n_fft: int) -> np.ndarray:
    """FFT bin indices of the first M predicted partials at this (f₀, β).

    Bin k of an n_fft-point FFT corresponds to frequency `k · fs / n_fft`,
    so frequency f maps to bin `round(f · n_fft / fs)`. MATLAB used 1-indexing
    and `round(f · 2N/fs + 1)` which is the same modulo the off-by-one.

    Indices ≥ spectrum_len are dropped (they fell off the single-sided
    spectrum's Nyquist edge).
    """
    phi = piano_model(f0, beta, M)
    indices = np.round(phi * n_fft / fs).astype(int)
    return indices[indices < spectrum_len]


# ---------------------------------------------------------------------------
# Pitch + (f₀, β) estimators
# ---------------------------------------------------------------------------


def harmonic_summation(X: np.ndarray, f0_limits: tuple[float, float], M: int, fs: int, n_fft: int) -> float:
    """Three-stage coarse-to-fine f₀ estimation, β=0.

    Mirror of icassp19_harmonic_summation. Grids:
      stage 1: 0.1 Hz between f0_limits[0] and f0_limits[1]
      stage 2: 0.01 Hz over ±4 Hz around stage-1 winner
      stage 3: 0.001 Hz over ±0.01 Hz around stage-2 winner
    """
    spec_len = len(X)

    def cost_at(f0):
        idx = inharmonic_index(spec_len, fs, M, f0, 0.0, n_fft)
        return X[idx].sum()

    grid1 = np.arange(f0_limits[0], f0_limits[1] + 1e-9, 0.1)
    pitch1 = grid1[np.argmax(np.array([cost_at(f) for f in grid1]))]

    grid2 = np.arange(pitch1 - 4, pitch1 + 4 + 1e-9, 0.01)
    pitch2 = grid2[np.argmax(np.array([cost_at(f) for f in grid2]))]

    grid3 = np.arange(pitch2 - 0.01, pitch2 + 0.01 + 1e-9, 0.001)
    pitch3 = grid3[np.argmax(np.array([cost_at(f) for f in grid3]))]

    return float(pitch3)


def inharmonic_summation(X: np.ndarray, pitch_initial: float, M: int, fs: int,
                         beta_grid: np.ndarray, n_fft: int) -> tuple[float, float, float]:
    """Joint (f₀, β) MLE via 2D grid search.

    Mirror of icassp19_inharmonic_summation. The pitch grid is centered on
    `pitch_initial` with half-width = 6·fs/2¹⁸ ≈ 1 Hz at fs=44.1k (covers
    the Gaussian window's mainlobe). Step is fs/n_fft = the FFT bin width.

    Returns (f0_est, beta_est, cost_max).
    """
    spec_len = len(X)
    pitch_width = 6 * fs / (2**18)  # ~1.01 Hz at fs=44.1k, independent of n_fft
    bin_spacing = fs / n_fft

    pitch_grid = np.arange(pitch_initial - pitch_width,
                           pitch_initial + pitch_width + bin_spacing / 2,
                           bin_spacing)

    cost = np.zeros((len(beta_grid), len(pitch_grid)))
    for i, beta in enumerate(beta_grid):
        for j, f0 in enumerate(pitch_grid):
            idx = inharmonic_index(spec_len, fs, M, f0, beta, n_fft)
            cost[i, j] = X[idx].sum()

    i_best, j_best = np.unravel_index(int(cost.argmax()), cost.shape)
    return float(pitch_grid[j_best]), float(beta_grid[i_best]), float(cost[i_best, j_best])


# ---------------------------------------------------------------------------
# Top-level estimator (40 ms segment in, β out)
# ---------------------------------------------------------------------------


def estimate_inharmonicity_from_segment(
    segment: np.ndarray, fs: int,
    f0_limits: tuple[float, float] = (75.0, 700.0),
    M: int = 25, M_initial: int = 5,
    beta_min: float = 1e-5, beta_max: float = 5e-4, beta_res: float = 1e-5,
    n_fft: int = 2**19,
) -> tuple[float, float, float]:
    """Run the full pipeline on one segment of audio. Returns (β, f₀, cost).

    Defaults match MATLAB's `recreate_plucking_experiment_WASPAA19.m`.
    `segment` should be ~40 ms of audio post-onset.
    """
    if len(segment) < 32:
        return float('nan'), float('nan'), float('nan')

    x = apply_gaussian_window(segment)
    x = hilbert_transform(x)
    _, X = compute_fft(x, fs, n_fft)

    f0_initial = harmonic_summation(X, f0_limits, M_initial, fs, n_fft)

    beta_grid = np.arange(beta_min, beta_max + beta_res / 2, beta_res)
    f0_est, beta_est, cost = inharmonic_summation(X, f0_initial, M, fs, beta_grid, n_fft)

    return beta_est, f0_est, cost


def estimate_inharmonicity_at_onset(
    audio: np.ndarray, fs: int, onset_seconds: float,
    segment_dur_s: float = 0.04,
    prepend_zeros: int = 1000,
    **kwargs,
) -> tuple[float, float, float]:
    """Estimate β at a specific onset time. Mirror of MATLAB's segment
    extraction in `icassp19_segment_from_all_onsets.m`.

    The MATLAB pipeline prepends 1000 zeros to the audio before computing
    onsets, so its onset times are in the prepended-signal time-frame. We
    match that by default — pass `prepend_zeros=0` if your onsets are in
    the original audio's time-frame.

    `kwargs` are forwarded to `estimate_inharmonicity_from_segment`.
    """
    padded = np.concatenate([np.zeros(prepend_zeros), audio]) if prepend_zeros else audio
    onset_sample = int(np.floor(onset_seconds * fs))
    seg_len = int(np.floor(segment_dur_s * fs)) + 2  # match MATLAB length exactly
    end = onset_sample + seg_len
    if end > len(padded):
        return float('nan'), float('nan'), float('nan')
    segment = padded[onset_sample:end]
    return estimate_inharmonicity_from_segment(segment, fs, **kwargs)


# ---------------------------------------------------------------------------
# Synthetic sanity check
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    """Quick synthetic sanity check: synthesize a signal with known (f₀, β)
    and verify the estimator recovers them within tolerance."""
    print('Synthetic β estimation sanity check (Hjerrild port)\n')
    print(f'{"f0":>6} {"β_true":>10} {"β_est":>10} {"f0_est":>8} {"rel_err%":>10}')

    rng = np.random.default_rng(0)
    fs = 44100
    duration = 0.04
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    cases = [
        (110.0, 5e-5),    # low E open-ish, small β
        (110.0, 5e-4),    # low E with high β (extreme)
        (220.0, 1e-4),    # A3
        (440.0, 5e-5),    # A4 low β
    ]
    for f0_true, beta_true in cases:
        sig = np.zeros_like(t)
        for k in range(1, 26):
            fk = k * f0_true * np.sqrt(1 + beta_true * k**2)
            if fk >= fs / 2:
                break
            sig += (1.0 / k) * np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))
        sig += 0.001 * rng.standard_normal(len(sig))

        beta_est, f0_est, _ = estimate_inharmonicity_from_segment(
            sig, fs, f0_limits=(max(75, f0_true - 30), min(700, f0_true + 30))
        )
        rel_err = abs(beta_est - beta_true) / beta_true * 100
        print(f'{f0_true:>6.0f} {beta_true:>10.5f} {beta_est:>10.5f} {f0_est:>8.2f} {rel_err:>10.2f}')

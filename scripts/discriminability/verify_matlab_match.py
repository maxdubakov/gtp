"""Validate the Python β port against MATLAB's exact output.

Two modes:
  --grids matlab  (default): load MATLAB-dumped per-class β bounds from
                  results/physical_models/class-bounds.txt — eliminates
                  RNG mismatch between our Monte Carlo and MATLAB's, gives
                  bit-exact comparison of the inharmonic-summation step.
  --grids python: run our own Monte Carlo simulation (physical_model.py) —
                  validates the simulation port itself.

Pipeline per segment (matches MATLAB's recreate_plucking_experiment_WASPAA19.m):
  1) Initial f₀ via harmonic summation
  2) Pitch-candidate selection via equal-tempered-scale lookup
  3) For each candidate (s, f), run inharmonic_summation with that class's
     β grid (= [min, max] for that class, step 1e-5)
  4) Pick the candidate with the highest cost

Usage:
  ./venv/bin/python scripts/discriminability/verify_matlab_match.py
  ./venv/bin/python scripts/discriminability/verify_matlab_match.py --grids python
"""

import argparse
import contextlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from inharmonicity import (
    apply_gaussian_window,
    compute_fft,
    harmonic_summation,
    hilbert_transform,
    inharmonic_summation,
)
from physical_model import obtain_pitch_candidates, simulate_features

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO = (REPO_ROOT / 'physical_models_for_fast_estimation_guitar_string_fret_and_plucking_position-master'
         / 'MATLAB_etc' / 'util' / 'hjerrild_ICASSP19' / 'recording_of_plucking_with_sudden_changes.wav')
ONSETS = REPO_ROOT / 'results' / 'physical_models' / 'onsets.txt'
MATLAB_BETAS = REPO_ROOT / 'results' / 'physical_models' / 'beta-estimations.txt'
MATLAB_BOUNDS = REPO_ROOT / 'results' / 'physical_models' / 'class-bounds.txt'

# Algorithm constants (match MATLAB defaults)
M_INITIAL = 5            # partials for initial harmonic summation
M = 25                   # partials for inharmonic summation
N_FFT = 2 ** 19          # zero-padded FFT length
F0_LIMITS = (75.0, 700.0)
SEGMENT_DUR_S = 0.04     # 40 ms
PREPEND_ZEROS = 1000     # MATLAB prepends 1000 zeros before onset detection
BETA_RES = 1e-5
N_REALIZATIONS = 500


def parse_onsets(path: Path) -> list[float]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or 'Onset' in s:
            continue
        with contextlib.suppress(ValueError):
            out.append(float(s))
    return out


def parse_matlab_betas(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or 'Estimated' in s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        with contextlib.suppress(ValueError):
            out.append({
                'idx': int(parts[0]),
                'string': int(parts[1]),
                'fret': int(parts[2]),
                'beta': float(parts[3]),
                'pitch': float(parts[4]),
                'pluck_cm': float(parts[5]),
            })
    return out


def load_matlab_grids(path: Path) -> dict:
    """Parse the MATLAB-dumped class-bounds file.

    Two sections separated by 'Mean f0 per class:':
      - first 78 lines: kk min max (β bounds)
      - next 78 lines: kk f0_mean

    Class index kk ∈ 1..78 maps to (row, fret) where row = (kk-1)//13 and
    fret = (kk-1) % 13. Row 0 = low E (matches MATLAB's w0Model rows after
    reshape+transpose).

    Returns dict matching `simulate_features` output:
      - 'f0_mean':  (6, 13)
      - 'beta_min': (6, 13)
      - 'beta_max': (6, 13)
    """
    f0_mean = np.zeros((6, 13))
    beta_min = np.zeros((6, 13))
    beta_max = np.zeros((6, 13))

    section = None
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        if 'Per-class' in s and 'bounds' in s:
            section = 'beta'
            continue
        if 'Mean' in s and 'class' in s:
            section = 'f0'
            continue
        parts = s.split()
        if section == 'beta' and len(parts) == 3:
            with contextlib.suppress(ValueError):
                kk = int(parts[0])
                row, fret = (kk - 1) // 13, (kk - 1) % 13
                beta_min[row, fret] = float(parts[1])
                beta_max[row, fret] = float(parts[2])
        elif section == 'f0' and len(parts) == 2:
            with contextlib.suppress(ValueError):
                kk = int(parts[0])
                row, fret = (kk - 1) // 13, (kk - 1) % 13
                f0_mean[row, fret] = float(parts[1])

    return {'f0_mean': f0_mean, 'beta_min': beta_min, 'beta_max': beta_max}


def estimate_with_class_grids(
    audio: np.ndarray, fs: int, onset_seconds: float, sim: dict,
) -> tuple[float, float, int, int]:
    """Faithful MATLAB-style estimator: candidate-driven inharmonic summation
    with per-class β grids. Returns (β, f₀, string_idx, fret_idx).

    String index follows MATLAB's `w0Model` row convention: 0 = low E.
    """
    padded = np.concatenate([np.zeros(PREPEND_ZEROS), audio]) if PREPEND_ZEROS else audio
    onset_sample = int(np.floor(onset_seconds * fs))
    seg_len = int(np.floor(SEGMENT_DUR_S * fs)) + 2
    seg = padded[onset_sample:onset_sample + seg_len]
    if len(seg) < 32:
        return float('nan'), float('nan'), -1, -1

    x = apply_gaussian_window(seg)
    x = hilbert_transform(x)
    _, X = compute_fft(x, fs, N_FFT)

    f0_initial = harmonic_summation(X, F0_LIMITS, M_INITIAL, fs, N_FFT)
    candidates = obtain_pitch_candidates(f0_initial, sim['f0_mean'])

    best = (float('-inf'), None, None, None, None)  # (cost, f0, β, string_idx, fret_idx)
    for s_idx, f_idx in candidates:
        b_min = sim['beta_min'][s_idx, f_idx]
        b_max = sim['beta_max'][s_idx, f_idx]
        # Match MATLAB's `start:step:stop`: includes start, then start+step, ...,
        # up to the LAST FULL STEP ≤ stop. Compute n explicitly to avoid
        # np.arange's float-precision quirks at the upper bound.
        n_steps = int(np.floor((b_max - b_min) / BETA_RES + 1e-12)) if b_max > b_min else 0
        beta_grid = b_min + np.arange(n_steps + 1) * BETA_RES
        f0_est, beta_est, cost = inharmonic_summation(X, f0_initial, M, fs, beta_grid, N_FFT)
        if cost > best[0]:
            best = (cost, f0_est, beta_est, s_idx, f_idx)

    return best[2], best[1], best[3], best[4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', choices=['matlab', 'python'], default='matlab',
                    help='matlab: load MATLAB-dumped class bounds (bit-exact match); '
                         'python: run our own Monte Carlo (validates simulation port)')
    args = ap.parse_args()

    print(f'Loading audio: {AUDIO.relative_to(REPO_ROOT)}')
    audio, fs = sf.read(str(AUDIO))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f'  fs={fs}, duration={len(audio) / fs:.2f}s')

    onsets = parse_onsets(ONSETS)
    matlab_rows = parse_matlab_betas(MATLAB_BETAS)
    print(f'  onsets: {len(onsets)}, MATLAB rows: {len(matlab_rows)}')

    if args.grids == 'matlab':
        if not MATLAB_BOUNDS.exists():
            raise SystemExit(f'Missing MATLAB bounds file: {MATLAB_BOUNDS}')
        print(f'\nLoading MATLAB-dumped class bounds: {MATLAB_BOUNDS.relative_to(REPO_ROOT)}')
        sim = load_matlab_grids(MATLAB_BOUNDS)
    else:
        print(f'\nSimulating physical-model features ({N_REALIZATIONS} realizations)...')
        sim = simulate_features(n_realizations=N_REALIZATIONS, seed=0)
    print(f'  f₀ range: [{sim["f0_mean"].min():.1f}, {sim["f0_mean"].max():.1f}] Hz')
    print(f'  β range:  [{sim["beta_min"].min():.2e}, {sim["beta_max"].max():.2e}]')

    print('\nPer-segment comparison:')
    print(f'  {"#":>3} | MATLAB: {"str":>3} {"fret":>4} {"β":>11} | '
          f'Python: {"str":>3} {"fret":>4} {"β":>11} | {"Δβ%":>6} {"class match":>11}')

    rel_errs = []
    abs_errs = []
    by_class = defaultdict(list)
    n_class_match = 0

    for onset, row in zip(onsets, matlab_rows, strict=True):
        beta_py, _f0_py, s_idx, f_idx = estimate_with_class_grids(audio, fs, onset, sim)
        # Convert Python's 0-indexed string to MATLAB's 1-indexed
        py_string = s_idx + 1 if s_idx >= 0 else -1
        py_fret = f_idx if f_idx >= 0 else -1

        beta_m = row['beta']
        rel_err = (beta_py - beta_m) / beta_m * 100 if beta_m > 0 else float('nan')
        rel_errs.append(rel_err)
        abs_errs.append(beta_py - beta_m)
        class_match = (row['string'] == py_string) and (row['fret'] == py_fret)
        if class_match:
            n_class_match += 1
        by_class[(row['string'], row['fret'])].append((beta_m, beta_py, class_match))

        match_str = 'OK' if class_match else f'-> ({py_string},{py_fret})'
        print(f'  {row["idx"]:>3d} | MATLAB: {row["string"]:>3d} {row["fret"]:>4d} {beta_m:>10.4e} | '
              f'Python: {py_string:>3d} {py_fret:>4d} {beta_py:>10.4e} | '
              f'{rel_err:>+5.1f}%  {match_str:>11}')

    print('\nPer-class summary:')
    print(f'  {"(s,f)":>8}  {"n":>3}  {"MATLAB med":>11}  {"Python med":>11}  '
          f'{"Δ%":>6}  {"class match":>11}')
    for (s, f), items in sorted(by_class.items()):
        m_betas = np.array([x[0] for x in items])
        p_betas = np.array([x[1] for x in items])
        n_match = sum(1 for x in items if x[2])
        m_med, p_med = float(np.median(m_betas)), float(np.median(p_betas))
        delta_pct = (p_med - m_med) / m_med * 100
        print(f'  ({s},{f:>2d})  {len(items):>3d}  {m_med:>11.4e}  {p_med:>11.4e}  '
              f'{delta_pct:>+5.1f}%  {n_match}/{len(items):<11}')

    rel_errs_arr = np.array(rel_errs)
    print('\nOverall:')
    print(f'  Class identification match: {n_class_match}/{len(matlab_rows)} segments')
    print(f'  β median rel error: {np.median(rel_errs_arr):+.2f}%')
    print(f'  β mean rel error  : {np.mean(rel_errs_arr):+.2f}%')
    print(f'  β max abs rel err : {np.max(np.abs(rel_errs_arr)):.2f}%')
    print(f'  segments within ±5% : {int(np.sum(np.abs(rel_errs_arr) <= 5))} / {len(rel_errs_arr)}')
    print(f'  segments within ±10%: {int(np.sum(np.abs(rel_errs_arr) <= 10))} / {len(rel_errs_arr)}')
    print(f'  segments within ±2% : {int(np.sum(np.abs(rel_errs_arr) <= 2))} / {len(rel_errs_arr)}')


if __name__ == '__main__':
    main()

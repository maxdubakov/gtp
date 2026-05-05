"""Physical-model β simulation (port of recreate_plucking_experiment_WASPAA19.m).

Monte Carlo over guitar string properties (Young's modulus, diameter, tension,
length, pluck deflection) → simulated (f₀, β) feature space per (string, fret)
class. Used to derive per-class β grids for the inharmonic summation, which is
how Hjerrild's MATLAB matches the search to physically-plausible β values per
class (rather than using a fixed grid across all classes).

Convention (matches MATLAB):
  - 6 strings x 13 frets (frets 0-12) = 78 classes
  - String index 1 = low E (thickest) ... 6 = high E (thinnest)
  - All `(strings, frets)` matrices have rows indexed 0..5 corresponding to
    MATLAB's strings 1..6, columns 0..12 corresponding to frets 0..12.
  - For pitch-candidate lookup, the MATLAB code reshapes/transposes the model
    to put low-E in row 1 — we expose `w0_model_reordered` for that.
"""

import numpy as np

# String material constants (exact values from
# recreate_plucking_experiment_WASPAA19.m lines 58-64)
E_STEEL = 2.27e11      # Young's modulus, Pa
G_SHEAR = 79.3e9       # shear modulus, Pa
RHO_CORE = 7950        # core (steel) density, kg/m³
RHO_WRAPPING = 6000    # wrapping (nickel-type) density, kg/m³

# Per-string nominal properties (string 1 = low E, ... string 6 = high E)
D_FULL = np.array([0.046, 0.036, 0.026, 0.0172, 0.013, 0.010]) * 0.0254  # m, full diameter
D_CORE = np.array([0.018, 0.016, 0.0146, 0.0172, 0.013, 0.010]) * 0.0254  # m, core diameter
T0_NOM = np.array([17, 20, 18.5, 17.5, 16, 16.5]) * 4.45                  # N, tension
# distance from nut to bridge per string (m). MATLAB had [1 2 9 4 5 2]*1e-3
# offset on top of 0.6411 — but their string-1 is low E in their loop, while
# we index string 1 = low E here. Need to match the same physical strings.
# MATLAB: L0 = 0.6411 + [1 2 9 4 5 2]'*1e-3, with strings 1..6 in tension order
# T0=[16.5, 16, 17.5, 18.5, 20, 17] *4.45. Their string 1 is the 16.5*4.45=73.4 N
# string with full-diameter 0.010" = 0.254 mm — that's HIGH E (thinnest).
# So MATLAB's "string 1" in their array = high E. We convert here.
L0_OFFSET = np.array([2, 5, 4, 9, 2, 1]) * 1e-3  # reversed from MATLAB's [1 2 9 4 5 2]
L0_NOM_OPEN = 0.6411 + L0_OFFSET                # m, length at fret 0 per string

# Pluck parameters
P_FRACTION = 1 / 3   # plucking position as fraction of string from bridge
FORCE_NOM = 0.05     # plucking force (N)

PCT_STD = 0.005      # 0.5% Gaussian noise on string properties per realization

N_STRINGS = 6
N_FRETS = 13


def simulate_features(n_realizations: int = 500, seed: int = 0) -> dict:
    """Run Monte Carlo over noisy string properties; return per-class (f₀, β).

    Output shape: (N_STRINGS, N_FRETS, n_realizations).
    String index 0 = low E (thickest), index 5 = high E (thinnest).

    Returns dict with:
      - 'f0':         array (6, 13, n) — fundamentals
      - 'beta':       array (6, 13, n) — total inharmonicity = β_intrinsic + β_pluck
      - 'beta_min':   array (6, 13)    — min β per class across realizations
      - 'beta_max':   array (6, 13)    — max β per class
      - 'f0_mean':    array (6, 13)    — mean f₀ per class (used as `w0Model`)
    """
    rng = np.random.default_rng(seed)

    # Length per (string, fret). Fret f reduces length by 2^(-f/12).
    fret_factor = 2.0 ** (-np.arange(N_FRETS) / 12)              # (13,)
    L0_per_fret = L0_NOM_OPEN[:, None] * fret_factor[None, :]    # (6, 13)

    f0_all = np.zeros((N_STRINGS, N_FRETS, n_realizations))
    beta_all = np.zeros((N_STRINGS, N_FRETS, n_realizations))

    for it in range(n_realizations):
        L0 = L0_per_fret * (1 + PCT_STD * rng.standard_normal(L0_per_fret.shape))
        T0 = T0_NOM * (1 + PCT_STD * rng.standard_normal(N_STRINGS))
        d_core = D_CORE * (1 + PCT_STD * rng.standard_normal(N_STRINGS))
        d_wrap_nom = (D_FULL - d_core) / 2
        d_wrap = d_wrap_nom * (1 + PCT_STD * rng.standard_normal(N_STRINGS))
        force = FORCE_NOM * (1 + PCT_STD * rng.standard_normal())

        a_core = np.pi * (d_core / 2) ** 2                        # (6,)
        mu = (a_core * RHO_CORE
              + RHO_WRAPPING * ((2 * d_wrap + d_core) ** 2 - d_core ** 2) * (np.pi / 4))  # (6,)

        # Composite-string analysis: split tension into core + wrapping
        D = d_core + d_wrap
        Tc_over_Tw = (8 * a_core * D ** 3 * E_STEEL) / (G_SHEAR * d_wrap ** 5)
        Tc = T0 / ((1 / Tc_over_Tw) + 1)                           # (6,)
        # Length-extension under tension (per fret needs broadcasting)
        deltaL = (L0 * Tc[:, None]) / (a_core[:, None] * E_STEEL + Tc[:, None])  # (6, 13)
        E_eff = (T0[:, None] / a_core[:, None]) / (deltaL / (L0 - deltaL))       # (6, 13)

        # Pluck deflection contribution
        deltaP = ((L0 * P_FRACTION * force * (1 - P_FRACTION)) / T0[:, None]) ** 2
        delta_dL = (np.sqrt((P_FRACTION * L0) ** 2 + deltaP ** 2)
                    + np.sqrt(((1 - P_FRACTION) * L0) ** 2 + deltaP ** 2)
                    - L0)
        delta_half = np.sqrt((delta_dL ** 2 + delta_dL * L0 * 2) / 4)            # (6, 13)

        K = (np.pi ** 3 * E_eff * d_core[:, None] ** 2) / (16 * T0[:, None] * L0 ** 2)
        beta_intrinsic = (K / 4) * d_core[:, None] ** 2
        beta_pluck = (K * 3 / 8) * delta_half ** 2
        beta_total = beta_intrinsic + beta_pluck                                  # (6, 13)

        f0 = np.sqrt(T0[:, None] / mu[:, None]) / L0 / 2                          # (6, 13)

        f0_all[:, :, it] = f0
        beta_all[:, :, it] = beta_total

    return {
        'f0': f0_all,
        'beta': beta_all,
        'f0_mean': f0_all.mean(axis=2),
        'beta_min': beta_all.min(axis=2),
        'beta_max': beta_all.max(axis=2),
    }


# ---------------------------------------------------------------------------
# Pitch-candidate selection
# ---------------------------------------------------------------------------


def obtain_pitch_candidates(observed_pitch: float, f0_model: np.ndarray) -> list[tuple[int, int]]:
    """Return (string_idx, fret_idx) candidates for `observed_pitch`. 0-indexed.

    Mirror of icassp19_obtain_pitch_candidates.m. The candidate set covers
    all positions on the guitar that produce a pitch close to `observed_pitch`
    in the equal-tempered scale.

    `f0_model` is the (6, 13) mean-f₀ matrix in MATLAB's `w0Model` convention
    (row 0 = low E, row 5 = high E), so this follows the MATLAB code directly.
    """
    log_obs = np.log(observed_pitch)
    log_model = np.log(f0_model)

    # Step 1: column-wise minimum across rows, then argmin column → fret
    col_min = np.min(np.abs(log_model - log_obs), axis=0)  # (13,)
    fret_idx = int(np.argmin(col_min))  # 0..12

    # Step 2: argmin row in that column → string
    string_idx = int(np.argmin(np.abs(log_model[:, fret_idx] - log_obs)))  # 0..5

    candidates: list[tuple[int, int]] = [(string_idx, fret_idx)]

    # MATLAB's enumeration of additional candidates per (string, fret) range.
    # MATLAB uses 1-indexed string and 0-indexed fret; we use 0-indexed string.
    s_m = string_idx + 1   # MATLAB-style 1-indexed
    f = fret_idx           # 0-indexed fret (matches MATLAB)

    extras: list[tuple[int, int]] = []
    if s_m == 1 and 4 < f < 10:
        extras.append((s_m + 1, f - 5))
    if s_m == 1 and f > 9:
        extras.append((s_m + 1, f - 5))
        extras.append((s_m + 2, f - 10))
    if s_m == 2 and f < 5:
        extras.append((s_m - 1, f + 5))
    if s_m == 2 and 4 < f < 8:
        extras.append((s_m + 1, f - 5))
        extras.append((s_m - 1, f + 5))
    if s_m == 2 and 7 < f < 10:
        extras.append((s_m + 1, f - 5))
    if s_m == 2 and f > 9:
        extras.append((s_m + 1, f - 5))
        extras.append((s_m + 2, f - 10))
    if s_m == 3 and f < 5:
        extras.append((s_m - 1, f + 5))
    if s_m == 3 and f < 3:
        extras.append((s_m - 2, f + 10))
    if s_m == 3 and 4 < f < 8:
        extras.append((s_m + 1, f - 5))
        extras.append((s_m - 1, f + 5))
    if s_m == 3 and 7 < f < 10:
        extras.append((s_m + 1, f - 5))
    if s_m == 3 and f > 8:
        extras.append((s_m + 1, f - 5))
        extras.append((s_m + 2, f - 9))
    if s_m == 4 and f < 4:
        extras.append((s_m - 1, f + 5))
    if s_m == 4 and f < 3:
        extras.append((s_m - 2, f + 10))
    if s_m == 4 and 3 < f < 8:
        extras.append((s_m + 1, f - 4))
        extras.append((s_m - 1, f + 5))
    if s_m == 4 and 7 < f < 10:
        extras.append((s_m + 1, f - 4))
    if s_m == 4 and f > 8:
        extras.append((s_m + 1, f - 4))
        extras.append((s_m + 2, f - 9))
    if s_m == 5 and f < 5:
        extras.append((s_m - 1, f + 4))
    if s_m == 5 and f < 4:
        extras.append((s_m - 2, f + 9))
    if s_m == 5 and 4 < f < 9:
        extras.append((s_m + 1, f - 5))
        extras.append((s_m - 1, f + 4))
    if s_m == 5 and f > 8:
        extras.append((s_m + 1, f - 5))
    if s_m == 6 and f < 4:
        extras.append((s_m - 1, f + 5))
        extras.append((s_m - 2, f + 9))
    if s_m == 6 and 3 < f < 8:
        extras.append((s_m - 1, f + 5))

    # Convert extras back to 0-indexed string and dedupe
    seen = {(string_idx, fret_idx)}
    for s_m_extra, f_extra in extras:
        cand = (s_m_extra - 1, f_extra)
        if 0 <= cand[0] <= 5 and 0 <= cand[1] <= 12 and cand not in seen:
            candidates.append(cand)
            seen.add(cand)

    # Sort by string ascending (matches MATLAB's final sort)
    candidates.sort(key=lambda c: c[0])
    return candidates


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    """Print mean (f0, β) per (string, fret) class — sanity-check the simulation."""
    sim = simulate_features(n_realizations=500, seed=0)
    print('Simulated mean (f₀, β) per (string, fret) class')
    print('  string indexing: 0=low E, 5=high E (matching MATLAB w0Model rows)\n')
    print(f'  {"s\\f":>4}  ' + '  '.join(f'{f:>5d}' for f in range(N_FRETS)))
    print('  -- f₀ mean (Hz) --')
    for s in range(N_STRINGS):
        print(f'  {s:>4}  ' + '  '.join(f'{sim["f0_mean"][s, f]:>5.0f}' for f in range(N_FRETS)))
    print('  -- β mean (x10⁻⁴) --')
    for s in range(N_STRINGS):
        print(f'  {s:>4}  ' + '  '.join(f'{sim["beta"].mean(axis=2)[s, f] * 1e4:>5.2f}' for f in range(N_FRETS)))
    print()
    # Sanity-check pitch-candidate function
    print('obtain_pitch_candidates(110.0, f0_mean):',
          obtain_pitch_candidates(110.0, sim['f0_mean']))
    print('obtain_pitch_candidates(330.0, f0_mean):',
          obtain_pitch_candidates(330.0, sim['f0_mean']))

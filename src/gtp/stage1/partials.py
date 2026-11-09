"""Pitch-normalized partial features of a note: levels, frequency deviations and inharmonicity of its first K partials.

With the pitch known, the partials are located in a high-resolution spectrum of consecutive frames after the onset.
Per frame and partial we keep the level (dB under the first frame's loudest partial) and the frequency deviation
(cents from k * f0, f0 fitted per note). A fit of f_k = k f0 sqrt(1 + B k^2) on the first frame gives f0 and the
inharmonicity B. These are the quantities string physics predicts differ between the candidate (string, fret) pairs of
one pitch: inharmonicity grows as the vibrating length shortens, and the pluck and pickup positions comb-filter the
partial levels.

`cfg` carries sample_rate-independent analysis settings: frame, n_frames, dev_frames, n_fft, n_partials, b_max,
tol_cents (conf/strings/harmonic_features.yaml).
"""

import librosa
import numpy as np


def fit_inharmonic(f, a, k):
    """Weighted least squares of (f_k / k)^2 = f0^2 (1 + B k^2) over the partials found; returns f0, B, residual."""
    ok = np.isfinite(f) & (a > np.nanmax(a) - 40)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    y, x, w = (f[ok] / k[ok]) ** 2, k[ok] ** 2, 10 ** (a[ok] / 40)
    A = np.stack([np.ones_like(x), x], 1) * w[:, None]
    (c0, c1), *_ = np.linalg.lstsq(A, y * w, rcond=None)
    if c0 <= 0:
        return np.nan, np.nan, np.nan
    f0, b = np.sqrt(c0), c1 / c0
    pred = k[ok] * f0 * np.sqrt(np.maximum(1 + b * k[ok] ** 2, 1e-6))
    return f0, b, float(np.average(np.abs(1200 * np.log2(f[ok] / pred)), weights=w))


def partials(spec, freqs, f0, n_partials, b_max, tol_cents):
    """Peak (frequency, dB) of partials 1..K of f0 in one magnitude spectrum; NaN above the band.

    Partials are tracked upward: each is searched around the position predicted by a fit of the ones below it,
    within a tolerance that shrinks with k so the search never reaches a neighbouring partial.
    """
    k = np.arange(1, n_partials + 1)
    df = freqs[1]
    f, a = np.full(n_partials, np.nan), np.full(n_partials, np.nan)
    log = 20 * np.log10(spec + 1e-9)
    f0_est, b_est = f0, 0.0
    for i in range(n_partials):
        pred = k[i] * f0_est * np.sqrt(1 + b_est * k[i] ** 2)
        tol = min(tol_cents, 600 / k[i])
        lo, hi = pred * 2 ** (-tol / 1200), pred * 2 ** (tol / 1200)
        if hi >= freqs[-1] * 0.95:
            break
        a0, a1 = int(lo / df), int(np.ceil(hi / df)) + 1
        j = a0 + int(np.argmax(spec[a0:a1]))
        y0, y1, y2 = log[j - 1], log[j], log[j + 1]
        d = np.clip(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12), -0.5, 0.5)
        f[i], a[i] = (j + d) * df, y1 - 0.25 * (y0 - y2) * d
        if i >= 2:
            fit_f0, fit_b, _ = fit_inharmonic(f, a, k)
            if np.isfinite(fit_f0):
                f0_est, b_est = fit_f0, float(np.clip(fit_b, 0.0, b_max))
    return f, a


def note_features(y, sr, onset, pitch, cfg):
    k = np.arange(1, cfg.n_partials + 1)
    window = np.hanning(cfg.frame)
    freqs = np.fft.rfftfreq(cfg.n_fft, 1 / sr)
    f0_nominal = librosa.midi_to_hz(pitch)
    start = round(onset * sr)
    frames = []
    for j in range(cfg.n_frames):
        s = start + j * cfg.frame
        seg = y[max(s, 0) : s + cfg.frame]
        if len(seg) < cfg.frame:
            seg = np.pad(seg, (0, cfg.frame - len(seg)))
        frames.append(np.abs(np.fft.rfft(seg * window, cfg.n_fft)))
    peaks = [partials(s, freqs, f0_nominal, cfg.n_partials, cfg.b_max, cfg.tol_cents) for s in frames]
    f0, b, resid = fit_inharmonic(*peaks[0], k)
    ref = f0 if np.isfinite(f0) else f0_nominal
    top = np.nanmax(peaks[0][1]) if np.isfinite(peaks[0][1]).any() else 0.0
    levels = np.stack([a - top for _, a in peaks])
    devs = np.stack([1200 * np.log2(f / (k * ref)) for f, _ in peaks[: cfg.dev_frames]])
    energy = np.array([20 * np.log10(np.sqrt(np.mean(s**2)) + 1e-9) for s in frames])
    scalars = np.array(
        [
            1200 * np.log2(ref / f0_nominal),
            np.log10(max(b, 1e-7)) if np.isfinite(b) else -7.0,
            resid if np.isfinite(resid) else 50.0,
            top,
        ]
    )
    return np.concatenate([levels.ravel(), devs.ravel(), energy - energy[0], scalars]).astype(np.float32)


def feature_names(cfg):
    k = range(1, cfg.n_partials + 1)
    return (
        [f'level_t{t}_k{i}' for t in range(cfg.n_frames) for i in k]
        + [f'dev_t{t}_k{i}' for t in range(cfg.dev_frames) for i in k]
        + [f'energy_t{t}' for t in range(cfg.n_frames)]
        + ['f0_cents', 'log10_b', 'fit_resid_cents', 'top_db']
    )

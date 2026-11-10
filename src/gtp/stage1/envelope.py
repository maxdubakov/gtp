"""Per-note harmonic levels, and the string-envelope model that reads the string from them.

A pickup at a fixed distance from the bridge cancels the partials whose half-wavelengths fit that distance. On
string s the nulls therefore sit at fixed frequencies, multiples of f_open_s / delta (delta = pickup distance over
scale length), whatever fret is played; the plucking point leaves a comb of the same kind. Every string colours its
notes with one curve H along the axis u = f / f_open_s, shared by the six strings, and the rest of the chain (amp,
room, microphone) adds one response E(f) common to all of them. A note's harmonic levels, in dB, are modelled as

    note(n) + H(f / f_open_s) + E(f) + noise

where note(n), a low-degree polynomial in log n, belongs to the note (how hard and with what it was plucked). H and E are fitted to the recordings themselves by EM over the notes' string assignments: no labels, and nothing
learned from other guitars. The comb survives amps and rooms, where partial frequencies (inharmonicity) blur.

`cfg`: conf/strings/harmonic_levels.yaml.
"""

import librosa
import numpy as np

from gtp.stage1.bprofile import occupied_bins, tuning_offset
from gtp.stage1.strings import OPEN_STRINGS


def note_spectrum(y, sr, onset, frame, cfg):
    """Power spectrum up to f_max of the `frame` samples that start start_s after the onset."""
    start = max(round((onset + cfg.start_s) * sr), 0)
    seg = y[start : start + frame]
    if len(seg) < frame:
        seg = np.pad(seg, (0, frame - len(seg)))
    return np.abs(np.fft.rfft(seg * np.hanning(frame), cfg.n_fft)[: int(cfg.f_max * cfg.n_fft / sr) + 1]) ** 2


def note_levels(y, sr, onsets, offsets, pitches, cfg):
    """(notes, n_harmonics) level in dB of each harmonic of each note; NaN above f_max, where another sounding
    note has a partial of its own, and past the note's last harmonic that is a spectral peak.

    The recording's own tuning is measured first, so that harmonics are looked for, and other notes' partials
    avoided, within a narrow range of where they must be: in a chord a wide range leaves almost nothing to read.
    Each note is read over the frame length, of those it lasts for, that leaves it the most harmonics: a long frame
    tells close partials apart, a short one ends before the next note of an arpeggio begins.
    """
    df = sr / cfg.n_fft
    n_bins = int(cfg.f_max / df)
    onsets, offsets = np.asarray(onsets), np.asarray(offsets)
    short = [note_spectrum(y, sr, on, min(cfg.frames), cfg) for on in onsets]
    pitches = np.asarray(pitches) + tuning_offset(short, pitches, df, cfg) / 100
    n = np.arange(1, cfg.n_harmonics + 1)
    out = np.full((len(onsets), cfg.n_harmonics), np.nan, np.float32)
    for i, (on, off, pitch) in enumerate(zip(onsets, offsets, pitches, strict=True)):
        f0 = librosa.midi_to_hz(pitch)
        lo = (n * f0 * 2 ** (-cfg.tuning_cents / 1200) / df).astype(int)
        hi = np.ceil(n * f0 * 2 ** (cfg.tuning_cents / 1200) * np.sqrt(1 + cfg.other_b_max * n**2) / df).astype(int) + 1
        half = max(int(f0 / 2 / df), 1)
        for frame in cfg.frames:
            end = on + cfg.start_s + frame / sr
            if frame > min(cfg.frames) and end > off + cfg.tail_s:
                break
            power = short[i] if frame == min(cfg.frames) else note_spectrum(y, sr, on, frame, cfg)
            sounding = (onsets < end - cfg.overlap_s) & (offsets > on + cfg.start_s + cfg.overlap_s)
            sounding[i] = False
            occ = occupied_bins(np.unique(pitches[sounding & (pitches != pitch)]), n_bins, df, cfg, 2 * sr / frame)
            row, last = np.full(cfg.n_harmonics, np.nan, np.float32), -1
            for k in range(cfg.n_harmonics):
                if hi[k] + half > n_bins:
                    break
                if not occ[lo[k] : hi[k]].any():
                    row[k] = 10 * np.log10(power[lo[k] : hi[k]].max() + 1e-12)
                    valley = max(power[max(lo[k] - half, 0) : lo[k] + 1].min(), power[hi[k] - 1 : hi[k] + half].min())
                    if row[k] - 10 * np.log10(valley + 1e-12) >= cfg.peak_db:
                        last = k
            # Past its last harmonic that stands out as a peak, a note is noise or the skirts of its lower harmonics
            # (a muted or dying note, a bend out of range): readings there are not the string's.
            row[last + 1 :] = np.nan
            if np.isfinite(row).sum() > np.isfinite(out[i]).sum():
                out[i] = row
    return out


def envelope_bins(pitch, cfg):
    """Bin of every (note, harmonic) on the E axis (log f), and on the H axis (log f / f_open_s) for each string."""
    freq = librosa.midi_to_hz(np.asarray(pitch))[:, None] * np.arange(1, cfg.n_harmonics + 1)[None]
    f_bin = np.round(cfg.bins_per_octave * np.log2(freq / cfg.f_min)).astype(int)
    u = freq[:, :, None] / librosa.midi_to_hz(OPEN_STRINGS)[None, None]
    return np.maximum(f_bin, 0), np.maximum(np.round(cfg.bins_per_octave * np.log2(u)).astype(int), 0)


def residuals(levels, f_bin, u_bin, h, e, cfg):
    """(notes, harmonics, 6) what each string's envelope leaves of the notes' levels, NaN where a harmonic is unusable.

    Each note keeps what the model does not explain about it for itself: a polynomial of degree `note_degree` in
    log harmonic number, fitted by least squares. Degree 0 is its loudness, 1 adds how fast its harmonics fall off,
    2 a bend in that fall: the pick, the fingers or a palm on the strings, not the string. A note with too few
    usable harmonics for its degree gets the loudness only.
    """
    ok = ~np.isnan(levels)
    r = (
        np.nan_to_num(levels)[:, :, None]
        - h[np.minimum(u_bin, len(h) - 1)]
        - e[np.minimum(f_bin, len(e) - 1)][:, :, None]
    )
    r = np.where(ok[:, :, None], r, 0.0)
    x = np.log2(np.arange(1, levels.shape[1] + 1))
    basis = np.stack([x**d for d in range(cfg.note_degree + 1)], axis=1)  # (harmonics, terms)
    basis = np.where(ok[:, :, None], basis[None], 0.0)  # (notes, harmonics, terms)
    enough = ok.sum(axis=1) >= cfg.min_harmonics + cfg.note_degree
    basis[~enough, :, 1:] = 0.0
    gram = np.einsum('nhp,nhq->npq', basis, basis) + 1e-6 * np.eye(basis.shape[2])
    coef = np.linalg.solve(gram, np.einsum('nhp,nhs->nps', basis, r))
    return np.where(ok[:, :, None], r - np.einsum('nhp,nps->nhs', basis, coef), np.nan)


def envelope_loglik(levels, f_bin, u_bin, model, allowed, cfg):
    """(notes, 6) log likelihood of each note's harmonic levels under each string (Laplace residuals).

    A note that no allowed string explains to within the model's residual scale (a muted or bent note, a harmonic)
    is scored on its own, wider scale, so it has less to say. One with too few usable harmonics says nothing.
    """
    h, e, sigma = model
    count = (~np.isnan(levels)).sum(axis=1)
    total = np.nansum(np.abs(residuals(levels, f_bin, u_bin, h, e, cfg)), axis=1)
    best = np.where(allowed, total, np.inf).min(axis=1) / np.maximum(count, 1)
    out = -total / np.maximum(sigma, best)[:, None]
    out[count < cfg.min_harmonics] = 0.0
    return out


def posterior(prior, loglik):
    logp = np.where(prior > 0, np.log(np.maximum(prior, 1e-12)) + loglik, -np.inf)
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def em(levels, f_bin, u_bin, prior, cfg):
    """(H, E, sigma) fitted by EM over the notes' strings, started from `prior`.

    Bins are shrunk towards zero by `shrink` pseudo-observations, so a handful of notes fits a nearly flat model.
    sigma, the residual scale in dB, is the one the fit leaves (never below sigma_min_db).
    """
    ok = ~np.isnan(levels)
    n_u, n_f = u_bin.max() + 1, f_bin.max() + 1
    h, e, sigma = np.zeros(n_u), np.zeros(n_f), cfg.sigma_start_db
    resp = prior / np.maximum(prior.sum(axis=1, keepdims=True), 1e-12)
    for _ in range(cfg.iters):
        w = np.where(ok[:, :, None], resp[:, None, :], 0.0)
        for _ in range(cfg.sweeps):
            r = np.nan_to_num(residuals(levels, f_bin, u_bin, h, e, cfg))
            total, count = np.zeros(n_u), np.zeros(n_u)
            np.add.at(total, u_bin, w * (r + h[u_bin]))
            np.add.at(count, u_bin, w)
            h = total / (count + cfg.shrink)
            r = np.nan_to_num(residuals(levels, f_bin, u_bin, h, e, cfg))
            total, count = np.zeros(n_f), np.zeros(n_f)
            np.add.at(total, f_bin, (w * (r + e[f_bin][:, :, None])).sum(axis=2))
            np.add.at(count, f_bin, w.sum(axis=2))
            e = total / (count + cfg.shrink)
        r = np.nan_to_num(residuals(levels, f_bin, u_bin, h, e, cfg))
        sigma = max((w * np.abs(r)).sum() / max(w.sum(), 1e-9), cfg.sigma_min_db)
        resp = posterior(prior, envelope_loglik(levels, f_bin, u_bin, (h, e, sigma), prior > 0, cfg))
    return h, e, sigma


def fit_envelopes(levels, pitch, prior, cfg):
    """(notes, 6) log likelihood of the notes' harmonic levels under each string, the model fitted to these notes
    themselves, starting from `prior` (zero on strings that cannot play the note). Also returns the residual scale (dB).
    """
    f_bin, u_bin = envelope_bins(pitch, cfg)
    model = em(levels, f_bin, u_bin, prior, cfg)
    return envelope_loglik(levels, f_bin, u_bin, model, prior > 0, cfg), model[2]

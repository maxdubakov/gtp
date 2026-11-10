"""Per-note inharmonicity likelihood profiles: how well each value of B explains a note's spectral peaks.

A point estimate of B fails in chords, where the peaks it is fitted to may belong to other notes, and gives no measure
of its own reliability. Here every frame of a note keeps a curve over a grid of log10(B) instead. For a hypothesis
(B, tuning offset) the partials are predicted at f_k = k f0 sqrt(1 + B k^2) and compared with the nearest detected
spectral peak. Each partial is either observed near its prediction (Laplace noise in Hz) or missing; the score is the
log likelihood ratio against "all partials missing", so a frame without usable peaks scores zero for every B and is
neutral. Predicted partials that fall where a concurrently sounding note has its own partials are skipped, which can
only remove reward, never add it.

`cfg`: sample_rate-independent settings (conf/strings/b_profiles.yaml).
"""

import librosa
import numpy as np
import scipy.signal


def grid(cfg):
    return np.arange(cfg.log_b_min, cfg.log_b_max + 1e-9, cfg.log_b_step)


def frame_peaks(y, start, cfg, window, sr):
    """Frequencies (Hz) of the spectral peaks of one frame, and the frame's bin spacing."""
    seg = y[max(start, 0) : start + cfg.frame]
    if len(seg) < cfg.frame:
        seg = np.pad(seg, (0, cfg.frame - len(seg)))
    df = sr / cfg.n_fft
    n_bins = int(cfg.f_max / df)
    log = 20 * np.log10(np.abs(np.fft.rfft(seg * window, cfg.n_fft))[:n_bins] + 1e-9)
    idx, _ = scipy.signal.find_peaks(log, height=log.max() - cfg.peak_floor_db, prominence=cfg.peak_prominence_db)
    idx = idx[(idx > 0) & (idx < n_bins - 1)]
    y0, y1, y2 = log[idx - 1], log[idx], log[idx + 1]
    shift = np.clip(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12), -0.5, 0.5)
    return (idx + shift) * df, df


def occupied_bins(others, n_bins, df, cfg, lobe):
    """Boolean mask over spectrum bins where another sounding note (MIDI pitches `others`) may have a partial."""
    occ = np.zeros(n_bins + 1, bool)
    for pitch in others:
        f0 = librosa.midi_to_hz(pitch)
        m = np.arange(1, int(cfg.f_max / f0) + 1)
        lo = m * f0 * 2 ** (-cfg.tuning_cents / 1200) - lobe
        hi = m * f0 * 2 ** (cfg.tuning_cents / 1200) * np.sqrt(1 + cfg.other_b_max * m**2) + lobe
        for a, b in zip((lo / df).astype(int), np.ceil(hi / df).astype(int), strict=True):
            occ[max(a, 0) : min(b, n_bins) + 1] = True
    return occ


def frame_profile(peaks, f0, occ, df, cfg, log_b):
    """Log likelihood ratio (vs no partial found) over the B grid for one frame, maximised over the tuning offset."""
    k = np.arange(1, cfg.n_partials + 1)
    stretch = np.sqrt(1 + 10 ** log_b[:, None] * k[None] ** 2)
    if len(peaks) == 0:
        return np.zeros(len(log_b))
    cents = np.arange(-cfg.tuning_cents, cfg.tuning_cents + 1e-9, cfg.tuning_step_cents)
    # A bent or slid note sits far from its nominal pitch: also try the offsets its lowest partials' peaks suggest.
    low = np.arange(1, cfg.bend_partials + 1)
    seen = 1200 * np.log2(peaks[:, None] / (low[None] * f0)).ravel()
    seen = seen[(seen > cents[-1]) & (seen < cfg.bend_cents)]
    cents = np.unique(np.round(np.concatenate([cents, (seen[:, None] + cents[None] / 5).ravel()])))
    offsets = 2 ** (cents / 1200)
    rho = (1 - cfg.missing) * f0 / (2 * cfg.noise_hz * cfg.missing)

    def score(pred):
        pos = np.clip(np.searchsorted(peaks, pred), 1, len(peaks) - 1) if len(peaks) > 1 else np.zeros(pred.shape, int)
        near = np.where(np.abs(peaks[pos - 1] - pred) < np.abs(peaks[pos] - pred), peaks[pos - 1], peaks[pos])
        usable = (pred < cfg.f_max) & ~occ[np.minimum((pred / df).astype(int), len(occ) - 1)]
        return np.where(usable, np.log1p(rho * np.exp(-np.abs(near - pred) / cfg.noise_hz)), 0.0), near, usable

    pred = offsets[None, :, None] * (f0 * k * stretch)[:, None, :]
    total = score(pred)[0].sum(axis=2)
    best = total.argmax(axis=1)
    # Refine the tuning offset per B: weighted median of the scale each matched peak implies.
    base = f0 * k * stretch
    _, near, usable = score(offsets[best][:, None] * base)
    out = np.empty(len(log_b))
    for i in range(len(log_b)):
        ok = usable[i] & (np.abs(near[i] - offsets[best[i]] * base[i]) < 4 * cfg.noise_hz)
        scale = offsets[best[i]]
        if ok.sum() >= 2:
            ratio, w = near[i][ok] / base[i][ok], base[i][ok]
            order = np.argsort(ratio)
            scale = ratio[order][np.searchsorted(np.cumsum(w[order]), w.sum() / 2)]
        out[i] = score(scale * base[i])[0].sum()
    return np.maximum(out, total.max(axis=1))


def recording_profiles(y, sr, onsets, offsets, pitches, cfg):
    """(notes, n_frames, grid) profiles for the notes of one recording; frames after a note's end are zero."""
    log_b = grid(cfg)
    window = np.hanning(cfg.frame)
    lobe = 2 * sr / cfg.frame
    onsets, offsets, pitches = np.asarray(onsets), np.asarray(offsets), np.asarray(pitches)
    out = np.zeros((len(onsets), cfg.n_frames, len(log_b)), np.float32)
    for i, (on, off, pitch) in enumerate(zip(onsets, offsets, pitches, strict=True)):
        for j in range(cfg.n_frames):
            t0 = on + j * cfg.frame / sr
            t1 = t0 + cfg.frame / sr
            if j > 0 and t1 > off + cfg.tail_s:
                break
            peaks, df = frame_peaks(y, round(t0 * sr), cfg, window, sr)
            sounding = (onsets < t1 - cfg.overlap_s) & (offsets + cfg.ring_s > t0 + cfg.overlap_s)
            sounding[i] = False
            others = np.unique(pitches[sounding & (pitches != pitch)])
            occ = occupied_bins(others, int(cfg.f_max / df), df, cfg, lobe)
            out[i, j] = frame_profile(peaks, librosa.midi_to_hz(pitch), occ, df, cfg, log_b)
    return out

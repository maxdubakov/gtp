"""Simulated room-microphone recordings, for evaluation only.

The room is measured, never synthetic: an impulse response of the ACE corpus (Eaton et al., WASPAA 2015; Zenodo
10.5281/zenodo.6257551, CC BY 4.0), recorded with a mock-up phone or a laptop's own microphones 1.3 to 3 m from a
loudspeaker in an office, meeting room or lecture room, and that device's recording of the same room's noise. So
the device's colouring and the noise come with the room. None of this shares code or parameters with chains.py,
whose synthetic rooms and microphones the string classifier is trained on.

A direct-input recording first goes through a clean amp and cabinet: a minimum-phase filter matched to a real rig,
the linear transfer from one guitar's direct-input recordings to its miked-amp recordings of the same takes.
"""

import hashlib

import librosa
import numpy as np
import scipy.signal
import soundfile as sf


def rng_for(name, seed):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}/{name}'.encode()).digest()[:8], 'little'))


def bank(ace_dir, devices, rooms):
    """Every (device, room, position, channel) of the ACE corpus under `ace_dir`, with its response and noise files."""
    out = []
    for device in devices:
        for room in rooms:
            for rir in sorted((ace_dir / device / room).glob('*/*_RIR.wav')):
                noises = {p.stem.rsplit('_', 1)[1]: str(p) for p in rir.parent.glob('*_Noise_*.wav')}
                out += [
                    {'device': device, 'room': room, 'position': int(rir.parent.name), 'channel': c,
                     'rir': str(rir), 'noises': noises}
                    for c in range(sf.info(rir).channels)
                ]  # fmt: skip
    assert out, f'no ACE responses under {ace_dir}'
    return out


def draw(rng, entries, noises, snr_db):
    """A device, room, position and channel drawn level by level (so each device and room is equally likely whatever
    its channel count), a noise recording of it, where in it to start, and the signal-to-noise ratio."""
    pool = entries
    for key in ('device', 'room', 'position', 'channel'):
        values = sorted({e[key] for e in pool})
        value = values[rng.integers(len(values))]
        pool = [e for e in pool if e[key] == value]
    entry = pool[0]
    kind = noises[rng.integers(len(noises))]
    return {
        **{k: entry[k] for k in ('device', 'room', 'position', 'channel', 'rir')},
        'noise': kind,
        'noise_file': entry['noises'][kind],
        'noise_start': float(rng.random()),
        'snr_db': float(rng.uniform(*snr_db)),
    }


def read_channel(path, channel, sr, start=0, frames=-1):
    x, file_sr = sf.read(path, start=start, frames=frames, dtype='float32', always_2d=True)
    return librosa.resample(x[:, channel], orig_sr=file_sr, target_sr=sr) if file_sr != sr else x[:, channel]


def response(path, channel, sr, length_s, pre_s=0.001):
    """The measured response from `pre_s` before its direct sound (its first sample within 6 dB of the peak)."""
    h = read_channel(path, channel, sr)
    first = int(np.argmax(np.abs(h) >= 0.5 * np.abs(h).max()))
    start = max(first - round(pre_s * sr), 0)
    h = h[start : start + round(length_s * sr)]
    fade = min(len(h), round(0.01 * sr))
    h[-fade:] *= np.hanning(2 * fade)[fade:]
    return h


def arrival(h, sr):
    """Seconds from a response's start to its direct sound."""
    return int(np.argmax(np.abs(h) >= 0.5 * np.abs(h).max())) / sr


def noise(path, channel, sr, n, start):
    """`n` samples of a noise recording from fraction `start` of it, looped when it is shorter."""
    info = sf.info(path)
    need = int(np.ceil(n / sr * info.samplerate)) + 64
    if need >= info.frames:
        x = read_channel(path, channel, sr)
        return np.tile(x, n // len(x) + 1)[:n]
    return read_channel(path, channel, sr, int(start * (info.frames - need)), need)[:n]


def active_rms(y, sr, frame_s=0.1):
    """RMS over the louder half of the 100 ms frames, so pauses don't lower a recording's level."""
    k = round(frame_s * sr)
    power = np.mean(y[: len(y) // k * k].reshape(-1, k) ** 2, axis=1)
    return float(np.sqrt(power[power >= np.median(power)].mean()))


def match_filter(pairs, sr, n_fft, smooth_octaves, top_hz, floor_db):
    """Minimum-phase FIR with the magnitude of the linear transfer |Σ P_xy| / Σ P_xx over (x, y) pairs (Welch
    spectra), smoothed over `smooth_octaves` and floored `floor_db` under its peak: the average linear colouring
    that turns each x into its y. Energy in y that x does not explain (hiss, distortion) is left out. Above `top_hz`,
    where a guitar's direct signal has too little energy to measure it, it falls by 24 dB per octave."""
    pxx, pxy = 0.0, 0.0
    for x, y in pairs:
        n = min(len(x), len(y))
        f, a = scipy.signal.welch(x[:n], sr, nperseg=n_fft)
        _, b = scipy.signal.csd(x[:n], y[:n], sr, nperseg=n_fft)
        pxx, pxy = pxx + a, pxy + b
    power = np.abs(pxy) ** 2 / (pxx**2 + 1e-30)
    lo, hi = f * 2 ** (-smooth_octaves / 2), f * 2 ** (smooth_octaves / 2)
    smooth = np.array([power[(f >= a) & (f <= b)].mean() for a, b in zip(lo, hi, strict=True)])
    smooth = np.where(f > top_hz, np.interp(top_hz, f, smooth) * (np.maximum(f, top_hz) / top_hz) ** -8.0, smooth)
    mag = np.sqrt(np.maximum(smooth, smooth.max() * 10 ** (floor_db / 10)))
    mag /= mag.max()
    cepstrum = np.fft.irfft(np.log(mag), n_fft)
    fold = np.zeros(n_fft)
    fold[0], fold[n_fft // 2] = cepstrum[0], cepstrum[n_fft // 2]
    fold[1 : n_fft // 2] = 2 * cepstrum[1 : n_fft // 2]
    h = np.fft.irfft(np.exp(np.fft.rfft(fold)), n_fft)[: n_fft // 2]
    h[-n_fft // 8 :] *= np.hanning(n_fft // 4)[n_fft // 8 :]
    return h.astype(np.float32)


def render(y, sr, p, cabinet, length_s):
    """(audio, seconds its notes moved): `y` through the cabinet (if given) and the drawn room, plus the room's noise
    at the drawn SNR against the active level, peak-normalised to 0.9."""
    h = response(p['rir'], p['channel'], sr, length_s)
    if cabinet is not None:
        h = scipy.signal.fftconvolve(cabinet, h)
    out = scipy.signal.fftconvolve(y, h)[: len(y)]
    n = noise(p['noise_file'], p['channel'], sr, len(out), p['noise_start'])
    out = out + n / (np.sqrt(np.mean(n**2)) + 1e-12) * active_rms(out, sr) * 10 ** (-p['snr_db'] / 20)
    return (out / (np.abs(out).max() + 1e-9) * 0.9).astype(np.float32), arrival(h, sr)

"""Random recording chains (pickup → amp → cabinet → room → mic) for string-classifier augmentation.

A chain is drawn from a seed, so a render is reproducible and its parameters can be shown or logged.
DI recordings get the whole chain; recordings already made through an amp or a mic get only the
room and mic stages, since they already carry an amp or body sound. A far-field chain swaps the light studio
reverb (`room`) for `far`: a microphone a metre or two from the source in an untreated room, with its noise.
"""

import numpy as np
import pedalboard as pb
import scipy.signal

DI_STAGES = ('pickup', 'amp', 'cabinet', 'room', 'mic')
MIC_STAGES = ('room', 'mic')
# The highest filter corner draw() can produce; audio sampled too low for it is rendered at twice its rate.
MAX_CORNER_HZ = 12000


def draw(rng, stages):
    """{stage: params} for one chain; a stage absent from the dict was skipped."""
    u = rng.uniform
    p = {}
    if 'pickup' in stages:
        p['pickup'] = {'resonance_hz': u(1500, 5000), 'resonance_db': u(0, 8), 'cable_lowpass_hz': u(4000, 12000)}
    if 'amp' in stages:
        drive = rng.random() < 0.6
        p['amp'] = {
            'drive_db': u(10, 35) if drive else u(0, 6),
            'bass_db': u(-6, 6),
            'mid_hz': u(400, 1500),
            'mid_db': u(-8, 6),
            'treble_db': u(-6, 6),
        }
    if 'cabinet' in stages:
        p['cabinet'] = {'highpass_hz': u(60, 120), 'lowpass_hz': u(3500, 7000), 'presence_db': u(-3, 6)}
    if 'room' in stages and rng.random() < 0.7:
        p['room'] = {'size': u(0.1, 0.8), 'damping': u(0.2, 0.8), 'wet': u(0.05, 0.4)}
    if 'far' in stages:
        p['far'] = {
            'rt60_s': u(0.15, 0.7),
            'direct_db': u(-3, 12),
            'tail_lowpass_hz': u(2000, 8000),
            'snr_db': u(8, 30),
        }
    if 'mic' in stages:
        p['mic'] = {
            'highpass_hz': u(60, 300),
            'lowpass_hz': u(6000, 11000),
            'resonance_hz': u(1000, 6000),
            'resonance_db': u(-6, 6),
            'compress': bool(rng.random() < 0.5),
            'snr_db': u(20, 50),
        }
    return p


def board(p):
    fx = []
    if 'pickup' in p:
        q = p['pickup']
        fx += [pb.PeakFilter(q['resonance_hz'], q['resonance_db'], 1.5), pb.LowpassFilter(q['cable_lowpass_hz'])]
    if 'amp' in p:
        q = p['amp']
        fx += [
            pb.Distortion(q['drive_db']),
            pb.LowShelfFilter(200, q['bass_db']),
            pb.PeakFilter(q['mid_hz'], q['mid_db'], 0.8),
            pb.HighShelfFilter(3000, q['treble_db']),
        ]
    if 'cabinet' in p:
        q = p['cabinet']
        fx += [
            pb.HighpassFilter(q['highpass_hz']),
            pb.LowpassFilter(q['lowpass_hz']),
            pb.LowpassFilter(q['lowpass_hz']),
            pb.PeakFilter(2500, q['presence_db'], 1.0),
        ]
    if 'room' in p:
        q = p['room']
        fx.append(pb.Reverb(q['size'], q['damping'], q['wet'], 1 - q['wet']))
    if 'mic' in p:
        q = p['mic']
        fx += [
            pb.HighpassFilter(q['highpass_hz']),
            pb.LowpassFilter(q['lowpass_hz']),
            pb.PeakFilter(q['resonance_hz'], q['resonance_db'], 2.0),
        ]
        if q['compress']:
            fx.append(pb.Compressor(-24, 4, 5, 100))
    return pb.Pedalboard(fx)


def impulse_response(sr, q, rng):
    """A room's response at a distant microphone: the direct sound, then a diffuse tail that decays by 60 dB in
    rt60_s and holds 10^(-direct_db / 10) of the direct sound's energy."""
    n = int(sr * 1.2 * q['rt60_s'])
    t = np.arange(n) / sr
    tail = (rng.standard_normal(n) * np.exp(-6.908 * t / q['rt60_s'])).astype(np.float32)
    tail = pb.LowpassFilter(q['tail_lowpass_hz'])(tail[None], sr)[0]
    tail[: int(0.003 * sr)] = 0
    h = tail * np.sqrt(10 ** (-q['direct_db'] / 10) / (np.sum(tail**2) + 1e-12))
    h[0] += 1.0
    return h


def pink_noise(n, rng):
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
    noise = np.fft.irfft(spectrum, n)
    return (noise / (noise.std() + 1e-12)).astype(np.float32)


def render(audio, sr, p, rng):
    """Mono float32 audio through chain `p`; noise for the far and mic stages, peak-normalised so drive and
    reverb don't shift the level the classifier sees."""
    y = audio.astype(np.float32)
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    over = 2 if sr < 2.2 * MAX_CORNER_HZ else 1
    if over > 1:
        y = scipy.signal.resample_poly(y, over, 1).astype(np.float32)
    y = board({k: v for k, v in p.items() if k != 'mic'})(y[None], over * sr)[0]
    if 'far' in p:
        h = impulse_response(over * sr, p['far'], rng)
        y = scipy.signal.fftconvolve(y, h)[: len(y)].astype(np.float32)
    if 'mic' in p:
        y = board({'mic': p['mic']})(y[None], over * sr)[0]
    if over > 1:
        y = scipy.signal.resample_poly(y, 1, over)[: len(audio)].astype(np.float32)
    rms = np.sqrt(np.mean(y**2)) + 1e-9
    if 'far' in p:
        y = y + pink_noise(len(y), rng) * rms * 10 ** (-p['far']['snr_db'] / 20)
    if 'mic' in p:
        y = y + rng.standard_normal(len(y)).astype(np.float32) * rms * 10 ** (-p['mic']['snr_db'] / 20)
    return y / (np.abs(y).max() + 1e-9) * 0.9


def stages_for(chain, far=False):
    stages = DI_STAGES if chain == 'di' else MIC_STAGES
    return tuple('far' if far and k == 'room' else k for k in stages)

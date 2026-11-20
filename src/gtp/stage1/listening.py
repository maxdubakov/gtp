"""Clips for listening pages: a stretch of a recording around a note, and its spectrogram with the note marked."""

import librosa
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def clip(y, sr, onset, before_s, after_s):
    """(samples, start time): the recording from `before_s` before the onset to `after_s` after, peak-normalised."""
    start = max(onset - before_s, 0.0)
    seg = y[int(start * sr) : int((onset + after_s) * sr)]
    ramp = np.minimum(1, np.minimum(np.arange(len(seg)), np.arange(len(seg))[::-1]) / (0.01 * sr))
    return seg / max(np.abs(seg).max(), 1e-6) * 0.9 * ramp, start


def spectrogram(y, sr, note, start, cfg, path):
    """Save the clip's log-frequency spectrogram with the note boxed; returns the plot area as fractions of the image."""
    fmin = librosa.midi_to_hz(cfg.fmin_midi)
    bins = (cfg.fmax_midi - cfg.fmin_midi) * 3
    c = librosa.amplitude_to_db(
        np.abs(librosa.cqt(y, sr=sr, hop_length=256, fmin=fmin, n_bins=bins, bins_per_octave=36)), ref=np.max
    )
    t = np.arange(c.shape[1]) * 256 / sr + start - note.onset
    ink = '#c9ceda'
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=110, facecolor='#14161c')
    ax.tick_params(colors=ink)
    ax.xaxis.label.set_color(ink)
    for side in ax.spines.values():
        side.set_color('#3a3f4b')
    ax.imshow(c, origin='lower', aspect='auto', cmap='magma', vmin=-70, vmax=0,
              extent=[t[0], t[-1], cfg.fmin_midi - 1 / 6, cfg.fmax_midi - 1 / 6])  # fmt: skip
    p, d = note.pitch, max(note.offset - note.onset, 0.05)
    ax.add_patch(plt.Rectangle((0, p - 0.5), d, 1, fill=False, ec='#39d0ff', lw=1.6))
    ax.axvline(0, color='#39d0ff', lw=0.8, ls=':')
    ticks = [m for m in range(cfg.fmin_midi, cfg.fmax_midi + 1) if m % 12 == 4]
    ax.set_yticks(ticks, [librosa.midi_to_note(m, unicode=False) for m in ticks])
    ax.set_xlabel('seconds from the note')
    ax.set_xlim(-cfg.before_s, cfg.after_s)
    fig.tight_layout()
    fig.savefig(path)
    box = ax.get_position()
    plt.close(fig)
    return {'x0': round(box.x0, 4), 'x1': round(box.x1, 4), 'y0': round(1 - box.y1, 4), 'y1': round(1 - box.y0, 4)}

"""Playing-technique labels of notes."""

import librosa
import numpy as np

from gtp.stage1.strings import SAMPLE_RATE


def class_of(techniques, classes, other_marks):
    """Index into `classes` per note, or -1 for a note to leave out."""
    out = np.full(len(techniques), -1)
    for i, marks in enumerate(techniques.fillna('').str.split(',')):
        named = [classes.index(m) for m in marks if m in classes]
        if named:
            out[i] = max(named)
        elif not set(marks) & set(other_marks):
            out[i] = classes.index('plain')
    return out


def head_labels(notes, heads, unlabelled):
    """{head: class index per note}, -1 where the head has no label for the note.

    A head's class is the last of its `classes` one of whose marks the note carries, the first class for a note
    with none; a note with a mark in the head's `unknown` has no label. Each rule of `unlabelled` removes the
    labels of its `heads` from the notes of a `dataset` (and `content`, and carrying one of `marks`, where given).
    """
    marks = [set(t.split(',')) - {''} if isinstance(t, str) else set() for t in notes['techniques']]

    def carrying(names):
        return np.array([bool(m & set(names)) for m in marks])

    out = {}
    for name, head in heads.items():
        label = np.zeros(len(notes), dtype=np.int64)
        for k, of in enumerate(head.classes.values()):
            label[carrying(of)] = k
        label[carrying(head.unknown)] = -1
        out[name] = label
    for rule in unlabelled:
        hit = (notes['dataset'] == rule.dataset).to_numpy()
        if rule.get('content'):
            hit = hit & (notes['content'] == rule.content).to_numpy()
        if rule.get('marks'):
            hit = hit & carrying(rule.marks)
        for name in rule.heads:
            out[name][hit] = -1
    return out


def pitch_comb(n_mels, harmonics):
    """(128, n_mels) table: for each MIDI pitch, ones at the log-mel bins (gtp.stage1.strings.spectrogram) of its
    first `harmonics` partials. As an input channel it shows a network which of the sounding notes is meant."""
    centres = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=30.0, fmax=SAMPLE_RATE / 2)[1:-1]
    table = np.zeros((128, n_mels), dtype=np.float32)
    for pitch in range(128):
        partials = librosa.midi_to_hz(pitch) * np.arange(1, harmonics + 1)
        partials = partials[(partials > centres[0]) & (partials < centres[-1])]
        table[pitch, np.abs(centres[None, :] - partials[:, None]).argmin(axis=1)] = 1
    return table

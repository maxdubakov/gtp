"""Re-time note labels to the audio with a Stage 1 checkpoint, as NoteEM's alignment step does.

Per recording: the labels shift by the median offset of their onsets from the onsets the checkpoint detects (same
pitch, within `lag_window_s`); then each labelled onset moves to a detected onset of its pitch within `max_shift_s`,
one detection per label (mir_eval's matching). Offsets shift with the recording, not per note. Pitches and the notes
kept are unchanged.
"""

import librosa
import mir_eval
import numpy as np

from gtp.stage1.postprocess import RegressionPostProcessor


def detected(model, audio, onset_threshold):
    """(m, 2) onset and MIDI pitch of the notes `model` finds in 16 kHz `audio`."""
    processor = RegressionPostProcessor(100, 88, onset_threshold=onset_threshold)
    events, _ = processor.output_dict_to_note_events(model._run_model(audio))
    return np.array([[v['onset_time'], v['midi_note']] for v in events], float).reshape(-1, 2)


def pairs(ref_onsets, ref_pitch, est_onsets, est_pitch, tolerance):
    def arrays(onsets, pitch):
        return np.c_[onsets, onsets + 0.1], librosa.midi_to_hz(pitch)

    return mir_eval.transcription.match_notes(*arrays(ref_onsets, ref_pitch), *arrays(est_onsets, est_pitch),
                                              onset_tolerance=tolerance, offset_ratio=None)  # fmt: skip


def retime(notes, est, lag_window_s, max_shift_s):
    """`notes` (n, 3: onset, offset, pitch) re-timed to `est`; with the recording's lag in s and the share moved."""
    on, pitch = notes[:, 0], notes[:, 2]
    est_on, est_pitch = est[:, 0], est[:, 1]
    matched = pairs(on, pitch, est_on, est_pitch, lag_window_s)
    lag = float(np.median([est_on[j] - on[i] for i, j in matched])) if matched else 0.0
    new_on = on + lag
    moved = pairs(new_on, pitch, est_on, est_pitch, max_shift_s)
    for i, j in moved:
        new_on[i] = est_on[j]
    return np.c_[new_on, np.maximum(notes[:, 1] + lag, new_on + 0.01), pitch], lag, len(moved) / max(len(on), 1)

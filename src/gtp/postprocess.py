# Adapted from https://github.com/bytedance/piano_transcription
# Original: piano_transcription-master/utils/utilities.py and piano_vad.py

import numpy as np

from gtp.log import trace

# These constants mirror those in inference.py and the pretrained checkpoint config
BEGIN_NOTE = 21       # A0, lowest piano MIDI note
VELOCITY_SCALE = 128  # Map [0,1] normalized velocity to MIDI 0-127


def note_detection_with_onset_offset_regress(
    frame_output,
    onset_output,
    onset_shift_output,
    offset_output,
    offset_shift_output,
    velocity_output,
    frame_threshold,
):
    """Convert per-pitch activation arrays to a list of (onset, offset, onset_shift,
    offset_shift, velocity) tuples for a single pitch class.

    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,) binarized, values 0 or 1
      onset_shift_output: (frames_num,) sub-frame onset correction in [-0.5, 0.5]
      offset_output: (frames_num,) binarized, values 0 or 1
      offset_shift_output: (frames_num,) sub-frame offset correction
      velocity_output: (frames_num,) normalized velocity in [0, 1]
      frame_threshold: float

    Returns:
      list of [bgn_frame, fin_frame, onset_shift, offset_shift, normalized_velocity]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            if bgn:
                # Consecutive onset: close the previous note immediately before this one
                fin = max(i - 1, 0)
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 0, velocity_output[bgn]])
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn and i > bgn:
            if frame_output[i] <= frame_threshold and not frame_disappear:
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                offset_occur = i

            if frame_disappear:
                if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                    # offset occurred closer to frame_disappear than to bgn: use frame_disappear
                    fin = offset_occur
                else:
                    fin = frame_disappear
                output_tuples.append([bgn, fin, onset_shift_output[bgn],
                                       offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                # No offset found within 6 seconds or reached end of audio
                fin = i
                output_tuples.append([bgn, fin, onset_shift_output[bgn],
                                       offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

    output_tuples.sort(key=lambda t: t[0])
    return output_tuples


def pedal_detection_with_onset_offset_regress(
    frame_output, offset_output, offset_shift_output, frame_threshold
):
    """Convert pedal activation arrays to (onset, offset, onset_shift, offset_shift) tuples."""
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            if not bgn:
                bgn = i

        if bgn and i > bgn:
            if frame_output[i] <= frame_threshold and not frame_disappear:
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                offset_occur = i

            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

            if frame_disappear and i - frame_disappear >= 10:
                fin = frame_disappear
                output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

    output_tuples.sort(key=lambda t: t[0])
    return output_tuples


class RegressionPostProcessor:
    """Convert model output activations into discrete MIDI note events.

    Uses Kong et al.'s high-resolution regression approach: onset/offset
    regression outputs are binarized by finding local maxima above threshold,
    then sub-frame precision is recovered via parabolic interpolation (the
    onset_shift / offset_shift values).
    """

    def __init__(
        self,
        frames_per_second,
        classes_num,
        onset_threshold=0.3,
        offset_threshold=0.3,
        frame_threshold=0.1,
        pedal_offset_threshold=0.2,
    ):
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.pedal_offset_threshold = pedal_offset_threshold
        self.begin_note = BEGIN_NOTE
        self.velocity_scale = VELOCITY_SCALE

    def output_dict_to_note_events(self, output_dict):
        """Main entry point. Convert model output_dict to a list of note event dicts.

        Args:
          output_dict: {
            'reg_onset_output':       (frames_num, classes_num),
            'reg_offset_output':      (frames_num, classes_num),
            'frame_output':           (frames_num, classes_num),
            'velocity_output':        (frames_num, classes_num),
            'reg_pedal_onset_output': (frames_num, 1),   # optional
            'reg_pedal_offset_output':(frames_num, 1),   # optional
            'pedal_frame_output':     (frames_num, 1),   # optional
          }

        Returns:
          note_events: list of dict, e.g.
            [{'onset_time': 0.51, 'offset_time': 0.82, 'midi_note': 45, 'velocity': 72}, ...]
          pedal_events: list of dict or None
        """
        onset_binary, onset_shift = self._binarize_regression(
            output_dict['reg_onset_output'], self.onset_threshold, neighbour=2
        )
        offset_binary, offset_shift = self._binarize_regression(
            output_dict['reg_offset_output'], self.offset_threshold, neighbour=4
        )

        trace("onset binarized", onset_binary, detections=int(onset_binary.sum()),
              threshold=self.onset_threshold)
        trace("offset binarized", offset_binary, detections=int(offset_binary.sum()),
              threshold=self.offset_threshold)

        augmented = dict(output_dict)
        augmented['onset_output'] = onset_binary
        augmented['onset_shift_output'] = onset_shift
        augmented['offset_output'] = offset_binary
        augmented['offset_shift_output'] = offset_shift

        # Process pedal offsets if present
        if 'reg_pedal_offset_output' in output_dict:
            pedal_offset_binary, pedal_offset_shift = self._binarize_regression(
                output_dict['reg_pedal_offset_output'], self.pedal_offset_threshold, neighbour=4
            )
            augmented['pedal_offset_output'] = pedal_offset_binary
            augmented['pedal_offset_shift_output'] = pedal_offset_shift

        est_on_off_note_vels = self._detect_notes(augmented)
        note_events = self._note_arrays_to_events(est_on_off_note_vels)

        if 'reg_pedal_onset_output' in output_dict:
            est_pedal_on_offs = self._detect_pedals(augmented)
            pedal_events = self._pedal_arrays_to_events(est_pedal_on_offs)
        else:
            pedal_events = None

        return note_events, pedal_events

    def _binarize_regression(self, reg_output, threshold, neighbour):
        """Find local maxima in reg_output above threshold and compute sub-frame shifts.

        The shift is derived from parabolic interpolation of the peak and its
        two immediate neighbours — see Section III-D of Kong et al. 2020.

        Args:
          reg_output: (frames_num, classes_num)
          threshold: float
          neighbour: int, number of frames on each side that must be monotonically
            increasing toward the peak

        Returns:
          binary_output: (frames_num, classes_num) values in {0, 1}
          shift_output:  (frames_num, classes_num) sub-frame correction in (-1, 1)
        """
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        frames_num, classes_num = reg_output.shape

        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self._is_local_max(x, n, neighbour):
                    binary_output[n, k] = 1
                    if x[n - 1] > x[n + 1]:
                        shift_output[n, k] = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift_output[n, k] = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2

        return binary_output, shift_output

    def _is_local_max(self, x, n, neighbour):
        """Return True if x[n] is a local maximum with monotonic flanks of width neighbour."""
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                return False
            if x[n + i] < x[n + i + 1]:
                return False
        return True

    def _detect_notes(self, augmented):
        """Run note detection across all pitch classes.

        Returns:
          (notes, 4) array: [onset_time, offset_time, midi_note, velocity]
        """
        est_tuples = []
        est_midi_notes = []

        for piano_note in range(self.classes_num):
            tuples = note_detection_with_onset_offset_regress(
                frame_output=augmented['frame_output'][:, piano_note],
                onset_output=augmented['onset_output'][:, piano_note],
                onset_shift_output=augmented['onset_shift_output'][:, piano_note],
                offset_output=augmented['offset_output'][:, piano_note],
                offset_shift_output=augmented['offset_shift_output'][:, piano_note],
                velocity_output=augmented['velocity_output'][:, piano_note],
                frame_threshold=self.frame_threshold,
            )
            est_tuples += tuples
            est_midi_notes += [piano_note + self.begin_note] * len(tuples)

        if len(est_tuples) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        est_tuples = np.array(est_tuples)           # (notes, 5)
        est_midi_notes = np.array(est_midi_notes)   # (notes,)

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]

        return np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1).astype(np.float32)

    def _detect_pedals(self, augmented):
        """Run pedal detection. Returns (pedal_events, 2) array or empty."""
        est_tuples = pedal_detection_with_onset_offset_regress(
            frame_output=augmented['pedal_frame_output'][:, 0],
            offset_output=augmented['pedal_offset_output'][:, 0],
            offset_shift_output=augmented['pedal_offset_shift_output'][:, 0],
            frame_threshold=0.5,
        )
        if len(est_tuples) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        est_tuples = np.array(est_tuples)
        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        return np.stack((onset_times, offset_times), axis=-1).astype(np.float32)

    def _note_arrays_to_events(self, est_on_off_note_vels):
        """Convert (notes, 4) array to list of note event dicts."""
        events = []
        for row in est_on_off_note_vels:
            events.append({
                'onset_time': float(row[0]),
                'offset_time': float(row[1]),
                'midi_note': int(row[2]),
                'velocity': int(row[3] * self.velocity_scale),
            })
        return events

    def _pedal_arrays_to_events(self, pedal_on_offs):
        """Convert (pedal_events, 2) array to list of pedal event dicts."""
        return [
            {'onset_time': float(row[0]), 'offset_time': float(row[1])}
            for row in pedal_on_offs
        ]


def write_events_to_midi(note_events, midi_path, pedal_events=None, start_time=0.0):
    """Write note (and optionally pedal) events to a MIDI file.

    Uses the same tempo/tick configuration as the MAESTRO dataset MIDIs
    (120 BPM, 384 ticks/beat) to match Kong's original output format.

    Args:
      note_events: list of dict with keys onset_time, offset_time, midi_note, velocity
      midi_path: str, output path
      pedal_events: list of dict with onset_time, offset_time, or None
      start_time: float, subtract this offset from all event times
    """
    from mido import Message, MidiFile, MidiTrack, MetaMessage

    ticks_per_beat = 384
    beats_per_second = 2  # 120 BPM
    ticks_per_second = ticks_per_beat * beats_per_second
    microseconds_per_beat = int(1e6 // beats_per_second)

    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat

    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)

    track1 = MidiTrack()
    message_roll = []

    for event in note_events:
        message_roll.append({'time': event['onset_time'], 'midi_note': event['midi_note'],
                              'velocity': event['velocity']})
        message_roll.append({'time': event['offset_time'], 'midi_note': event['midi_note'],
                              'velocity': 0})

    if pedal_events:
        for event in pedal_events:
            message_roll.append({'time': event['onset_time'], 'control_change': 64, 'value': 127})
            message_roll.append({'time': event['offset_time'], 'control_change': 64, 'value': 0})

    message_roll.sort(key=lambda m: m['time'])

    previous_ticks = 0
    for message in message_roll:
        this_ticks = int((message['time'] - start_time) * ticks_per_second)
        if this_ticks >= 0:
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            if 'midi_note' in message:
                track1.append(Message('note_on', note=message['midi_note'],
                                       velocity=message['velocity'], time=diff_ticks))
            elif 'control_change' in message:
                track1.append(Message('control_change', channel=0,
                                       control=message['control_change'],
                                       value=message['value'], time=diff_ticks))

    track1.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track1)
    midi_file.save(midi_path)

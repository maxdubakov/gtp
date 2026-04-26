"""Guitar transcription data pipeline.

Loads GAPS and GuitarSet audio + annotations into a PyTorch Dataset that
returns 10-second segments with the target roll format expected by Kong's
training loop.
"""

import csv
import os

import jams
import librosa
import numpy as np
import pretty_midi
from torch.utils.data import Dataset

from gtp.log import trace

# Piano MIDI range: A0 (21) to C8 (108), 88 keys.
BEGIN_NOTE = 21
CLASSES_NUM = 88

# GuitarSet: note_midi annotations alternate with pitch_contour at indices 1,3,5,7,9,11.
_GUITARSET_NOTE_MIDI_INDICES = [1, 3, 5, 7, 9, 11]


# ---------------------------------------------------------------------------
# TargetProcessor (adapted from Kong et al., pedal handling removed)
# ---------------------------------------------------------------------------

class TargetProcessor:
    """Convert note events into target rolls for training.

    Based on Kong et al. "High-resolution Piano Transcription with Pedals by
    Regressing Onsets and Offsets Times" (2020). Pedal logic removed because
    guitar has no sustain pedal.

    Input note events are pre-parsed dicts:
        {'onset_time': float, 'offset_time': float, 'midi_note': int, 'velocity': int}
    """

    def __init__(self, segment_seconds=10.0, frames_per_second=100,
                 begin_note=BEGIN_NOTE, classes_num=CLASSES_NUM):
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num
        self.max_piano_note = classes_num - 1

    def process_notes(self, start_time, notes):
        """Build target rolls for a segment from a list of note events.

        Notes that began before `start_time` (cross-segment notes) are
        masked out in mask_roll so the model is not penalised for them.

        Args:
            start_time: float, segment start in seconds (absolute)
            notes: list of {'onset_time', 'offset_time', 'midi_note', 'velocity'}

        Returns:
            dict with keys: onset_roll, offset_roll, reg_onset_roll,
            reg_offset_roll, frame_roll, velocity_roll, mask_roll
            each of shape (frames_num, classes_num)
        """
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        end_time = start_time + self.segment_seconds

        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        # reg rolls initialised to 1 (the "no event" sentinel used by get_regression)
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))

        # Filter to notes that overlap this segment
        segment_notes = [
            n for n in notes
            if n['offset_time'] > start_time and n['onset_time'] < end_time
        ]

        trace("process_notes", notes=len(segment_notes),
              start_time=start_time, end_time=end_time)

        for note in segment_notes:
            piano_note = np.clip(
                note['midi_note'] - self.begin_note, 0, self.max_piano_note
            )

            bgn_frame = int(round(
                (note['onset_time'] - start_time) * self.frames_per_second
            ))
            fin_frame = int(round(
                (note['offset_time'] - start_time) * self.frames_per_second
            ))
            fin_frame = min(fin_frame, frames_num - 1)

            note_extends_past_end = note['offset_time'] > end_time

            if fin_frame >= 0:
                frame_roll[max(bgn_frame, 0):fin_frame + 1, piano_note] = 1
                velocity_roll[max(bgn_frame, 0):fin_frame + 1, piano_note] = (
                    note['velocity']
                )

                if not note_extends_past_end:
                    offset_roll[fin_frame, piano_note] = 1
                    reg_offset_roll[fin_frame, piano_note] = (
                        (note['offset_time'] - start_time)
                        - (fin_frame / self.frames_per_second)
                    )

                if bgn_frame >= 0:
                    onset_roll[bgn_frame, piano_note] = 1
                    reg_onset_roll[bgn_frame, piano_note] = (
                        (note['onset_time'] - start_time)
                        - (bgn_frame / self.frames_per_second)
                    )
                else:
                    # Note started before segment: mask those frames out
                    mask_roll[:fin_frame + 1, piano_note] = 0

        # Mask notes whose offset extends past the segment end: the model
        # should not be penalised for a missing offset it cannot observe.
        for note in segment_notes:
            if note['offset_time'] > end_time:
                piano_note = np.clip(
                    note['midi_note'] - self.begin_note, 0, self.max_piano_note
                )
                bgn_frame = int(round(
                    (note['onset_time'] - start_time) * self.frames_per_second
                ))
                if bgn_frame >= 0:
                    mask_roll[bgn_frame:, piano_note] = 0

        # Apply Gaussian-like smoothing to build regression targets
        for k in range(self.classes_num):
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        return {
            'onset_roll': onset_roll,
            'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll,
            'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll,
            'velocity_roll': velocity_roll,
            'mask_roll': mask_roll,
        }

    def get_regression(self, input):
        """Produce a smooth triangular regression target centred on each event.

        For frames with no event the output is 0. Frames near an event ramp
        up to 1 (at the event frame) then back down, clipped to a ±0.05s
        window scaled to [0, 1].  See Fig. 2 of Kong et al. 2020.

        Args:
            input: (frames_num,) array; event frames have values < 0.5
                   (sub-frame offset in seconds); non-event frames are 1.0

        Returns:
            (frames_num,) array in [0, 1]
        """
        step = 1.0 / self.frames_per_second
        output = np.ones_like(input)

        locts = np.where(input < 0.5)[0]
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]
                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0.0, 0.05) * 20
        output = 1.0 - output
        return output


# ---------------------------------------------------------------------------
# Annotation loaders
# ---------------------------------------------------------------------------

def load_gaps_notes(midi_path):
    """Parse a GAPS MIDI file into a flat list of note dicts.

    GAPS MIDI files have uniform velocity (all 100), so velocity is kept
    as-is (the velocity head is low-priority; see task brief).

    Returns:
        list of {'onset_time', 'offset_time', 'midi_note', 'velocity'}
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            notes.append({
                'onset_time': float(note.start),
                'offset_time': float(note.end),
                'midi_note': int(note.pitch),
                'velocity': int(note.velocity),
            })
    trace("load_gaps_notes", notes=len(notes), path=midi_path)
    return notes


def load_guitarset_notes(jams_path):
    """Parse a GuitarSet JAMS file into a flat list of note dicts.

    Combines notes from all 6 strings. MIDI pitch values are fractional
    (microtonal); they are rounded to the nearest semitone.  Confidence
    fields are typically None, so velocity defaults to 64.

    Returns:
        list of {'onset_time', 'offset_time', 'midi_note', 'velocity'}
    """
    score = jams.load(jams_path)
    notes = []

    for ann_idx in _GUITARSET_NOTE_MIDI_INDICES:
        ann = score.annotations[ann_idx]
        for obs in ann.data:
            onset = float(obs.time)
            duration = float(obs.duration)
            midi_note = int(round(float(obs.value)))
            # confidence is typically None in GuitarSet; fall back to 64
            velocity = int(obs.confidence) if obs.confidence is not None else 64
            notes.append({
                'onset_time': onset,
                'offset_time': onset + duration,
                'midi_note': midi_note,
                'velocity': velocity,
            })

    notes.sort(key=lambda n: n['onset_time'])
    trace("load_guitarset_notes", notes=len(notes), path=jams_path)
    return notes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GuitarDataset(Dataset):
    """PyTorch Dataset for guitar transcription training.

    Each item is a 10-second audio segment with corresponding target rolls.
    All valid start positions are pre-enumerated with a 1-second stride.

    Args:
        items: list of (audio_path, notes) where notes is a list of note
               dicts as returned by load_gaps_notes / load_guitarset_notes
        segment_seconds: segment duration in seconds (default 10)
        frames_per_second: frame rate for target rolls (default 100)
        sample_rate: target audio sample rate in Hz (default 16000)
    """

    def __init__(self, items, segment_seconds=10.0, frames_per_second=100,
                 sample_rate=16000):
        self.items = items
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)

        self.target_processor = TargetProcessor(
            segment_seconds=segment_seconds,
            frames_per_second=frames_per_second,
        )

        # Build flat segment index: [(audio_path, notes, start_time), ...]
        self.segments = []
        hop_seconds = 1.0
        for audio_path, notes in items:
            duration = self._audio_duration(audio_path)
            start = 0.0
            while start + segment_seconds <= duration:
                self.segments.append((audio_path, notes, start))
                start += hop_seconds

        trace("GuitarDataset init",
              files=len(items), segments=len(self.segments))

    def _audio_duration(self, audio_path):
        """Return audio duration in seconds without loading the full file."""
        info = librosa.get_duration(path=audio_path)
        return info

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        audio_path, notes, start_time = self.segments[idx]

        trace("__getitem__", idx=idx, audio_path=audio_path,
              start_time=start_time)

        # Load the 10-second window, resampling to 16 kHz
        offset = start_time
        waveform, _ = librosa.load(
            audio_path,
            sr=self.sample_rate,
            offset=offset,
            duration=self.segment_seconds,
            mono=True,
        )

        # Pad if the last segment is slightly short (edge of file)
        if len(waveform) < self.segment_samples:
            waveform = np.pad(
                waveform, (0, self.segment_samples - len(waveform))
            )

        trace("waveform", waveform)

        targets = self.target_processor.process_notes(start_time, notes)

        for key, arr in targets.items():
            trace(key, arr)

        return {
            'waveform': waveform.astype(np.float32),
            **{k: v.astype(np.float32) for k, v in targets.items()},
        }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(gaps_dir, guitarset_dir, split='train',
                  segment_seconds=10.0, frames_per_second=100,
                  sample_rate=16000):
    """Build a GuitarDataset from GAPS and GuitarSet for the given split.

    GAPS split: determined by `split` column in gaps_metadata_with_splits.csv.
    GuitarSet split: player-based.
      train      → players 00–04
      validation → player 05
      test       → player 05  (same as validation; GuitarSet has no held-out test set)

    Args:
        gaps_dir: path to GAPS data root (contains audio/, midi/, gaps_metadata_with_splits.csv)
        guitarset_dir: path to GuitarSet root (contains audio_mono-mic/, annotation/)
        split: 'train' | 'validation' | 'test'
        segment_seconds: segment duration (default 10)
        frames_per_second: frame rate (default 100)
        sample_rate: target audio sample rate (default 16000)

    Returns:
        GuitarDataset
    """
    assert split in ('train', 'validation', 'test'), \
        f"split must be 'train', 'validation', or 'test'; got {split!r}"

    items = []

    # ------ GAPS ------
    meta_path = os.path.join(gaps_dir, 'gaps_metadata_with_splits.csv')
    gaps_split_label = 'train' if split == 'train' else 'test'

    split_counts: dict[str, int] = {}
    with open(meta_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['split'] if row['split'] else 'excluded'
            split_counts[label] = split_counts.get(label, 0) + 1
            if row['split'] != gaps_split_label:
                continue
            audio_path = os.path.join(gaps_dir, row['audio_path'])
            midi_path = os.path.join(gaps_dir, row['midi_path'])
            if not os.path.exists(audio_path) or not os.path.exists(midi_path):
                continue
            notes = load_gaps_notes(midi_path)
            items.append((audio_path, notes))

    trace("GAPS split distribution", **split_counts)
    trace("GAPS items loaded", items=len(items), split=gaps_split_label)

    # ------ GuitarSet ------
    # Player IDs 00–04 → train; 05 → validation/test
    if split == 'train':
        player_ids = {'00', '01', '02', '03', '04'}
    else:
        player_ids = {'05'}

    audio_dir = os.path.join(guitarset_dir, 'audio_mono-mic')
    ann_dir = os.path.join(guitarset_dir, 'annotation')

    for wav_name in sorted(os.listdir(audio_dir)):
        if not wav_name.endswith('_mic.wav'):
            continue
        player_id = wav_name[:2]
        if player_id not in player_ids:
            continue
        stem = wav_name.replace('_mic.wav', '')
        audio_path = os.path.join(audio_dir, wav_name)
        jams_path = os.path.join(ann_dir, stem + '.jams')
        if not os.path.exists(jams_path):
            continue
        notes = load_guitarset_notes(jams_path)
        items.append((audio_path, notes))

    trace("total items after GuitarSet", items=len(items))

    return GuitarDataset(
        items,
        segment_seconds=segment_seconds,
        frames_per_second=frames_per_second,
        sample_rate=sample_rate,
    )

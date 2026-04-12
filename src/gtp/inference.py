# Adapted from https://github.com/bytedance/piano_transcription

import numpy as np
import torch

from gtp.model.kong import Note_pedal
from gtp.model.utils import forward
from gtp.postprocess import RegressionPostProcessor, write_events_to_midi

# Constants matching the pretrained checkpoint configuration
SAMPLE_RATE = 16000
CLASSES_NUM = 88        # Number of piano notes
FRAMES_PER_SECOND = 100
BEGIN_NOTE = 21         # MIDI note of A0, the lowest piano note


class PianoTranscription:
    """Transcribes audio to piano note events using Kong et al.'s CRNN model.

    The model is loaded from a pretrained checkpoint and runs inference on
    10-second audio segments.
    """

    def __init__(
        self,
        checkpoint_path,
        device=None,
        segment_samples=SAMPLE_RATE * 10,
    ):
        """
        Args:
          checkpoint_path: str, path to the .pth checkpoint file
          device: torch.device or None; if None, auto-selects MPS > CUDA > CPU
          segment_samples: int, number of audio samples per inference segment
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.segment_samples = segment_samples
        self.frames_per_second = FRAMES_PER_SECOND
        self.classes_num = CLASSES_NUM

        # Build model
        self.model = Note_pedal(
            frames_per_second=self.frames_per_second,
            classes_num=self.classes_num,
        )

        # Load pretrained weights — checkpoint stores note/pedal sub-dicts
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, audio, midi_path=None):
        """Run model inference on a raw audio array and post-process to note events.

        Args:
          audio: np.ndarray, shape (audio_samples,), float32, 16 kHz mono
          midi_path: str or None; if given, write predicted notes to this MIDI file

        Returns:
          dict with keys:
            'output_dict':    raw model activations (frames x classes per output head)
            'note_events':    list of {'onset_time', 'offset_time', 'midi_note', 'velocity'}
            'pedal_events':   list of pedal event dicts, or None
        """
        output_dict = self._run_model(audio)

        post_processor = RegressionPostProcessor(
            frames_per_second=self.frames_per_second,
            classes_num=self.classes_num,
        )
        note_events, pedal_events = post_processor.output_dict_to_note_events(output_dict)

        if midi_path:
            write_events_to_midi(note_events, midi_path, pedal_events=pedal_events)

        return {
            'output_dict': output_dict,
            'note_events': note_events,
            'pedal_events': pedal_events,
        }

    def _run_model(self, audio):
        """Run the neural network forward pass and return raw activation arrays.

        Args:
          audio: np.ndarray, shape (audio_samples,), float32, 16 kHz mono

        Returns:
          output_dict: dict mapping output key -> np.ndarray
            'reg_onset_output':       (audio_frames, classes_num)
            'reg_offset_output':      (audio_frames, classes_num)
            'frame_output':           (audio_frames, classes_num)
            'velocity_output':        (audio_frames, classes_num)
            'reg_pedal_onset_output': (audio_frames, 1)
            'reg_pedal_offset_output':(audio_frames, 1)
            'pedal_frame_output':     (audio_frames, 1)
        """
        audio = audio[None, :]  # (1, audio_samples)
        audio_len = audio.shape[1]

        # Pad to a multiple of segment_samples
        pad_len = (
            int(np.ceil(audio_len / self.segment_samples)) * self.segment_samples
            - audio_len
        )
        audio = np.concatenate((audio, np.zeros((1, pad_len), dtype=audio.dtype)), axis=1)

        # Split into overlapping segments
        segments = self._enframe(audio, self.segment_samples)  # (N, segment_samples)

        # Run batched forward pass
        output_dict = forward(self.model, segments, batch_size=1)

        # Reassemble segments back to original frame length.
        # The hop size used by the feature extractor is sample_rate // fps, and
        # the model produces exactly audio_samples // hop_size valid frames.
        hop_size = SAMPLE_RATE // self.frames_per_second
        expected_frames = audio_len // hop_size
        for key in output_dict.keys():
            output_dict[key] = self._deframe(output_dict[key])[:expected_frames]

        return output_dict

    def _enframe(self, x, segment_samples):
        """Split (1, audio_samples) into overlapping (N, segment_samples) array."""
        assert x.shape[1] % segment_samples == 0
        batch = []
        pointer = 0
        while pointer + segment_samples <= x.shape[1]:
            batch.append(x[:, pointer: pointer + segment_samples])
            pointer += segment_samples // 2
        return np.concatenate(batch, axis=0)

    def _deframe(self, x):
        """Reassemble (N, segment_frames, classes_num) into (audio_frames, classes_num)."""
        if x.shape[0] == 1:
            return x[0]

        x = x[:, :-1, :]  # remove the extra frame at each segment end
        (N, segment_frames, classes_num) = x.shape
        assert segment_frames % 4 == 0

        y = [x[0, : int(segment_frames * 0.75)]]
        for i in range(1, N - 1):
            y.append(x[i, int(segment_frames * 0.25): int(segment_frames * 0.75)])
        y.append(x[-1, int(segment_frames * 0.25):])
        return np.concatenate(y, axis=0)

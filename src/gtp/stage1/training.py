"""Helpers shared by the Stage 1 training scripts: batching and loading the note model."""

import numpy as np
import torch

from gtp.stage1.model.kong import Regress_onset_offset_frame_velocity_CRNN

TARGET_KEYS = [
    'reg_onset_roll',
    'reg_offset_roll',
    'frame_roll',
    'velocity_roll',
    'onset_roll',
    'mask_roll',
]


def collate_fn(batch):
    """Convert list of numpy-array dicts to a batched tensor dict."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        arrays = [item[key] for item in batch]
        result[key] = torch.from_numpy(np.stack(arrays, axis=0))
    return result


def load_note_model(checkpoint_path):
    """Load the note model from either a pretrained or fine-tuned checkpoint.

    Pretrained checkpoints have nested format: checkpoint['model']['note_model'].
    Fine-tuned checkpoints have flat format: checkpoint['model'] is a state dict.

    Returns (model, checkpoint) where checkpoint is the full dict (may be used
    by the caller to restore optimizer state and step count).
    """
    model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=100, classes_num=88)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = checkpoint['model']
    if 'note_model' in state:
        model.load_state_dict(state['note_model'])
    else:
        model.load_state_dict(state)
    return model, checkpoint

"""Project-wide device-selection helpers."""

import torch


def auto_device() -> str:
    """Pick the best available device. MPS first (Mac dev), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

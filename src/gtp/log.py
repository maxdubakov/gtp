"""Simple verbose logging for tracing data flow through the pipeline."""

import numpy as np

_verbose = False


def set_verbose(enabled: bool):
    global _verbose
    _verbose = enabled


def is_verbose() -> bool:
    return _verbose


def trace(label: str, data=None, **kwargs):
    """Print a trace line when verbose mode is on.

    Usage:
        trace("loaded audio", audio, sr=16000)
        trace("onset binary", onset_binary, nonzero=np.count_nonzero(onset_binary))
    """
    if not _verbose:
        return

    parts = [f"  [{label}]"]

    if data is not None:
        if isinstance(data, np.ndarray):
            parts.append(f"shape={data.shape} dtype={data.dtype} min={data.min():.4f} max={data.max():.4f} mean={data.mean():.4f}")
        elif isinstance(data, list):
            parts.append(f"len={len(data)}")
        elif isinstance(data, dict):
            parts.append(f"keys={list(data.keys())}")
        else:
            parts.append(str(data))

    for k, v in kwargs.items():
        parts.append(f"{k}={v}")

    print(" ".join(parts))

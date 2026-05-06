"""Simple logging utilities used across stages.

Two entry points:

  * `info(msg)`       — always-on, timestamped progress output. Use for training
                        loops, eval cycles, dataset-build progress, etc.
  * `trace(label, …)` — verbose-mode-only tracing for inner data flow. Used
                        primarily in Stage 1 audio/MIDI pipelines. Toggled on
                        with `set_verbose(True)`.

Both prepend an `[HH:MM:SS]` local-time tag so multi-hour training logs are
greppable for "when did X happen".
"""

from datetime import datetime

import numpy as np

_verbose = False


def set_verbose(enabled: bool):
    global _verbose
    _verbose = enabled


def is_verbose() -> bool:
    return _verbose


def _ts() -> str:
    """HH:MM:SS local-time stamp."""
    return datetime.now().strftime('%H:%M:%S')


def info(msg: str) -> None:
    """Print a timestamped log line. Always emitted. flush=True so logs are
    immediately visible on RunPod / tee'd files."""
    print(f'[{_ts()}] {msg}', flush=True)


def trace(label: str, data=None, **kwargs):
    """Print a timestamped trace line when verbose mode is on.

    Usage:
        trace("loaded audio", audio, sr=16000)
        trace("onset binary", onset_binary, nonzero=np.count_nonzero(onset_binary))
    """
    if not _verbose:
        return

    parts = [f'[{_ts()}]  [{label}]']

    if data is not None:
        if isinstance(data, np.ndarray):
            parts.append(
                f'shape={data.shape} dtype={data.dtype} '
                f'min={data.min():.4f} max={data.max():.4f} mean={data.mean():.4f}'
            )
        elif isinstance(data, list):
            parts.append(f'len={len(data)}')
        elif isinstance(data, dict):
            parts.append(f'keys={list(data.keys())}')
        else:
            parts.append(str(data))

    for k, v in kwargs.items():
        parts.append(f'{k}={v}')

    print(' '.join(parts), flush=True)

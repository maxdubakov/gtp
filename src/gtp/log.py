"""Simple logging utilities used across stages.

Two entry points:

  * `info(msg)`       — always-on, timestamped progress output. Use for training
                        loops, eval cycles, dataset-build progress, etc.
  * `trace(label, …)` — verbose-mode-only tracing for inner data flow. Used
                        primarily in Stage 1 audio/MIDI pipelines. Toggled on
                        with `set_verbose(True)`.

Both prepend a `[YYYY-MM-DD HH:MM:SS]` local-time tag so multi-day training
logs are unambiguous and greppable for "when did X happen".
"""

from datetime import datetime

import numpy as np

_verbose = False
_log_file = None


def set_verbose(enabled: bool):
    global _verbose
    _verbose = enabled


def is_verbose() -> bool:
    return _verbose


def set_log_file(path) -> None:
    """Also append every `info` line to `path` (None stops it)."""
    global _log_file
    _log_file = path


def _ts() -> str:
    """YYYY-MM-DD HH:MM:SS local-time stamp."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def info(msg: str) -> None:
    """Print a timestamped log line. Always emitted. flush=True so logs are
    immediately visible on RunPod / tee'd files."""
    line = f'[{_ts()}] {msg}'
    print(line, flush=True)
    if _log_file is not None:
        with open(_log_file, 'a') as f:
            f.write(line + '\n')


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

"""Cache per-note spectrogram windows for the string classifier.

Writes data/guitarset/string_cache/<source>_<feature>.npy with shape
(notes, channels, freq, frames), float16, rows in `load_notes()` order, plus a
<source>_<feature>.json with the layout and code version it was built with.

Usage:
  ./venv/bin/python scripts/stage1/strings/build_cache.py --source mic --feature logmel229
  ./venv/bin/python scripts/stage1/strings/build_cache.py --source hex --feature logmel64
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import librosa
import numpy as np

from gtp.log import info
from gtp.stage1.strings import (
    CACHE_DIR,
    FEATURES,
    SOURCES,
    cache_layout,
    cache_path,
    load_audio,
    load_notes,
    note_windows,
    spectrogram,
)
from gtp.track import git_state


def track_windows(args):
    track, source, feature, onsets = args
    return note_windows(spectrogram(load_audio(track, source), feature), onsets).astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=SOURCES, required=True)
    parser.add_argument('--feature', choices=FEATURES, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    notes = load_notes()
    tracks = list(dict.fromkeys(notes.track))
    jobs = [(t, args.source, args.feature, notes.onset[notes.track == t]) for t in tracks]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cache_path(args.source, args.feature)
    tmp_path = out_path.with_suffix('.tmp.npy')
    out = None
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (track, windows) in enumerate(zip(tracks, pool.map(track_windows, jobs), strict=True)):
            if out is None:
                out = np.lib.format.open_memmap(
                    tmp_path, mode='w+', dtype=np.float16, shape=(len(notes.track), *windows.shape[1:])
                )
            out[np.flatnonzero(notes.track == track)] = windows
            if (i + 1) % 60 == 0:
                info(f'{args.source}/{args.feature}: {i + 1}/{len(tracks)} tracks')
    out.flush()
    del out
    tmp_path.rename(out_path)
    sha, diff = git_state()
    meta = {
        'source': args.source,
        'feature': args.feature,
        'layout': cache_layout(),
        'librosa': librosa.__version__,
        'git_sha': sha,
        'git_dirty': bool(diff),
        'built': datetime.now().isoformat(timespec='seconds'),
    }
    out_path.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    info(f'Wrote {out_path} {np.load(out_path, mmap_mode="r").shape}')


if __name__ == '__main__':
    main()

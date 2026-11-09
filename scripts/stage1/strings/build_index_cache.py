"""Cache per-note spectrogram windows for every note in the string index, with augmented copies for training.

Every note gets its real window (variant 0). Training recordings also get `augment` more renders, each
through its own random chain (chains.py) drawn from a seed fixed by (recording, variant), so a rebuild is
identical and a window can be traced back to its chain. Val and test stay real recordings only.

Writes to <index_dir>/cache/:
  <feature>.npy        (rows, 1, freq, PRE+MAX_POST frames) float16
  <feature>_rows.csv.gz  note (row in notes.csv.gz), variant
  <feature>_chains.jsonl one line per rendered (recording, variant): its chain parameters
  <feature>.json        layout, augment count and code version

Usage:
  uv run scripts/stage1/strings/build_index_cache.py
  uv run scripts/stage1/strings/build_index_cache.py augment=0 limit=50
"""

import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import hydra
import librosa
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1 import chains
from gtp.stage1.strings import SAMPLE_RATE, cache_layout, note_windows, spectrogram
from gtp.track import git_state


def chain_seed(rec, variant):
    return zlib.crc32(f'{rec}#{variant}'.encode())


def recording_windows(job):
    rec, audio, chain, onsets, variants, feature = job
    y, _ = librosa.load(REPO_ROOT / audio, sr=SAMPLE_RATE, mono=True)
    out, params = [], []
    for v in variants:
        if v == 0:
            x = y
        else:
            rng = np.random.default_rng(chain_seed(rec, v))
            p = chains.draw(rng, chains.stages_for(chain))
            x = chains.render(y, SAMPLE_RATE, p, rng)
            params.append({'rec': rec, 'variant': v, 'chain': p})
        out.append(note_windows(spectrogram(x[None], feature), onsets).astype(np.float16))
    return out, params


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='index_cache')
def main(cfg):
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    notes = pd.read_csv(index / 'notes.csv.gz')
    if cfg.limit:
        recs = recs.groupby('dataset').head(cfg.limit)
    by_rec = notes.groupby('rec').indices

    jobs, rows = [], []
    for r in recs.itertuples():
        idx = by_rec[r.rec]
        variants = list(range(1 + (cfg.augment if r.split == 'train' else 0)))
        jobs.append((r.rec, r.audio, r.chain, notes['onset'].to_numpy()[idx], variants, cfg.feature))
        rows += [(i, v) for v in variants for i in idx]
    rows = pd.DataFrame(rows, columns=['note', 'variant'])
    info(f'{len(recs)} recordings -> {len(rows)} windows ({cfg.augment} augmented copies per training note)')

    out_dir = index / 'cache'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{cfg.feature}.npy'
    tmp = path.with_suffix('.tmp.npy')
    x, k = None, 0
    with ProcessPoolExecutor(cfg.workers) as pool, open(out_dir / f'{cfg.feature}_chains.jsonl', 'w') as log:
        for i, (windows, params) in enumerate(pool.map(recording_windows, jobs, chunksize=8)):
            if x is None:
                x = np.lib.format.open_memmap(
                    tmp, mode='w+', dtype=np.float16, shape=(len(rows), *windows[0].shape[1:])
                )
            for w in windows:
                x[k : k + len(w)] = w
                k += len(w)
            log.writelines(json.dumps(p) + '\n' for p in params)
            if (i + 1) % 500 == 0:
                info(f'{i + 1}/{len(jobs)} recordings')
    assert k == len(rows), (k, len(rows))
    x.flush()
    del x
    tmp.rename(path)
    rows.to_csv(out_dir / f'{cfg.feature}_rows.csv.gz', index=False)
    sha, diff = git_state()
    meta = {
        'feature': cfg.feature,
        'augment': cfg.augment,
        'limit': cfg.limit,
        'layout': cache_layout(),
        'n_windows': len(rows),
        'librosa': librosa.__version__,
        'git_sha': sha,
        'git_dirty': bool(diff),
        'built': datetime.now().isoformat(timespec='seconds'),
    }
    path.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    info(f'Wrote {path} ({path.stat().st_size / 1e9:.1f} GB)')


if __name__ == '__main__':
    main()

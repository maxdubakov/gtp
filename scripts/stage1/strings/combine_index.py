"""Join cached string indexes (build_index.py or build_goat_index.py, then build_index_cache.py) into one.

The recordings and notes are concatenated in the order given, and so are the feature caches: their windows are
copied, not rendered again, so the first index's training windows stay byte for byte those its runs trained on. The
caches must share their layout and augmentation counts, so a variant number means the same chain family in each.

Usage:
  uv run scripts/stage1/strings/combine_index.py \
      "indexes=[data/string_index,data/string_index_goat_train_di]" out_dir=data/string_index_plus_goat_di
"""

import json
from datetime import datetime

import hydra
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info
from gtp.track import git_state


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='combine_index')
def main(cfg):
    dirs = [REPO_ROOT / d for d in cfg.indexes]
    recs = [pd.read_csv(d / 'recordings.csv') for d in dirs]
    notes = [pd.read_csv(d / 'notes.csv.gz') for d in dirs]
    metas = [json.loads((d / 'cache' / f'{cfg.feature}.json').read_text()) for d in dirs]
    for m in metas[1:]:
        assert all(m[k] == metas[0][k] for k in ('layout', 'augment', 'augment_far', 'limit')), (metas[0], m)
    assert not pd.concat(recs)['rec'].duplicated().any()
    offsets = np.cumsum([0] + [len(n) for n in notes[:-1]])
    rows = pd.concat(
        [pd.read_csv(d / 'cache' / f'{cfg.feature}_rows.csv.gz').assign(note=lambda r, o=o: r['note'] + o)
         for d, o in zip(dirs, offsets, strict=True)], ignore_index=True
    )  # fmt: skip

    out = REPO_ROOT / cfg.out_dir
    (out / 'cache').mkdir(parents=True, exist_ok=False)
    pd.concat(recs, ignore_index=True).to_csv(out / 'recordings.csv', index=False)
    pd.concat(notes, ignore_index=True).to_csv(out / 'notes.csv.gz', index=False, float_format='%.5f')
    path = out / 'cache' / f'{cfg.feature}.npy'
    sources = [np.load(d / 'cache' / f'{cfg.feature}.npy', mmap_mode='r') for d in dirs]
    x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(len(rows), *sources[0].shape[1:]))
    k = 0
    for src in sources:
        for i in range(0, len(src), cfg.chunk):
            part = src[i : i + cfg.chunk]
            x[k : k + len(part)] = part
            k += len(part)
    assert k == len(rows), (k, len(rows))
    x.flush()
    rows.to_csv(out / 'cache' / f'{cfg.feature}_rows.csv.gz', index=False)
    with open(out / 'cache' / f'{cfg.feature}_chains.jsonl', 'w') as fh:
        for d in dirs:
            fh.write((d / 'cache' / f'{cfg.feature}_chains.jsonl').read_text())
    sha, diff = git_state()
    meta = metas[0] | {
        'n_windows': len(rows),
        'sources': {str(d.relative_to(REPO_ROOT)): m for d, m in zip(dirs, metas, strict=True)},
        'git_sha': sha,
        'git_dirty': bool(diff),
        'built': datetime.now().isoformat(timespec='seconds'),
    }
    path.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    info(f'{sum(map(len, recs))} recordings, {len(rows)} windows -> {out} ({path.stat().st_size / 1e9:.1f} GB)')


if __name__ == '__main__':
    main()

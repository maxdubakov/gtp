"""Measure how far each recording's labels sit from the notes Stage 1 hears in it.

The onset-envelope alignment of build_index.py can lock one strum note off, and takes of one Guitar-TECHS group
differ by up to 70 ms, so neither a recording's own envelope lag nor its group's median is safe on chords. Here every
label is matched to the transcribed note of the same pitch nearest in time, which a strum cannot confuse; the
recording's lag is the median difference. It is stored against the raw labels (before the index's shift), so
build_index.py can use it in place of the envelope lag.

A recording listed in `reuse` chains takes the lag of the same take's `source_chain` recording: the two chains of a
Guitar-TECHS take are one performance with one set of labels.

Writes <index_dir>/note_lags.csv: rec, pairs, lag_raw_ms, mad_ms, source.

Usage:
  uv run scripts/stage1/strings/note_lags.py
"""

from concurrent.futures import ProcessPoolExecutor

import hydra
import numpy as np
import pandas as pd
import torch

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.inference import PianoTranscription, transcribe_cached

MODEL = None


def lag_of(job):
    global MODEL
    rec, audio, onsets, pitches, shift_ms, cfg, cache = job
    if MODEL is None:
        torch.set_num_threads(cfg.threads)
        MODEL = PianoTranscription(str(REPO_ROOT / cfg.stage1_checkpoint), device='cpu')
    est = pd.DataFrame(transcribe_cached(MODEL, REPO_ROOT / audio, cache), columns=['pitch', 'start', 'end'])
    gaps = []
    for onset, pitch in zip(onsets, pitches, strict=True):
        d = est.loc[est['pitch'] == pitch, 'start'].to_numpy() - onset
        if len(d) and np.abs(d).min() < cfg.max_gap_s:
            gaps.append(d[np.abs(d).argmin()])
    if not gaps:
        return rec, 0, np.nan, np.nan
    gaps = 1000 * np.array(gaps)
    lag = float(np.median(gaps))
    return rec, len(gaps), lag + shift_ms, float(np.median(np.abs(gaps - lag)))


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='note_lags')
def main(cfg):
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    notes = dict(tuple(pd.read_csv(index / 'notes.csv.gz').groupby('rec')))
    cache = REPO_ROOT / 'results' / 'e2e' / 'stage1' / f'index_{(REPO_ROOT / cfg.stage1_checkpoint).stem}'
    chosen = recs[recs['dataset'].isin(list(cfg.datasets))]
    reused = chosen['chain'].isin(list(cfg.reuse.chains)) & chosen['dataset'].isin(list(cfg.reuse.datasets))
    jobs = [
        (r.rec, r.audio, notes[r.rec]['onset'].to_numpy(), notes[r.rec]['pitch'].to_numpy(), r.shift_ms, cfg, cache / f'{r.rec}.json')
        for r in chosen[~reused].itertuples()
    ]  # fmt: skip
    info(f'{len(jobs)} recordings to align, {chosen[~reused]["duration"].sum() / 3600:.1f} hours')
    with ProcessPoolExecutor(cfg.workers) as pool:
        rows = list(pool.map(lag_of, jobs, chunksize=1))
    out = pd.DataFrame(rows, columns=['rec', 'pairs', 'lag_raw_ms', 'mad_ms']).assign(source='notes')
    by_rec = out.set_index('rec')
    for r in chosen[reused].itertuples():
        source = r.rec.replace(f'/{r.chain}/', f'/{cfg.reuse.source_chain}/')
        if source in by_rec.index:
            out.loc[len(out)] = [
                r.rec,
                *by_rec.loc[source, ['pairs', 'lag_raw_ms', 'mad_ms']],
                f'take:{cfg.reuse.source_chain}',
            ]
    out.to_csv(index / 'note_lags.csv', index=False)
    m = out.merge(recs[['rec', 'dataset', 'player', 'chain', 'content']], on='rec')
    stats = m.groupby(['dataset', 'player', 'chain', 'content']).agg(recs=('rec', 'size'), lag=('lag_raw_ms', 'median'), lag_min=('lag_raw_ms', 'min'), lag_max=('lag_raw_ms', 'max'), mad=('mad_ms', 'median'))  # fmt: skip
    info(f'Wrote {index / "note_lags.csv"}\n{stats.round(1).to_string()}')


if __name__ == '__main__':
    main()

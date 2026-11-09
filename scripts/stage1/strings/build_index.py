"""Build the multi-dataset string index: one recordings table and one notes table, cleaned and aligned.

Writes data/string_index/{recordings.csv, notes.csv.gz, report.json}. Each recording's label lag is
measured against its onset envelope; GuitarSet's hex-pickup labels set the detector's own bias, and a
recording whose lag departs from that bias by more than `shift_threshold_ms` has its labels shifted.
Recordings whose lag hits the edge of the search window are dropped.

Usage:
  uv run scripts/stage1/strings/build_index.py
  uv run scripts/stage1/strings/build_index.py 'datasets=[guitarset,guitar_techs]'
"""

import json
from concurrent.futures import ProcessPoolExecutor

import hydra
import librosa
import numpy as np

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1 import string_data as sd
from gtp.track import git_state


def measure(job):
    audio, onsets = job
    y, _ = librosa.load(REPO_ROOT / audio, sr=sd.ALIGN_SR, mono=True)
    lag, dist = sd.alignment(y, onsets)
    ok = ~np.isnan(dist)
    return {
        'duration': len(y) / sd.ALIGN_SR,
        'lag_ms': lag * 1000,
        'matched': float(ok.mean()) if len(ok) else 0.0,
        'residual_ms': float(np.median(np.abs(dist[ok])) * 1000) if ok.any() else float('nan'),
    }


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='index')
def main(cfg):
    recs, notes = sd.build(tuple(cfg.datasets))
    raw_counts = notes.groupby(notes['rec'].map(recs.set_index('rec')['dataset'])).size().to_dict()
    notes, dropped = sd.clean(recs, notes)
    recs = sd.drop_empty(recs, notes)
    info(f'{len(recs)} recordings, {len(notes)} notes after cleaning')

    grouped = notes.groupby('rec')['onset']
    jobs = [(a, grouped.get_group(r).to_numpy()) for r, a in zip(recs['rec'], recs['audio'], strict=True)]
    with ProcessPoolExecutor(cfg.workers) as pool:
        stats = list(pool.map(measure, jobs, chunksize=4))
    for key in stats[0]:
        recs[key] = [s[key] for s in stats]

    # A lag at the edge of the search window means the true offset lies beyond it: unusable labels.
    unresolved = recs['lag_ms'].abs() >= sd.MAX_LAG * 1000 - 1000 * sd.ALIGN_HOP / sd.ALIGN_SR
    unresolved_counts = recs[unresolved].groupby('dataset').size().to_dict()
    recs = recs[~unresolved].reset_index(drop=True)
    notes = notes[notes['rec'].isin(recs['rec'])].reset_index(drop=True)

    bias = float(recs.loc[recs['dataset'] == 'guitarset', 'lag_ms'].median())
    recs['shift_ms'] = np.where(
        (recs['lag_ms'] - bias).abs() > cfg.shift_threshold_ms, -(recs['lag_ms'] - bias), 0.0
    ).round(1)
    shift = notes['rec'].map(recs.set_index('rec')['shift_ms']) / 1000
    notes['onset'] += shift
    notes['offset'] += shift

    out = REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out / 'recordings.csv', index=False)
    notes.to_csv(out / 'notes.csv.gz', index=False, float_format='%.5f')
    sha, diff = git_state()
    report = {
        'datasets': list(cfg.datasets),
        'raw_notes': raw_counts,
        'dropped': dropped,
        'detector_bias_ms': bias,
        'shift_threshold_ms': cfg.shift_threshold_ms,
        'unresolved_recordings': unresolved_counts,
        'shifted_recordings': recs.groupby('dataset')['shift_ms'].apply(lambda s: int((s != 0).sum())).to_dict(),
        'git_sha': sha,
        'git_dirty': bool(diff),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    info(f'detector bias {bias:+.1f} ms; shifted: {report["shifted_recordings"]}')
    info(recs.groupby(['dataset', 'split', 'chain']).size().to_string())


if __name__ == '__main__':
    main()

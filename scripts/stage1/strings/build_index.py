"""Build the multi-dataset string index: one recordings table and one notes table, cleaned and aligned.

Writes data/string_index/{recordings.csv, notes.csv.gz, report.json}. Each recording's label lag is
measured against its onset envelope; GuitarSet's hex-pickup labels set the detector's own bias, and a
recording whose lag departs from that bias by more than `shift_threshold_ms` has its labels shifted.
One-note recordings keep their own lag; denser ones take their player/chain/content group median, since
the search can lock one strum note off. Recordings that Stage 1 has been run over (note_lags.py) are placed
by its notes instead, which is the reliable measure on chords, and dropped if its notes fit no single lag.
Datasets whose onsets come from their own audio are never shifted. Recordings whose lag hits the search edge
are dropped. Guitar-TECHS's room recordings (`room_mics`) are added last, as chains `ego` and `exo` carrying
the notes of their direct-input recordings.

Usage:
  uv run scripts/stage1/strings/build_index.py
  uv run scripts/stage1/strings/build_index.py 'datasets=[guitarset,guitar_techs]'
"""

import json
from concurrent.futures import ProcessPoolExecutor

import hydra
import librosa
import numpy as np
import pandas as pd

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
    own = recs['dataset'].isin(sd.AUDIO_ONSETS)
    unresolved = (recs['lag_ms'].abs() >= sd.MAX_LAG * 1000 - 1000 * sd.ALIGN_HOP / sd.ALIGN_SR) & ~own
    unresolved_counts = recs[unresolved].groupby('dataset').size().to_dict()
    recs = recs[~unresolved].reset_index(drop=True)
    own = recs['dataset'].isin(sd.AUDIO_ONSETS)
    notes = notes[notes['rec'].isin(recs['rec'])].reset_index(drop=True)

    # On strummed or dense recordings the lag search can lock one note off (the DI and amp-mic copies of one
    # Guitar-TECHS take disagree by ~50 ms), while lags within a player/chain/content group agree; so only a
    # one-note recording keeps its own lag and the rest take their group's median.
    n_notes = recs['rec'].map(notes.groupby('rec').size())
    group = recs.groupby(['dataset', 'player', 'chain', 'content'])['lag_ms'].transform('median')
    recs['lag_used_ms'] = np.where(own, np.nan, np.where(n_notes == 1, recs['lag_ms'], group))
    bias = float(recs.loc[recs['dataset'] == 'guitarset', 'lag_ms'].median())
    off = recs['lag_used_ms'] - bias
    recs['shift_ms'] = np.where(~own & (off.abs() > cfg.shift_threshold_ms), -off, 0.0).round(1)
    recs['lag_source'] = np.where(own, 'own audio', np.where(n_notes == 1, 'envelope', 'group envelope'))

    # Where Stage 1 has been run over a recording (note_lags.py), its labels are placed by the notes it hears:
    # matching by pitch cannot lock a strum note off, and it sees that takes of one group differ. GuitarSet's
    # labels are the reference there too, so its median lag is Stage 1's own bias and GuitarSet itself stays put.
    scattered_counts = {}
    lags_path = REPO_ROOT / cfg.out_dir / 'note_lags.csv'
    if cfg.use_note_lags and lags_path.exists():
        lags = pd.read_csv(lags_path).set_index('rec')
        lags = lags[lags['pairs'] >= cfg.min_note_pairs]
        # Matched notes that scatter widely around their lag mean the labels don't follow the audio at all
        # (two GuitarSet takes are annotated a beat off in places): no shift can repair those.
        scattered = recs['rec'].map(lags['mad_ms']) > cfg.max_note_mad_ms
        scattered_counts = recs[scattered].groupby('dataset').size().to_dict()
        recs = recs[~scattered].reset_index(drop=True)
        own = recs['dataset'].isin(sd.AUDIO_ONSETS)
        notes = notes[notes['rec'].isin(recs['rec'])].reset_index(drop=True)
        note_lag = recs['rec'].map(lags['lag_raw_ms'])
        note_bias = float(note_lag[recs['dataset'] == 'guitarset'].median())
        by_notes = note_lag.notna() & ~own & (recs['dataset'] != 'guitarset')
        move = note_lag - note_bias
        recs['shift_ms'] = np.where(
            by_notes, np.where(move.abs() > cfg.shift_threshold_ms, move, 0.0), recs['shift_ms']
        ).round(1)
        recs['lag_source'] = np.where(by_notes, 'notes', recs['lag_source'])
        info(f'{int(by_notes.sum())} recordings placed by Stage 1 notes (its bias on GuitarSet {note_bias:+.1f} ms)')
    shift = notes['rec'].map(recs.set_index('rec')['shift_ms']) / 1000
    notes['onset'] += shift
    notes['offset'] += shift
    outside = (notes['onset'] < 0) | (notes['onset'] >= notes['rec'].map(recs.set_index('rec')['duration']))
    dropped['starts outside the audio'] = (
        outside.groupby(notes['rec'].map(recs.set_index('rec')['dataset'])).sum().to_dict()
    )
    notes = notes[~outside].reset_index(drop=True)
    recs = sd.drop_empty(recs, notes)

    if cfg.room_mics:
        room_recs, room_notes = sd.room_recordings(recs, notes, list(cfg.room_mics), cfg.workers)
        recs = pd.concat([recs, room_recs], ignore_index=True)
        notes = pd.concat([notes, room_notes], ignore_index=True)
        stats = room_recs.groupby(['player', 'chain']).agg(recordings=('rec', 'size'), lag_ms=('room_lag_ms', 'median'), lag_min=('room_lag_ms', 'min'), lag_max=('room_lag_ms', 'max'), corr=('room_corr', 'median'), corr_min=('room_corr', 'min'))  # fmt: skip
        info(f'{len(room_recs)} room recordings, {len(room_notes)} notes\n{stats.round(2).to_string()}')

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
        'scattered_recordings': scattered_counts,
        'shifted_recordings': recs.groupby('dataset')['shift_ms'].apply(lambda s: int((s != 0).sum())).to_dict(),
        'git_sha': sha,
        'git_dirty': bool(diff),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    info(f'detector bias {bias:+.1f} ms; shifted: {report["shifted_recordings"]}')
    info(recs.groupby(['dataset', 'split', 'chain']).size().to_string())


if __name__ == '__main__':
    main()

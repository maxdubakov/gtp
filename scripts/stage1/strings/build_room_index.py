"""A string index of Guitar-TECHS's room recordings: the same performances heard by microphones in the room.

Guitar-TECHS films every performance with a head-mounted camera (`ego`) and a camera about 5 feet in front of the
player (`exo`); their audio is a real far-field recording of the labelled performance. Each room recording takes the
notes of its direct-input recording in the main index, moved by the lag between the two audio files (the peak of
the cross-correlation of their onset envelopes).

Writes <out_dir>/recordings.csv and notes.csv.gz in the main index's format, with chain = ego or exo.

Usage:
  uv run scripts/stage1/strings/build_room_index.py
"""

from concurrent.futures import ProcessPoolExecutor

import hydra
import librosa
import numpy as np
import pandas as pd
import scipy.signal

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.string_data import ALIGN_HOP, ALIGN_SR, onset_envelope


def lag_of(job):
    """Seconds the room audio lags its direct recording, the correlation at that lag, and the room audio's length."""
    direct, room, max_lag = job
    a, _ = librosa.load(REPO_ROOT / direct, sr=ALIGN_SR, mono=True)
    b, _ = librosa.load(REPO_ROOT / room, sr=ALIGN_SR, mono=True)
    ea, eb = onset_envelope(a), onset_envelope(b)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    corr = scipy.signal.correlate(eb, ea, mode='full')
    lags = scipy.signal.correlation_lags(n, n, mode='full')
    near = np.abs(lags) <= round(max_lag * ALIGN_SR / ALIGN_HOP)
    k = np.argmax(corr[near])
    peak = corr[near][k] / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-12)
    return lags[near][k] * ALIGN_HOP / ALIGN_SR, float(peak), len(b) / ALIGN_SR


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='room_index')
def main(cfg):
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    notes = pd.read_csv(index / 'notes.csv.gz')
    direct = recs[(recs['dataset'] == 'guitar_techs') & (recs['chain'] == 'di')]
    rows = []
    for r in direct.itertuples():
        _, part, _, stem = r.rec.split('/')
        for mic in cfg.mics:
            audio = f'data/guitar_techs/{part}/video/{mic}/{mic}_{stem}.mp3'
            if (REPO_ROOT / audio).exists():
                rows.append(
                    {
                        **r._asdict(),
                        'direct_rec': r.rec,
                        'direct_audio': r.audio,
                        'rec': f'guitar_techs/{part}/{mic}/{stem}',
                        'audio': audio,
                        'chain': mic,
                    }
                )
    out = pd.DataFrame(rows).drop(columns=['Index'])
    with ProcessPoolExecutor(cfg.workers) as pool:
        lags = list(pool.map(lag_of, [(r.direct_audio, r.audio, cfg.max_lag_s) for r in out.itertuples()], chunksize=4))
    out['lag_s'], out['lag_corr'], out['duration'] = zip(*lags, strict=True)
    by_rec = dict(tuple(notes.groupby('rec')))
    moved = []
    for r in out.itertuples():
        n = by_rec[r.direct_rec].assign(rec=r.rec)
        n[['onset', 'offset']] = n[['onset', 'offset']] + r.lag_s
        moved.append(n[(n['onset'] >= 0) & (n['onset'] < r.duration)])
    moved = pd.concat(moved, ignore_index=True)
    out_dir = REPO_ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out.drop(columns=['direct_audio']).to_csv(out_dir / 'recordings.csv', index=False)
    moved.to_csv(out_dir / 'notes.csv.gz', index=False)
    stats = out.groupby(['player', 'chain']).agg(recordings=('rec', 'size'), lag_ms=('lag_s', lambda s: 1000 * s.median()), lag_min=('lag_s', lambda s: 1000 * s.min()), lag_max=('lag_s', lambda s: 1000 * s.max()), corr=('lag_corr', 'median'), corr_min=('lag_corr', 'min'))  # fmt: skip
    info(f'{len(out)} room recordings, {len(moved)} notes -> {out_dir}\n{stats.round(2).to_string()}')


if __name__ == '__main__':
    main()

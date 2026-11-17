"""A string index of simulated room-microphone copies of another index's recordings, for evaluation only.

Each recording goes through a measured room and device (gtp.stage1.simroom, ACE corpus), direct-input ones through
a clean amp and cabinet matched to a real rig first; the room, device, noise and SNR are drawn from a seed fixed by
the recording's name, so a rebuild is identical. The labels are the source's, moved by the time from the response's
start to its direct sound (about a millisecond). As a check, each copy's lag behind its source is measured on their
onset envelopes (string_data.room_lag; ~6 ms resolution). The new recordings are named <prefix>/<source rec>, so
Stage 1's per-recording cache never mixes them with their sources.

Writes <out_dir>/{recordings.csv, notes.csv.gz, report.json, cabinet.npy, audio/}.

Usage:
  uv run scripts/stage1/strings/build_simroom_index.py
  uv run scripts/stage1/strings/build_simroom_index.py simroom=music
  uv run scripts/stage1/strings/build_simroom_index.py \
      'sources=[{index_dir:data/string_index_synthtab,where:{}}]' out_dir=data/string_index_simroom_synthtab
"""

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import hydra
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1 import simroom
from gtp.stage1 import string_data as sd
from gtp.track import git_state


def cabinet_pairs(cfg, sr):
    recs = pd.read_csv(REPO_ROOT / cfg.index_dir / 'recordings.csv')
    recs = recs[recs['guitar'] == cfg.guitar]
    direct, miked = (
        recs[recs['chain'] == c].assign(key=lambda r, c=c: r['rec'].str.replace(f'/{c}/', '/')).set_index('key')
        for c in (cfg.direct, cfg.miked)
    )
    keys = direct.index.intersection(miked.index)
    assert len(keys), f'no {cfg.direct}/{cfg.miked} pairs of {cfg.guitar}'
    for k in keys:
        yield (librosa.load(REPO_ROOT / direct.loc[k, 'audio'], sr=sr, mono=True)[0],
               librosa.load(REPO_ROOT / miked.loc[k, 'audio'], sr=sr, mono=True)[0])  # fmt: skip


def render_one(job, cabinet, sr, length_s):
    rec, params = job
    y, _ = librosa.load(REPO_ROOT / rec['audio'], sr=sr, mono=True)
    out, latency = simroom.render(y, sr, params, cabinet if rec['direct'] else None, length_s)
    path = REPO_ROOT / rec['out_audio']
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, out, sr, subtype='PCM_16')
    lag, corr, _ = sd.room_lag((rec['audio'], rec['out_audio']))
    return {'duration': len(out) / sr, 'latency_ms': 1000 * latency, 'check_lag_ms': 1000 * lag, 'check_corr': corr}


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='simroom_index')
def main(cfg):
    recs, notes = [], []
    for source in cfg.sources:
        r = pd.read_csv(REPO_ROOT / source.index_dir / 'recordings.csv')
        for column, values in source.where.items():
            r = r[r[column].isin(list(values))]
        n = pd.read_csv(REPO_ROOT / source.index_dir / 'notes.csv.gz')
        recs.append(r[sd.REC_COLUMNS])
        notes.append(n[n['rec'].isin(r['rec'])])
    recs, notes = pd.concat(recs, ignore_index=True), pd.concat(notes, ignore_index=True)
    assert len(recs) and not recs['rec'].duplicated().any()

    out = REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    recs['direct'] = recs['chain'].isin(list(cfg.direct_chains))
    cabinet = None
    if recs['direct'].any():
        cabinet = simroom.match_filter(
            cabinet_pairs(cfg.cabinet, cfg.sample_rate), cfg.sample_rate, cfg.cabinet.n_fft,
            cfg.cabinet.smooth_octaves, cfg.cabinet.top_hz, cfg.cabinet.floor_db,
        )  # fmt: skip
        np.save(out / 'cabinet.npy', cabinet)
        gain = 20 * np.log10(np.abs(np.fft.rfft(cabinet, cfg.cabinet.n_fft)) + 1e-12)
        freqs = np.fft.rfftfreq(cfg.cabinet.n_fft, 1 / cfg.sample_rate)
        bands = [62.5, 125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000]
        info('cabinet dB at ' + ', '.join(f'{f:.0f} Hz {np.interp(f, freqs, gain):+.1f}' for f in bands))

    entries = simroom.bank(REPO_ROOT / cfg.ace_dir, list(cfg.devices), list(cfg.rooms))
    params = [
        simroom.draw(simroom.rng_for(r, cfg.seed), entries, list(cfg.noises), list(cfg.snr_db)) for r in recs['rec']
    ]
    recs['out_rec'] = cfg.prefix + '/' + recs['rec']
    recs['out_audio'] = [str((out / 'audio' / f'{r}.wav').relative_to(REPO_ROOT)) for r in recs['rec']]
    render = partial(render_one, cabinet=cabinet, sr=cfg.sample_rate, length_s=cfg.response_s)
    with ProcessPoolExecutor(cfg.workers) as pool:
        stats = list(pool.map(render, zip(recs.to_dict('records'), params, strict=True), chunksize=1))

    drawn = pd.DataFrame(params).drop(columns=['noise_file']).assign(rir=lambda d: d['rir'].map(sd.rel))
    table = pd.concat(
        [recs[sd.REC_COLUMNS].assign(source_rec=recs['rec'], source_chain=recs['chain'], source_audio=recs['audio'],
                                     rec=recs['out_rec'], audio=recs['out_audio'], chain=cfg.chain,
                                     amp=np.where(recs['direct'], 'cabinet', 'none')),
         drawn, pd.DataFrame(stats)], axis=1
    )  # fmt: skip
    moved = notes['rec'].map(table.set_index('source_rec')['latency_ms']) / 1000
    notes = notes.assign(
        rec=cfg.prefix + '/' + notes['rec'], onset=notes['onset'] + moved, offset=notes['offset'] + moved
    )
    table.to_csv(out / 'recordings.csv', index=False, float_format='%.4f')
    notes.to_csv(out / 'notes.csv.gz', index=False, float_format='%.5f')
    sha, diff = git_state()
    report = {
        'config': OmegaConf.to_container(cfg),
        'recordings': len(table),
        'notes': len(notes),
        'by': {'/'.join(k): int(v) for k, v in table.groupby(['dataset', 'guitar', 'source_chain']).size().items()},
        'git_sha': sha,
        'git_dirty': bool(diff),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    info(f'{len(table)} recordings, {len(notes)} notes -> {out}')
    info(table.groupby(['device', 'room']).size().to_string())
    info(table[['latency_ms', 'check_lag_ms', 'check_corr', 'snr_db']].describe().round(2).to_string())


if __name__ == '__main__':
    main()

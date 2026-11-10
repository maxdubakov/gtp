"""Gather every labelled recording Stage 1 can train on into one folder of 16 kHz audio and a manifest.

Sources, with the splits the string index uses (by player or guitar):
  * GAPS (classical guitar): its own train split trains, its test split validates;
  * every recording of the string index except EGFxSet's single notes;
  * the room recordings of Guitar-TECHS (build_room_index.py);
  * EGFxSet's single notes, stitched into short sequences so that they fill a training segment. Each note is cut to
    a random length, so the sequences have onsets and endings at the density of played music.

Writes <out_dir>/audio/<id>.flac (16 kHz mono) and manifest.jsonl, one recording per line:
  id, audio, dataset, chain, guitar, content, split, duration, notes [{onset_time, offset_time, midi_note, velocity}]

Usage:
  uv run scripts/stage1/build_notes_dataset.py
"""

import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor

import hydra
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.data import load_gaps_notes

SR = 16000
VELOCITY = 64


def slug(rec):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', rec)


def convert(job):
    """One source file to 16 kHz mono FLAC; returns its duration."""
    src, dst = job
    y, _ = librosa.load(src, sr=SR, mono=True)
    sf.write(dst, y, SR, subtype='PCM_16', format='FLAC')
    return len(y) / SR


def stitch(job):
    """Single-note files cut to random lengths and joined; returns (duration, notes)."""
    files, onsets, pitches, dst, seed, cfg = job
    rng = np.random.default_rng(seed)
    fade = np.linspace(1.0, 0.0, round(cfg.fade_s * SR))
    parts, notes, at = [], [], 0.0
    for path, onset, pitch in zip(files, onsets, pitches, strict=True):
        y, _ = librosa.load(path, sr=SR, mono=True)
        start = max(0.0, onset - cfg.lead_s)
        length = rng.uniform(cfg.min_note_s, cfg.max_note_s)
        cut = y[round(start * SR) : round((start + length) * SR)].copy()
        cut[-len(fade) :] *= fade[-len(cut) :]
        notes.append(
            {
                'onset_time': at + onset - start,
                'offset_time': at + len(cut) / SR,
                'midi_note': int(pitch),
                'velocity': VELOCITY,
            }
        )
        parts.append(cut)
        at += len(cut) / SR
    sf.write(dst, np.concatenate(parts), SR, subtype='PCM_16', format='FLAC')
    return at, notes


def index_entries(index_dir, exclude=()):
    recs = pd.read_csv(index_dir / 'recordings.csv')
    notes = dict(tuple(pd.read_csv(index_dir / 'notes.csv.gz').groupby('rec')))
    for r in recs[~recs['dataset'].isin(list(exclude))].itertuples():
        yield (
            r,
            [
                {
                    'onset_time': float(n.onset),
                    'offset_time': float(n.offset),
                    'midi_note': int(n.pitch),
                    'velocity': VELOCITY,
                }
                for n in notes[r.rec].itertuples()
            ],
        )


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='notes_dataset')
def main(cfg):
    out = REPO_ROOT / cfg.out_dir
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    entries, jobs = [], []

    def add(rec_id, src, notes, **meta):
        audio = f'audio/{slug(rec_id)}.flac'
        entries.append({'id': rec_id, 'audio': audio, **meta, 'notes': notes})
        jobs.append((str(src), str(out / audio)))

    gaps = REPO_ROOT / cfg.gaps_dir
    with open(gaps / 'gaps_metadata_with_splits.csv', newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            split = {'train': 'train', 'test': 'val'}.get(row['split'])
            if split and (gaps / row['audio_path']).exists() and (gaps / row['midi_path']).exists():
                add(f'gaps/{row["id"]}', gaps / row['audio_path'], load_gaps_notes(str(gaps / row['midi_path'])),
                    dataset='gaps', chain='acoustic-mic', guitar='gaps', content='pieces', split=split)  # fmt: skip
    for index_dir, exclude in ((cfg.index_dir, ['egfxset']), (cfg.room_index_dir, [])):
        for r, notes in index_entries(REPO_ROOT / index_dir, exclude):
            add(r.rec, REPO_ROOT / r.audio, notes, dataset=r.dataset, chain=r.chain, guitar=r.guitar, content=r.content, split=r.split)  # fmt: skip
    with ProcessPoolExecutor(cfg.workers) as pool:
        for e, duration in zip(entries, pool.map(convert, jobs, chunksize=8), strict=True):
            e['duration'] = duration

    # EGFxSet: sequences of single notes, per effect and pickup, each note used cfg.stitch.passes times.
    recs = pd.read_csv(REPO_ROOT / cfg.index_dir / 'recordings.csv')
    notes = pd.read_csv(REPO_ROOT / cfg.index_dir / 'notes.csv.gz').set_index('rec')
    single = recs[recs['dataset'] == 'egfxset'].assign(group=lambda d: d['rec'].str.rsplit('/', n=1).str[0])
    stitched, stitch_jobs = [], []
    rng = np.random.default_rng(cfg.stitch.seed)
    for group, g in single.groupby('group'):
        order = np.concatenate([rng.permutation(len(g)) for _ in range(cfg.stitch.passes)])
        for k in range(0, len(order), cfg.stitch.notes_per_sequence):
            part = g.iloc[order[k : k + cfg.stitch.notes_per_sequence]]
            rec_id = f'{group}/seq{k // cfg.stitch.notes_per_sequence:03d}'
            audio = f'audio/{slug(rec_id)}.flac'
            r = part.iloc[0]
            stitched.append({'id': rec_id, 'audio': audio, 'dataset': 'egfxset', 'chain': r['chain'], 'guitar': r['guitar'], 'content': r['content'], 'split': r['split']})  # fmt: skip
            stitch_jobs.append(
                ([str(REPO_ROOT / a) for a in part['audio']], notes.loc[part['rec'], 'onset'].tolist(),
                 notes.loc[part['rec'], 'pitch'].tolist(), str(out / audio), int(rng.integers(1 << 31)), cfg.stitch)
            )  # fmt: skip
    with ProcessPoolExecutor(cfg.workers) as pool:
        for e, (duration, seq_notes) in zip(stitched, pool.map(stitch, stitch_jobs, chunksize=4), strict=True):
            e['duration'], e['notes'] = duration, seq_notes
    entries += stitched

    with open(out / 'manifest.jsonl', 'w') as fh:
        fh.writelines(json.dumps(e) + '\n' for e in entries)
    table = pd.DataFrame(
        [{k: e[k] for k in ('dataset', 'chain', 'split', 'duration')} | {'notes': len(e['notes'])} for e in entries]
    )
    summary = table.groupby(['split', 'dataset', 'chain']).agg(recordings=('notes', 'size'), hours=('duration', lambda s: s.sum() / 3600), notes=('notes', 'sum'))  # fmt: skip
    info(f'{len(entries)} recordings -> {out}\n{summary.round(2).to_string()}')


if __name__ == '__main__':
    main()

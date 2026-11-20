"""A string index of GOAT items, for scoring the pipeline's tabs on GOAT as Noise2Fret does or for training on them.

GOAT's labels in time are its fine-aligned MIDI; its strings are in the Guitar Pro 5 tab, which runs in score time
and lists tied notes separately. The tab's chords (notes sharing a score time) are aligned to the MIDI's onset
clusters by dynamic time warping on pitch-set overlap, and inside each aligned pair notes are matched by pitch; a MIDI
note without a match stays out of the index, so the labels keep GOAT's timing and only gain strings. Every render
(DI and the five amps) is a recording of its own chain, so each is calibrated apart. With `retimed_notes` set, each
render's notes take the times re-timed to its audio (scripts/stage1/retime_goat_test.py) in place of the MIDI's.
With `manifest` set, the Stage 1 notes manifest's GOAT entries choose the items and their train/val split and give
the re-timed times, so the index holds exactly the items Stage 1 trained and validated on. With `renders_per_item`,
each item keeps that many of its renders, taken in turn so every render covers an equal share of the items.

Usage:
  uv run scripts/stage1/strings/build_goat_index.py
  uv run scripts/stage1/strings/build_goat_index.py retimed_notes=data/goat/test_retimed/notes.csv.gz \
      out_dir=data/string_index_goat_retimed
  uv run scripts/stage1/strings/build_goat_index.py manifest=data/stage1_notes_z_refined/manifest.jsonl \
      "splits=[train,val]" "tuning=[64,59,55,50,45,40]" out_dir=data/string_index_goat_trainval
"""

import json

import hydra
import librosa
import numpy as np
import pandas as pd
import pretty_midi

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.tab_alignment import strings_by_alignment
from gtp.stage2.guitarpro import parse_gp_track


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='goat_index')
def main(cfg):
    root = REPO_ROOT / cfg.goat_dir
    meta = pd.read_csv(root / 'metadata.csv')
    retimed = {}
    if cfg.manifest:
        with open(REPO_ROOT / cfg.manifest) as fh:
            entries = [e for e in map(json.loads, fh) if e['dataset'] == 'goat']
        retimed = {e['id']: [(n['onset_time'], n['offset_time'], n['midi_note']) for n in e['notes']] for e in entries}
        meta['split'] = meta['item'].map({e['id'].split('/')[1]: e['split'] for e in entries})
    elif cfg.retimed_notes:
        retimed = {rec: g[['onset', 'offset', 'pitch']].to_numpy() for rec, g in pd.read_csv(REPO_ROOT / cfg.retimed_notes).groupby('rec', sort=False)}  # fmt: skip
    renders = list(cfg.renders.items())
    kept = cfg.renders_per_item or len(renders)
    recs, notes, coverage, skipped = [], [], [], {}
    for r in meta[meta.split.isin(list(cfg.splits))].itertuples():
        try:
            tab = parse_gp_track(root / 'data' / r.gp5_path.split('/', 1)[1], 0)
        except ValueError as e:
            skipped[r.item] = f'tab unreadable: {e}'
            continue
        if cfg.tuning and list(tab['tuning']) != list(cfg.tuning):
            skipped[r.item] = f'tuning {tab["tuning"]}'
            continue
        midi = [n for i in pretty_midi.PrettyMIDI(str(root / 'data' / r.finealigned_midi_path.split('/', 1)[1])).instruments
                for n in i.notes]  # fmt: skip
        matched = strings_by_alignment(tab['notes'], midi, cfg)
        coverage.append({'item': r.item, 'split': r.split, 'notes': len(midi), 'with_string': len(matched)})
        for column, chain in (renders[(len(coverage) * kept + k) % len(renders)] for k in range(kept)):
            audio = root / 'data' / getattr(r, column).split('/', 1)[1]
            rec = f'goat/{r.item}/{chain}'
            recs.append({'rec': rec, 'dataset': 'goat', 'audio': str(audio.relative_to(REPO_ROOT)), 'player': r.player,
                         'guitar': f'goat-{r.guitar_brand}', 'chain': chain, 'content': 'pieces', 'split': r.split,
                         'duration': librosa.get_duration(path=audio)})  # fmt: skip
            times = [(n.start, n.end) for n in midi]
            if retimed:
                times = np.asarray(retimed[f'goat/{r.item}/{audio.stem}'])
                assert len(times) == len(midi) and (times[:, 2] == [n.pitch for n in midi]).all(), rec
            notes += [{'rec': rec, 'onset': times[k][0], 'offset': times[k][1], 'pitch': midi[k].pitch,
                       'string': t['string'], 'fret': midi[k].pitch - tab['tuning'][t['string'] - 1]}
                      for k, t in sorted(matched.items())]  # fmt: skip
    out = REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(out / 'recordings.csv', index=False)
    pd.DataFrame(notes).to_csv(out / 'notes.csv.gz', index=False)
    coverage = pd.DataFrame(coverage)
    (out / 'report.json').write_text(json.dumps({'items': coverage.to_dict('records'), 'skipped': skipped}, indent=1))
    info(
        f'{len(recs)} recordings -> {out}; notes with a string: {coverage.with_string.sum()} of {coverage.notes.sum()}; '
        f'{len(skipped)} items skipped'
    )


if __name__ == '__main__':
    main()

"""A string index of GOAT's test split, for scoring the pipeline's tabs on GOAT as Noise2Fret does.

GOAT's labels in time are its fine-aligned MIDI; its strings are in the Guitar Pro 5 tab, which runs in score time
and lists tied notes separately. The tab's chords (notes sharing a score time) are aligned to the MIDI's onset
clusters by dynamic time warping on pitch-set overlap, and inside each aligned pair notes are matched by pitch; a MIDI
note without a match stays out of the index, so the labels keep GOAT's timing and only gain strings. Every render
(DI and the five amps) is a recording of its own chain, so each is calibrated apart.

Usage:
  uv run scripts/stage1/strings/build_goat_index.py
"""

import json

import hydra
import librosa
import numpy as np
import pandas as pd
import pretty_midi

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage2.guitarpro import parse_gp_track


def clusters(notes, start, gap):
    out = []
    for n in notes:
        if out and start(n) - start(out[-1][-1]) <= gap:
            out[-1].append(n)
        else:
            out.append([n])
    return out


def strings_by_alignment(tab, midi, cfg):
    """{index of a MIDI note: tab note} for the MIDI notes the tab's chords align to."""
    chords = clusters(sorted(tab, key=lambda n: (n['start'], n['pitch'])), lambda n: n['start'], 1e-4)
    order = sorted(range(len(midi)), key=lambda k: (midi[k].start, midi[k].pitch))
    onsets = clusters(order, lambda k: midi[k].start, cfg.cluster_gap_s)
    pa = [{n['pitch'] for n in c} for c in chords]
    pb = [{midi[k].pitch for k in c} for c in onsets]
    n, m = len(chords), len(onsets)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    step = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        for j in range(max(1, i - cfg.band), min(m, i + cfg.band) + 1):
            options = (cost[i - 1, j - 1] + 1 - len(pa[i - 1] & pb[j - 1]) / len(pa[i - 1] | pb[j - 1]),
                       cost[i - 1, j] + 0.5, cost[i, j - 1] + 0.5)  # fmt: skip
            step[i, j] = int(np.argmin(options))
            cost[i, j] = options[step[i, j]]
    matched, i, j = {}, n, m
    while i > 0 and j > 0:
        if step[i, j] == 0:
            pool = {}
            for t in chords[i - 1]:
                pool.setdefault(t['pitch'], []).append(t)
            for k in onsets[j - 1]:
                if pool.get(midi[k].pitch):
                    matched[k] = pool[midi[k].pitch].pop(0)
            i, j = i - 1, j - 1
        elif step[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return matched


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='goat_index')
def main(cfg):
    root = REPO_ROOT / cfg.goat_dir
    meta = pd.read_csv(root / 'metadata.csv')
    recs, notes, coverage = [], [], []
    for r in meta[meta.split == 'test'].itertuples():
        tab = parse_gp_track(root / 'data' / r.gp5_path.split('/', 1)[1], 0)
        midi = [n for i in pretty_midi.PrettyMIDI(str(root / 'data' / r.finealigned_midi_path.split('/', 1)[1])).instruments
                for n in i.notes]  # fmt: skip
        matched = strings_by_alignment(tab['notes'], midi, cfg)
        coverage.append({'item': r.item, 'notes': len(midi), 'with_string': len(matched)})
        for column, chain in cfg.renders.items():
            audio = root / 'data' / getattr(r, column).split('/', 1)[1]
            rec = f'goat/{r.item}/{chain}'
            recs.append({'rec': rec, 'dataset': 'goat', 'audio': str(audio.relative_to(REPO_ROOT)), 'player': r.player,
                         'guitar': f'goat-{r.guitar_brand}', 'chain': chain, 'content': 'pieces', 'split': 'test',
                         'duration': librosa.get_duration(path=audio)})  # fmt: skip
            notes += [{'rec': rec, 'onset': midi[k].start, 'offset': midi[k].end, 'pitch': midi[k].pitch,
                       'string': t['string'], 'fret': midi[k].pitch - tab['tuning'][t['string'] - 1]}
                      for k, t in sorted(matched.items())]  # fmt: skip
    out = REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(out / 'recordings.csv', index=False)
    pd.DataFrame(notes).to_csv(out / 'notes.csv.gz', index=False)
    coverage = pd.DataFrame(coverage)
    (out / 'report.json').write_text(json.dumps(coverage.to_dict('records'), indent=1))
    info(
        f'{len(recs)} recordings -> {out}; notes with a string: {coverage.with_string.sum()} of {coverage.notes.sum()}'
    )


if __name__ == '__main__':
    main()

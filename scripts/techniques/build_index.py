"""A note index with playing-technique labels, in the string index's format, for the technique classifier.

Recordings and note times come from indexes that are already aligned to the audio; this adds a `techniques` column
(comma-separated marks, empty for a plain note):
  * IDMT: its per-note styles, and the notes the string index leaves out (dead notes, bends, slides, harmonics),
    placed with the recording's shift; a recording of such notes only is not in the string index and takes the
    median shift of its guitar's recordings;
  * Guitar-TECHS: one technique per take, from the take's name;
  * GOAT: the tab's marks, as build_goat_index.py matched them to the audio.
Splits are the sources' own: by guitar (IDMT), by player (Guitar-TECHS, GOAT's test player), GOAT validation by item.

Writes <out_dir>/recordings.csv and notes.csv.gz; scripts/stage1/strings/build_index_cache.py caches its windows.

Usage:
  uv run scripts/stage1/strings/build_goat_index.py manifest=data/stage1_notes_z_refined/manifest.jsonl \
      "splits=[train,val]" out_dir=data/technique_index/goat_trainval
  uv run scripts/stage1/strings/build_goat_index.py retimed_notes=data/goat/test_retimed/notes.csv.gz \
      out_dir=data/technique_index/goat_test
  uv run scripts/techniques/build_index.py
  uv run scripts/stage1/strings/build_index_cache.py index_dir=data/technique_index augment=2 augment_far=1
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import soundfile as sf

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1 import string_data

IDMT_STYLES = {'MU': 'palm_mute', 'BE': 'bend', 'SL': 'slide', 'VI': 'vibrato', 'HA': 'harmonic', 'DN': 'dead'}
TECHS_TAKES = {'PalmMute': 'palm_mute', 'Vibrato': 'vibrato'}
RECORDING_COLUMNS = ['rec', 'dataset', 'audio', 'player', 'guitar', 'chain', 'content', 'split', 'duration']


def idmt_events(audio):
    xml = Path(audio).parent.parent / 'annotation' / f'{Path(audio).stem}.xml'
    events = []
    for e in ET.parse(REPO_ROOT / xml).getroot().iter('event'):
        styles = (e.findtext('excitationStyle'), e.findtext('expressionStyle'))
        onset = float(e.findtext('onsetSec'))
        events.append(
            {
                'onset': onset,
                'offset': float(e.findtext('offsetSec') or onset + 0.5),
                'pitch': int(e.findtext('pitch')),
                'string': 7 - int(e.findtext('stringNumber')),
                'fret': int(e.findtext('fretNumber')),
                'techniques': ','.join(IDMT_STYLES[s] for s in styles if s in IDMT_STYLES),
            }
        )
    return pd.DataFrame(events)


def idmt_notes(rec, audio, notes, match_s):
    """The recording's index notes with their styles, plus the annotated notes the index leaves out."""
    events = idmt_events(audio)
    gaps = []
    for n in notes.itertuples():
        same = events.loc[events['pitch'] == n.pitch, 'onset'].to_numpy()
        if len(same):
            gaps.append(n.onset - same[np.abs(same - n.onset).argmin()])
    shift = float(np.median(gaps)) if gaps else 0.0
    notes, taken, extra = notes.assign(techniques=''), set(), []
    for e in events.to_dict('records'):
        gap = (notes['onset'] - e['onset'] - shift).abs()
        near = gap[(notes['pitch'] == e['pitch']) & (gap <= match_s) & ~notes.index.isin(taken)]
        if len(near):
            notes.loc[near.idxmin(), 'techniques'] = e['techniques']
            taken.add(near.idxmin())
        elif re.search(r'dead|bend|slide|harmonic', e['techniques']):
            extra.append({**e, 'rec': rec, 'onset': e['onset'] + shift, 'offset': e['offset'] + shift})
    return pd.concat([notes, pd.DataFrame(extra)], ignore_index=True)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='index')
def main(cfg):
    base = REPO_ROOT / cfg.string_index
    recs = pd.read_csv(base / 'recordings.csv')
    recs = recs[recs['dataset'].isin(list(cfg.datasets))]
    notes = pd.read_csv(base / 'notes.csv.gz')
    by_rec = dict(tuple(notes[notes['rec'].isin(recs['rec'])].groupby('rec', sort=False)))
    parts = []
    for r in recs.itertuples():
        if r.dataset == 'idmt':
            parts.append(idmt_notes(r.rec, r.audio, by_rec[r.rec], cfg.match_s))
        else:
            take = r.rec.rsplit('/', 1)[1] if r.content == 'techniques' else ''
            parts.append(by_rec[r.rec].assign(techniques=TECHS_TAKES.get(take, '')))
    if 'idmt' in cfg.datasets:
        every = string_data.idmt()[0]
        shifts = recs[recs['dataset'] == 'idmt'].groupby('guitar')['shift_ms'].median() / 1000
        left_out = every[~every['rec'].isin(recs['rec'])].copy()
        left_out['duration'] = [sf.info(str(REPO_ROOT / a)).duration for a in left_out['audio']]
        for r in left_out.itertuples():
            shift = shifts.get(r.guitar, 0.0)
            events = idmt_events(r.audio)
            parts.append(events.assign(rec=r.rec, onset=events['onset'] + shift, offset=events['offset'] + shift))
        recs = pd.concat([recs, left_out])
    recs = [recs[RECORDING_COLUMNS]]
    for d in cfg.goat_indexes:
        goat = pd.read_csv(REPO_ROOT / d / 'recordings.csv')
        goat = goat[goat['chain'].isin(list(cfg.goat_chains))]
        goat_notes = pd.read_csv(REPO_ROOT / d / 'notes.csv.gz', keep_default_na=False)
        recs.append(goat[RECORDING_COLUMNS])
        parts.append(goat_notes[goat_notes['rec'].isin(goat['rec'])])
    recs, notes = pd.concat(recs, ignore_index=True), pd.concat(parts, ignore_index=True)
    notes = notes.sort_values(['rec', 'onset', 'pitch'], kind='stable').reset_index(drop=True)
    out = REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out / 'recordings.csv', index=False)
    notes.to_csv(out / 'notes.csv.gz', index=False)
    marks = notes.assign(
        split=notes['rec'].map(recs.set_index('rec')['split']), dataset=notes['rec'].str.split('/').str[0]
    )
    marks = (
        marks.assign(mark=marks['techniques'].str.split(','))
        .explode('mark')
        .reset_index(drop=True)
        .replace({'mark': {'': '(plain)'}})
    )
    info(
        f'{len(recs)} recordings, {len(notes)} notes -> {out}\n{pd.crosstab(marks["mark"], [marks["dataset"], marks["split"]])}'
    )


if __name__ == '__main__':
    main()

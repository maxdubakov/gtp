"""Count the playing-technique labels each dataset carries, by split.

A dataset labels techniques per note (IDMT's annotations, GOAT's tab matched to its aligned MIDI, AG-PT-set's onset
labels, EG-IPT's one file per note), per recording (Guitar-TECHS records one technique per take) or only in a score
the player read (GAPS, GOAT's unmatched tab notes). DadaGP has no audio: a sample of its tabs shows how often tab
writers mark each technique. AG-PT-set and EG-IPT have no split of ours yet: theirs is the guitar (AG-PT-set) or the
pickup setting (EG-IPT, whose six channels of one performance count once).

Writes `out`: dataset, split, labelled, technique, notes, recordings; technique `(all notes)` is the total.

Usage:
  uv run scripts/techniques/survey.py
  uv run scripts/techniques/survey.py "datasets=[idmt,guitar_techs]"
"""

import random
import re
import xml.etree.ElementTree as ET

import hydra
import pandas as pd
import pretty_midi

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.string_data import DATA, IDMT_ROOT, IDMT_VAL_GUITAR, TECHS_SPLIT, TECHS_TRACKS, midi_notes, slug
from gtp.stage1.tab_alignment import strings_by_alignment
from gtp.stage2.guitarpro import parse_gp_track

TOTAL = '(all notes)'
IDMT_STYLES = {'MU': 'palm_mute', 'BE': 'bend', 'SL': 'slide', 'VI': 'vibrato', 'HA': 'harmonic', 'DN': 'dead'}
TECHS_TAKES = {
    'Bendings': 'bend',
    'Harmonics': 'harmonic',
    'PinchHarmonics': 'pinch_harmonic',
    'PalmMute': 'palm_mute',
    'Vibrato': 'vibrato',
}
XML_MARKS = {
    'harmonic': 'harmonic',
    'hammer-on': 'hammer',
    'pull-off': 'hammer',
    'slur': 'slur',
    'glissando': 'slide',
    'slide': 'slide',
    'bend': 'bend',
    'staccato': 'staccato',
    'arpeggiate': 'arpeggiate',
    'tremolo': 'tremolo_picking',
    'trill-mark': 'trill',
}
AG_PT_MARKS = {
    'Palm Mute': ['palm_mute'],
    'nat_harm': ['harmonic'],
    'near_bridge': [],
    'over_soundhole': [],
    'snare-b': ['muted_string_hit'],
    'kick': ['body_hit'],
    'snare-a': ['body_hit'],
    'tom': ['body_hit'],
}
EG_IPT_MARKS = {'ordinario': [], 'muted': ['palm_mute'], 'harmonics': ['harmonic']}
DADAGP_GUITAR = re.compile(r'^(clean|distorted)\d+:note:')
DADAGP_MARKS = {'ghost_note': 'ghost', 'tremoloPicking': 'tremolo_picking', 'tremolo_picking': 'tremolo_picking'}


def idmt(cfg):
    for xml in sorted(IDMT_ROOT.glob('dataset[123]/**/annotation/*.xml')):
        if not (xml.parent.parent / 'audio' / f'{xml.stem}.wav').exists():
            continue
        root = ET.parse(xml).getroot()
        guitar = slug(root.findtext('globalParameter/instrumentModel') or '')
        split = 'val' if guitar == IDMT_VAL_GUITAR else 'train'
        for e in root.iter('event'):
            styles = (e.findtext('excitationStyle'), e.findtext('expressionStyle'))
            yield 'idmt', split, 'note', str(xml), [IDMT_STYLES[s] for s in styles if s in IDMT_STYLES]


def guitar_techs(cfg):
    for midi in sorted((DATA / 'guitar_techs').glob('P*_*/midi/*.mid')):
        player, content = midi.parts[-3].split('_', 1)
        technique = TECHS_TAKES.get(midi.stem.removeprefix('midi_')) if content == 'techniques' else None
        for _ in midi_notes(midi, TECHS_TRACKS):
            yield 'guitar_techs', TECHS_SPLIT[player], 'recording', str(midi), [technique] if technique else []


def goat(cfg):
    """Every tab note counts as read from the score; the ones matched to the aligned MIDI also as labelled per note."""
    root = REPO_ROOT / cfg.goat_dir
    meta = pd.read_csv(root / 'metadata.csv')
    for r in meta.dropna(subset=['finealigned_midi_path']).itertuples():
        try:
            tab = parse_gp_track(root / 'data' / r.gp5_path.split('/', 1)[1], 0, techniques=True)
        except ValueError:
            continue
        for n in tab['notes']:
            yield 'goat', r.split, 'score', r.item, n['techniques']
        midi_path = root / 'data' / r.finealigned_midi_path.split('/', 1)[1]
        if not midi_path.is_file() or (r.split == 'train' and r.alignment_f_measure_fine < cfg.goat_min_alignment):
            continue
        midi = [n for i in pretty_midi.PrettyMIDI(str(midi_path)).instruments for n in i.notes]
        for n in strings_by_alignment(tab['notes'], midi, cfg).values():
            yield 'goat', r.split, 'note', r.item, n['techniques']


def gaps(cfg):
    """The first part of each score (the second repeats it as a tab)."""
    root = REPO_ROOT / cfg.gaps_dir
    splits = pd.read_csv(root / 'gaps_metadata_with_splits.csv').set_index('id')['split'].fillna('unassigned')
    for xml in sorted((root / 'musicxml').glob('*.xml')):
        for note in ET.parse(xml).getroot().find('part').iter('note'):
            if note.find('pitch') is None:
                continue
            marks = {XML_MARKS[e.tag] for e in note.iter() if e.tag in XML_MARKS and e.get('type') != 'stop'}
            if note.find("tie[@type='stop']") is not None:
                marks.add('tie')
            if note.find('grace') is not None:
                marks.add('grace')
            yield 'gaps', splits.get(xml.stem, 'unassigned'), 'score', xml.stem, sorted(marks)


def ag_pt(cfg):
    meta = REPO_ROOT / cfg.ag_pt_dir / 'metadata'
    names = pd.read_csv(meta / 'expressive_techniques.csv').set_index('expressive_technique_id')['name_nospace']
    guitar = pd.read_csv(meta / 'files.csv').set_index('filename')['guitar_id']
    for n in pd.read_csv(meta / 'note_labels.csv').itertuples():
        marks = AG_PT_MARKS[names[n.expressive_technique_id]]
        yield 'ag_pt', f'guitar{guitar[n.audio_file_path]}', 'note', n.audio_file_path, marks


def eg_ipt(cfg):
    for wav in sorted((REPO_ROOT / cfg.eg_ipt_dir).glob('*/DI/*/*.wav')):
        pickup, _, technique = wav.parts[-4:-1]
        yield 'eg_ipt', pickup, 'note', str(wav), EG_IPT_MARKS.get(technique, [technique.replace('-', '_')])


def dadagp(cfg):
    files = sorted((REPO_ROOT / cfg.dadagp_dir).rglob('*.tokens.txt'))
    for path in random.Random(cfg.seed).sample(files, min(cfg.dadagp_files, len(files))):
        notes = []
        for token in path.read_text().split('\n'):
            guitar = DADAGP_GUITAR.match(token)
            if guitar:
                notes.append((guitar.group(1), []))
            elif token.startswith('nfx:') and notes and notes[-1] is not None:
                name = token.split(':')[1]
                notes[-1][1].append(DADAGP_MARKS.get(name, name))
            elif ':note:' in token:
                notes.append(None)
        for sound, marks in filter(None, notes):
            yield 'dadagp', sound, 'tab only', str(path), marks


LOADERS = {
    'idmt': idmt,
    'guitar_techs': guitar_techs,
    'goat': goat,
    'gaps': gaps,
    'dadagp': dadagp,
    'ag_pt': ag_pt,
    'eg_ipt': eg_ipt,
}


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='survey')
def main(cfg):
    rows = []
    for name in cfg.datasets:
        for dataset, split, labelled, rec, techniques in LOADERS[name](cfg):
            rows += [(dataset, split, labelled, rec, t) for t in [TOTAL, *dict.fromkeys(techniques)]]
        info(f'{name}: read')
    notes = pd.DataFrame(rows, columns=['dataset', 'split', 'labelled', 'rec', 'technique'])
    counts = (
        notes.groupby(['dataset', 'split', 'labelled', 'technique'])['rec']
        .agg(notes='size', recordings='nunique')
        .reset_index()
    )
    out = REPO_ROOT / cfg.out
    out.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out, index=False)
    table = counts.pivot_table(
        index='technique', columns=['dataset', 'labelled', 'split'], values='notes', fill_value=0
    )
    order = table.sum(axis=1).sort_values(ascending=False).index
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        info(f'notes per technique -> {out}\n{table.loc[order].astype(int)}')


if __name__ == '__main__':
    main()

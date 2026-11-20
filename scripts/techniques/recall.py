"""How the note model treats notes played with each technique, on its validation recordings.

Every labelled note is looked up in the transcription three ways: a note of its pitch starting within
`onset_tolerance` (found), a note within `near_pitch` semitones (near), any note at all (any). Labels come from the
sources themselves, so they include the notes the note model's training labels leave out: IDMT's dead notes, bends,
slides and harmonics, and Guitar-TECHS's takes of bends and harmonics (gtp.stage1.string_data). Each dataset is also
scored twice, against all its labels and against the ones training keeps.

Writes `out`: dataset, technique, notes, found, near, any; and prints both tables.

Usage:
  uv run scripts/techniques/recall.py
"""

import json
import xml.etree.ElementTree as ET

import hydra
import mir_eval
import numpy as np
import pandas as pd
import pretty_midi

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.inference import PianoTranscription, transcribe_cached
from gtp.stage1.string_data import (
    DATA,
    IDMT_ROOT,
    IDMT_UNFRETTED,
    IDMT_VAL_GUITAR,
    TECHS_SPLIT,
    TECHS_TRACKS,
    UNFRETTED,
    midi_notes,
    slug,
)
from gtp.stage1.tab_alignment import strings_by_alignment
from gtp.stage2.guitarpro import parse_gp_track

IDMT_STYLES = {'MU': 'palm_mute', 'BE': 'bend', 'SL': 'slide', 'VI': 'vibrato', 'HA': 'harmonic', 'DN': 'dead'}
TECHS_TAKES = {
    'Bendings': 'bend',
    'Harmonics': 'harmonic',
    'PinchHarmonics': 'pinch_harmonic',
    'PalmMute': 'palm_mute',
    'Vibrato': 'vibrato',
}
# Marks that say how long a note rings, not how it is played.
DURATION_MARKS = {'let_ring', 'staccato'}
PLAIN, UNMATCHED = '(plain)', '(not in the tab)'


def label(onset, pitch, techniques, in_training=True):
    return {'onset': onset, 'pitch': pitch, 'techniques': techniques, 'in_training': in_training}


def idmt(cfg):
    for xml in sorted(IDMT_ROOT.glob('dataset[123]/**/annotation/*.xml')):
        root = ET.parse(xml).getroot()
        audio = xml.parent.parent / 'audio' / f'{xml.stem}.wav'
        if slug(root.findtext('globalParameter/instrumentModel') or '') != IDMT_VAL_GUITAR or not audio.exists():
            continue
        labels = []
        for e in root.iter('event'):
            styles = (e.findtext('excitationStyle'), e.findtext('expressionStyle'))
            labels.append(
                label(
                    float(e.findtext('onsetSec')),
                    int(e.findtext('pitch')),
                    [IDMT_STYLES[s] for s in styles if s in IDMT_STYLES],
                    styles[1] not in IDMT_UNFRETTED,
                )
            )
        yield f'idmt/{xml.stem}', audio, labels, True


def guitar_techs(cfg):
    for midi in sorted((DATA / 'guitar_techs').glob('P*_*/midi/*.mid')):
        part, stem = midi.parts[-3], midi.stem.removeprefix('midi_')
        player, content = part.split('_', 1)
        if TECHS_SPLIT[player] != cfg.split:
            continue
        technique = TECHS_TAKES.get(stem) if content == 'techniques' else None
        kept = not (content == 'techniques' and UNFRETTED.search(stem))
        labels = [
            label(start, pitch, [technique] if technique else [], kept)
            for start, _, pitch, _ in midi_notes(midi, TECHS_TRACKS)
        ]
        audio = midi.parents[1] / 'audio' / 'directinput' / f'directinput_{stem}.wav'
        if audio.exists():
            yield f'guitar_techs/{part}/{stem}', audio, labels, True


def goat(cfg):
    """The manifest's notes are GOAT's aligned MIDI re-timed to the audio, in the MIDI's order."""
    root = REPO_ROOT / cfg.goat_dir
    meta = pd.read_csv(root / 'metadata.csv').set_index('item')
    manifest = REPO_ROOT / cfg.manifest
    with open(manifest) as fh:
        entries = [json.loads(line) for line in fh]
    for e in entries:
        if (e['dataset'], e['split'], e['chain']) != ('goat', cfg.split, 'di'):
            continue
        r = meta.loc[e['id'].split('/')[1]]
        tab = parse_gp_track(root / 'data' / r.gp5_path.split('/', 1)[1], 0, techniques=True)
        midi = [
            n
            for i in pretty_midi.PrettyMIDI(str(root / 'data' / r.finealigned_midi_path.split('/', 1)[1])).instruments
            for n in i.notes
        ]
        assert [n.pitch for n in midi] == [n['midi_note'] for n in e['notes']], e['id']
        matched = strings_by_alignment(tab['notes'], midi, cfg)
        labels = [
            label(n['onset_time'], n['midi_note'], matched[k]['techniques'] if k in matched else [UNMATCHED])
            for k, n in enumerate(e['notes'])
        ]
        yield e['id'], manifest.parent / e['audio'], labels, False


LOADERS = {'idmt': idmt, 'guitar_techs': guitar_techs, 'goat': goat}


def lag(labels, est, max_lag):
    gaps = []
    for onset, pitch in zip(labels['onset'], labels['pitch'], strict=True):
        d = est.loc[est['pitch'] == pitch, 'start'].to_numpy() - onset
        if len(d) and np.abs(d).min() < max_lag:
            gaps.append(d[np.abs(d).argmin()])
    return float(np.median(gaps)) if gaps else 0.0


def matches(onsets, pitches, est, tolerance):
    """Indices of the labels and of the transcribed notes paired one to one on pitch and onset."""
    if not len(onsets) or not len(est):
        return np.zeros(0, int), np.zeros(0, int)
    pairs = mir_eval.transcription.match_notes(
        np.column_stack([onsets, onsets + 0.1]),
        mir_eval.util.midi_to_hz(pitches),
        np.column_stack([est['start'], np.maximum(est['end'], est['start'] + 0.01)]),
        mir_eval.util.midi_to_hz(est['pitch'].to_numpy()),
        onset_tolerance=tolerance,
        offset_ratio=None,
    )
    return np.array([p[0] for p in pairs], int), np.array([p[1] for p in pairs], int)


def prf(found, labels, transcribed):
    p, r = found / max(transcribed, 1), found / max(labels, 1)
    return {'labels': labels, 'transcribed': transcribed, 'P': p, 'R': r, 'F1': 2 * p * r / max(p + r, 1e-9)}


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='recall')
def main(cfg):
    checkpoint = REPO_ROOT / cfg.stage1_checkpoint
    model = PianoTranscription(str(checkpoint), device=auto_device())
    cache = REPO_ROOT / 'results' / 'techniques' / 'stage1' / checkpoint.stem
    rows, scores = [], []
    for name in cfg.datasets:
        totals = {'all labels': np.zeros(3, int), 'training labels': np.zeros(3, int)}
        for rec, audio, labels, shift in LOADERS[name](cfg):
            est = pd.DataFrame(
                transcribe_cached(model, audio, cache / f'{slug(rec)}.json'), columns=['pitch', 'start', 'end']
            )
            labels = pd.DataFrame(labels)
            if shift:
                labels['onset'] += lag(labels, est, cfg.max_lag_s)
            onsets, pitches = labels['onset'].to_numpy(), labels['pitch'].to_numpy()
            found = np.zeros(len(labels), bool)
            found[matches(onsets, pitches, est, cfg.onset_tolerance)[0]] = True
            close = np.abs(est['start'].to_numpy()[None] - onsets[:, None]) <= cfg.onset_tolerance
            near = close & (np.abs(est['pitch'].to_numpy()[None] - pitches[:, None]) <= cfg.near_pitch)
            for k, techniques in enumerate(labels['techniques']):
                played = [t for t in techniques if t not in DURATION_MARKS] or [PLAIN]
                rows += [(name, t, found[k], near[k].any(), close[k].any()) for t in [*played, '(all)']]
            totals['all labels'] += (int(found.sum()), len(labels), len(est))
            kept = labels['in_training'].to_numpy()
            totals['training labels'] += (
                len(matches(onsets[kept], pitches[kept], est, cfg.onset_tolerance)[0]),
                int(kept.sum()),
                len(est),
            )
        scores += [{'dataset': name, 'scored against': k, **prf(*v)} for k, v in totals.items()]
        info(f'{name}: done')
    table = (
        pd.DataFrame(rows, columns=['dataset', 'technique', 'found', 'near', 'any'])
        .groupby(['dataset', 'technique'])
        .agg(notes=('found', 'size'), found=('found', 'mean'), near=('near', 'mean'), any=('any', 'mean'))
        .reset_index()
        .sort_values(['dataset', 'notes'], ascending=[True, False])
    )
    out = REPO_ROOT / cfg.out
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    scores = pd.DataFrame(scores)
    scores.to_csv(out.with_name(f'{out.stem}_scores.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.float_format', '{:.3f}'.format):
        info(f'{checkpoint.name} -> {out}\n{table.to_string(index=False)}\n\n{scores.to_string(index=False)}')


if __name__ == '__main__':
    main()

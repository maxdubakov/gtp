"""Tab notes for Stage 2 from the audio datasets that ship tablature: GOAT, GAPS and Guitar-TECHS.

Writes one JSON per piece in the processed format of the other Stage 2 sources (tempo, tuning high to low, capo, notes
with pitch, string, fret, start, end; pitch is the open string plus the fret, the capo kept apart, as for DadaGP) to
`out_dirs`:
  * GOAT: the Guitar Pro 5 file of each item outside its test split (the transcriber's fingering of the performance);
  * GAPS: the tab part of each MusicXML score (ClassClef tabs, crowdsourced) whose score is not in GAPS's test split,
    leaving out scores that look generated (`gaps_max_lowest_share` of notes on their lowest fret);
  * Guitar-TECHS: the string-labelled notes of the index's train recordings, one chain per take.

Usage:
  uv run scripts/stage2/data/build_tab_sources.py
"""

import json
import xml.etree.ElementTree as ET

import hydra
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage2.guitarpro import parse_gp_track

STEPS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def number(text):
    return int(float((text or '0').strip() or 0))


def write(out_dir, name, piece):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{name}.json').write_text(json.dumps(piece))


def goat(cfg):
    meta = pd.read_csv(REPO_ROOT / cfg.goat_dir / 'metadata.csv')
    kept, unreadable = 0, []
    for r in meta[meta.split != 'test'].itertuples():
        path = REPO_ROOT / cfg.goat_dir / 'data' / str(r.gp5_path).split('/', 1)[1]
        if not path.is_file():
            continue
        try:
            tab = parse_gp_track(path, 0)
        except ValueError as e:
            unreadable.append((r.item, str(e)))
            continue
        if len(tab['tuning']) != 6 or len(tab['notes']) < 5:
            continue
        write(
            REPO_ROOT / cfg.out_dirs.goat,
            r.item,
            {'source': 'goat', 'file': str(r.gp5_path), 'genre': 'unknown', **tab},
        )
        kept += 1
    if unreadable:
        info(f'GOAT files pyguitarpro cannot read: {unreadable}')
    return kept


def musicxml_tab(path):
    """(tuning high to low, capo, notes) of a score's tab part, or None if it has none."""
    root = ET.parse(path).getroot()
    for part in root.findall('part'):
        details = part.find('measure/attributes/staff-details')
        if details is None:
            continue
        lines = {
            int(t.get('line')): 12 * (number(t.findtext('tuning-octave')) + 1)
            + STEPS[t.findtext('tuning-step').strip()]
            + number(t.findtext('tuning-alter'))
            for t in details.findall('staff-tuning')
        }
        tuning = [lines[k] for k in sorted(lines, reverse=True)]
        capo = number(details.findtext('capo'))
        notes, divisions, tempo, measure_start = [], 1, 120.0, 0.0
        for measure in part.findall('measure'):
            pos = longest = last_start = 0
            for el in measure:
                if el.tag == 'attributes' and el.find('divisions') is not None:
                    divisions = number(el.findtext('divisions')) or divisions
                sound = el.find('sound') if el.tag == 'direction' else (el if el.tag == 'sound' else None)
                if sound is not None and sound.get('tempo'):
                    tempo = float(sound.get('tempo'))
                if el.tag == 'backup':
                    pos -= number(el.findtext('duration'))
                elif el.tag == 'forward':
                    pos += number(el.findtext('duration'))
                elif el.tag == 'note':
                    if el.find('grace') is not None:
                        continue
                    duration = number(el.findtext('duration'))
                    start = last_start if el.find('chord') is not None else pos
                    if el.find('chord') is None:
                        last_start, pos = pos, pos + duration
                    string, fret = el.find('notations/technical/string'), el.find('notations/technical/fret')
                    tie_stop = any(t.get('type') == 'stop' for t in el.findall('tie'))
                    if el.find('pitch') is not None and string is not None and fret is not None and not tie_stop:
                        s, f = int(string.text), int(fret.text)
                        seconds = 60.0 / tempo / divisions
                        notes.append({'pitch': tuning[s - 1] + f, 'string': s, 'fret': f,
                                      'start': round(measure_start + start * seconds, 4),
                                      'end': round(measure_start + (start + duration) * seconds, 4)})  # fmt: skip
                longest = max(longest, pos)
            measure_start += longest * 60.0 / tempo / divisions
        return tuning, capo, notes
    return None


def gaps(cfg):
    meta = pd.read_csv(REPO_ROOT / cfg.gaps_dir / 'gaps_metadata_with_splits.csv')
    test = meta[meta.split == 'test']
    held = set(test.id.str.split('_').str[1]) | set(test.title.dropna())
    titles = dict(zip(meta.id, meta.title, strict=True))
    kept = generated = 0
    for path in sorted((REPO_ROOT / cfg.gaps_dir / 'musicxml').glob('*.xml')):
        if path.stem.split('_')[-1] in held or titles.get(path.stem) in held:
            continue
        tab = musicxml_tab(path)
        if tab is None or len(tab[0]) != 6 or len(tab[2]) < 5:
            continue
        tuning, capo, notes = tab
        lowest = [n['fret'] == min(n['pitch'] - o for o in tuning if n['pitch'] >= o) for n in notes]
        if sum(lowest) / len(lowest) >= cfg.gaps_max_lowest_share:
            generated += 1
            continue
        write(REPO_ROOT / cfg.out_dirs.gaps, path.stem,
              {'source': 'gaps', 'file': path.name, 'tempo': None, 'tuning': tuning, 'capo': capo, 'genre': 'classical',
               'notes': notes})  # fmt: skip
        kept += 1
    return kept, generated


def guitar_techs(cfg):
    recs = pd.read_csv(REPO_ROOT / cfg.index_dir / 'recordings.csv')
    recs = recs[(recs.dataset == 'guitar_techs') & (recs.split == 'train') & (recs.chain == cfg.guitar_techs_chain)]
    notes = pd.read_csv(REPO_ROOT / cfg.index_dir / 'notes.csv.gz').groupby('rec')
    for r in recs.itertuples():
        g = notes.get_group(r.rec).sort_values('onset')
        write(REPO_ROOT / cfg.out_dirs.guitar_techs, r.rec.replace('/', '__'),
              {'source': 'guitar_techs', 'file': r.rec, 'tempo': None, 'tuning': [64, 59, 55, 50, 45, 40], 'capo': 0,
               'genre': 'unknown',
               'notes': [{'pitch': int(n.pitch), 'string': int(n.string), 'fret': int(n.fret),
                          'start': round(float(n.onset), 4), 'end': round(float(n.offset), 4)} for n in g.itertuples()]})  # fmt: skip
    return len(recs)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage2'), config_name='tab_sources')
def main(cfg):
    n_goat = goat(cfg)
    n_gaps, n_generated = gaps(cfg)
    n_gt = guitar_techs(cfg)
    info(f'GOAT {n_goat} pieces; GAPS {n_gaps} scores ({n_generated} left out as generated); Guitar-TECHS {n_gt} takes')


if __name__ == '__main__':
    main()

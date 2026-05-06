"""Build GuitarSet tab dataset from JAMS annotations.

GuitarSet has per-string note_midi annotations from hexaphonic pickups,
giving us ground-truth string assignment. Fret is computed from
round(pitch) - open_string_midi.
"""

import argparse
import json
import re

import jams
import pretty_midi

from gtp import REPO_ROOT
from gtp.stage2.genres import UNKNOWN

ANNOTATION_DIR = REPO_ROOT / 'data' / 'guitarset' / 'annotation'
OUTPUT_DIR = REPO_ROOT / 'data' / 'guitarset' / 'processed'

# JAMS annotation indices for note_midi per string (string 6=low E to string 1=high E)
NOTE_MIDI_INDICES = [1, 3, 5, 7, 9, 11]
STRING_NUMBERS = [6, 5, 4, 3, 2, 1]
OPEN_PITCHES = [40, 45, 50, 55, 59, 64]
TUNING = [64, 59, 55, 50, 45, 40]  # string 1 (high E) to string 6 (low E)

# Filename format: <player>_<Style><N>-<bpm>-<key>_<comp|solo>.json
# e.g. '00_Jazz1-150-C_solo.json'. 5 style codes, mapped to canonical genres.
GS_STYLE_TO_GENRE: dict[str, str] = {
    'Rock': 'rock',
    'Jazz': 'jazz',
    'Funk': 'funk',
    'BN':   'jazz',     # Bossa Nova, sub-bucket of jazz
    'SS':   'folk',     # Singer-Songwriter
}
_GS_FILENAME_RE = re.compile(r'^\d{2}_([A-Za-z]+)\d')


def classify_guitarset(filename: str) -> str:
    """Extract style code from GuitarSet filename and map to canonical bucket."""
    m = _GS_FILENAME_RE.match(filename)
    if not m:
        return UNKNOWN
    return GS_STYLE_TO_GENRE.get(m.group(1), UNKNOWN)


def process_one(jams_path):
    """Extract tab data from a GuitarSet JAMS file.

    Returns (notes, tempo) where notes is a list of {pitch, string, fret, start, end}
    and tempo is the BPM from the JAMS tempo annotation (audited to be present and
    high-confidence for all 360 files), or None if missing.
    """
    score = jams.load(str(jams_path))
    notes = []

    for string_num, ann_idx, open_pitch in zip(STRING_NUMBERS, NOTE_MIDI_INDICES, OPEN_PITCHES, strict=True):
        ann = score.annotations[ann_idx]
        for obs in ann.data:
            pitch = round(float(obs.value))
            fret = pitch - open_pitch
            notes.append(
                {
                    'pitch': pitch,
                    'string': string_num,
                    'fret': fret,
                    'start': round(float(obs.time), 4),
                    'end': round(float(obs.time) + float(obs.duration), 4),
                }
            )

    notes.sort(key=lambda n: (n['start'], n['pitch']))

    tempo_anns = score.search(namespace='tempo')
    tempo = float(tempo_anns[0].data[0].value) if tempo_anns and tempo_anns[0].data else None

    return notes, tempo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--info', action='store_true')
    args = parser.parse_args()

    jams_files = sorted(ANNOTATION_DIR.glob('*.jams'))
    print(f'JAMS files: {len(jams_files)}')

    if args.info:
        return

    entries = jams_files[: args.limit] if args.limit else jams_files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    failed = 0
    total_notes = 0

    for i, jams_path in enumerate(entries):
        name = jams_path.stem
        json_path = OUTPUT_DIR / f'{name}.json'
        mid_path = OUTPUT_DIR / f'{name}.mid'

        if json_path.exists() and mid_path.exists():
            skipped += 1
            continue

        try:
            notes, tempo = process_one(jams_path)
        except Exception as e:
            failed += 1
            print(f'[{i + 1:3d}/{len(entries)}] FAIL {name}: {e}')
            continue

        if len(notes) < 5:
            failed += 1
            continue

        tab_data = {
            'source': 'guitarset',
            'tuning': TUNING,
            'tempo': tempo,
            'genre': classify_guitarset(f'{name}.json'),
            'notes': notes,
        }
        with open(json_path, 'w') as f:
            json.dump(tab_data, f, indent=2)

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo or 120.0)
        guitar = pretty_midi.Instrument(program=25)
        for n in notes:
            guitar.notes.append(
                pretty_midi.Note(velocity=80, pitch=n['pitch'], start=n['start'], end=max(n['start'] + 0.01, n['end']))
            )
        midi.instruments.append(guitar)
        midi.write(str(mid_path))

        done += 1
        total_notes += len(notes)

        if (i + 1) % 50 == 0:
            print(f'  [{i + 1:3d}/{len(entries)}] done={done}')

    print('\n=== Summary ===')
    print(f'Processed: {done} ({total_notes:,} total notes)')
    print(f'Skipped: {skipped}')
    print(f'Failed: {failed}')
    print(f'Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

"""Build GuitarSet tab dataset from JAMS annotations.

GuitarSet has per-string note_midi annotations from hexaphonic pickups,
giving us ground-truth string assignment. Fret is computed from
round(pitch) - open_string_midi.

Usage:
    python scripts/data/guitarset/build_dataset.py
    python scripts/data/guitarset/build_dataset.py --limit 10
    python scripts/data/guitarset/build_dataset.py --info
"""

import os
import json
import argparse
import numpy as np
import jams
import pretty_midi
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ANNOTATION_DIR = REPO_ROOT / 'data' / 'guitarset' / 'annotation'
OUTPUT_DIR = REPO_ROOT / 'data' / 'guitarset' / 'processed'

# JAMS annotation indices for note_midi per string (string 6=low E to string 1=high E)
NOTE_MIDI_INDICES = [1, 3, 5, 7, 9, 11]
STRING_NUMBERS = [6, 5, 4, 3, 2, 1]
OPEN_PITCHES = [40, 45, 50, 55, 59, 64]
TUNING = [64, 59, 55, 50, 45, 40]  # string 1 (high E) to string 6 (low E)


def process_one(jams_path):
    """Extract tab data from a GuitarSet JAMS file.

    Returns (notes, n_notes) where notes is a list of {pitch, string, fret, start, end}.
    """
    score = jams.load(str(jams_path))
    notes = []

    for string_num, ann_idx, open_pitch in zip(STRING_NUMBERS, NOTE_MIDI_INDICES, OPEN_PITCHES):
        ann = score.annotations[ann_idx]
        for obs in ann.data:
            pitch = round(float(obs.value))
            fret = pitch - open_pitch
            notes.append({
                'pitch': pitch,
                'string': string_num,
                'fret': fret,
                'start': round(float(obs.time), 4),
                'end': round(float(obs.time) + float(obs.duration), 4),
            })

    notes.sort(key=lambda n: (n['start'], n['pitch']))
    return notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--info', action='store_true')
    args = parser.parse_args()

    jams_files = sorted(ANNOTATION_DIR.glob('*.jams'))
    print(f'JAMS files: {len(jams_files)}')

    if args.info:
        return

    entries = jams_files[:args.limit] if args.limit else jams_files
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
            notes = process_one(jams_path)
        except Exception as e:
            failed += 1
            print(f'[{i+1:3d}/{len(entries)}] FAIL {name}: {e}')
            continue

        if len(notes) < 5:
            failed += 1
            continue

        tab_data = {
            'source': 'guitarset',
            'tuning': TUNING,
            'notes': notes,
        }
        with open(json_path, 'w') as f:
            json.dump(tab_data, f, indent=2)

        midi = pretty_midi.PrettyMIDI()
        guitar = pretty_midi.Instrument(program=25)
        for n in notes:
            guitar.notes.append(pretty_midi.Note(
                velocity=80, pitch=n['pitch'],
                start=n['start'], end=max(n['start'] + 0.01, n['end'])))
        midi.instruments.append(guitar)
        midi.write(str(mid_path))

        done += 1
        total_notes += len(notes)

        if (i + 1) % 50 == 0:
            print(f'  [{i+1:3d}/{len(entries)}] done={done}')

    print(f'\n=== Summary ===')
    print(f'Processed: {done} ({total_notes:,} total notes)')
    print(f'Skipped: {skipped}')
    print(f'Failed: {failed}')
    print(f'Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

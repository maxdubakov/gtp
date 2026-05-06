"""Build the GuitarToday dataset from fetched Soundslice JSONs.

For each slice, produces:
  - {slice_id}.mid   — MIDI file (pitch + timing, for Stage 1)
  - {slice_id}.json  — Tab annotations (pitch + string + fret + timing, for Stage 2)

The tab JSON format per note:
  {"pitch": 43, "string": 6, "fret": 3, "start": 0.232, "end": 0.812}
"""

import argparse
import csv
import json

# Import the converter — it lives in the same directory
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from soundslice_to_midi import estimate_tempo_from_sync, notes_to_midi, parse_soundslice

from gtp import REPO_ROOT
from gtp.stage2.genres import UNKNOWN


# GuitarToday is one player's covers; not assignable to a genre.
def classify_guitartoday() -> str:
    return UNKNOWN

TEMPO_MIN = 40
TEMPO_MAX = 240

CATALOG_CSV = REPO_ROOT / 'data' / 'guitartoday' / 'posts.csv'
SLICES_DIR = REPO_ROOT / 'data' / 'guitartoday' / 'slices'
OUTPUT_DIR = REPO_ROOT / 'data' / 'guitartoday' / 'processed'


def load_catalog():
    rows = []
    with open(CATALOG_CSV) as f:
        for row in csv.DictReader(f):
            if row.get('soundslice_id', '').strip():
                rows.append(row)
    return rows


def process_slice(slice_id):
    """Convert one fetched slice to MIDI + tab JSON. Returns (status, note_count)."""
    slice_dir = SLICES_DIR / slice_id
    data_path = slice_dir / 'data.json'

    if not data_path.exists():
        return 'not fetched', 0

    midi_path = OUTPUT_DIR / f'{slice_id}.mid'
    tab_path = OUTPUT_DIR / f'{slice_id}.json'

    if midi_path.exists() and tab_path.exists():
        return 'skip', 0

    with open(data_path) as f:
        data = json.load(f)

    sync_path = slice_dir / 'sync.json'
    syncpoints = None
    if sync_path.exists():
        with open(sync_path) as f:
            syncpoints = json.load(f)

    # Estimate real tempo from sync (median across per-bar-pair segments).
    # Mark unknown if no sync, no result, or estimate is implausible.
    tempo = estimate_tempo_from_sync(syncpoints, data['bars']) if syncpoints else None
    if tempo is not None and not (TEMPO_MIN <= tempo <= TEMPO_MAX):
        tempo = None

    try:
        notes, _string_pitches = parse_soundslice(data, syncpoints=syncpoints, tempo_bpm=tempo or 120)
    except Exception as e:
        return f'parse error: {e}', 0

    if not notes:
        return 'no notes', 0

    notes_to_midi(notes, str(midi_path), tempo_bpm=tempo or 120)

    # Save tab annotations with string/fret info
    track_info = next((t for t in data['tracks'] if 'pitches' in t), data['tracks'][0])
    tuning = track_info['pitches']
    tab_data = {
        'slice_id': slice_id,
        'tuning': tuning,
        'tuning_names': [f'MIDI {p}' for p in tuning],
        'n_strings': track_info.get('strings', 6),
        'tempo': round(tempo, 2) if tempo is not None else None,
        'has_sync': syncpoints is not None,
        'genre': classify_guitartoday(),
        'notes': [
            {
                'pitch': n['pitch'],
                'string': n['string'],
                'fret': n['fret'],
                'start': round(n['start'], 4),
                'end': round(n['end'], 4),
            }
            for n in notes
        ],
    }
    with open(tab_path, 'w') as f:
        json.dump(tab_data, f, indent=2)

    return 'ok', len(notes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--info', action='store_true', help='Count available slices and exit')
    args = parser.parse_args()

    catalog = load_catalog()
    print(f'Catalog: {len(catalog)} entries with Soundslice IDs')

    fetched = sum(1 for r in catalog if (SLICES_DIR / r['soundslice_id'] / 'data.json').exists())
    print(f'Fetched: {fetched}')

    if args.info:
        return

    entries = catalog[: args.limit] if args.limit else catalog
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    not_fetched = 0
    failed = 0
    total_notes = 0

    for i, row in enumerate(entries):
        slice_id = row['soundslice_id']
        status, n_notes = process_slice(slice_id)

        if status == 'ok':
            done += 1
            total_notes += n_notes
        elif status == 'skip':
            skipped += 1
        elif status == 'not fetched':
            not_fetched += 1
        else:
            failed += 1

        if status not in ('skip', 'not fetched') or (i + 1) % 50 == 0:
            title = row.get('title', '')[:40]
            print(f'[{i + 1:3d}/{len(entries)}] {slice_id}: {status:<20} {title}')

    print('\n=== Summary ===')
    print(f'Processed: {done} ({total_notes} total notes)')
    print(f'Skipped (already exists): {skipped}')
    print(f'Not fetched yet: {not_fetched}')
    print(f'Failed: {failed}')
    print(f'Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

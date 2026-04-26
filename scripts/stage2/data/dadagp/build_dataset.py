"""Build DadaGP dataset by parsing GP3/4/5 files directly with pyguitarpro.

Reads acoustic_tracks.csv catalog (from filter_acoustic.py), parses each
GP file to extract pitch, string, fret, and timing from the beat durations.

Usage:
    python scripts/data/dadagp/build_dataset.py
    python scripts/data/dadagp/build_dataset.py --limit 50
    python scripts/data/dadagp/build_dataset.py --info
"""

import csv
import json
import argparse
import pretty_midi
import guitarpro as gp
from pathlib import Path

from gtp import REPO_ROOT
DADAGP_DIR = REPO_ROOT / 'data' / 'DadaGP-v1.1'
CATALOG_CSV = REPO_ROOT / 'data' / 'dadagp' / 'acoustic_tracks.csv'
OUTPUT_DIR = REPO_ROOT / 'data' / 'dadagp' / 'processed'

TICKS_PER_QUARTER = 960


def load_catalog():
    rows = []
    with open(CATALOG_CSV) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def parse_gp_track(gp_path, track_idx):
    """Parse a specific track from a GP file.

    Returns dict with: tempo, tuning, notes [{pitch, string, fret, start, end}]
    """
    song = gp.parse(str(gp_path))
    track = song.tracks[track_idx]
    tuning = [s.value for s in track.strings]
    n_strings = len(tuning)
    tempo = song.tempo

    notes = []
    current_tick = 0

    for measure in track.measures:
        header = measure.header
        if header.tempo and header.tempo.value:
            tempo = header.tempo.value

        tps = (tempo * TICKS_PER_QUARTER) / 60.0

        for voice in measure.voices:
            beat_tick = current_tick
            for beat in voice.beats:
                dur_ticks = beat.duration.time

                start_sec = beat_tick / tps
                end_sec = (beat_tick + dur_ticks) / tps

                for note in beat.notes:
                    # pyguitarpro: note.string is 1-indexed from highest string
                    # note.value is the fret number
                    pitch = tuning[note.string - 1] + note.value + track.offset

                    notes.append({
                        'pitch': pitch,
                        'string': note.string,
                        'fret': note.value,
                        'start': round(start_sec, 4),
                        'end': round(end_sec, 4),
                    })

                beat_tick += dur_ticks

        time_sig = header.timeSignature
        bar_ticks = int(TICKS_PER_QUARTER * 4 * time_sig.numerator / time_sig.denominator.value)
        current_tick += bar_ticks

    return {
        'tempo': tempo,
        'tuning': tuning,
        'notes': notes,
    }


def process_one(gp_rel_path, track_idx):
    """Process one track. Returns (status, n_notes)."""
    safe_name = gp_rel_path.replace('/', '_').replace(' ', '_')
    safe_name = Path(safe_name).stem
    json_path = OUTPUT_DIR / f'{safe_name}.json'
    mid_path = OUTPUT_DIR / f'{safe_name}.mid'

    if json_path.exists() and mid_path.exists():
        return 'skip', 0

    gp_path = DADAGP_DIR / gp_rel_path

    try:
        result = parse_gp_track(gp_path, track_idx)
    except Exception as e:
        return f'error: {e}', 0

    notes = result['notes']
    if len(notes) < 5:
        return 'too few notes', 0

    tab_data = {
        'source': 'dadagp',
        'file': gp_rel_path,
        'tempo': result['tempo'],
        'tuning': result['tuning'],
        'notes': notes,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(tab_data, f, indent=2)

    midi = pretty_midi.PrettyMIDI(initial_tempo=result['tempo'])
    guitar = pretty_midi.Instrument(program=25)
    for n in notes:
        if n['end'] <= n['start']:
            continue
        guitar.notes.append(pretty_midi.Note(
            velocity=80, pitch=n['pitch'],
            start=n['start'], end=n['end']))
    midi.instruments.append(guitar)
    midi.write(str(mid_path))

    return 'ok', len(notes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--info', action='store_true')
    args = parser.parse_args()

    catalog = load_catalog()
    print(f'Catalog: {len(catalog)} acoustic tracks')

    seen_files = set()
    unique_entries = []
    for row in catalog:
        if row['file'] not in seen_files:
            seen_files.add(row['file'])
            unique_entries.append(row)
    print(f'Unique files: {len(unique_entries)}')

    if args.info:
        return

    entries = unique_entries[:args.limit] if args.limit else unique_entries
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    too_few = 0
    failed = 0
    total_notes = 0

    for i, row in enumerate(entries):
        status, n_notes = process_one(row['file'], int(row['track_idx']))

        if status == 'ok':
            done += 1
            total_notes += n_notes
        elif status == 'skip':
            skipped += 1
        elif status == 'too few notes':
            too_few += 1
        else:
            failed += 1

        if (i + 1) % 500 == 0:
            print(f'  [{i+1:5d}/{len(entries)}] done={done} skip={skipped} '
                  f'too_few={too_few} fail={failed}')

    print(f'\n=== Summary ===')
    print(f'Processed: {done} ({total_notes:,} total notes)')
    print(f'Skipped (already exists): {skipped}')
    print(f'Too few notes: {too_few}')
    print(f'Failed: {failed}')
    print(f'Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

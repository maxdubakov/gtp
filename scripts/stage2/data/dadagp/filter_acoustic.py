"""Filter DadaGP for acoustic guitar tracks.

Scans all GP3/4/5 files, finds tracks with acoustic guitar MIDI instruments
(nylon=24/25, steel=26), and outputs a catalog CSV.
"""

import argparse
import csv
import os

import guitarpro as gp

from gtp import REPO_ROOT

DADAGP_DIR = REPO_ROOT / 'data' / 'DadaGP-v1.1'
OUTPUT_CSV = REPO_ROOT / 'data' / 'dadagp' / 'acoustic_tracks.csv'

# General MIDI guitar instruments:
# 24 = Acoustic Guitar (nylon)
# 25 = Acoustic Guitar (steel)
# 26 = Electric Guitar (jazz/clean)
# 27 = Electric Guitar (clean)
# 28 = Electric Guitar (muted)
# 29 = Overdriven Guitar
# 30 = Distortion Guitar
# 31 = Guitar Harmonics
DEFAULT_INSTRUMENTS = "25,26"


def find_gp_files():
    """Find all original GP files (skip pyguitarpro re-exports and token round-trips)."""
    gp_files = []
    for root, dirs, files in os.walk(DADAGP_DIR):
        for f in files:
            if not f.endswith(('.gp3', '.gp4', '.gp5')):
                continue
            if '.pygp.' in f or '.gp2tokens' in f:
                continue
            gp_files.append(os.path.join(root, f))
    return sorted(gp_files)


def scan_file(gp_path, instruments):
    """Parse a GP file and return list of matching guitar tracks.

    Returns list of dicts: {track_idx, track_name, instrument, n_strings, tuning, n_notes}
    """
    song = gp.parse(gp_path)
    results = []

    for i, track in enumerate(song.tracks):
        if track.isPercussionTrack:
            continue

        instrument = track.channel.instrument if hasattr(track.channel, 'instrument') else -1

        if instrument not in instruments:
            continue

        n_strings = len(track.strings) if hasattr(track, 'strings') else 0
        if n_strings < 4 or n_strings > 8:
            continue

        tuning = [s.value for s in track.strings] if hasattr(track, 'strings') else []

        n_notes = 0
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    n_notes += len(beat.notes)

        if n_notes < 10:
            continue

        results.append({
            'track_idx': i,
            'track_name': track.name,
            'instrument': instrument,
            'n_strings': n_strings,
            'tuning': ','.join(str(t) for t in tuning),
            'n_notes': n_notes,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', default=DEFAULT_INSTRUMENTS,
                        help=f'Comma-separated MIDI instrument IDs to filter for (default: {DEFAULT_INSTRUMENTS})')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--info', action='store_true')
    args = parser.parse_args()

    instruments = {int(x) for x in args.instruments.split(',')}

    gp_files = find_gp_files()
    print(f'Total GP files: {len(gp_files)}')
    print(f'Filtering for MIDI instruments: {sorted(instruments)}')

    if args.info:
        return

    if args.limit:
        gp_files = gp_files[:args.limit]
        print(f'Limited to {args.limit}')

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    scanned = 0
    errors = 0

    for i, gp_path in enumerate(gp_files):
        rel_path = os.path.relpath(gp_path, DADAGP_DIR)

        try:
            tracks = scan_file(gp_path, instruments)
            scanned += 1
        except Exception:
            errors += 1
            continue

        for t in tracks:
            rows.append({
                'file': rel_path,
                **t,
            })

        if (i + 1) % 500 == 0:
            print(f'  [{i+1:5d}/{len(gp_files)}] scanned={scanned} acoustic_tracks={len(rows)} errors={errors}')

    # Write catalog
    fields = ['file', 'track_idx', 'track_name', 'instrument', 'n_strings', 'tuning', 'n_notes']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print('\n=== Summary ===')
    print(f'Scanned: {scanned}')
    print(f'Errors: {errors}')
    print(f'Acoustic guitar tracks found: {len(rows)}')
    print(f'Unique files with acoustic guitar: {len(set(r["file"] for r in rows))}')

    # Instrument breakdown
    from collections import Counter
    inst_counts = Counter(r['instrument'] for r in rows)
    print('\nBy instrument:')
    inst_names = {24: 'Nylon Guitar', 25: 'Steel Guitar', 26: 'Jazz Guitar'}
    for inst, count in inst_counts.most_common():
        print(f'  {inst_names.get(inst, inst)}: {count}')

    print(f'\nCatalog written: {OUTPUT_CSV}')


if __name__ == '__main__':
    main()

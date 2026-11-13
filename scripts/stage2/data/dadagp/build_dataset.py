"""Build DadaGP dataset by parsing GP3/4/5 files directly with pyguitarpro.

Reads acoustic_tracks.csv catalog (from filter_acoustic.py), parses each
GP file to extract pitch, string, fret, and timing from the beat durations.
"""

import argparse
import csv
import json
from pathlib import Path

import pretty_midi

from gtp import REPO_ROOT
from gtp.stage2.genres import UNKNOWN
from gtp.stage2.guitarpro import parse_gp_track

DADAGP_DIR = REPO_ROOT / 'data' / 'DadaGP-v1.1'
CATALOG_CSV = REPO_ROOT / 'data' / 'dadagp' / 'acoustic_tracks.csv'
DADAGP_META_PATH = DADAGP_DIR / '_DadaGP_all_metadata.json'
OUTPUT_DIR = REPO_ROOT / 'data' / 'dadagp' / 'processed'

GENRE_RULES: list[tuple[str, list[str]]] = [
    ('classical', ['classical', 'baroque', 'romantic', 'opera']),
    ('jazz', ['jazz', 'bebop', 'swing', 'bossa', 'fusion', 'cool_jazz', 'latin_jazz']),
    ('funk', ['funk']),
    ('folk', ['folk', 'singer_songwriter', 'americana', 'bluegrass', 'celtic']),
    ('blues', ['blues']),
    ('metal', ['metal', 'thrash', 'doom', 'grindcore', 'djent', 'progressive_metal', 'death_metal', 'black_metal']),
    ('punk', ['punk', 'hardcore']),
    ('pop', ['pop', 'mellow_gold', 'permanent_wave']),
    ('rock', ['rock', 'grunge', 'shoegaze']),
]


def _load_dadagp_metadata() -> dict:
    """Map gp4_path → metadata dict (with `genre_tokens`).

    DadaGP keys are stored with a `.tokens.txt` suffix for LM training; we
    strip it so the keys match the relative GP path stored in our catalog.
    """
    if not DADAGP_META_PATH.exists():
        return {}
    raw = json.loads(DADAGP_META_PATH.read_text())
    out = {}
    for k, v in raw.items():
        key = k[: -len('.tokens.txt')] if k.endswith('.tokens.txt') else k
        out[key] = v
    return out


_DADAGP_META = _load_dadagp_metadata()


def classify_dadagp(genre_tokens: list[str] | None) -> str:
    """Coarse-grain a list of DadaGP `genre:*` tokens to one canonical bucket.

    Returns UNKNOWN when the tokens are missing, mark the piece as unknown,
    or fail to match any GENRE_RULES bucket (catches niche tags like
    `genre:indie_quebecois`, `genre:regional_mexican`, etc).
    """
    if not genre_tokens:
        return UNKNOWN
    tags = [g.replace('genre:', '') for g in genre_tokens if g.startswith('genre:')]
    if 'unknown_genre' in tags:
        return UNKNOWN
    for bucket, keywords in GENRE_RULES:
        for tag in tags:
            for kw in keywords:
                if kw in tag:
                    return bucket
    return UNKNOWN


def load_catalog():
    rows = []
    with open(CATALOG_CSV) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


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
        'capo': result['capo'],
        'genre': classify_dadagp(_DADAGP_META.get(gp_rel_path, {}).get('genre_tokens')),
        'notes': notes,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(tab_data, f, indent=2)

    midi = pretty_midi.PrettyMIDI(initial_tempo=result['tempo'] if result['tempo'] is not None else 120)
    guitar = pretty_midi.Instrument(program=25)
    for n in notes:
        if n['end'] <= n['start']:
            continue
        guitar.notes.append(pretty_midi.Note(velocity=80, pitch=n['pitch'], start=n['start'], end=n['end']))
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

    entries = unique_entries[: args.limit] if args.limit else unique_entries
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
            print(f'  [{i + 1:5d}/{len(entries)}] done={done} skip={skipped} too_few={too_few} fail={failed}')

    print('\n=== Summary ===')
    print(f'Processed: {done} ({total_notes:,} total notes)')
    print(f'Skipped (already exists): {skipped}')
    print(f'Too few notes: {too_few}')
    print(f'Failed: {failed}')
    print(f'Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

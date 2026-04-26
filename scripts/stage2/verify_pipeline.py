"""Pipeline verification script.

Independently re-parses original data sources and compares note-by-note against
our processed JSON to catch pipeline bugs.

Datasets verified:
  - GuitarSet: JAMS → our JSON (exact match expected, same library)
  - Leduc: GP → MuseScore MIDI → our JSON (cross-tool sanity check)
  - DadaGP: GP → pyguitarpro → our JSON (exact pitch/string/fret, ±50ms timing)

Usage:
    venv/bin/python scripts/stage2/verify_pipeline.py [--dataset {guitarset,leduc,dadagp,all}] [--limit N]
"""

import argparse
import csv
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import guitarpro as gp
import jams
import pretty_midi

from gtp import REPO_ROOT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GUITARSET_ANNOTATION_DIR = REPO_ROOT / 'data' / 'guitarset' / 'annotation'
GUITARSET_PROCESSED_DIR = REPO_ROOT / 'data' / 'guitarset' / 'processed'

LEDUC_GP_DIR = REPO_ROOT / 'data' / 'leduc' / 'gp_files'
LEDUC_PROCESSED_DIR = REPO_ROOT / 'data' / 'leduc' / 'processed'

DADAGP_DIR = REPO_ROOT / 'data' / 'DadaGP-v1.1'
DADAGP_CATALOG_CSV = REPO_ROOT / 'data' / 'dadagp' / 'acoustic_tracks.csv'
DADAGP_PROCESSED_DIR = REPO_ROOT / 'data' / 'dadagp' / 'processed'

MSCORE = '/opt/homebrew/bin/mscore'
TICKS_PER_QUARTER = 960

# Guitar MIDI program numbers recognised by MuseScore output
GUITAR_PROGRAMS = {24, 25, 26, 27, 28}

# ---------------------------------------------------------------------------
# GuitarSet
# ---------------------------------------------------------------------------
NOTE_MIDI_INDICES = [1, 3, 5, 7, 9, 11]
STRING_NUMBERS = [6, 5, 4, 3, 2, 1]
OPEN_PITCHES = [40, 45, 50, 55, 59, 64]


def parse_guitarset_jams(jams_path):
    """Re-parse a GuitarSet JAMS file; returns sorted list of note dicts."""
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
    return notes


def verify_guitarset(limit=None):
    print('=== GuitarSet Verification ===')

    jams_files = sorted(GUITARSET_ANNOTATION_DIR.glob('*.jams'))
    if limit:
        jams_files = jams_files[:limit]

    checked = 0
    perfect = 0
    mismatches = []

    for jams_path in jams_files:
        name = jams_path.stem
        json_path = GUITARSET_PROCESSED_DIR / f'{name}.json'

        if not json_path.exists():
            continue

        checked += 1
        with open(json_path) as f:
            proc = json.load(f)['notes']

        try:
            fresh = parse_guitarset_jams(jams_path)
        except Exception as e:
            mismatches.append(f'  {name}: parse error — {e}')
            continue

        if len(fresh) != len(proc):
            mismatches.append(f'  {name}: note count mismatch (expected {len(proc)}, got {len(fresh)})')
            continue

        mismatch_detail = None
        for i, (f, p) in enumerate(zip(fresh, proc, strict=True)):
            if f['pitch'] != p['pitch']:
                mismatch_detail = f'  {name}: pitch mismatch at note {i} (expected {p["pitch"]}, got {f["pitch"]})'
                break
            if f['string'] != p['string']:
                mismatch_detail = f'  {name}: string mismatch at note {i} (expected {p["string"]}, got {f["string"]})'
                break
            if f['fret'] != p['fret']:
                mismatch_detail = f'  {name}: fret mismatch at note {i} (expected {p["fret"]}, got {f["fret"]})'
                break
            if f['start'] != p['start'] or f['end'] != p['end']:
                mismatch_detail = (
                    f'  {name}: timing mismatch at note {i} (expected start={p["start"]}, got {f["start"]})'
                )
                break

        if mismatch_detail:
            mismatches.append(mismatch_detail)
        else:
            perfect += 1

    pct = 100 * perfect / checked if checked else 0.0
    print(f'Files checked: {checked}')
    print(f'Perfect matches: {perfect} ({pct:.1f}%)')
    print(f'Mismatches: {len(mismatches)}')
    for m in mismatches:
        print(m)
    print()


# ---------------------------------------------------------------------------
# Leduc
# ---------------------------------------------------------------------------


def mscore_notes(gp_path):
    """Convert a GP file with MuseScore and return (notes, error_string|None).

    Returns (list_of_pretty_midi_notes, None) on success, or (None, error_msg) on failure.
    """
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        mid_path = f.name
    try:
        result = subprocess.run(
            [MSCORE, str(gp_path), '-o', mid_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None, f'mscore exit {result.returncode}'
        if not os.path.exists(mid_path) or os.path.getsize(mid_path) == 0:
            return None, 'empty MIDI output'

        pm = pretty_midi.PrettyMIDI(mid_path)
        notes = []
        for inst in pm.instruments:
            if not inst.is_drum and inst.program in GUITAR_PROGRAMS:
                notes.extend(inst.notes)
        notes.sort(key=lambda n: n.start)
        return notes, None
    except subprocess.TimeoutExpired:
        return None, 'mscore timeout'
    except Exception as e:
        return None, str(e)
    finally:
        if os.path.exists(mid_path):
            os.unlink(mid_path)


def verify_leduc(limit=None):
    print('=== Leduc Verification (cross-tool: MuseScore MIDI) ===')

    gp_files = sorted(LEDUC_GP_DIR.glob('*.gp'))
    if limit:
        gp_files = gp_files[:limit]

    mscore_failures = 0
    checked = 0

    count_within_5pct = 0
    count_major_discrepancy = 0  # >20% difference
    pitch_set_matches = 0
    timing_matches = 0

    count_details = []
    pitch_details = []

    for gp_path in gp_files:
        name = gp_path.stem
        json_path = LEDUC_PROCESSED_DIR / f'{name}.json'
        if not json_path.exists():
            continue

        with open(json_path) as f:
            proc = json.load(f)

        ms_notes, _err = mscore_notes(gp_path)
        if ms_notes is None:
            mscore_failures += 1
            continue

        checked += 1
        proc_notes = proc['notes']

        # --- note count comparison ---
        proc_count = len(proc_notes)
        ms_count = len(ms_notes)
        if proc_count > 0:
            pct_diff = abs(ms_count - proc_count) / proc_count
        else:
            pct_diff = 1.0

        if pct_diff <= 0.05:
            count_within_5pct += 1
        elif pct_diff > 0.20:
            count_major_discrepancy += 1
            count_details.append(f'  {name}: proc={proc_count} mscore={ms_count} ({pct_diff:.0%} diff)')

        # --- pitch set comparison ---
        proc_pitches = set(n['pitch'] for n in proc_notes)
        ms_pitches = set(n.pitch for n in ms_notes)
        if proc_pitches == ms_pitches:
            pitch_set_matches += 1
        else:
            only_proc = proc_pitches - ms_pitches
            only_ms = ms_pitches - proc_pitches
            pitch_details.append(f'  {name}: only_in_proc={sorted(only_proc)[:5]} only_in_mscore={sorted(only_ms)[:5]}')

        # --- timing comparison (best-effort: align by onset order) ---
        proc_starts = sorted(n['start'] for n in proc_notes)
        ms_starts = sorted(n.start for n in ms_notes)
        n_compare = min(len(proc_starts), len(ms_starts))
        if n_compare > 0:
            within_100ms = sum(
                1
                for ps, ms in zip(proc_starts[:n_compare], ms_starts[:n_compare], strict=True)
                if abs(ps - ms) <= 0.100
            )
            timing_matches += within_100ms / n_compare >= 0.80  # file passes if 80% of notes within 100ms

    print(f'Files checked: {checked} ({mscore_failures} MuseScore failures skipped)')
    print(f'Note count: {count_within_5pct} within ±5%, {count_major_discrepancy} major discrepancies (>20%)')
    print(f'Pitch set match: {pitch_set_matches}/{checked}')
    print(f'Timing match (≥80% notes within 100ms): {timing_matches}/{checked}')

    if count_details:
        print('Major count discrepancies:')
        for d in count_details:
            print(d)
    if pitch_details:
        print('Pitch set mismatches:')
        for d in pitch_details:
            print(d)
    print()


# ---------------------------------------------------------------------------
# DadaGP
# ---------------------------------------------------------------------------


def load_dadagp_catalog():
    """Return (unique_rows, file_to_track_idx_map)."""
    with open(DADAGP_CATALOG_CSV) as f:
        rows = list(csv.DictReader(f))
    seen = set()
    unique = []
    for row in rows:
        if row['file'] not in seen:
            seen.add(row['file'])
            unique.append(row)
    file_to_track = {row['file']: int(row['track_idx']) for row in unique}
    return unique, file_to_track


def dadagp_json_path(gp_rel_path):
    safe_name = gp_rel_path.replace('/', '_').replace(' ', '_')
    safe_name = Path(safe_name).stem
    return DADAGP_PROCESSED_DIR / f'{safe_name}.json'


def parse_dadagp_track(gp_path, track_idx):
    """Re-parse a DadaGP GP file; returns list of note dicts."""
    song = gp.parse(str(gp_path))
    track = song.tracks[track_idx]
    tuning = [s.value for s in track.strings]
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
                    pitch = tuning[note.string - 1] + note.value + track.offset
                    notes.append(
                        {
                            'pitch': pitch,
                            'string': note.string,
                            'fret': note.value,
                            'start': round(start_sec, 4),
                            'end': round(end_sec, 4),
                        }
                    )
                beat_tick += dur_ticks

        time_sig = header.timeSignature
        bar_ticks = int(TICKS_PER_QUARTER * 4 * time_sig.numerator / time_sig.denominator.value)
        current_tick += bar_ticks

    return notes, track.offset


def compare_dadagp(fresh, proc):
    """Return (is_match, detail_string|None).

    Match criteria:
      - note count exact
      - all pitches exact (in order)
      - all string/fret exact (in order)
      - all timings within 50ms
    """
    if len(fresh) != len(proc):
        return False, f'note count: expected {len(proc)}, got {len(fresh)}'

    for i, (f, p) in enumerate(zip(fresh, proc, strict=True)):
        if f['pitch'] != p['pitch']:
            return False, f'note {i}: pitch expected {p["pitch"]}, got {f["pitch"]}'
        if f['string'] != p['string']:
            return False, f'note {i}: string expected {p["string"]}, got {f["string"]}'
        if f['fret'] != p['fret']:
            return False, f'note {i}: fret expected {p["fret"]}, got {f["fret"]}'
        start_mismatch = abs(f['start'] - p['start']) > 0.050
        end_mismatch = abs(f['end'] - p['end']) > 0.050
        if start_mismatch or end_mismatch:
            parts = []
            if start_mismatch:
                parts.append(f'start expected={p["start"]} got={f["start"]} (diff={abs(f["start"] - p["start"]):.4f}s)')
            if end_mismatch:
                parts.append(f'end expected={p["end"]} got={f["end"]} (diff={abs(f["end"] - p["end"]):.4f}s)')
            return False, f'note {i}: timing ' + '; '.join(parts)

    return True, None


def detect_capo_stems():
    """Return the set of processed JSON stems where ALL notes have the same
    non-zero constant diff ``pitch - (tuning[string-1] + fret)``.

    Reading only the small processed JSONs (no GP parsing) makes this fast even
    over thousands of files.
    """
    capo_stems = set()
    for jpath in DADAGP_PROCESSED_DIR.glob('*.json'):
        try:
            with open(jpath) as f:
                d = json.load(f)
            notes = d.get('notes', [])
            tuning = d.get('tuning', [])
            if not notes or not tuning:
                continue
            diffs = set()
            for n in notes:
                diffs.add(n['pitch'] - (tuning[n['string'] - 1] + n['fret']))
                if len(diffs) > 1:
                    break  # heterogeneous — not a uniform-offset file
            if len(diffs) == 1 and next(iter(diffs)) != 0:
                capo_stems.add(jpath.stem)
        except Exception:
            pass
    return capo_stems


def verify_dadagp(limit=None):
    print('=== DadaGP Verification ===')

    unique, file_to_track = load_dadagp_catalog()

    # Identify capo files by checking the processed JSONs: a file is a capo file
    # if every note satisfies pitch == tuning[string-1] + fret + constant_offset
    # where constant_offset != 0.  This is fast (JSON-only, no GP parsing) and
    # correctly captures all ~258 capo files regardless of track name.
    capo_stems = detect_capo_stems()

    # Map each catalog row to its processed JSON stem so we can tag it.
    def row_stem(r):
        return Path(dadagp_json_path(r['file'])).stem

    capo_rows = [r for r in unique if row_stem(r) in capo_stems]
    non_capo = [r for r in unique if row_stem(r) not in capo_stems]

    random.seed(42)
    random_sample = random.sample(non_capo, min(200, len(non_capo)))

    # Combine: random sample + all capo files detected from processed JSONs
    target_files = set(r['file'] for r in random_sample) | set(r['file'] for r in capo_rows)
    target_rows = [r for r in unique if r['file'] in target_files]

    if limit:
        target_rows = target_rows[:limit]

    parse_errors = 0
    checked = 0
    perfect = 0
    mismatches = []

    capo_offset_count = 0  # how many had nonzero GP track.offset

    for row in target_rows:
        gp_rel = row['file']
        jpath = dadagp_json_path(gp_rel)

        if not jpath.exists():
            continue

        with open(jpath) as f:
            proc_data = json.load(f)
        proc = proc_data['notes']

        # Use the file path stored in the processed JSON (the actual source GP file used
        # during build) rather than the catalog row's file — they can differ when
        # multiple GP versions of the same song exist (e.g. .gp3 and .gp4).
        actual_gp_rel = proc_data.get('file', gp_rel)
        track_idx = file_to_track.get(actual_gp_rel, int(row['track_idx']))
        gp_path = DADAGP_DIR / actual_gp_rel

        try:
            fresh, offset = parse_dadagp_track(gp_path, track_idx)
        except Exception:
            parse_errors += 1
            continue

        if offset != 0:
            capo_offset_count += 1

        checked += 1
        ok, detail = compare_dadagp(fresh, proc)

        if ok:
            perfect += 1
        else:
            mismatches.append(
                f'  {Path(gp_rel).stem[:60]}{" [capo offset=" + str(offset) + "]" if offset != 0 else ""}: {detail}'
            )

    pct = 100 * perfect / checked if checked else 0.0
    print(
        f'Files checked: {checked} ({len(random_sample)} random + {len(capo_rows)} capo-detected, '
        f'{capo_offset_count} had nonzero GP offset, {parse_errors} parse errors skipped)'
    )
    print(f'Perfect matches: {perfect} ({pct:.1f}%)')
    print(f'Mismatches: {len(mismatches)}')

    if mismatches:
        # Look for patterns
        capo_mismatches = [m for m in mismatches if '[capo' in m]
        non_capo_mismatches = [m for m in mismatches if '[capo' not in m]
        if capo_mismatches:
            print(f'  Capo-related mismatches ({len(capo_mismatches)}):')
            for m in capo_mismatches:
                print(m)
        if non_capo_mismatches:
            print(f'  Other mismatches ({len(non_capo_mismatches)}):')
            for m in non_capo_mismatches:
                print(m)

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='Verify preprocessing pipeline by re-parsing original sources.')
    parser.add_argument(
        '--dataset',
        choices=['guitarset', 'leduc', 'dadagp', 'all'],
        default='all',
        help='Which dataset to verify (default: all)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Max files per dataset (for quick test runs)',
    )
    args = parser.parse_args()

    run_all = args.dataset == 'all'
    run_guitarset = run_all or args.dataset == 'guitarset'
    run_leduc = run_all or args.dataset == 'leduc'
    run_dadagp = run_all or args.dataset == 'dadagp'

    if run_guitarset:
        verify_guitarset(limit=args.limit)
    if run_leduc:
        verify_leduc(limit=args.limit)
    if run_dadagp:
        verify_dadagp(limit=args.limit)


if __name__ == '__main__':
    main()

"""Stage 2 data audit: stream processed JSONs and print compact quality stats."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage2.metrics import pitch_of

DATASETS = ['dadagp', 'guitarset', 'guitartoday', 'leduc']
DATA_ROOT = REPO_ROOT / 'data'

STANDARD_TUNING = [64, 59, 55, 50, 45, 40]
DUR_BINS = np.logspace(-2, 2, 80)
ONSET_GAP_BINS = np.logspace(-3, 2, 80)


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def json_files(dataset: str) -> list[Path]:
    return sorted((DATA_ROOT / dataset / 'processed').glob('*.json'))


def add_flag(bucket, path, **extra):
    bucket.append({'file': path.name, **extra})


def classify_inconsistency(path, n_notes, diffs, buckets):
    unique = sorted(set(diffs))
    row = {'file': path.name, 'n_notes': n_notes, 'n_inconsistent': len(diffs), 'diffs': unique}

    if len(unique) == 1 and len(diffs) == n_notes:
        buckets['capo'].append({**row, 'offset': unique[0]})
    elif len(unique) == 1 or all(d in (12, 19, 24) for d in unique):
        buckets['harmonic'].append(row)
    else:
        buckets['bug'].append(row)


def analyse_dataset(dataset: str) -> dict:
    files = json_files(dataset)
    info(f'[ds:{dataset}] {len(files)} files')

    stats = {
        'dataset': dataset,
        'n_files': len(files),
        'total_notes': 0,
        'total_inconsistent': 0,
        'pitch_counts': Counter(),
        'string_counts': Counter(),
        'fret_counts': Counter(),
        'notes_per_piece': [],
        'piece_durations': [],
        'tuning_counts': Counter(),
        'std_tuning_notes': 0,
        'dur_hist': np.zeros(len(DUR_BINS) - 1, dtype=np.int64),
        'dur_min': np.inf,
        'dur_max': -np.inf,
        'onset_gap_hist': np.zeros(len(ONSET_GAP_BINS) - 1, dtype=np.int64),
        'onset_gap_large': 0,
        'capo_files': [],
        'harmonic_files': [],
        'bug_files': [],
        'neg_fret_notes': [],
        'high_fret_notes': [],
        'zero_dur_notes': [],
        'long_notes': [],
        'short_pieces': [],
    }
    inconsistency_buckets = {
        'capo': stats['capo_files'],
        'harmonic': stats['harmonic_files'],
        'bug': stats['bug_files'],
    }

    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            info(f'[ds:{dataset}] WARN cannot read {path.name}: {exc}')
            continue

        tuning = data.get('tuning', STANDARD_TUNING)
        notes = data.get('notes', [])
        n_notes = len(notes)

        stats['total_notes'] += n_notes
        stats['notes_per_piece'].append(n_notes)
        stats['tuning_counts'][tuple(tuning)] += 1
        if tuple(tuning) == tuple(STANDARD_TUNING):
            stats['std_tuning_notes'] += n_notes
        if n_notes < 10:
            stats['short_pieces'].append(path.name)

        starts = []
        piece_end = 0.0
        diffs = []
        durations = []

        for note in notes:
            pitch = note['pitch']
            string = note['string']
            fret = note['fret']
            start = note['start']
            end = note['end']
            dur = end - start

            stats['pitch_counts'][pitch] += 1
            stats['string_counts'][string] += 1
            stats['fret_counts'][fret] += 1
            starts.append(start)
            durations.append(dur)
            piece_end = max(piece_end, end)

            if dur <= 0:
                add_flag(stats['zero_dur_notes'], path)
            if dur > 10:
                add_flag(stats['long_notes'], path, duration=round(dur, 2))
            if fret < 0:
                add_flag(stats['neg_fret_notes'], path, fret=fret)
            if fret > 24:
                add_flag(stats['high_fret_notes'], path, fret=fret)

            expected = pitch_of((string, fret), tuning)
            if expected is not None and pitch != expected:
                diffs.append(pitch - expected)

        if durations:
            durs = np.array(durations)
            stats['dur_min'] = min(stats['dur_min'], float(durs.min()))
            stats['dur_max'] = max(stats['dur_max'], float(durs.max()))
            stats['dur_hist'] += np.histogram(durs[durs > 0], bins=DUR_BINS)[0]

        if len(starts) > 1:
            gaps = np.diff(np.sort(np.array(starts)))
            gaps_pos = gaps[gaps > 0]
            stats['onset_gap_hist'] += np.histogram(gaps_pos, bins=ONSET_GAP_BINS)[0]
            stats['onset_gap_large'] += int((gaps_pos > 30).sum())

        stats['piece_durations'].append(piece_end)
        stats['total_inconsistent'] += len(diffs)
        if diffs:
            classify_inconsistency(path, n_notes, diffs, inconsistency_buckets)

    pitches = sorted(stats['pitch_counts'])
    frets = sorted(stats['fret_counts'])
    stats['pitch_range'] = (min(pitches), max(pitches)) if pitches else (None, None)
    stats['fret_range'] = (min(frets), max(frets)) if frets else (None, None)
    stats['dur_min'] = stats['dur_min'] if np.isfinite(stats['dur_min']) else 0.0
    stats['dur_max'] = stats['dur_max'] if np.isfinite(stats['dur_max']) else 0.0
    return stats


def range_str(pair):
    lo, hi = pair
    return f'{lo}-{hi}' if lo is not None else '-'


def print_summary_table(stats_list: list[dict]):
    header = (
        f'{"Dataset":<14} {"Files":>6} {"Notes":>9} {"Pitch":>10} {"Fret":>8} '
        f'{"Dur (s)":>14} {"Piece (s)":>14} {"% Std":>7} {"% Bad":>7}'
    )
    print('\n=== SUMMARY ===')
    print(header)
    print('-' * len(header))

    totals = defaultdict(int)
    pitch_min, pitch_max = np.inf, -np.inf
    fret_min, fret_max = np.inf, -np.inf
    dur_min, dur_max = np.inf, -np.inf
    piece_min, piece_max = np.inf, -np.inf

    for stats in stats_list:
        pieces = stats['piece_durations']
        pmin, pmax = stats['pitch_range']
        fmin, fmax = stats['fret_range']
        pdmin, pdmax = (min(pieces), max(pieces)) if pieces else (0.0, 0.0)

        print(
            f'{stats["dataset"]:<14} {stats["n_files"]:>6,} {stats["total_notes"]:>9,} '
            f'{range_str(stats["pitch_range"]):>10} {range_str(stats["fret_range"]):>8} '
            f'{stats["dur_min"]:>6.2f}-{stats["dur_max"]:<6.1f} '
            f'{pdmin:>6.0f}-{pdmax:<6.0f} '
            f'{pct(stats["std_tuning_notes"], stats["total_notes"]):>6.1f}% '
            f'{pct(stats["total_inconsistent"], stats["total_notes"]):>6.2f}%'
        )

        totals['files'] += stats['n_files']
        totals['notes'] += stats['total_notes']
        totals['bad'] += stats['total_inconsistent']
        totals['std'] += stats['std_tuning_notes']
        if pmin is not None:
            pitch_min, pitch_max = min(pitch_min, pmin), max(pitch_max, pmax)
        if fmin is not None:
            fret_min, fret_max = min(fret_min, fmin), max(fret_max, fmax)
        dur_min, dur_max = min(dur_min, stats['dur_min']), max(dur_max, stats['dur_max'])
        piece_min, piece_max = min(piece_min, pdmin), max(piece_max, pdmax)

    print('-' * len(header))
    print(
        f'{"combined":<14} {totals["files"]:>6,} {totals["notes"]:>9,} '
        f'{int(pitch_min):>4}-{int(pitch_max):<5} {int(fret_min):>3}-{int(fret_max):<4} '
        f'{dur_min:>6.2f}-{dur_max:<6.1f} {piece_min:>6.0f}-{piece_max:<6.0f} '
        f'{pct(totals["std"], totals["notes"]):>6.1f}% {pct(totals["bad"], totals["notes"]):>6.2f}%'
    )


def file_counts(items):
    return Counter(item['file'] for item in items)


def flag_counts(items, value_key=None):
    by_file = defaultdict(lambda: [0, set()])
    for item in items:
        row = by_file[item['file']]
        row[0] += 1
        if value_key:
            row[1].add(item[value_key])

    rows = []
    for file, (count, values) in by_file.items():
        vals = sorted(values)
        rows.append((file, count, vals))
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def flag_fmt(label):
    def _fmt(row):
        file, count, values = row
        value = ''
        if values:
            value = f' {label}={values[0]}' if len(values) == 1 else f' {label}={values[0]}..{values[-1]}'
        return f'{count} notes{value}  {file}'

    return _fmt


def show_rows(rows, limit, fmt):
    for row in rows[:limit]:
        print('  ' + fmt(row))
    if len(rows) > limit:
        print(f'  ... {len(rows) - limit} more')


def print_consistency_report(stats_list: list[dict]):
    print('\n=== PITCH CONSISTENCY ===')
    for stats in stats_list:
        notes = stats['total_notes']
        bad = stats['total_inconsistent']
        print(
            f'{stats["dataset"]:<12} bad={bad:>7,}/{notes:<9,} ({pct(bad, notes):>5.2f}%)  '
            f'capo={len(stats["capo_files"]):>3}  harmonic={len(stats["harmonic_files"]):>3}  '
            f'mixed={len(stats["bug_files"]):>3}'
        )

    print('\nDetails:')
    for stats in stats_list:
        rows = (
            [('capo', r, f'offset={r["offset"]:+d}') for r in stats['capo_files']]
            + [('harmonic', r, f'diffs={r["diffs"]}') for r in stats['harmonic_files']]
            + [('mixed', r, f'diffs={r["diffs"]}') for r in stats['bug_files']]
        )
        if not rows:
            continue
        print(f'[{stats["dataset"]}]')
        show_rows(
            rows,
            15,
            lambda x: f'{x[0]:<8} n={x[1]["n_inconsistent"]:>5,}/{x[1]["n_notes"]:<5,} {x[2]}  {x[1]["file"]}',
        )


def print_red_flags(stats_list: list[dict]):
    print('\n=== RED FLAGS ===')
    groups = [
        ('negative fret', 'neg_fret_notes', lambda rows: flag_counts(rows, 'fret'), flag_fmt('fret')),
        ('fret > 24', 'high_fret_notes', lambda rows: flag_counts(rows, 'fret'), flag_fmt('fret')),
        (
            'zero/negative duration',
            'zero_dur_notes',
            lambda rows: file_counts(rows).most_common(),
            lambda r: f'{r[1]} notes  {r[0]}',
        ),
        ('duration > 10s', 'long_notes', lambda rows: flag_counts(rows, 'duration'), flag_fmt('dur')),
        ('<10 notes', 'short_pieces', lambda rows: rows, lambda r: r),
    ]

    for label, key, make_rows, formatter in groups:
        print(f'\n{label}:')
        for stats in stats_list:
            raw = stats[key]
            rows = make_rows(raw)
            total = len(raw)
            print(f'  {stats["dataset"]:<12} {total:>6,}')
            show_rows(rows, 10, formatter)


def main():
    info('[analyze-data] start')
    stats_list = [analyse_dataset(ds) for ds in DATASETS]
    print_summary_table(stats_list)
    print_consistency_report(stats_list)
    print_red_flags(stats_list)
    info('[analyze-data] done')


if __name__ == '__main__':
    main()

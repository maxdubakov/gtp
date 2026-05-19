"""Stage 2 tempo audit: bucket processed piece tempos by source."""

import argparse
import json
import statistics
from collections import Counter

from gtp.log import info
from gtp.stage2.paths import PROCESSED_DIRS
from gtp.utils import pct

DEFAULT_TEMPO = 120


def bucket(tempo):
    return None if tempo is None else round(float(tempo))


def fmt(value):
    return '-' if value is None else f'{value:.1f}'


def tempo_label(tempo):
    return 'unknown' if tempo is None else f'{tempo:>3d}'


def analyse_source(source):
    path = PROCESSED_DIRS[source]
    tempos = [json.loads(file.read_text()).get('tempo') for file in sorted(path.glob('*.json'))]
    known = [float(t) for t in tempos if t is not None]
    buckets = Counter(bucket(t) for t in tempos)

    return {
        'source': source,
        'n': len(tempos),
        'known': len(known),
        'unknown': buckets[None],
        'default': buckets[DEFAULT_TEMPO],
        'min': min(known) if known else None,
        'median': statistics.median(known) if known else None,
        'mean': statistics.mean(known) if known else None,
        'max': max(known) if known else None,
        'n_buckets': sum(1 for b in buckets if b is not None),
        'buckets': buckets,
    }


def main():
    ap = argparse.ArgumentParser(description='Audit tempo distribution in processed Stage 2 JSONs')
    ap.add_argument('--top', type=int, default=10, help='Show top N tempo buckets per source. Use 0 to hide.')
    args = ap.parse_args()

    info('[analyze-tempo] start')
    rows = [analyse_source(src) for src, path in PROCESSED_DIRS.items() if path.exists()]

    header = (
        f'{"Source":<14} {"Pieces":>6} {"Known":>6} {"Unk":>6} {"Unk%":>6} {"120":>6} {"120%":>6} '
        f'{"Min":>7} {"Median":>7} {"Mean":>7} {"Max":>7} {"Buckets":>7}'
    )
    print('\n=== TEMPO SUMMARY ===')
    print(header)
    print('-' * len(header))

    totals = Counter()
    all_buckets = Counter()
    for row in rows:
        totals.update({k: row[k] for k in ('n', 'known', 'unknown', 'default')})
        all_buckets.update(row['buckets'])
        print(
            f'{row["source"]:<14} {row["n"]:>6,} {row["known"]:>6,} '
            f'{row["unknown"]:>6,} {pct(row["unknown"], row["n"]):>5.1f}% '
            f'{row["default"]:>6,} {pct(row["default"], row["n"]):>5.1f}% '
            f'{fmt(row["min"]):>7} {fmt(row["median"]):>7} {fmt(row["mean"]):>7} '
            f'{fmt(row["max"]):>7} {row["n_buckets"]:>7}'
        )

    print('-' * len(header))
    print(
        f'{"combined":<14} {totals["n"]:>6,} {totals["known"]:>6,} '
        f'{totals["unknown"]:>6,} {pct(totals["unknown"], totals["n"]):>5.1f}% '
        f'{totals["default"]:>6,} {pct(totals["default"], totals["n"]):>5.1f}% '
        f'{"-":>7} {"-":>7} {"-":>7} {"-":>7} {sum(1 for b in all_buckets if b is not None):>7}'
    )

    if args.top > 0:
        print(f'\n=== TOP {args.top} BUCKETS ===')
        for row in rows:
            print(f'\n[{row["source"]}]')
            for tempo, count in row['buckets'].most_common(args.top):
                marker = ' default' if tempo == DEFAULT_TEMPO else ''
                print(f'  tempo={tempo_label(tempo)}  pieces={count:>5,}  ({pct(count, row["n"]):>4.1f}%){marker}')

    info('[analyze-tempo] done')


if __name__ == '__main__':
    main()

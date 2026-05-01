"""Tempo metadata distribution per source.

Reads only the `tempo` field from each processed JSON to surface how often
the MIDI default of 120 dominates a source — a strong signal that the
original transcriber didn't bother setting tempo, so the value is unreliable.
"""

import argparse
import json
from collections import Counter

from gtp.stage2.data import PROCESSED_DIRS

DEFAULT_TEMPO = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=10, help='Show top N most common tempos per source')
    args = ap.parse_args()

    grand_total = Counter()
    for source in PROCESSED_DIRS:
        path = PROCESSED_DIRS[source]
        if not path.exists():
            continue

        tempos = []
        for f in sorted(path.glob('*.json')):
            with open(f) as fh:
                data = json.load(fh)
            tempos.append(data.get('tempo', DEFAULT_TEMPO))

        n = len(tempos)
        counts = Counter(tempos)
        grand_total.update(tempos)
        default_count = counts.get(DEFAULT_TEMPO, 0)
        unique = len(counts)
        non_default = n - default_count

        print(f'\n{source}: {n} pieces, {unique} unique tempos')
        print(f'  tempo=120 (default): {default_count} ({100 * default_count / n:.1f}%)')
        print(f'  non-default:         {non_default} ({100 * non_default / n:.1f}%)')
        print(f'  top {args.top} tempos:')
        for tempo, count in counts.most_common(args.top):
            marker = '  ← default' if tempo == DEFAULT_TEMPO else ''
            print(f'    tempo={tempo:>5.1f}  pieces={count:>5}  ({100 * count / n:.1f}%){marker}')

    n_total = sum(grand_total.values())
    if n_total > 0:
        default_total = grand_total.get(DEFAULT_TEMPO, 0)
        print(f'\noverall: {n_total} pieces, tempo=120 in {default_total} ({100 * default_total / n_total:.1f}%)')


if __name__ == '__main__':
    main()

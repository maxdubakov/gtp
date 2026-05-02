"""Tempo distribution per source.

After the tempo-extraction fixes (Leduc MP3 + librosa, GuitarSet JAMS, GuitarToday
sync), tempo values can be:
  - integer (DadaGP, GuitarSet — set values from authoring tools)
  - float (Leduc, GuitarToday — derived from audio analysis)
  - null (unknown — Leduc files with no MP3, GuitarToday outliers)

Histogram bucketing rounds tempos to nearest integer so float vs. integer pieces are
counted comparably. `tempo=null` is reported separately.
"""

import argparse
import json
import statistics
from collections import Counter

from gtp.stage2.paths import PROCESSED_DIRS

DEFAULT_TEMPO = 120


def bucket(tempo):
    return None if tempo is None else round(float(tempo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=10, help='Show top N most common tempo buckets per source')
    args = ap.parse_args()

    grand_total = Counter()
    grand_unknown = 0
    grand_default = 0
    for source in PROCESSED_DIRS:
        path = PROCESSED_DIRS[source]
        if not path.exists():
            continue

        tempos_raw = []
        for f in sorted(path.glob('*.json')):
            with open(f) as fh:
                data = json.load(fh)
            tempos_raw.append(data.get('tempo'))

        n = len(tempos_raw)
        unknown = sum(1 for t in tempos_raw if t is None)
        known = [float(t) for t in tempos_raw if t is not None]
        buckets = Counter(bucket(t) for t in tempos_raw)
        default_count = buckets.get(DEFAULT_TEMPO, 0)

        grand_total.update(buckets)
        grand_unknown += unknown
        grand_default += default_count

        print(f'\n{source}: {n} pieces')
        print(f'  unknown (null):    {unknown} ({100 * unknown / n:.1f}%)')
        print(f'  tempo≈120 bucket:  {default_count} ({100 * default_count / n:.1f}%)')
        if known:
            print(
                f'  known stats: min={min(known):.1f}  max={max(known):.1f}  '
                f'median={statistics.median(known):.1f}  mean={statistics.mean(known):.1f}  '
                f'unique buckets={len([b for b in buckets if b is not None])}'
            )
        print(f'  top {args.top} buckets:')
        for tempo, count in buckets.most_common(args.top):
            label = 'unknown' if tempo is None else f'{tempo:>3d}'
            marker = '  ← MIDI default' if tempo == DEFAULT_TEMPO else ''
            print(f'    tempo={label}  pieces={count:>5}  ({100 * count / n:.1f}%){marker}')

    n_total = sum(grand_total.values())
    if n_total > 0:
        print(f'\noverall: {n_total} pieces')
        print(f'  unknown:   {grand_unknown} ({100 * grand_unknown / n_total:.1f}%)')
        print(f'  ≈120 bucket: {grand_default} ({100 * grand_default / n_total:.1f}%)')


if __name__ == '__main__':
    main()

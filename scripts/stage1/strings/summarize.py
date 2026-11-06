"""Tabulate every string-classifier run in results/strings, sorted by val NLL (the selection metric).

Usage:
  ./venv/bin/python scripts/stage1/strings/summarize.py
  ./venv/bin/python scripts/stage1/strings/summarize.py --filter mic_
"""

import argparse
import json

from gtp import REPO_ROOT

RESULTS_DIR = REPO_ROOT / 'results' / 'strings'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', default='')
    args = parser.parse_args()

    rows = []
    for path in sorted(RESULTS_DIR.glob(f'*{args.filter}*/metrics.json')):
        m = json.loads(path.read_text())
        t = m['test']
        rows.append(
            (
                min(h['val_nll_bits'] for h in m['history']),
                path.parent.name,
                m['n_params'] / 1e6,
                m['best_epoch'],
                t['multi']['model']['acc'],
                t['multi']['prior']['acc'],
                t['multi']['bits_gained'],
                t['multi']['model']['ece'],
                t['multi_comp']['model']['acc'],
                t['multi_solo']['model']['acc'],
            )
        )
    header = f'{"run":38s} {"params":>6s} {"ep":>3s} {"valNLL":>7s} {"acc":>6s} {"prior":>6s} {"bits+":>6s} {"ece":>6s} {"comp":>6s} {"solo":>6s}'
    print(header)
    for val_nll, name, params, ep, acc, prior, gained, ece, comp, solo in sorted(rows):
        print(
            f'{name:38s} {params:5.2f}M {ep:3d} {val_nll:7.3f} {acc:6.3f} {prior:6.3f} {gained:+6.3f} {ece:6.3f} {comp:6.3f} {solo:6.3f}'
        )


if __name__ == '__main__':
    main()

"""Frame-level tablature metrics of an end-to-end run, as GuitarSet tablature papers report them (TabCNN's protocol).

Time is cut into frames of hop_s. A frame's reference is the set of (string, pitch) sounding in it, its estimate the
transcribed notes with the strings a method gave them. Reported per split, guitar, dataset and chain:
multi-pitch F1 (strings ignored), tablature precision, recall and F1, and the tablature disambiguation rate
(TDR): of the frames' correctly detected pitches, the share on the right string.

Writes <run>/frame_metrics.csv, and the per-recording counts behind it to <run>/frame_counts.csv (some papers average
per track instead of pooling frames).

Usage:
  uv run scripts/e2e/frame_metrics.py run=results/e2e/index/<run>
"""

import json

import hydra
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info

PITCHES = 128


def roll(notes, n_frames, hop):
    """(frames, 6, pitches) bool: which (string, pitch) sound in each frame."""
    out = np.zeros((n_frames, 6, PITCHES), bool)
    for start, end, pitch, string in notes:
        out[
            int(np.ceil(start / hop)) : max(int(np.ceil(end / hop)), int(np.ceil(start / hop)) + 1),
            int(string) - 1,
            int(pitch),
        ] = True
    return out


def f1(tp, n_est, n_ref):
    p, r = tp / max(n_est, 1), tp / max(n_ref, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'e2e'), config_name='frame_metrics')
def main(cfg):
    run = REPO_ROOT / cfg.run
    est = pd.read_csv(run / 'notes.csv.gz')
    est = est[est['input'] == 'stage1']
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv').set_index('rec')
    ref = dict(tuple(pd.read_csv(index / 'notes.csv.gz').groupby('rec')))
    cache = REPO_ROOT / 'results' / 'e2e' / 'stage1' / f'index_{(REPO_ROOT / cfg.stage1_checkpoint).stem}'
    counts = []
    for rec, e in est.groupby('rec'):
        ends = {(n['pitch'], round(n['start'], 4)): n['end'] for n in json.loads((cache / f'{rec}.json').read_text())}
        end = np.array([ends[(p, round(s, 4))] for p, s in zip(e['pitch'], e['start'], strict=True)])
        r = ref[rec]
        n_frames = int(np.ceil(max(end.max(), r['offset'].max()) / cfg.hop_s)) + 1
        truth = roll(zip(r['onset'], r['offset'], r['pitch'], r['string'], strict=True), n_frames, cfg.hop_s)
        for m in cfg.methods:
            guess = roll(zip(e['start'], end, e['pitch'], e[m], strict=True), n_frames, cfg.hop_s)
            counts.append(
                {
                    'rec': rec,
                    'method': m,
                    'tab_tp': int((guess & truth).sum()),
                    'tab_est': int(guess.sum()),
                    'tab_ref': int(truth.sum()),
                    'pitch_tp': int((guess.any(axis=1) & truth.any(axis=1)).sum()),
                    'pitch_est': int(guess.any(axis=1).sum()),
                    'pitch_ref': int(truth.any(axis=1).sum()),
                }
            )
    counts = pd.DataFrame(counts).join(recs[['split', 'guitar', 'dataset', 'chain']], on='rec')
    counts.to_csv(run / 'frame_counts.csv', index=False)
    rows = []
    for key, c in counts.groupby(['split', 'guitar', 'dataset', 'chain', 'method']):
        s = c.sum(numeric_only=True)
        tab_p, tab_r, tab_f = f1(s.tab_tp, s.tab_est, s.tab_ref)
        rows.append(
            dict(zip(['split', 'guitar', 'dataset', 'chain', 'method'], key, strict=True))
            | {
                'recordings': len(c),
                'pitch_f1': f1(s.pitch_tp, s.pitch_est, s.pitch_ref)[2],
                'tab_p': tab_p,
                'tab_r': tab_r,
                'tab_f1': tab_f,
                'tdr': s.tab_tp / max(s.pitch_tp, 1),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(run / 'frame_metrics.csv', index=False)
    info(f'{run}\n{out.round(3).to_string(index=False)}')


if __name__ == '__main__':
    main()

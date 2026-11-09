"""Fuse the string classifier with self-calibrated inharmonicity evidence on the index's real recordings.

For every guitar of the requested splits, the six open-string inharmonicity constants are calibrated without labels
(inharmonicity.calibrate), pooling the guitar's recordings or one recording at a time, and the classifier's
probabilities are multiplied by the resulting likelihood. `oracle` calibrates from the labels of the guitar's other
recordings instead (two folds), as a ceiling for the calibration step.

Reports accuracy on notes with more than one candidate string, per split, guitar, dataset and content, and writes
the fused probabilities to <out>/fused.npz (note = row of notes.csv.gz).

Usage:
  uv run scripts/stage1/strings/physics_fusion.py "runs=['results/strings/runs/<run>', ...]"
"""

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.inharmonicity import calibrate, fit_strings, fuse, implied_constants
from gtp.stage1.strings import candidates


def one_hot(strings):
    return np.eye(6)[np.asarray(strings) - 1]


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='physics_fusion')
def main(cfg):
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    notes = pd.read_csv(index / 'notes.csv.gz').join(
        recs.set_index('rec')[['dataset', 'guitar', 'chain', 'content', 'split']], on='rec'
    )
    runs = [np.load(REPO_ROOT / r / 'index_probs.npz') for r in cfg.runs]
    note = runs[0]['note']
    assert all((r['note'] == note).all() for r in runs)
    d = notes.iloc[note].reset_index(drop=True)
    d = d.assign(note=note)
    probs = np.mean([r['probs'] for r in runs], axis=0)

    harm = np.load(index / 'cache' / cfg.features)
    names = list(harm['names'])
    log_b, resid = harm['x'][note, names.index('log10_b')], harm['x'][note, names.index('fit_resid_cents')]
    valid = (log_b > cfg.min_log_b) & (resid < cfg.max_resid_cents)
    implied = implied_constants(log_b, d['pitch'])

    fused = {k: probs.copy() for k in ('per_guitar', 'per_recording', 'oracle')}
    for guitar, g in d.groupby('guitar').indices.items():
        fused['per_guitar'][g], const, _ = calibrate(probs[g], implied[g], valid[g], cfg.iters, cfg.weight)
        info(f'{guitar}: constants {const.round(2)}')
        rec = d['rec'].to_numpy()[g]
        for r in np.unique(rec):
            i = g[rec == r]
            fused['per_recording'][i], *_ = calibrate(probs[i], implied[i], valid[i], cfg.iters, cfg.weight)
        fold = pd.factorize(rec)[0] % 2
        for f in (0, 1):
            fit, i = g[(fold != f) & valid[g]], g[fold == f]
            const, scale = fit_strings(implied[fit], one_hot(d['string'].to_numpy()[fit]))
            fused['oracle'][i] = fuse(probs[i], implied[i], valid[i], const, scale, cfg.weight)

    multi = candidates(d['pitch'].to_numpy()).sum(axis=1) > 1
    label = d['string'].to_numpy() - 1
    table = d.assign(multi=multi, valid=valid, classifier=probs.argmax(1) == label)
    for k, p in fused.items():
        table[k] = p.argmax(1) == label
    cols = ['classifier', *fused]
    keys = ['split', 'guitar', 'dataset', 'content']
    summary = (
        table[multi]
        .groupby(keys)
        .agg(notes=('multi', 'size'), valid=('valid', 'mean'), **{c: (c, 'mean') for c in cols})
    )
    info('\n' + summary.round(3).to_string())
    by_split = table[multi].groupby(['split', 'guitar']).agg(notes=('multi', 'size'), **{c: (c, 'mean') for c in cols})
    info('\n' + by_split.round(3).to_string())

    out = HydraConfig.get().runtime.output_dir
    summary.to_csv(f'{out}/summary.csv')
    np.savez(f'{out}/fused.npz', note=note, classifier=probs, **fused)
    info(f'Saved {out}')


if __name__ == '__main__':
    main()

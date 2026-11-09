"""Collect leave-one-guitar-out runs (train_index.py with `holdout`) into one set of held-out string probabilities.

Every note of the index is scored by the runs that never saw its guitar. A guitar's temperature is fitted on the
held-out logits of all the other guitars, so nothing about a guitar's own labels touches its probabilities. The
temperature is also written into each run's metrics.json, so StringClassifier.from_run() is calibrated.

Writes <out>/probs.npz: `note` (rows of notes.csv.gz), `classifier` (seed average) and one array per seed (`s0`, ...).

Usage:
  uv run scripts/stage1/strings/logo_probs.py
"""

import glob
import json

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.strings import candidates, fit_temperature, prior_from_counts, summarize


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='logo_probs')
def main(cfg):
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    notes = pd.read_csv(index / 'notes.csv.gz').join(recs.set_index('rec')[['dataset', 'guitar', 'content']], on='rec')
    label = notes['string'].to_numpy() - 1
    folds = {}
    for run in sorted(glob.glob(str(REPO_ROOT / cfg.runs))):
        rc = OmegaConf.load(f'{run}/.hydra/config.yaml')
        z = np.load(f'{run}/holdout_logits.npz')
        order = np.argsort(z['note'])
        folds[(rc.holdout[0], int(rc.seed))] = (run, z['note'][order], z['logits'][order])
    guitars, seeds = sorted({g for g, _ in folds}), sorted({s for _, s in folds})
    info(f'{len(folds)} runs: {len(guitars)} guitars x seeds {seeds}')

    note = np.concatenate([folds[(g, seeds[0])][1] for g in guitars])
    per_seed = {s: [] for s in seeds}
    rows = []
    for g in guitars:
        temps = []
        for s in seeds:
            others = [folds[(o, s)] for o in guitars if o != g and (o, s) in folds]
            logits = torch.from_numpy(np.concatenate([f[2] for f in others]))
            t = fit_temperature(logits, torch.from_numpy(label[np.concatenate([f[1] for f in others])]))
            run, ids, own = folds[(g, s)]
            assert (ids == folds[(g, seeds[0])][1]).all()
            per_seed[s].append(torch.softmax(torch.from_numpy(own) / t, dim=1).numpy())
            metrics = json.loads((REPO_ROOT / run / 'metrics.json').read_text())
            metrics['temperature'] = t
            (REPO_ROOT / run / 'metrics.json').write_text(json.dumps(metrics, indent=1))
            temps.append(t)
        ids = folds[(g, seeds[0])][1]
        multi = candidates(notes['pitch'].to_numpy()[ids]).sum(axis=1) > 1
        mean = np.mean([per_seed[s][-1] for s in seeds], axis=0)
        counts = np.load(REPO_ROOT / folds[(g, seeds[0])][0] / 'classifier.npz')['prior_counts']
        prior = summarize(prior_from_counts(counts, notes['pitch'].to_numpy()[ids])[multi], label[ids][multi])
        m = summarize(mean[multi], label[ids][multi])
        accs = [summarize(per_seed[s][-1][multi], label[ids][multi])['acc'] for s in seeds]
        rows.append(
            {'guitar': g, 'notes': int(multi.sum()), 'prior': prior['acc'], 'seed mean': np.mean(accs),
             'seed sd': np.std(accs), 'ensemble': m['acc'], 'nll bits': m['nll_bits'], 'ece': m['ece'],
             'temperature': np.mean(temps)}
        )  # fmt: skip
    table = pd.DataFrame(rows)
    info('held-out string accuracy on notes with more than one candidate\n' + table.round(3).to_string(index=False))
    out = HydraConfig.get().runtime.output_dir
    stacked = {f's{s}': np.concatenate(per_seed[s]) for s in seeds}
    np.savez(f'{out}/probs.npz', note=note, classifier=np.mean(list(stacked.values()), axis=0), **stacked)
    table.to_csv(f'{out}/summary.csv', index=False)
    info(f'Saved {out}')


if __name__ == '__main__':
    main()

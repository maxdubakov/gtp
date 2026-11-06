"""Tabulate string-classifier runs: one row per config, aggregated over seeds and folds.

A config is labelled by how its saved config differs from the current defaults, minus
seed / test_player / naming keys, so seeds and leave-one-player-out folds of the same config
share a row (--by-fold splits folds). Mean ± std is over runs. T, bits+ (T) and ECE (T) are
after temperature scaling fit on the val player. The 95% CI and the delta vs `--baseline`
come from a bootstrap over test tracks on the pooled per-note predictions (paired on the
notes both configs predicted).

Usage:
  uv run scripts/stage1/strings/report.py
  uv run scripts/stage1/strings/report.py v0_grid v0_seeds --baseline defaults --format md
  uv run scripts/stage1/strings/report.py lopo --format latex > paper/tables/strings_lopo.tex
"""

import argparse
import json

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.stage1.strings import MAX_FRET, OPEN_STRINGS

RESULTS_DIR = REPO_ROOT / 'results' / 'strings'
POOLED_KEYS = ('seed', 'test_player')
NAMING_KEYS = ('name', 'wandb.')


def flatten(cfg):
    return {k: v for k, v in pd.json_normalize(OmegaConf.to_container(cfg, resolve=True), sep='.').iloc[0].items()}


def current_defaults():
    with initialize_config_dir(version_base='1.3', config_dir=str(REPO_ROOT / 'conf' / 'strings')):
        return flatten(compose('train'))


def config_label(run_cfg, defaults, pooled=POOLED_KEYS):
    """The run's differences from the *current* defaults, so labels stay right when defaults change."""
    diff = [f'{k}={v}' for k, v in run_cfg.items() if defaults.get(k) != v and not k.startswith(pooled + NAMING_KEYS)]
    return ' '.join(sorted(diff)) or 'defaults'


def load_runs(experiments, pooled):
    defaults = current_defaults()
    runs = []
    for metrics_path in sorted(
        [*RESULTS_DIR.glob('runs/*/metrics.json'), *RESULTS_DIR.glob('sweeps/*/*/*/metrics.json')]
    ):
        run_dir = metrics_path.parent
        provenance = json.loads((run_dir / 'provenance.json').read_text())
        if experiments and provenance['experiment'] not in experiments:
            continue
        run_cfg = flatten(OmegaConf.load(run_dir / '.hydra' / 'config.yaml'))
        runs.append(
            {
                'dir': run_dir,
                'experiment': provenance['experiment'],
                'config': config_label(run_cfg, defaults, pooled),
                'dirty': provenance['git_dirty'],
                'metrics': json.loads(metrics_path.read_text()),
            }
        )
    return runs


def note_table(run_dirs, subset):
    """Per-note correctness and NLL (bits), averaged over runs that predicted the same note."""
    frames = []
    for d in run_dirs:
        z = np.load(d / 'test_probs.npz')
        label = z['string'] - 1
        fret = z['pitch'][:, None] - OPEN_STRINGS[None, :]
        multi = ((fret >= 0) & (fret <= MAX_FRET)).sum(axis=1) > 1
        p_true = z['probs'][np.arange(len(label)), label]
        df = pd.DataFrame(
            {
                'track': z['track'],
                'onset': z['onset'],
                'pitch': z['pitch'],
                'correct': (z['probs'].argmax(axis=1) == label).astype(float),
                'nll': -np.log2(np.clip(p_true, 1e-12, None)),
            }
        )
        frames.append(df[multi] if subset == 'multi' else df)
    return pd.concat(frames).groupby(['track', 'onset', 'pitch'], as_index=False)[['correct', 'nll']].mean()


def bootstrap_tracks(notes, column, n_boot=2000, seed=0):
    """95% CI of the note-weighted mean of `column`, resampling whole tracks."""
    per_track = notes.groupby('track')[column].agg(['sum', 'count'])
    sums, counts = per_track['sum'].to_numpy(), per_track['count'].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    stats = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return np.percentile(stats, [2.5, 97.5])


def summarize_group(group, subset):
    tests = [r['metrics']['test'][subset] for r in group]
    acc = np.array([t['model']['acc'] for t in tests])
    gained = np.array([t['bits_gained'] for t in tests])
    ece = np.array([t['model']['ece'] for t in tests])
    notes = note_table([r['dir'] for r in group], subset)
    lo, hi = bootstrap_tracks(notes, 'correct')
    return {
        'experiment': group[0]['experiment'],
        'config': group[0]['config'],
        'runs': len(group),
        'params_M': group[0]['metrics']['n_params'] / 1e6,
        'val_nll': np.mean([r['metrics']['best_val_nll_bits'] for r in group]),
        'acc': acc.mean(),
        'acc_std': acc.std(ddof=1) if len(acc) > 1 else np.nan,
        'acc_ci': (lo, hi),
        'prior_acc': np.mean([t['prior']['acc'] for t in tests]),
        'bits_gained': gained.mean(),
        'bits_std': gained.std(ddof=1) if len(gained) > 1 else np.nan,
        'ece': ece.mean(),
        'bits_gained_cal': np.mean([t.get('bits_gained_calibrated', np.nan) for t in tests]),
        'ece_cal': np.mean([t['calibrated']['ece'] if 'calibrated' in t else np.nan for t in tests]),
        'temperature': np.mean([r['metrics'].get('temperature', np.nan) for r in group]),
        'comp_acc': np.mean([r['metrics']['test'][f'{subset}_comp']['model']['acc'] for r in group])
        if subset == 'multi'
        else np.nan,
        'solo_acc': np.mean([r['metrics']['test'][f'{subset}_solo']['model']['acc'] for r in group])
        if subset == 'multi'
        else np.nan,
        'dirty': any(r['dirty'] for r in group),
        '_notes': notes,
    }


def paired_delta(notes, base_notes):
    both = notes.merge(base_notes, on=['track', 'onset', 'pitch'], suffixes=('', '_base'))
    both['diff'] = both['correct'] - both['correct_base']
    lo, hi = bootstrap_tracks(both, 'diff')
    return both['diff'].mean(), lo, hi


def pm(mean, std, scale=1.0, digits=1):
    s = f'{mean * scale:.{digits}f}'
    return s if np.isnan(std) else f'{s} ± {std * scale:.{digits}f}'


def render(rows, baseline):
    table = []
    for r in rows:
        row = {
            'experiment': r['experiment'],
            'config': r['config'] + (' *' if r['dirty'] else ''),
            'runs': r['runs'],
            'params': f'{r["params_M"]:.2f}M',
            'val NLL': f'{r["val_nll"]:.3f}',
            'acc %': pm(r['acc'], r['acc_std'], 100),
            '95% CI': f'[{r["acc_ci"][0] * 100:.1f}, {r["acc_ci"][1] * 100:.1f}]',
            'prior %': f'{r["prior_acc"] * 100:.1f}',
            'bits+': pm(r['bits_gained'], r['bits_std'], digits=2),
            'ECE': f'{r["ece"]:.3f}',
            'T': f'{r["temperature"]:.2f}',
            'bits+ (T)': f'{r["bits_gained_cal"]:.2f}',
            'ECE (T)': f'{r["ece_cal"]:.3f}',
            'comp %': f'{r["comp_acc"] * 100:.1f}',
            'solo %': f'{r["solo_acc"] * 100:.1f}',
        }
        if baseline is not None:
            d, lo, hi = r['delta']
            row['Δacc vs base'] = (
                '—' if r['config'] == baseline else f'{d * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]'
            )
        table.append(row)
    return pd.DataFrame(table)


def to_markdown(df):
    lines = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)]
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', nargs='*', help='experiment names to include (default: all)')
    parser.add_argument('--subset', choices=('multi', 'all'), default='multi')
    parser.add_argument('--baseline', default=None, help='config label to compute paired deltas against')
    parser.add_argument('--format', choices=('text', 'md', 'latex'), default='text')
    parser.add_argument('--by-fold', action='store_true', help='one row per test player instead of pooling folds')
    args = parser.parse_args()

    groups = {}
    pooled = ('seed',) if args.by_fold else POOLED_KEYS
    for run in load_runs(set(args.experiments), pooled):
        groups.setdefault((run['experiment'], run['config']), []).append(run)
    rows = [summarize_group(g, args.subset) for g in groups.values()]
    if not rows:
        raise SystemExit(f'No finished runs under {RESULTS_DIR} for {args.experiments or "any experiment"}')
    if args.baseline is not None:
        base = [r for r in rows if r['config'] == args.baseline]
        if not base:
            raise SystemExit(f'No config labelled {args.baseline!r}; labels: {sorted({r["config"] for r in rows})}')
        base_notes = pd.concat([r['_notes'] for r in base]).groupby(['track', 'onset', 'pitch'], as_index=False).mean()
        for r in rows:
            r['delta'] = paired_delta(r['_notes'], base_notes)
    rows.sort(key=lambda r: (r['experiment'], r['val_nll']))

    df = render(rows, args.baseline)
    if args.format == 'md':
        print(to_markdown(df))
    elif args.format == 'latex':
        print(
            df.drop(columns='experiment')
            .to_latex(index=False, escape=True)
            .replace('±', r'$\pm$')
            .replace('Δ', r'$\Delta$')
        )
    else:
        print(df.to_string(index=False))
        if any(r['dirty'] for r in rows):
            print('\n* at least one run was trained from a dirty tree (see git.diff in its run dir)')


if __name__ == '__main__':
    main()

"""Compare two or more Stage 2 training runs side-by-side.

Reads each run's `config.json`, `metrics.jsonl`, and `eval_sampled.json` and
prints:

  1. Run-level metadata (label, conditioning, rebalancing, device, params)
  2. Final-step val metrics (tab_strict / tab_equivalent / pitch_pp / val_loss)
  3. Per-source autoregressive tab_acc_pp at the final step
  4. Drift-bucket distribution at the final step
  5. Trajectory through eval_sampled.json (9 sample points for typical runs)
  6. Train/val gap trajectory (from metrics.jsonl)

The first run on the command line is treated as the BASELINE. Other runs are
shown with deltas vs baseline. Useful for "did Exp 1 / Exp 2a actually move
the needle?" at a glance.

Usage:
    python scripts/stage2/compare_runs.py \\
        runs/stage2_baseline \\
        runs/stage2_002_exp1_genre \\
        runs/stage2_003_exp2a_rebalance

    # Optional CSV dump for plotting:
    python scripts/stage2/compare_runs.py runs/* --csv-dir results/comparison/
"""

import argparse
import csv
import json
from pathlib import Path

from gtp.stage2.config import RunConfig


def load_run(run_dir: Path) -> dict:
    """Read all the artifact JSONs we know about from a run directory."""
    out = {'name': run_dir.name, 'dir': run_dir}
    cfg_path = run_dir / 'config.json'
    out['config'] = RunConfig.load(cfg_path) if cfg_path.exists() else None

    metrics_path = run_dir / 'metrics.jsonl'
    out['metrics'] = []
    if metrics_path.exists():
        with metrics_path.open() as f:
            out['metrics'] = [json.loads(line) for line in f]

    eval_path = run_dir / 'eval_sampled.json'
    out['eval_sampled'] = []
    if eval_path.exists():
        out['eval_sampled'] = json.loads(eval_path.read_text())

    final_path = run_dir / 'final_eval.json'
    out['final_eval'] = json.loads(final_path.read_text()) if final_path.exists() else None
    return out


def fmt_delta(cur, base, fmt='.4f', as_pp=True):
    """Pretty-print absolute + delta. `as_pp`: render delta as percentage points."""
    if base is None or cur is None:
        return f'{cur:{fmt}}' if cur is not None else '--'
    delta = cur - base
    sign = '+' if delta >= 0 else ''
    if as_pp:
        return f'{cur:{fmt}}  ({sign}{delta * 100:.2f}pp)'
    return f'{cur:{fmt}}  ({sign}{delta:{fmt}})'


def header(title: str) -> None:
    print(f'\n=== {title} ===')


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def print_run_metadata(runs):
    header('Run metadata')
    print(f'  {"name":<35s} {"label":<32s} {"params":>10s} {"GPU":<22s} {"genre":>5s} {"rebal":>5s}')
    for r in runs:
        cfg = r['config']
        if cfg is None:
            print(f'  {r["name"]:<35s} (no config.json)')
            continue
        print(
            f'  {r["name"]:<35s} '
            f'{(cfg.experiment_label or "")[:32]:<32s} '
            f'{cfg.model.params:>10,d} '
            f'{(cfg.device.cuda_name or cfg.device.type or "")[:22]:<22s} '
            f'{"Y" if cfg.conditioning.genre else "":>5s} '
            f'{"Y" if cfg.rebalancing.enabled else "":>5s}'
        )


def _final_summary(run):
    """Pull the highest-step `summary_val` from eval_sampled.json."""
    if not run['eval_sampled']:
        return None
    rec = max(run['eval_sampled'], key=lambda r: r['step'])
    return rec.get('summary_val'), rec.get('splits', {}).get('val'), rec['step']


def print_final_topline(runs):
    header('Final-step val metrics (autoregressive + post-processed)')

    # Reference run = first
    base = runs[0]
    base_summary, _, base_step = _final_summary(base) or (None, None, None)
    if base_summary is None:
        print('  (no eval_sampled.json in baseline run; skipping topline)')
        return

    # Header
    print(f'  {"metric":<22s} {base["name"]:<22s} ' + ' '.join(f'{r["name"]:<32s}' for r in runs[1:]))

    rows = [
        ('tab_strict_acc', '.4f', True),
        ('tab_equivalent_acc', '.4f', True),
        ('pitch_pp_acc', '.4f', True),
        ('recovered_by_alt', ',d', False),
    ]
    for key, fmt, as_pp in rows:
        cells = [f'{base_summary.get(key, "--"):{fmt}}']
        for r in runs[1:]:
            s, _, _ = _final_summary(r) or (None, None, None)
            if s is None:
                cells.append('--')
                continue
            cells.append(fmt_delta(s.get(key), base_summary.get(key), fmt=fmt, as_pp=as_pp))
        print(f'  {key:<22s} ' + '  '.join(f'{c:<22s}' for c in cells))


def print_per_source(runs):
    header('Per-source tab_acc_pp at final step (autoregressive)')

    base = runs[0]
    _, base_splits, _ = _final_summary(base) or (None, None, None)
    if not base_splits:
        print('  (no per-source split data)')
        return

    sources = sorted([s for s in base_splits if s != '_overall'])
    sources.append('_overall')

    print(f'  {"source":<14s} {base["name"]:<22s} ' + ' '.join(f'{r["name"]:<32s}' for r in runs[1:]))
    for src in sources:
        base_val = base_splits.get(src, {}).get('tab_acc_pp')
        cells = [f'{base_val:.4f}' if base_val is not None else '--']
        for r in runs[1:]:
            _, splits, _ = _final_summary(r) or (None, None, None)
            cur = (splits or {}).get(src, {}).get('tab_acc_pp') if splits else None
            cells.append(fmt_delta(cur, base_val, fmt='.4f', as_pp=True))
        print(f'  {src:<14s} ' + '  '.join(f'{c:<22s}' for c in cells))


def _best_step(run, key: str):
    """Find the (value, step) where `summary_val[key]` is maximal across eval_sampled.

    Returns (None, None) if no eval_sampled.json or key missing.
    """
    if not run['eval_sampled']:
        return None, None
    best_val, best_step = None, None
    for rec in run['eval_sampled']:
        v = rec.get('summary_val', {}).get(key)
        if v is None:
            continue
        if best_val is None or v > best_val:
            best_val, best_step = v, rec['step']
    return best_val, best_step


def print_best_checkpoint(runs):
    """Headline: the best each run achieves across the eval_sampled trajectory.

    More honest than final-step: training often peaks earlier than the last
    checkpoint due to mild overfitting (baseline peaked around step 45k).
    """
    header('Best-checkpoint val metrics (across eval_sampled trajectory)')
    base = runs[0]

    print(f'  {"metric":<22s} {base["name"]:<28s} ' + ' '.join(f'{r["name"]:<32s}' for r in runs[1:]))
    for key in ('tab_strict_acc', 'tab_equivalent_acc'):
        base_val, base_step = _best_step(base, key)
        cells = [f'{base_val:.4f} @ step {base_step}' if base_val is not None else '--']
        for r in runs[1:]:
            v, s = _best_step(r, key)
            if v is None:
                cells.append('--')
                continue
            sign = '+' if v >= base_val else ''
            cells.append(f'{v:.4f} @ step {s}  ({sign}{(v - base_val) * 100:+.2f}pp)')
        print(f'  {key:<22s} ' + '  '.join(f'{c:<32s}' for c in cells))


def print_drift_buckets(runs):
    header('Drift bucket distribution at final step (n_pieces)')

    base = runs[0]
    base_summary, _, _ = _final_summary(base) or (None, None, None)
    if not base_summary or 'drift_buckets' not in base_summary:
        print('  (no drift bucket data)')
        return

    buckets = ['perfect', 'consistent_alt', 'partial_alt', 'inconsistent']
    print(f'  {"bucket":<16s} {base["name"]:<22s} ' + ' '.join(f'{r["name"]:<32s}' for r in runs[1:]))
    for b in buckets:
        base_n = base_summary['drift_buckets'].get(b, 0)
        cells = [f'{base_n:>5d}']
        for r in runs[1:]:
            s, _, _ = _final_summary(r) or (None, None, None)
            if s is None:
                cells.append('--')
                continue
            cur = s.get('drift_buckets', {}).get(b, 0)
            sign = '+' if cur >= base_n else ''
            cells.append(f'{cur:>5d}  ({sign}{cur - base_n:+d})')
        print(f'  {b:<16s} ' + '  '.join(f'{c:<22s}' for c in cells))


def print_trajectory(runs, key='tab_strict_acc'):
    header(f'Trajectory: {key} (from eval_sampled.json)')

    # Use the steps from the first run as the reference
    base = runs[0]
    base_steps = sorted({rec['step'] for rec in base['eval_sampled']})
    if not base_steps:
        print('  (no eval_sampled.json data)')
        return

    print(f'  {"step":>6s}  ' + '  '.join(f'{r["name"]:<32s}' for r in runs))
    for s in base_steps:
        cells = []
        base_val = None
        for i, r in enumerate(runs):
            recs = [rec for rec in r['eval_sampled'] if rec['step'] == s]
            # If multiple at same step (e.g. _final variant), prefer last (final)
            rec = recs[-1] if recs else None
            cur = rec.get('summary_val', {}).get(key) if rec else None
            if i == 0:
                base_val = cur
                cells.append(f'{cur:.4f}' if cur is not None else '--')
            else:
                cells.append(fmt_delta(cur, base_val, fmt='.4f', as_pp=True))
        print(f'  {s:>6d}  ' + '  '.join(f'{c:<32s}' for c in cells))


def print_train_val_gap(runs):
    header('Train/val gap at selected steps (val_loss − train_loss_since_last_eval)')

    sample_steps = [5000, 10000, 25000, 30000, 45000, 60000]
    print(f'  {"step":>6s}  ' + '  '.join(f'{r["name"]:<32s}' for r in runs))
    for s in sample_steps:
        cells = []
        for r in runs:
            rec = next((m for m in r['metrics'] if m['step'] == s), None)
            if rec is None or rec.get('train_loss_since_last_eval') is None:
                cells.append('--')
                continue
            gap = rec['val_loss'] - rec['train_loss_since_last_eval']
            sign = '+' if gap >= 0 else ''
            cells.append(f'{rec["train_loss_since_last_eval"]:.4f} → {rec["val_loss"]:.4f}  '
                         f'(gap {sign}{gap:+.4f})')
        print(f'  {s:>6d}  ' + '  '.join(f'{c:<32s}' for c in cells))


# ---------------------------------------------------------------------------
# CSV dump (for matplotlib / spreadsheet plotting)
# ---------------------------------------------------------------------------


def dump_csv(runs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Trajectory across all eval_sampled steps, all summary metrics
    keys = ['tab_strict_acc', 'tab_equivalent_acc', 'pitch_pp_acc', 'recovered_by_alt']
    csv_path = out_dir / 'eval_sampled_trajectory.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'step', *keys])
        for r in runs:
            for rec in r['eval_sampled']:
                s = rec.get('summary_val', {})
                w.writerow([r['name'], rec['step'], *(s.get(k) for k in keys)])
    print(f'  wrote {csv_path}')

    # metrics.jsonl trajectory
    csv_path = out_dir / 'training_trajectory.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'step', 'val_loss', 'train_loss_since_last_eval', 'tab_acc_tf', 'pitch_acc_tf'])
        for r in runs:
            for m in r['metrics']:
                w.writerow([
                    r['name'], m['step'], m.get('val_loss'),
                    m.get('train_loss_since_last_eval'),
                    m.get('tab_acc_tf'), m.get('pitch_acc_tf'),
                ])
    print(f'  wrote {csv_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='Run directories. First is treated as baseline.')
    ap.add_argument('--csv-dir', default=None,
                    help='If set, dump trajectory data as CSVs to this directory.')
    args = ap.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    runs = [load_run(d) for d in run_dirs]

    print_run_metadata(runs)
    print_best_checkpoint(runs)
    print_final_topline(runs)
    print_per_source(runs)
    print_drift_buckets(runs)
    print_trajectory(runs, key='tab_strict_acc')
    print_trajectory(runs, key='tab_equivalent_acc')
    print_train_val_gap(runs)

    if args.csv_dir:
        header('CSV dumps')
        dump_csv(runs, Path(args.csv_dir))


if __name__ == '__main__':
    main()

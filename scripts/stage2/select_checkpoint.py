"""Pick a Stage 2 checkpoint on GuitarSet val (never test).

Every `every` steps (plus the final checkpoint) the run's checkpoint decodes the val pieces along
the reference skeleton; the score is string accuracy, averaged over classifier seeds for a model
trained with evidence (which is also scored with the evidence removed).

Usage (on the pod, where the checkpoints are):
  python scripts/stage2/select_checkpoint.py run_dir=runs/stage2/<timestamp>_p05_audio
"""

import json
import re
from pathlib import Path

import hydra
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage2.evidence import evidence_runs
from gtp.stage2.fusion import decode_structured
from gtp.stage2.guitarset_eval import correct, load_distributions, prepare, rows_for
from gtp.stage2.inference import load_checkpoint
from gtp.track import start_run


def checkpoints(run_dir, every):
    out = {}
    for path in sorted(Path(run_dir, 'checkpoints').glob('step_*.pth')):
        step = int(re.match(r'step_(\d+)', path.stem)[1])
        if step % every == 0 or path.stem.endswith('_final'):
            out.setdefault(step, path)
    return out


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'fusion'), config_name='select')
def main(cfg):
    device = auto_device()
    run_dir = REPO_ROOT / cfg.run_dir
    ckpts = checkpoints(run_dir, cfg.every)
    run = start_run(cfg, job_type='select', device=device, extra={'run_dir': str(run_dir)})
    dists = load_distributions(evidence_runs(cfg.evidence), cfg.evidence.calibrated)
    _model, vocab, _step, _max_seq_len = load_checkpoint(next(iter(ckpts.values())), 'cpu')
    pieces, df = prepare(cfg, vocab, dists)
    multi = df.n_cand > 1
    info(f'{len(ckpts)} checkpoints of {run_dir.name}, {len(pieces)} pieces, {len(df)} notes')

    rows = []
    for step, path in ckpts.items():
        model, vocab, _, max_seq_len = load_checkpoint(path, device)
        record = {'step': step}
        if hasattr(model, 'embed'):
            accs = [
                correct(decode_structured(model, rows_for(pieces, vocab, max_seq_len, i), device, cfg.batch), df)
                for i in range(len(cfg.seeds))
            ]
            noaudio = correct(decode_structured(model, rows_for(pieces, vocab, max_seq_len, -1), device, cfg.batch), df)
            record |= {'noaudio_acc': noaudio.mean(), 'noaudio_acc_multi': noaudio[multi].mean()}
        else:
            accs = [correct(decode_structured(model, rows_for(pieces, vocab, max_seq_len), device, cfg.batch), df)]
        record |= {'acc': np.mean([a.mean() for a in accs]), 'acc_multi': np.mean([a[multi].mean() for a in accs])}
        rows.append(record)
        run.log({f'select/{k}': v for k, v in record.items() if k != 'step'}, step=step)
        info(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in record.items()))

    table = pd.DataFrame(rows)
    best = table.loc[table.acc.idxmax()]
    summary = {'best_step': int(best.step), 'best_acc': float(best.acc), 'best_checkpoint': str(ckpts[int(best.step)])}
    info(f'\n{table.round(4).to_string(index=False)}\nbest: {summary}')
    table.to_csv(run.dir / 'select.csv', index=False)
    (run.dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    run.finish(summary)


if __name__ == '__main__':
    main()

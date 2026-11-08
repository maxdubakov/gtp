"""Check the per-note string evidence Stage 2 will train on (src/gtp/stage2/evidence.py).

1. Fidelity: each held-in player's GuitarSet notes are simulated from the other held-in players and
   compared with that player's real classifier outputs. The test player is compared the same way.
2. Coverage: share of GuitarSet notes in the player-split dataset that find their real output.
3. DadaGP: what the simulator gives on DadaGP training pieces.

Statistics are over notes with more than one candidate string, where the evidence has to choose.

Usage:
  uv run scripts/stage2/check_evidence.py
"""

import json
from itertools import islice

import hydra
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.strings import GUITARSET_DIR, MAX_FRET, PLAYERS, split_players
from gtp.stage2.evidence import Pool, StringEvidence, load_evidence, rows_or_none, sweep_runs
from gtp.track import start_run


def guitarset_pieces(player):
    for path in sorted((GUITARSET_DIR / 'processed').glob(f'{player}_*.json')):
        yield {**json.loads(path.read_text()), 'source': 'guitarset', 'filename': path.name}


def jsonl_pieces(path, source):
    with open(path) as fh:
        for line in fh:
            if f'"source": "{source}"' in line[:200]:
                yield json.loads(line)


def collect(pieces, ev, rng, as_source=None):
    rows = []
    for piece in pieces:
        if as_source:
            piece = {**piece, 'source': as_source}
        for n, p in zip(piece['notes'], rows_or_none(ev.sample(piece, rng)), strict=True):
            rows.append(
                {
                    'piece': f'{piece["filename"]}:{piece.get("capo", 0)}',
                    'string': n['string'],
                    'fret': n['fret'],
                    'probs': p,
                }
            )
    return pd.DataFrame(rows)


def ece(conf, correct, bins=10):
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx))


def summarize(df):
    have = df[df.probs.notna()]
    probs = np.stack(have.probs.to_numpy()) if len(have) else np.zeros((0, 6))
    multi = (probs > 0).sum(axis=1) > 1
    probs, have = probs[multi], have[multi]
    true = probs[np.arange(len(probs)), have.string.to_numpy() - 1]
    correct = probs.argmax(axis=1) == have.string.to_numpy() - 1
    conf = probs.max(axis=1)
    per_piece = pd.Series(correct, index=have.piece.to_numpy()).groupby(level=0).mean()
    return {
        'notes': len(df),
        'missing': df.probs.isna().mean(),
        'fret_over_classifier_max': (df.fret > MAX_FRET).mean(),
        'multi': len(probs),
        'acc': correct.mean(),
        'confidence': conf.mean(),
        'nll_bits': -np.log2(np.maximum(true, 1e-4)).mean(),
        'ece': ece(conf, correct),
        'piece_acc_std': per_piece.std(),
    }


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'evidence'), config_name='check')
def main(cfg):
    run = start_run(cfg, job_type='evidence', device='cpu')
    rng = np.random.default_rng(cfg.seed)
    test = PLAYERS[cfg.test_player]
    heldin = sweep_runs(cfg.heldin_sweep)
    test_runs = {k: v for k, v in sweep_runs(cfg.test_sweep).items() if k[0] == test}
    stage2_train = split_players(cfg.test_player)['train']
    info(f'held-in runs {sorted(heldin)}, test runs {sorted(test_runs)}, donors from {stage2_train}')

    def runs_of(runs, players):
        return {k: v for k, v in runs.items() if k[0] in players}

    rows = []
    for player in [*sorted({p for p, _ in heldin}), test]:
        real = runs_of(test_runs if player == test else heldin, [player])
        donors = runs_of(heldin, [p for p in stage2_train if p != player])
        ev = StringEvidence(Pool.load(real, cfg.calibrated), Pool.load(donors, cfg.calibrated))
        pieces = list(guitarset_pieces(player))
        for kind, as_source in (('real', None), ('simulated', 'simulated')):
            rows.append(
                {
                    'check': 'fidelity',
                    'player': player,
                    'evidence': kind,
                    **summarize(collect(pieces, ev, rng, as_source)),
                }
            )
        info(f'fidelity {player}: {rows[-2]["acc"]:.3f} real vs {rows[-1]["acc"]:.3f} simulated')

    ev = load_evidence(cfg)
    data_dir = REPO_ROOT / cfg.data_dir
    for split in ('train', 'val', 'test'):
        df = collect(jsonl_pieces(data_dir / f'{split}.jsonl', 'guitarset'), ev, rng)
        rows.append({'check': 'coverage', 'player': split, 'evidence': 'real', **summarize(df)})
        info(f'coverage {split}: {1 - rows[-1]["missing"]:.4f} of {rows[-1]["notes"]} GuitarSet notes')
    dadagp = islice(jsonl_pieces(data_dir / 'train.jsonl', 'dadagp'), cfg.dadagp_pieces)
    rows.append({'check': 'dadagp', 'player': 'train', 'evidence': 'simulated', **summarize(collect(dadagp, ev, rng))})

    summary = pd.DataFrame(rows)
    info(f'\n{summary.round(4).to_string(index=False)}')
    summary.to_csv(run.dir / 'summary.csv', index=False)
    flat = {
        f'{r["check"]}/{r["player"]}/{r["evidence"]}/{k}': v for r in rows for k, v in r.items() if isinstance(v, float)
    }
    (run.dir / 'summary.json').write_text(json.dumps(flat, indent=1))
    run.finish(flat)


if __name__ == '__main__':
    main()

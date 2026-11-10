"""Stage 2 on the string index's real recordings, with per-note string evidence from any source.

Each recording of the requested splits becomes a Stage 2 piece (reference notes, standard tuning, no capo). For every
evidence column of `evidence` (an .npz of per-note string probabilities, e.g. physics_fusion.py's fused.npz), a model
without evidence input is rescored at decode time (string_fusion.py's rule, weight lam), and an EvidenceT5 reads the
probabilities in its encoder. Reports, per split, guitar, dataset and content, strict string accuracy and the
equivalent accuracy of metrics.py: a recording's most common wrong string shift is forgiven, as a consistent
alternative fingering.

Usage:
  uv run scripts/stage2/index_fusion.py evidence=results/strings/physics/<run>/fused.npz
"""

import re

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.strings import OPEN_STRINGS, candidates, prior_from_counts
from gtp.stage2.evidence import restrict
from gtp.stage2.fusion import decode_beam, decode_structured, jump_of, log_ratio, piece_rows, valid_tabs
from gtp.stage2.inference import load_checkpoint
from gtp.stage2.metrics import aggregate_drift_buckets

TUNING = [int(p) for p in OPEN_STRINGS]


def tempo_of(rec, dataset, default):
    """GuitarSet names carry the tempo (05_BN2-166-Ab_solo); other datasets use `default` (None = unknown)."""
    m = re.search(r'-(\d+)-', rec) if dataset == 'guitarset' else None
    return float(m[1]) if m else default


def decode(model, vocab, max_seq_len, pieces, probs, prior, cfg, device, lam):
    """{note: string} for one model; `probs` (row -> (6,)) feeds the encoder of an EvidenceT5, else the lam rescoring."""
    rows = []
    for piece, ids in pieces:
        evidence, ratios = None, [None] * len(ids)
        if probs is not None and hasattr(model, 'embed'):
            evidence = restrict(probs[ids], [n['pitch'] for n in piece['notes']], TUNING)
        elif probs is not None and lam:
            ratios = [
                log_ratio(probs[i][None], prior[i], valid_tabs(n['pitch'], TUNING, vocab)[0], cfg.prob_floor)
                for i, n in zip(ids, piece['notes'], strict=True)
            ]
        elif hasattr(model, 'embed'):
            evidence = np.full((len(ids), 6), np.nan)
        rows.extend(piece_rows(piece, vocab, max_seq_len, ids, ratios, evidence))
    if cfg.beams > 1:
        return decode_beam(
            model, rows, device, cfg.batch, lam, beams=cfg.beams, exclusive=cfg.exclusive_s, jump=jump_of(cfg)
        )
    return decode_structured(model, rows, device, cfg.batch, lam)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'fusion'), config_name='index_fusion')
def main(cfg):
    device = auto_device()
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    notes = pd.read_csv(index / 'notes.csv.gz').join(
        recs.set_index('rec')[['dataset', 'guitar', 'chain', 'content', 'split']], on='rec'
    )
    ev = np.load(REPO_ROOT / cfg.evidence)
    probs = {m: np.full((len(notes), 6), np.nan) for m in cfg.methods}
    for m in cfg.methods:
        probs[m][ev['note']] = ev[m]
    prior = prior_from_counts(np.load(REPO_ROOT / cfg.prior_run / 'classifier.npz')['prior_counts'], notes['pitch'])

    d = notes[notes['split'].isin(list(cfg.splits)) & notes.index.isin(ev['note'])]
    d = d[~d['dataset'].isin(list(cfg.exclude_datasets))]
    if cfg.guitars:
        d = d[d['guitar'].isin(list(cfg.guitars))]
    d = d.sort_values(['rec', 'onset', 'pitch'], kind='stable')
    pieces = []
    for rec, g in d.groupby('rec', sort=False):
        piece = {
            'source': 'index',
            'filename': rec,
            'tuning': TUNING,
            'tempo': tempo_of(rec, g['dataset'].iloc[0], cfg.tempo),
            'capo': 0,
            'notes': [
                {'pitch': int(n.pitch), 'string': int(n.string), 'fret': int(n.fret), 'start': n.onset, 'end': n.offset}
                for n in g.itertuples()
            ],
        }
        pieces.append((piece, g.index.to_numpy()))
    info(f'{len(pieces)} recordings, {len(d)} notes')

    table = d.assign(multi=candidates(d['pitch'].to_numpy()).sum(axis=1) > 1)
    picks = {m: pd.Series(probs[m][d.index].argmax(axis=1) + 1, index=d.index) for m in cfg.methods}
    for name, mcfg in cfg.models.items():
        model, vocab, step, max_seq_len = load_checkpoint(REPO_ROOT / mcfg.checkpoint, device)
        reads_evidence = hasattr(model, 'embed')
        info(f'{name}: step {step}, evidence {reads_evidence}')
        picks[f'{name}:none'] = pd.Series(decode(model, vocab, max_seq_len, pieces, None, prior, cfg, device, 0.0))
        for m in cfg.methods:
            for lam in [0.0] if reads_evidence else mcfg.lambdas:
                col = f'{name}:{m}' + ('' if reads_evidence else f':lam{lam:g}')
                picks[col] = pd.Series(decode(model, vocab, max_seq_len, pieces, probs[m], prior, cfg, device, lam))
                info(f'{col} done')

    keys = ['split', 'guitar', 'dataset', 'content']
    rows = []
    for col, pick in picks.items():
        pick = pick.reindex(d.index).astype(int)
        table[f'pick:{col}'] = pick
        records = pd.DataFrame(
            {
                'piece_id': d['rec'],
                'error_type_pp': np.where(pick == d['string'], 'correct', 'wrong'),
                'delta_string_pp': pick - d['string'],
                'delta_fret_pp': (d['pitch'] - OPEN_STRINGS[pick.to_numpy() - 1]) - d['fret'],
            }
        )
        groups = [
            *d.groupby(keys).groups.items(),
            *(((*k, 'all', 'all'), g) for k, g in d.groupby(keys[:2]).groups.items()),
        ]
        for key, idx in groups:
            drift = aggregate_drift_buckets(records.loc[idx].to_dict('records'), min_notes=cfg.min_notes_for_drift)
            rows.append(
                dict(zip(keys, key, strict=True))
                | {
                    'method': col,
                    'notes': len(idx),
                    'strict': float((pick.loc[idx] == d.loc[idx, 'string']).mean()),
                    'strict_multi': float((pick.loc[idx] == d.loc[idx, 'string'])[table.loc[idx, 'multi']].mean()),
                    'equivalent': drift['tab_equivalent_acc'],
                    'equivalent_notes': drift['n_notes'],
                }
            )
    summary = pd.DataFrame(rows)
    wide = summary.pivot_table(index='method', columns=keys, values=['strict', 'equivalent'], sort=False)
    info('strict accuracy, all notes\n' + wide['strict'].round(3).to_string())
    info(
        'equivalent accuracy (one consistent string shift per recording forgiven)\n'
        + wide['equivalent'].round(3).to_string()
    )
    out = HydraConfig.get().runtime.output_dir
    summary.to_csv(f'{out}/summary.csv', index=False)
    table.drop(columns=['offset']).to_csv(f'{out}/notes.csv.gz', index_label='note')
    info(f'Saved {out}')


if __name__ == '__main__':
    main()

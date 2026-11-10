"""Stage 2 on the string index's real recordings, with per-note string evidence from any source.

Each recording of the requested splits becomes a Stage 2 piece (reference notes, standard tuning, no capo). For every
evidence column of `evidence` (an .npz of per-note string probabilities, e.g. physics_fusion.py's fused.npz), a model
without evidence input is rescored at decode time (string_fusion.py's rule, weight lam), and an EvidenceT5 reads the
probabilities in its encoder. Reports string accuracy per split, guitar, dataset and content.

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
from gtp.stage2.fusion import decode_structured, log_ratio, piece_rows, valid_tabs
from gtp.stage2.inference import load_checkpoint

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
    for m in cfg.methods:
        table[m] = probs[m][d.index].argmax(axis=1) + 1 == d['string']
    for name, mcfg in cfg.models.items():
        model, vocab, step, max_seq_len = load_checkpoint(REPO_ROOT / mcfg.checkpoint, device)
        reads_evidence = hasattr(model, 'embed')
        info(f'{name}: step {step}, evidence {reads_evidence}')
        picks = decode(model, vocab, max_seq_len, pieces, None, prior, cfg, device, 0.0)
        table[f'{name}:none'] = pd.Series(picks).reindex(d.index) == d['string']
        for m in cfg.methods:
            for lam in [0.0] if reads_evidence else mcfg.lambdas:
                picks = decode(model, vocab, max_seq_len, pieces, probs[m], prior, cfg, device, lam)
                col = f'{name}:{m}' + ('' if reads_evidence else f':lam{lam:g}')
                table[col] = pd.Series(picks).reindex(d.index) == d['string']
                info(f'{col}: {table.loc[table.multi, col].mean():.4f}')

    cols = [c for c in table.columns if table[c].dtype == bool and c != 'multi']
    keys = ['split', 'guitar', 'dataset', 'content']
    summary = table[table.multi].groupby(keys)[cols].mean().assign(notes=table[table.multi].groupby(keys).size())
    info('multi-candidate notes\n' + summary.round(3).T.to_string())
    info('all notes\n' + table.groupby(['split', 'guitar'])[cols].mean().round(3).T.to_string())
    out = HydraConfig.get().runtime.output_dir
    summary.to_csv(f'{out}/summary.csv')
    table.drop(columns=['offset']).to_csv(f'{out}/notes.csv.gz', index_label='note')
    info(f'Saved {out}')


if __name__ == '__main__':
    main()

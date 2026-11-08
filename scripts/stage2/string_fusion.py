"""Stage 2 with audio string evidence on GuitarSet's held-out players (task 4).

Each model decodes the GuitarSet val/test pieces along the reference decoder skeleton
(TIME_SHIFT tokens and note order, both fixed by the input notes), so it only chooses each
note's string among the pitch-valid TAB tokens. That choice can be rescored at decode time with
the classifier's out-of-fold string distribution (lam > 0):

    score(s) = log p_stage2(TAB<s, f_s>) + lam * (log p_audio(s) - log p_prior(s))

A model trained with evidence (EvidenceT5) also reads the distribution in its encoder; it is
evaluated per classifier seed, and once with the evidence removed (`noaudio`). The free greedy
decode with post-processing (the README metric) is reported for models without evidence.

Usage:
  uv run scripts/stage2/string_fusion.py
  uv run scripts/stage2/string_fusion.py 'models.audio.lambdas=[0.0]' wandb.mode=disabled
"""

import json

import hydra
import numpy as np
import pandas as pd
import torch

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage2.data import load_jsonl_pieces
from gtp.stage2.evidence import evidence_runs, restrict
from gtp.stage2.fusion import decode_structured, log_ratio, padded, piece_rows, sorted_notes, valid_tabs
from gtp.stage2.inference import load_checkpoint
from gtp.stage2.postprocess import correct_tabs
from gtp.stage2.tokenizer import extract_input_pitches, extract_tabs
from gtp.track import start_run

N_STRINGS = 6


def load_distributions(runs, calibrated):
    """(track, onset, capo-0 pitch) -> {'audio': {seed: probs}, 'prior': probs}."""
    key = 'probs_calibrated' if calibrated else 'probs'
    dists = {}
    for (_player, seed), run_dir in sorted(runs.items()):
        z = np.load(f'{run_dir}/test_probs.npz')
        for track, onset, pitch, probs, prior in zip(
            z['track'], z['onset'], z['pitch'], z[key], z['prior'], strict=True
        ):
            entry = dists.setdefault((str(track), round(float(onset), 4), int(pitch)), {'audio': {}, 'prior': prior})
            entry['audio'][seed] = probs
    return dists


def guitarset_pieces(cfg, split):
    for piece in load_jsonl_pieces(REPO_ROOT / cfg.data_dir / f'{split}.jsonl'):
        if piece['source'] == 'guitarset' and (piece['capo'] == 0 or not cfg.capo0_only):
            yield piece


def prepare(cfg, vocab, dists):
    """Per-note table plus, per piece, what decoding needs: notes in decoder order, log ratios, evidence."""
    table, pieces = [], []
    for split in cfg.splits:
        for piece in guitarset_pieces(cfg, split):
            track, tuning, capo = piece['filename'].removesuffix('.json'), piece['tuning'], piece['capo']
            note_ids, ratios, evidence = [], [], []
            for n in sorted_notes(piece):
                strings, _ = valid_tabs(n['pitch'], tuning, vocab)
                d = dists.get((track, round(n['start'], 4), n['pitch'] - capo))
                record = {
                    'split': split,
                    'piece': f'{track}:capo{capo}',
                    'track': track,
                    'pitch': n['pitch'],
                    'true_string': n['string'],
                    'n_cand': len(strings),
                    'has_audio': d is not None and len(tuning) == N_STRINGS,
                }
                ratio, ev = None, np.full((len(cfg.seeds), N_STRINGS), np.nan)
                if record['has_audio']:
                    audio = np.stack([d['audio'][s] for s in cfg.seeds])
                    ratio = log_ratio(audio, d['prior'], strings, cfg.prob_floor)
                    ev = restrict(audio, [n['pitch']] * len(cfg.seeds), tuning)
                    for i, s in enumerate(cfg.seeds):
                        record[f'audio_s{s}'] = int(strings[audio[i, strings - 1].argmax()])
                note_ids.append(len(table))
                ratios.append(ratio)
                evidence.append(ev)
                table.append(record)
            pieces.append((piece, note_ids, ratios, np.stack(evidence, axis=1)))
    return pieces, pd.DataFrame(table)


def rows_for(pieces, vocab, max_seq_len, seed_idx=None):
    """Decoder rows for every piece; with `seed_idx`, the encoder also gets that seed's evidence (NaN = none)."""
    rows = []
    for piece, note_ids, ratios, evidence in pieces:
        ev = None
        if seed_idx is not None:
            ev = evidence[seed_idx] if seed_idx >= 0 else np.full_like(evidence[0], np.nan)
        rows.extend(piece_rows(piece, vocab, max_seq_len, note_ids, ratios, ev))
    return rows


@torch.no_grad()
def decode_free(model, rows, vocab, device, batch, max_seq_len):
    """The README metric: free greedy generation + pitch post-processing; returns {note: correct}."""
    correct = {}
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        input_ids, mask = padded([r.enc for r in chunk], vocab.pad_id, device)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            max_new_tokens=max_seq_len,
            num_beams=1,
            do_sample=False,
            pad_token_id=vocab.pad_id,
            eos_token_id=vocab.eos_id,
        ).tolist()
        for r, gen in zip(chunk, generated, strict=True):
            gt = extract_tabs(r.dec, vocab)
            fixed = correct_tabs(extract_input_pitches(r.enc, vocab), extract_tabs(gen[1:], vocab), r.tuning)
            for j, slot in enumerate(sorted(r.slots.items())):
                correct[slot[1].note] = j < len(fixed) and fixed[j] == gt[j]
    return correct


def correct(picks, df):
    assert len(picks) == len(df), f'{len(picks)} picks for {len(df)} notes'
    return pd.Series(picks).reindex(df.index) == df.true_string


def piece_bootstrap(df, diff_col, n_boot=2000, seed=0):
    per_piece = df.groupby('piece')[diff_col].agg(['sum', 'count'])
    sums, counts = per_piece['sum'].to_numpy(), per_piece['count'].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    return np.percentile(sums[idx].sum(axis=1) / counts[idx].sum(axis=1), [2.5, 97.5])


def evaluate_model(name, mcfg, cfg, pieces, df, device):
    """Adds one boolean column per (lam, seed), named `<name>_lam<lam>_s<seed>`; returns {method: columns}."""
    model, vocab, step, max_seq_len = load_checkpoint(REPO_ROOT / mcfg.checkpoint, device)
    uses_evidence = hasattr(model, 'embed')
    info(f'{name}: step {step}, evidence {uses_evidence}')
    methods = {}
    if uses_evidence:
        df[f'{name}_noaudio'] = correct(
            decode_structured(model, rows_for(pieces, vocab, max_seq_len, -1), device, cfg.batch), df
        )
        methods[f'{name}_noaudio'] = [f'{name}_noaudio']
    else:
        plain = rows_for(pieces, vocab, max_seq_len)
        free = decode_free(model, plain, vocab, device, cfg.batch, max_seq_len)
        df[f'{name}_free'] = df.index.map(free).astype(bool)
        methods[f'{name}_free'] = [f'{name}_free']
    for lam in mcfg.lambdas:
        cols = []
        for i, s in enumerate(cfg.seeds):
            col = f'{name}_lam{lam:g}_s{s}'
            if uses_evidence:
                rows = rows_for(pieces, vocab, max_seq_len, i)
            elif lam == 0 and i > 0:
                df[col] = df[f'{name}_lam0_s{cfg.seeds[0]}']
                cols.append(col)
                continue
            else:
                rows = plain
            df[col] = correct(decode_structured(model, rows, device, cfg.batch, lam, i), df)
            cols.append(col)
        methods[f'{name}_lam{lam:g}'] = cols
        info(f'{name} lam {lam:g}: {np.mean([df[c].mean() for c in cols]):.4f}')
    return methods


def summarize(df, cfg, methods, comparisons):
    rows = []
    for split, part in df.groupby('split'):
        for subset, sel in {'all': part, 'multi': part[part.n_cand > 1]}.items():
            row = {'split': split, 'subset': subset, 'notes': len(sel)}
            row['audio_only'] = np.mean([sel[f'audio_s{s}_ok'].mean() for s in cfg.seeds])
            for method, cols in methods.items():
                accs = [sel[c].mean() for c in cols]
                row[method] = np.mean(accs)
                if len(cols) > 1:
                    row[f'{method}_std'] = np.std(accs, ddof=1)
            for label, (a, b) in comparisons.items():
                gain = sel[methods[a]].mean(axis=1) - sel[methods[b]].mean(axis=1)
                row[f'gain_{label}'] = gain.mean()
                row[f'gain_{label}_ci_lo'], row[f'gain_{label}_ci_hi'] = piece_bootstrap(sel.assign(gain=gain), 'gain')
            rows.append(row)
    return pd.DataFrame(rows)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'fusion'), config_name='fusion')
def main(cfg):
    device = auto_device()
    run = start_run(cfg, job_type='fusion', device=device)
    dists = load_distributions(evidence_runs(cfg.evidence), cfg.evidence.calibrated)
    _model, vocab, _step, _max_seq_len = load_checkpoint(REPO_ROOT / next(iter(cfg.models.values())).checkpoint, 'cpu')
    pieces, df = prepare(cfg, vocab, dists)
    info(f'{len(pieces)} pieces, {len(df)} notes, {df.has_audio.mean():.2%} with audio, {len(dists)} distributions')
    for s in cfg.seeds:
        df[f'audio_s{s}_ok'] = df[f'audio_s{s}'] == df.true_string

    methods = {}
    for name, mcfg in cfg.models.items():
        methods |= evaluate_model(name, mcfg, cfg, pieces, df, device)

    val = df[df.split == 'val']
    best = {}
    for name, mcfg in cfg.models.items():
        fused = [lam for lam in mcfg.lambdas if lam]
        if fused:
            best[name] = max(fused, key=lambda lam, n=name: val[methods[f'{n}_lam{lam:g}']].mean().mean())
    comparisons = {}
    for label, (a, b) in cfg.comparisons.items():
        resolve = {f'{n}_best': f'{n}_lam{lam:g}' for n, lam in best.items()}
        comparisons[label] = (resolve.get(a, a), resolve.get(b, b))
    summary = summarize(df, cfg, methods, comparisons)
    info(f'best lam on val: {best}\n{summary.round(4).T.to_string()}')

    df.to_csv(run.dir / 'notes.csv.gz', index=False)
    summary.to_csv(run.dir / 'summary.csv', index=False)
    flat = {f'best_lam/{n}': lam for n, lam in best.items()}
    for r in summary.to_dict('records'):
        flat |= {f'{r["split"]}/{r["subset"]}/{k}': v for k, v in r.items() if k not in ('split', 'subset')}
    (run.dir / 'summary.json').write_text(json.dumps(flat, indent=1, default=float))
    run.finish(flat)


if __name__ == '__main__':
    main()

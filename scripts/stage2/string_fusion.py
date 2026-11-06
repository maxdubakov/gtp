"""Decode-time fusion of Stage 2 with the per-note string classifier (task 4, step 0).

Stage 2 decodes each GuitarSet val/test piece along the reference decoder skeleton
(TIME_SHIFT tokens and note order, both fixed by the input notes), so it only chooses each
note's string among the pitch-valid TAB tokens. That choice is rescored with the classifier's
out-of-fold string distribution from the leave-one-player-out sweep:

    score(s) = log p_stage2(TAB<s, f_s>) + lam * (log p_audio(s) - log p_prior(s))

lam = 0 is Stage 2 alone under pitch-constrained decoding. The free greedy decode with
post-processing (the README metric) is reported as a reference.

Usage:
  uv run scripts/stage2/string_fusion.py
  uv run scripts/stage2/string_fusion.py calibrated=false 'lambdas=[0.0,1.0]'
"""

import glob
import json
import re
from dataclasses import dataclass

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage2.data import load_jsonl_pieces
from gtp.stage2.inference import load_checkpoint
from gtp.stage2.paths import AUG_DATA_DIR
from gtp.stage2.postprocess import correct_tabs
from gtp.stage2.tokenizer import MAX_FRET, TAB, Token, extract_input_pitches, extract_tabs, tokenize_piece
from gtp.track import start_run

N_STRINGS = 6


@dataclass
class Slot:
    note: int
    token_ids: np.ndarray
    strings: np.ndarray
    log_ratio: np.ndarray | None  # (n_seeds, n_valid): log p_audio - log p_prior


@dataclass
class Row:
    enc: list[int]
    dec: list[int]
    tuning: list[int]
    slots: dict[int, Slot]  # decoder position -> slot


def load_distributions(sweep_dir, calibrated):
    """(track, onset, capo-0 pitch) -> {'audio': {seed: probs}, 'prior': probs}."""
    key = 'probs_calibrated' if calibrated else 'probs'
    dists = {}
    for path in sorted(glob.glob(str(REPO_ROOT / sweep_dir / '*' / '*' / 'test_probs.npz'))):
        seed = int(re.search(r'seed=(\d+)', path)[1])
        z = np.load(path)
        for track, onset, pitch, probs, prior in zip(
            z['track'], z['onset'], z['pitch'], z[key], z['prior'], strict=True
        ):
            entry = dists.setdefault((str(track), round(float(onset), 4), int(pitch)), {'audio': {}, 'prior': prior})
            entry['audio'][seed] = probs
    return dists


def floored(p, floor):
    p = np.maximum(p, floor)
    return p / p.sum(axis=-1, keepdims=True)


def prepare(cfg, vocab, max_seq_len, dists):
    """Tokenize GuitarSet pieces of the requested splits; one Slot per note, one table row per note."""
    table, rows = [], []
    for split in cfg.splits:
        for piece in load_jsonl_pieces(AUG_DATA_DIR / f'{split}.jsonl'):
            if piece['source'] != 'guitarset':
                continue
            track, tuning, capo = piece['filename'].removesuffix('.json'), piece['tuning'], piece['capo']
            notes = sorted(piece['notes'], key=lambda n: (n['start'], n['pitch']))
            k = 0
            for enc, dec in tokenize_piece(piece, vocab, max_seq_len=max_seq_len):
                slots = {}
                for t, tid in enumerate(dec):
                    if not vocab.decode(tid).startswith(TAB):
                        continue
                    n = notes[k]
                    k += 1
                    strings = np.array(
                        [s for s in range(1, len(tuning) + 1) if 0 <= n['pitch'] - tuning[s - 1] <= MAX_FRET]
                    )
                    token_ids = np.array(
                        [vocab.encode(Token(TAB, f'{s},{n["pitch"] - tuning[s - 1]}')) for s in strings]
                    )
                    assert tid in token_ids and n['string'] in strings, (track, n)
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
                    log_ratio = None
                    if record['has_audio']:
                        audio = floored(np.stack([d['audio'][s] for s in cfg.seeds])[:, strings - 1], cfg.prob_floor)
                        prior = floored(d['prior'][strings - 1], cfg.prob_floor)
                        log_ratio = np.log(audio) - np.log(prior)[None]
                        for i, s in enumerate(cfg.seeds):
                            record[f'audio_s{s}'] = int(strings[audio[i].argmax()])
                    slots[t] = Slot(len(table), token_ids, strings, log_ratio)
                    table.append(record)
                rows.append(Row(enc, dec, tuning, slots))
            assert k == len(notes), (track, k, len(notes))
    return rows, pd.DataFrame(table)


def padded(seqs, pad, device):
    width = max(len(s) for s in seqs)
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=device)
    return ids, (ids != pad).long()


@torch.no_grad()
def decode_structured(model, rows, device, batch, lam=0.0, seed_idx=0):
    """Greedy decode along each row's reference skeleton; returns {note: chosen string}."""
    pad = model.config.pad_token_id
    picks = {}
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        input_ids, mask = padded([r.enc for r in chunk], pad, device)
        encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=mask)
        cur = torch.full((len(chunk), 1), pad, device=device)
        past = None
        for t in range(max(len(r.dec) for r in chunk)):
            out = model(
                encoder_outputs=encoder_outputs,
                attention_mask=mask,
                decoder_input_ids=cur,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logp = F.log_softmax(out.logits[:, -1].float(), dim=-1).cpu().numpy()
            nxt = []
            for b, r in enumerate(chunk):
                slot = r.slots.get(t)
                if t >= len(r.dec):
                    nxt.append(pad)
                elif slot is None:
                    nxt.append(r.dec[t])
                else:
                    score = logp[b, slot.token_ids]
                    if lam and slot.log_ratio is not None:
                        score = score + lam * slot.log_ratio[seed_idx]
                    j = int(score.argmax())
                    picks[slot.note] = int(slot.strings[j])
                    nxt.append(int(slot.token_ids[j]))
            cur = torch.tensor(nxt, device=device)[:, None]
    return picks


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


def summarize(df, cfg, best_lam):
    rows = []
    for split, part in df.groupby('split'):
        for subset, sel in {'all': part, 'multi': part[part.n_cand > 1]}.items():
            row = {'split': split, 'subset': subset, 'notes': len(sel), 'stage2_free_pp': sel['free'].mean()}
            row['stage2_constrained'] = sel['lam0'].mean()
            row['audio_only'] = np.mean([sel[f'audio_s{s}_ok'].mean() for s in cfg.seeds])
            for lam in cfg.lambdas:
                if lam:
                    accs = [sel[f'lam{lam:g}_s{s}'].mean() for s in cfg.seeds]
                    row[f'fused_lam{lam:g}'] = np.mean(accs)
                    row[f'fused_lam{lam:g}_std'] = np.std(accs, ddof=1)
            row['either_right'] = np.mean([(sel['lam0'] | sel[f'audio_s{s}_ok']).mean() for s in cfg.seeds])
            wrong = sel[~sel['lam0']]
            row['stage2_wrong_audio_right'] = np.mean([wrong[f'audio_s{s}_ok'].mean() for s in cfg.seeds])
            if best_lam:
                sel = sel.assign(gain=sel[[f'lam{best_lam:g}_s{s}' for s in cfg.seeds]].mean(axis=1) - sel['lam0'])
                row['gain_best_lam'] = sel['gain'].mean()
                row['gain_ci_lo'], row['gain_ci_hi'] = piece_bootstrap(sel, 'gain')
            rows.append(row)
    return pd.DataFrame(rows)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'fusion'), config_name='fusion')
def main(cfg):
    device = auto_device()
    model, vocab, step, max_seq_len = load_checkpoint(REPO_ROOT / cfg.checkpoint, device)
    run = start_run(cfg, job_type='fusion', device=device, extra={'stage2_step': step})
    dists = load_distributions(cfg.strings_sweep, cfg.calibrated)
    rows, df = prepare(cfg, vocab, max_seq_len, dists)
    info(
        f'{len(rows)} sub-sequences, {len(df)} notes, {df.has_audio.mean():.2%} with audio, {len(dists)} distributions'
    )

    free = decode_free(model, rows, vocab, device, cfg.batch, max_seq_len)
    df['free'] = df.index.map(free).astype(bool)
    df['lam0'] = correct(decode_structured(model, rows, device, cfg.batch), df)
    for s in cfg.seeds:
        df[f'audio_s{s}_ok'] = df[f'audio_s{s}'] == df.true_string
    info(f'stage 2 free+pp {df.free.mean():.4f}, constrained {df.lam0.mean():.4f}')
    for lam in cfg.lambdas:
        if not lam:
            continue
        for i, s in enumerate(cfg.seeds):
            df[f'lam{lam:g}_s{s}'] = correct(decode_structured(model, rows, device, cfg.batch, lam, i), df)
        info(f'lam {lam:g}: {np.mean([df[f"lam{lam:g}_s{s}"].mean() for s in cfg.seeds]):.4f}')

    val = df[df.split == 'val']
    fused = [lam for lam in cfg.lambdas if lam]
    best_lam = max(fused, key=lambda lam: np.mean([val[f'lam{lam:g}_s{s}'].mean() for s in cfg.seeds])) if fused else 0
    summary = summarize(df, cfg, best_lam)
    info(f'best lam on val: {best_lam:g}\n{summary.round(4).to_string(index=False)}')

    df.to_csv(run.dir / 'notes.csv.gz', index=False)
    summary.to_csv(run.dir / 'summary.csv', index=False)
    flat = {'best_lam': best_lam}
    for r in summary.to_dict('records'):
        flat |= {f'{r["split"]}/{r["subset"]}/{k}': v for k, v in r.items() if k not in ('split', 'subset')}
    (run.dir / 'summary.json').write_text(json.dumps(flat, indent=1, default=float))
    run.finish(flat)


if __name__ == '__main__':
    main()

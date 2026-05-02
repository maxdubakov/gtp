"""Stage 2 evaluation: autoregressive generation + post-processing + per-source metrics.

Computes pitch_acc and tab_acc, both raw (model output) and post-processed (paper's
±5 neighbor correction → first viable fallback), per source, on val (and optionally test).

Single checkpoint:
  python scripts/stage2/eval.py --checkpoint runs/stage2_001/step_0030000_final.pth

All checkpoints in a directory:
  python scripts/stage2/eval.py --checkpoint-dir runs/stage2_001/

Include test split (default is val only):
  python scripts/stage2/eval.py --checkpoint-dir runs/stage2_001/ --include-test

Persist results for plotting / later analysis:
  python scripts/stage2/eval.py --checkpoint-dir runs/stage2_001/ --output results/eval.json
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gtp.stage2.data import build_datasets
from gtp.stage2.model import build_model
from gtp.stage2.postprocess import correct_tabs
from gtp.stage2.tokenizer import (
    EOS,
    NOTE_ON,
    PAD,
    TAB,
    TUNING_END,
    TUNING_START,
    VOCAB,
    parse_token_str,
)

# ---------------------------------------------------------------------------
# Token-stream helpers
# ---------------------------------------------------------------------------


def parse_tuning_from_enc(enc_ids):
    """Walk encoder IDs, return the tuning block as a list of pitches. None if missing."""
    in_tuning = False
    tuning = []
    for tid in enc_ids:
        t, v = parse_token_str(VOCAB.decode(int(tid)))
        if t == TUNING_START:
            in_tuning = True
            tuning = []
        elif t == TUNING_END:
            return tuning if tuning else None
        elif t == NOTE_ON and in_tuning:
            tuning.append(int(v))
    return None


def extract_input_pitches(enc_ids):
    """Walk encoder IDs, return the body's NOTE_ON pitches in order (skips tuning block)."""
    in_tuning = False
    pitches = []
    for tid in enc_ids:
        t, v = parse_token_str(VOCAB.decode(int(tid)))
        if t == TUNING_START:
            in_tuning = True
        elif t == TUNING_END:
            in_tuning = False
        elif t == NOTE_ON and not in_tuning:
            pitches.append(int(v))
        elif t == EOS:
            break
    return pitches


def extract_tabs(token_ids):
    """Walk decoder IDs, return list of (string, fret) from TAB tokens. Stops at EOS."""
    tabs = []
    for tid in token_ids:
        t, v = parse_token_str(VOCAB.decode(int(tid)))
        if t == EOS:
            break
        if t == PAD:
            continue
        if t == TAB:
            ss, ff = v.split(',')
            tabs.append((int(ss), int(ff)))
    return tabs


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def auto_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def evaluate_split(model, loader, device):
    """Run one split and return per-source counts.

    Returns dict {source: {n_input_notes, tab_correct_raw, tab_correct_pp,
                           pitch_correct_raw, pitch_correct_pp}}.
    Both raw and pp metrics use the same denominator (= number of input notes per piece),
    so they're directly comparable. Missing predictions count as wrong for raw.
    """
    metrics = defaultdict(
        lambda: {
            'n_input_notes': 0,
            'tab_correct_raw': 0,
            'tab_correct_pp': 0,
            'pitch_correct_raw': 0,
            'pitch_correct_pp': 0,
        }
    )

    with torch.no_grad():
        for batch in loader:
            enc, dec, sources = batch
            input_ids = enc.to(device)
            attention_mask = (input_ids != VOCAB.pad_id).long()

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=enc.size(1),
                num_beams=1,
                do_sample=False,
                pad_token_id=VOCAB.pad_id,
                eos_token_id=VOCAB.eos_id,
            )

            inputs_cpu = input_ids.cpu().tolist()
            dec_cpu = dec.cpu().tolist()
            gen_cpu = generated.cpu().tolist()

            for b in range(input_ids.size(0)):
                src = sources[b]
                tuning = parse_tuning_from_enc(inputs_cpu[b])
                if not tuning:
                    continue

                input_pitches = extract_input_pitches(inputs_cpu[b])
                gt_tabs = extract_tabs(dec_cpu[b])
                pred_tabs = extract_tabs(gen_cpu[b][1:])  # skip decoder_start (PAD)

                if not gt_tabs or not input_pitches:
                    continue

                # By construction, len(gt_tabs) should equal len(input_pitches).
                # If they differ, use the shorter (defensive).
                n = min(len(input_pitches), len(gt_tabs))

                corrected_tabs = correct_tabs(input_pitches[:n], pred_tabs, tuning)

                m = metrics[src]
                m['n_input_notes'] += n

                for j in range(n):
                    gs, gf = gt_tabs[j]
                    g_pitch = tuning[gs - 1] + gf if 1 <= gs <= len(tuning) else None

                    # --- raw: pred_tabs[j] if it exists, else miss ---
                    if j < len(pred_tabs):
                        ps, pf = pred_tabs[j]
                        if (ps, pf) == (gs, gf):
                            m['tab_correct_raw'] += 1
                        if (
                            g_pitch is not None
                            and 1 <= ps <= len(tuning)
                            and tuning[ps - 1] + pf == g_pitch
                        ):
                            m['pitch_correct_raw'] += 1

                    # --- post-processed: corrected_tabs[j] (always defined for in-range pitch) ---
                    cor = corrected_tabs[j]
                    if cor is not None:
                        cs, cf = cor
                        if (cs, cf) == (gs, gf):
                            m['tab_correct_pp'] += 1
                        if (
                            g_pitch is not None
                            and 1 <= cs <= len(tuning)
                            and tuning[cs - 1] + cf == g_pitch
                        ):
                            m['pitch_correct_pp'] += 1

    return dict(metrics)


def aggregate(metrics):
    """Add an `_overall` row summing across sources, return a clean per-source view with rates."""
    overall = {
        'n_input_notes': 0,
        'tab_correct_raw': 0,
        'tab_correct_pp': 0,
        'pitch_correct_raw': 0,
        'pitch_correct_pp': 0,
    }
    rows = {}
    for src, m in sorted(metrics.items()):
        rows[src] = _to_rates(m)
        for k in overall:
            overall[k] += m[k]
    rows['_overall'] = _to_rates(overall)
    return rows


def _to_rates(m):
    n = max(1, m['n_input_notes'])
    return {
        'n_input_notes': m['n_input_notes'],
        'tab_acc_raw': m['tab_correct_raw'] / n,
        'tab_acc_pp': m['tab_correct_pp'] / n,
        'pitch_acc_raw': m['pitch_correct_raw'] / n,
        'pitch_acc_pp': m['pitch_correct_pp'] / n,
    }


def print_metrics(label, rows):
    print(f'  {label}:')
    print(
        f'    {"source":<12} {"n_notes":>10}  '
        f'{"tab_raw":>8} {"tab_pp":>8}   {"pitch_raw":>10} {"pitch_pp":>10}'
    )
    for src, r in rows.items():
        marker = ' ← overall' if src == '_overall' else ''
        print(
            f'    {src:<12} {r["n_input_notes"]:>10,}  '
            f'{r["tab_acc_raw"]:>8.3f} {r["tab_acc_pp"]:>8.3f}   '
            f'{r["pitch_acc_raw"]:>10.3f} {r["pitch_acc_pp"]:>10.3f}{marker}'
        )


# ---------------------------------------------------------------------------
# Checkpoint discovery + main loop
# ---------------------------------------------------------------------------


def find_checkpoints(path):
    """Resolve --checkpoint or --checkpoint-dir to a list of (label, path) pairs."""
    p = Path(path)
    if p.is_file():
        return [(p.stem, p)]
    if p.is_dir():
        files = sorted(p.glob('step_*.pth'))
        return [(f.stem, f) for f in files]
    raise FileNotFoundError(f'Not a file or directory: {path}')


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    meta = ckpt.get('tokenizer_meta', {})
    if meta.get('vocab_size') and meta['vocab_size'] != len(VOCAB):
        raise ValueError(
            f'Vocab mismatch: checkpoint has {meta["vocab_size"]} tokens, '
            f'current vocab has {len(VOCAB)}.'
        )
    model = build_model().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt.get('iteration', None)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--checkpoint', help='Single .pth checkpoint to evaluate')
    src.add_argument('--checkpoint-dir', help='Evaluate every step_*.pth in this directory')
    ap.add_argument('--include-test', action='store_true', help='Also evaluate on test split')
    ap.add_argument('--datasets', nargs='+', default=None, help='Filter to specific sources')
    ap.add_argument('--device', default=None)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--output', default=None, help='Optional JSON path to save aggregated results')
    args = ap.parse_args()

    device = args.device or auto_device()
    print(f'Device: {device}')

    print('Building datasets...')
    _train_ds, val_ds, test_ds, _stats = build_datasets(datasets=args.datasets)
    print(f'  val sequences: {len(val_ds)}, test sequences: {len(test_ds)}')

    pin_memory = device == 'cuda'
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if args.include_test
        else None
    )

    ckpts = find_checkpoints(args.checkpoint or args.checkpoint_dir)
    print(f'Evaluating {len(ckpts)} checkpoint(s)')

    all_results = []
    for label, path in ckpts:
        print(f'\n===== {label} =====')
        t0 = time.time()
        model, step = load_checkpoint(path, device)

        record = {'checkpoint': str(path), 'label': label, 'step': step, 'splits': {}}

        val_metrics = evaluate_split(model, val_loader, device)
        val_rows = aggregate(val_metrics)
        print_metrics('val', val_rows)
        record['splits']['val'] = val_rows

        if test_loader is not None:
            test_metrics = evaluate_split(model, test_loader, device)
            test_rows = aggregate(test_metrics)
            print_metrics('test', test_rows)
            record['splits']['test'] = test_rows

        elapsed = time.time() - t0
        print(f'  elapsed: {elapsed:.1f}s')
        all_results.append(record)

        # Free model memory before next checkpoint
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nSaved results to {out_path}')


if __name__ == '__main__':
    main()

"""Stage 2 evaluation: autoregressive generation + post-processing + a bunch of metrics."""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gtp.device import auto_device
from gtp.log import info
from gtp.stage2.config import RunConfig, find_run_config
from gtp.stage2.data import build_datasets
from gtp.stage2.inference import load_checkpoint
from gtp.stage2.metrics import (
    classify_error,
    compute_eval_summary,
    difficulty_score,
    pitch_correct,
    pitch_of,
    tab_correct,
)
from gtp.stage2.postprocess import correct_tabs
from gtp.stage2.tokenizer import (
    Vocabulary,
    extract_input_pitches,
    extract_tabs,
    parse_tuning_from_enc,
)


def evaluate_split(model, loader, vocab, device, fallback='first_viable'):
    """Returns (per_source_counts, per_note_records).
    `per_source_counts`: dictionary containing counts of correctly predicted tabs/pitches (raw and post-processed)
    `per_note_records`: list of dictionaries, contains a lot of meta-information about this tab prediction
    """
    metrics = defaultdict(
        lambda: {
            'n_input_notes': 0,
            'tab_correct_raw': 0,
            'tab_correct_pp': 0,
            'pitch_correct_raw': 0,
            'pitch_correct_pp': 0,
            'difficulty_sum_gt': 0.0,
            'difficulty_sum_raw': 0.0,
            'difficulty_sum_pp': 0.0,
            'difficulty_n_subseqs_gt': 0,
            'difficulty_n_subseqs_raw': 0,
            'difficulty_n_subseqs_pp': 0,
        }
    )
    note_records: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            enc, dec, sources, piece_ids = batch
            input_ids = enc.to(device)
            attention_mask = (input_ids != vocab.pad_id).long()

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=enc.size(1),
                num_beams=1,
                do_sample=False,
                pad_token_id=vocab.pad_id,
                eos_token_id=vocab.eos_id,
            )

            inputs_cpu = input_ids.cpu().tolist()
            dec_cpu = dec.cpu().tolist()
            gen_cpu = generated.cpu().tolist()

            for seq_idx in range(input_ids.size(0)):
                src = sources[seq_idx]
                pid = piece_ids[seq_idx]
                tuning = parse_tuning_from_enc(inputs_cpu[seq_idx], vocab)
                if not tuning:
                    info(f'[ERROR] no tuning for piece_id={pid}. Skipped sub-seq')
                    continue

                input_pitches = extract_input_pitches(inputs_cpu[seq_idx], vocab)
                gt_tabs = extract_tabs(dec_cpu[seq_idx], vocab)
                pred_tabs = extract_tabs(gen_cpu[seq_idx][1:], vocab)  # skip decoder_start (PAD)

                if not gt_tabs or not input_pitches:
                    info(f'[ERROR] empty gt_tabs or input_pitches for piece_id={pid}. Skipped sub-seq')
                    continue
                if len(gt_tabs) != len(input_pitches):
                    info(f'[ERROR] gt_tabs/input_pitches length mismatch for piece_id={pid}. Skipped sub-seq')
                    continue
                n = len(gt_tabs)

                corrected_tabs = correct_tabs(input_pitches, pred_tabs, tuning, fallback=fallback)

                m = metrics[src]
                m['n_input_notes'] += n

                for j in range(n):
                    true_s, true_f = gt_tabs[j]
                    g_pitch = pitch_of((true_s, true_f), tuning)

                    raw = pred_tabs[j] if j < len(pred_tabs) else None
                    cor = corrected_tabs[j] if j < len(corrected_tabs) else None

                    if raw is not None:
                        if tab_correct(raw, gt_tabs[j]):
                            m['tab_correct_raw'] += 1
                        if g_pitch is not None and pitch_correct(raw, g_pitch, tuning):
                            m['pitch_correct_raw'] += 1
                    if cor is not None:
                        if tab_correct(cor, gt_tabs[j]):
                            m['tab_correct_pp'] += 1
                        if g_pitch is not None and pitch_correct(cor, g_pitch, tuning):
                            m['pitch_correct_pp'] += 1

                    raw_s, raw_f = raw if raw else (None, None)
                    raw_pitch = pitch_of(raw, tuning)
                    pp_s, pp_f = cor if cor else (None, None)
                    pp_pitch = pitch_of(cor, tuning)

                    note_records.append(
                        {
                            'piece_id': pid,
                            'source': src,
                            'pitch': g_pitch,
                            'true_string': true_s,
                            'true_fret': true_f,
                            'pred_raw_string': raw_s,
                            'pred_raw_fret': raw_f,
                            'pred_raw_pitch': raw_pitch,
                            'pred_pp_string': pp_s,
                            'pred_pp_fret': pp_f,
                            'pred_pp_pitch': pp_pitch,
                            'error_type_raw': classify_error(true_s, true_f, g_pitch, raw_s, raw_f, raw_pitch),
                            'error_type_pp': classify_error(true_s, true_f, g_pitch, pp_s, pp_f, pp_pitch),
                            'delta_string_pp': (pp_s - true_s) if pp_s is not None else None,
                            'delta_fret_pp': (pp_f - true_f) if pp_f is not None else None,
                        }
                    )

                d_gt = difficulty_score(gt_tabs)
                d_raw = difficulty_score(pred_tabs)
                d_pp = difficulty_score(corrected_tabs)
                if d_gt is not None:
                    m['difficulty_sum_gt'] += d_gt
                    m['difficulty_n_subseqs_gt'] += 1
                if d_raw is not None:
                    m['difficulty_sum_raw'] += d_raw
                    m['difficulty_n_subseqs_raw'] += 1
                if d_pp is not None:
                    m['difficulty_sum_pp'] += d_pp
                    m['difficulty_n_subseqs_pp'] += 1

    return dict(metrics), note_records


def _to_rates(m):
    n = max(1, m['n_input_notes'])

    def _safe_div(num, den):
        return num / den if den > 0 else None

    return {
        'n_input_notes': m['n_input_notes'],
        'tab_acc_raw': m['tab_correct_raw'] / n,
        'tab_acc_pp': m['tab_correct_pp'] / n,
        'pitch_acc_raw': m['pitch_correct_raw'] / n,
        'pitch_acc_pp': m['pitch_correct_pp'] / n,
        'diff_gt': _safe_div(m['difficulty_sum_gt'], m['difficulty_n_subseqs_gt']),
        'diff_raw': _safe_div(m['difficulty_sum_raw'], m['difficulty_n_subseqs_raw']),
        'diff_pp': _safe_div(m['difficulty_sum_pp'], m['difficulty_n_subseqs_pp']),
        'difficulty_n_subseqs': m['difficulty_n_subseqs_gt'],
    }


def aggregate(metrics):
    counter_keys = [
        'n_input_notes',
        'tab_correct_raw',
        'tab_correct_pp',
        'pitch_correct_raw',
        'pitch_correct_pp',
        'difficulty_sum_gt',
        'difficulty_sum_raw',
        'difficulty_sum_pp',
        'difficulty_n_subseqs_gt',
        'difficulty_n_subseqs_raw',
        'difficulty_n_subseqs_pp',
    ]
    overall = dict.fromkeys(counter_keys, 0)
    rows = {}
    for src, m in sorted(metrics.items()):
        rows[src] = _to_rates(m)
        for k in counter_keys:
            overall[k] += m.get(k, 0)
    rows['_overall'] = _to_rates(overall)
    return rows


def _print_split(label, summary):
    if not summary or summary.get('n_notes', 0) == 0:
        info(f'[{label}] (no records)')
        return
    info(
        f'[{label}] tab_strict={summary["tab_strict_acc"]:.4f}  '
        f'tab_equivalent={summary["tab_equivalent_acc"]:.4f}  '
        f'pitch_pp={summary["pitch_pp_acc"]:.4f}  '
        f'n_pieces={summary["n_pieces_qualified"]}'
    )


_STEP_RE = re.compile(r'step_(\d+)')


def filter_by_steps(items, steps):
    by_step: dict[int, tuple] = {}
    for label, path in items:
        m = _STEP_RE.search(label)
        if not m:
            continue
        s = int(m.group(1))
        if s not in by_step or '_final' in label:
            by_step[s] = (label, path)
    missing = [s for s in steps if s not in by_step]
    if missing:
        raise SystemExit(f'Requested checkpoint-steps not found: {missing}. Available: {sorted(by_step)}')
    return [by_step[s] for s in steps]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--checkpoint', help='Single .pth checkpoint to evaluate')
    src.add_argument('--run-dir', help='Run directory; scans <run-dir>/checkpoints/ for step_*.pth files')
    ap.add_argument(
        '--checkpoint-steps',
        type=int,
        nargs='+',
        default=None,
        metavar='STEP',
        help='When --run-dir is set, restrict to these exact step values (e.g. --checkpoint-steps 5000 10000 ...)',
    )
    ap.add_argument('--output', required=True, help='JSON path to save aggregated results')
    ap.add_argument('--device', default=None)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument(
        '--fallback',
        choices=['first_viable', 'nearest_viable'],
        default='first_viable',
        help='Post-processing fallback strategy when no model tab in ±window matches the input pitch',
    )
    args = ap.parse_args()
    if args.checkpoint_steps and not args.run_dir:
        ap.error('--checkpoint-steps requires --run-dir')

    device = args.device or auto_device()
    info(f'Device: {device}')

    # Resolve checkpoints from --checkpoint (single file) or --run-dir (+ optional --checkpoint-steps filter)
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f'Not a file: {ckpt}')
        ckpts = [(ckpt.stem, ckpt)]
    else:
        ckpt_dir = Path(args.run_dir) / 'checkpoints'
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f'No checkpoints/ subdir at {args.run_dir}')
        ckpts = [(f.stem, f) for f in sorted(ckpt_dir.glob('step_*.pth'))]
        if args.checkpoint_steps:
            ckpts = filter_by_steps(ckpts, args.checkpoint_steps)
    info(f'Evaluating {len(ckpts)} checkpoint(s)')
    cfg = RunConfig.load(find_run_config(ckpts[0][1]))
    vocab = Vocabulary(include_genre=cfg.conditioning.genre)
    max_seq_len = cfg.train.max_seq_len
    info(f'Vocab: {len(vocab)} tokens (include_genre={cfg.conditioning.genre})  max_seq_len: {max_seq_len}')

    # Build datasets
    datasets, _stats = build_datasets(vocab, splits=('val', 'test'), max_seq_len=max_seq_len)
    val_ds, test_ds = datasets['val'], datasets['test']
    info(f'Val sequences: {len(val_ds)}, test sequences: {len(test_ds)}')

    pin_memory = device == 'cuda'
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    all_results = []
    for label, path in ckpts:
        info('\n' + '-' * 20 + f' {label} ' + '-' * 20)
        t0 = time.time()
        model, _vocab_ckpt, step, _max_seq_len = load_checkpoint(path, device)

        record = {'checkpoint': str(path), 'label': label, 'step': step, 'splits': {}, 'fallback': args.fallback}

        for split, loader in (('val', val_loader), ('test', test_loader)):
            metrics, records = evaluate_split(model, loader, vocab, device, fallback=args.fallback)
            summary = compute_eval_summary(records)
            _print_split(split, summary)
            record['splits'][split] = aggregate(metrics)
            record[f'summary_{split}'] = summary

        elapsed = time.time() - t0
        info(f'  elapsed: {elapsed:.1f}s')
        all_results.append(record)

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    info(f'\nSaved results to {out_path}')


if __name__ == '__main__':
    main()

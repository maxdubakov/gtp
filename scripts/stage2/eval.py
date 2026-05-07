"""Stage 2 evaluation: autoregressive generation + post-processing + per-source metrics.

Computes pitch_acc and tab_acc, both raw (model output) and post-processed (paper's
±5 neighbor correction → first viable fallback), per source, on val (and optionally test).

Single checkpoint:
  python scripts/stage2/eval.py --checkpoint runs/stage2_baseline/checkpoints/step_0060000_final.pth

All checkpoints in a directory:
  python scripts/stage2/eval.py --checkpoint-dir runs/stage2_baseline/

Include test split (default is val only):
  python scripts/stage2/eval.py --checkpoint-dir runs/stage2_baseline/ --include-test

Persist results for plotting / later analysis:
  python scripts/stage2/eval.py --checkpoint-dir runs/stage2_baseline/ --output results/eval.json
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gtp.log import info
from gtp.stage2.config import RunConfig, find_run_config
from gtp.stage2.data import TabDataset, load_jsonl_pieces
from gtp.stage2.metrics import (
    classify_error,
    compute_eval_summary,
    difficulty_score,
    pitch_correct,
    tab_correct,
)
from gtp.stage2.model import build_model
from gtp.stage2.paths import AUG_DATA_DIR
from gtp.stage2.postprocess import correct_tabs
from gtp.stage2.tokenizer import (
    EOS,
    NOTE_ON,
    PAD,
    TAB,
    TUNING_END,
    TUNING_START,
    Vocabulary,
    parse_token_str,
)

# ---------------------------------------------------------------------------
# Token-stream helpers
# ---------------------------------------------------------------------------


def parse_tuning_from_enc(enc_ids, vocab):
    """Walk encoder IDs, return the tuning block as a list of pitches. None if missing."""
    in_tuning = False
    tuning = []
    for tid in enc_ids:
        t, v = parse_token_str(vocab.decode(int(tid)))
        if t == TUNING_START:
            in_tuning = True
            tuning = []
        elif t == TUNING_END:
            return tuning if tuning else None
        elif t == NOTE_ON and in_tuning:
            tuning.append(int(v))
    return None


def extract_input_pitches(enc_ids, vocab):
    """Walk encoder IDs, return the body's NOTE_ON pitches in order (skips tuning block)."""
    in_tuning = False
    pitches = []
    for tid in enc_ids:
        t, v = parse_token_str(vocab.decode(int(tid)))
        if t == TUNING_START:
            in_tuning = True
        elif t == TUNING_END:
            in_tuning = False
        elif t == NOTE_ON and not in_tuning:
            pitches.append(int(v))
        elif t == EOS:
            break
    return pitches


def extract_tabs(token_ids, vocab):
    """Walk decoder IDs, return list of (string, fret) from TAB tokens. Stops at EOS."""
    tabs = []
    for tid in token_ids:
        t, v = parse_token_str(vocab.decode(int(tid)))
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


def evaluate_split(model, loader, vocab, device, fallback='first_viable'):
    """Run one split. Returns (per_source_counts, per_note_records).

    `per_source_counts`: dict {source: {n_input_notes, tab_correct_raw,
        tab_correct_pp, pitch_correct_raw, pitch_correct_pp, difficulty_sums,
        difficulty_n_subseqs}}.

    `per_note_records`: list of dicts (one per ground-truth note across all
        sub-sequences), each with the fields needed by
        gtp.stage2.metrics.compute_eval_summary:
            piece_id, source, pitch, true_string, true_fret,
            pred_raw_string, pred_raw_fret, pred_raw_pitch,
            pred_pp_string, pred_pp_fret, pred_pp_pitch,
            error_type_raw, error_type_pp,
            delta_string_pp, delta_fret_pp.

    Difficulty is aggregated per-subseq (mean of per-subseq means), since
    difficulty is naturally a sequence-level quantity.
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

            for b in range(input_ids.size(0)):
                src = sources[b]
                pid = piece_ids[b]
                tuning = parse_tuning_from_enc(inputs_cpu[b], vocab)
                if not tuning:
                    continue

                input_pitches = extract_input_pitches(inputs_cpu[b], vocab)
                gt_tabs = extract_tabs(dec_cpu[b], vocab)
                pred_tabs = extract_tabs(gen_cpu[b][1:], vocab)  # skip decoder_start (PAD)

                if not gt_tabs or not input_pitches:
                    continue

                # By construction, len(gt_tabs) should equal len(input_pitches).
                # If they differ, use the shorter (defensive).
                n = min(len(input_pitches), len(gt_tabs))

                corrected_tabs = correct_tabs(input_pitches[:n], pred_tabs, tuning, fallback=fallback)

                m = metrics[src]
                m['n_input_notes'] += n

                for j in range(n):
                    true_s, true_f = gt_tabs[j]
                    g_pitch = tuning[true_s - 1] + true_f if 1 <= true_s <= len(tuning) else None

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
                    raw_pitch = tuning[raw_s - 1] + raw_f if raw_s is not None and 1 <= raw_s <= len(tuning) else None
                    pp_s, pp_f = cor if cor else (None, None)
                    pp_pitch = tuning[pp_s - 1] + pp_f if pp_s is not None and 1 <= pp_s <= len(tuning) else None

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

                # --- difficulty: per-subseq means ---
                d_gt = difficulty_score(gt_tabs[:n])
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


def aggregate(metrics):
    """Add an `_overall` row summing across sources, return a clean per-source view with rates."""
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


def _to_rates(m):
    n = max(1, m['n_input_notes'])
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


def _safe_div(num, den):
    return num / den if den > 0 else None


def print_metrics(label, rows):
    print(f'  {label}:')
    print(
        f'    {"source":<12} {"n_notes":>10}  '
        f'{"tab_raw":>8} {"tab_pp":>8}   {"pitch_raw":>10} {"pitch_pp":>10}   '
        f'{"diff_gt":>8} {"diff_raw":>9} {"diff_pp":>8}'
    )
    for src, r in rows.items():
        marker = ' ← overall' if src == '_overall' else ''
        print(
            f'    {src:<12} {r["n_input_notes"]:>10,}  '
            f'{r["tab_acc_raw"]:>8.3f} {r["tab_acc_pp"]:>8.3f}   '
            f'{r["pitch_acc_raw"]:>10.3f} {r["pitch_acc_pp"]:>10.3f}   '
            f'{_fmt(r["diff_gt"]):>8} {_fmt(r["diff_raw"]):>9} {_fmt(r["diff_pp"]):>8}{marker}'
        )


def _print_summary(label, s):
    """Print the rich Stage-2 metrics summary returned by compute_eval_summary."""
    if not s or s.get('n_notes', 0) == 0:
        print(f'  [{label}] (no records)')
        return
    print(
        f'\n  [{label}] tab_strict={s["tab_strict_acc"]:.3f}  '
        f'tab_equivalent={s["tab_equivalent_acc"]:.3f}  '
        f'pitch_pp={s["pitch_pp_acc"]:.3f}  '
        f'recovered_by_alt={s["recovered_by_alt"]:,}'
    )
    bc = s.get('drift_buckets', {})
    bn = s.get('drift_buckets_notes', {})
    n_qualified = max(s.get('n_pieces_qualified', 1), 1)
    n_notes_total = max(sum(bn.values()), 1)
    print(f'    drift buckets ({s.get("n_pieces_qualified", 0)} pieces ≥20 notes):')
    for b in ('perfect', 'consistent_alt', 'partial_alt', 'inconsistent'):
        p = bc.get(b, 0)
        n = bn.get(b, 0)
        print(
            f'      {b:<16s}  {p:>4d} pcs ({100 * p / n_qualified:>5.1f}%)  '
            f'{n:>8,d} notes ({100 * n / n_notes_total:>5.1f}%)'
        )
    if s.get('consistent_alt_drift_histogram'):
        print('    top consistent_alt drifts:')
        for drift, n_pcs in list(s['consistent_alt_drift_histogram'].items())[:5]:
            print(f'      ({drift})  {n_pcs} pcs')


def _fmt(x):
    return '   --   ' if x is None else f'{x:.3f}'


# ---------------------------------------------------------------------------
# Checkpoint discovery + main loop
# ---------------------------------------------------------------------------


def find_checkpoints(path):
    """Resolve --checkpoint or --checkpoint-dir to a list of (label, path) pairs.

    Supports both layouts:
      - new:    <run-dir>/checkpoints/step_*.pth
      - legacy: <run-dir>/step_*.pth
    When given a run dir, prefers the `checkpoints/` subdir if it exists.
    """
    p = Path(path)
    if p.is_file():
        return [(p.stem, p)]
    if p.is_dir():
        ckpt_dir = p / 'checkpoints'
        search_dir = ckpt_dir if ckpt_dir.is_dir() else p
        files = sorted(search_dir.glob('step_*.pth'))
        return [(f.stem, f) for f in files]
    raise FileNotFoundError(f'Not a file or directory: {path}')


_STEP_RE = re.compile(r'step_(\d+)')


def filter_by_steps(items, steps):
    """Keep only `(label, path)` items whose label encodes one of the requested steps.

    Use case: aligning eval sample points across runs with different save cadences
    so the resulting eval_sampled.json files are directly comparable. Errors if
    any requested step is not found among `items`.

    When duplicate labels exist for the same step (e.g. step_0030000.pth and
    step_0030000_final.pth), prefers the `_final` variant.
    """
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
        raise SystemExit(
            f'Requested checkpoint-steps not found: {missing}. Available: {sorted(by_step)}'
        )
    return [by_step[s] for s in steps]


def load_checkpoint(path, device):
    """Load checkpoint + matching Vocabulary. Returns (model, vocab, iteration)."""
    from gtp.stage2.config import RunConfig, find_run_config

    include_genre = False
    config_path = find_run_config(path)
    if config_path is not None:
        try:
            cfg = RunConfig.load(config_path)
            include_genre = cfg.conditioning.genre
        except Exception as e:
            print(f'  WARN: could not parse {config_path}: {e}. Assuming no genre conditioning.')
    vocab = Vocabulary(include_genre=include_genre)

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    meta = ckpt.get('tokenizer_meta', {})
    if meta.get('vocab_size') and meta['vocab_size'] != len(vocab):
        raise ValueError(f'Vocab mismatch: checkpoint has {meta["vocab_size"]} tokens, current vocab has {len(vocab)}.')
    model = build_model(vocab).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocab, ckpt.get('iteration', None)


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
    ap.add_argument(
        '--fallback',
        choices=['first_viable', 'nearest_viable'],
        default='first_viable',
        help='Post-processing fallback strategy when no model tab in ±window matches the '
        'input pitch. first_viable = paper-faithful (high-string-first lowest-fret). '
        'nearest_viable = deviation: Manhattan-nearest realization to the model raw output.',
    )
    ap.add_argument(
        '--checkpoint-steps',
        type=int,
        nargs='+',
        default=None,
        metavar='STEP',
        help='Eval only checkpoints at these exact step values (e.g. '
             '--checkpoint-steps 5000 10000 15000 ...). Errors if any requested step '
             "is not saved. Useful for aligning eval points across runs with "
             'different save cadences. Default: eval all available checkpoints.',
    )
    args = ap.parse_args()

    device = args.device or auto_device()
    info(f'Device: {device}')

    # Build vocab from the first checkpoint's sibling config.json. All
    # checkpoints in --checkpoint-dir mode are assumed to share a vocab
    # (same training run). Falls back to no-genre vocab if no config.json.
    # find_run_config handles both layouts (checkpoints/ subdir or flat).
    ckpts = find_checkpoints(args.checkpoint or args.checkpoint_dir)
    n_total = len(ckpts)
    if args.checkpoint_steps:
        ckpts = filter_by_steps(ckpts, args.checkpoint_steps)
        info(f'Selecting {len(ckpts)} of {n_total} checkpoints '
             f'(--checkpoint-steps={args.checkpoint_steps})')
    info(f'Evaluating {len(ckpts)} checkpoint(s)')
    config_path = find_run_config(ckpts[0][1])
    include_genre = False
    if config_path is not None:
        try:
            cfg = RunConfig.load(config_path)
            include_genre = cfg.conditioning.genre
        except Exception as e:
            print(f'  WARN: could not parse {config_path}: {e}.')
    vocab = Vocabulary(include_genre=include_genre)
    info(f'Vocab: {len(vocab)} tokens (include_genre={include_genre})')

    info('Loading val/test pieces (skipping train.jsonl)...')
    val_pieces = load_jsonl_pieces(AUG_DATA_DIR / 'val.jsonl')
    test_pieces = load_jsonl_pieces(AUG_DATA_DIR / 'test.jsonl') if args.include_test else []
    if args.datasets:
        wanted = set(args.datasets)
        val_pieces = [p for p in val_pieces if p['source'] in wanted]
        test_pieces = [p for p in test_pieces if p['source'] in wanted]
    val_ds = TabDataset(val_pieces, vocab, augment=False)
    test_ds = TabDataset(test_pieces, vocab, augment=False) if args.include_test else None
    print(f'  val sequences: {len(val_ds)}', end='')
    print(f', test sequences: {len(test_ds)}' if test_ds else '')

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
        if test_ds is not None
        else None
    )

    all_results = []
    for label, path in ckpts:
        info(f'\n===== {label} =====')
        t0 = time.time()
        model, _vocab_ckpt, step = load_checkpoint(path, device)
        # _vocab_ckpt should match `vocab` since all ckpts share config.json;
        # we use the outer `vocab` for consistency with the pre-built datasets.

        record = {'checkpoint': str(path), 'label': label, 'step': step, 'splits': {}}

        val_metrics, val_records = evaluate_split(model, val_loader, vocab, device, fallback=args.fallback)
        val_rows = aggregate(val_metrics)
        print_metrics('val', val_rows)
        val_summary = compute_eval_summary(val_records)
        _print_summary('val', val_summary)
        record['splits']['val'] = val_rows
        record['summary_val'] = val_summary
        record['fallback'] = args.fallback

        if test_loader is not None:
            test_metrics, test_records = evaluate_split(model, test_loader, vocab, device, fallback=args.fallback)
            test_rows = aggregate(test_metrics)
            print_metrics('test', test_rows)
            test_summary = compute_eval_summary(test_records)
            _print_summary('test', test_summary)
            record['splits']['test'] = test_rows
            record['summary_test'] = test_summary

        elapsed = time.time() - t0
        info(f'  elapsed: {elapsed:.1f}s')
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
        info(f'\nSaved results to {out_path}')


if __name__ == '__main__':
    main()

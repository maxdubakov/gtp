"""Train T5 model with self-adaptive Adafactor.

Mirrors the Stage 1 training pattern: argparse, infinite iterator over the loader,
periodic eval/save, ETA logging. Differences:
  - HF T5ForConditionalGeneration (returns loss when given labels)
  - Adafactor in self-adaptive mode (lr=None, no manual schedule)
  - Per-source val loss tracking via the source tag in TabDataset batches

Usage:
  python scripts/stage2/train.py --device cuda --num-workers 4
  python scripts/stage2/train.py --datasets guitarset --batch-size 2 --max-steps 50  # smoke test
  python scripts/stage2/train.py --resume runs/stage2_001/step_0010000.pth
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Adafactor

from gtp import REPO_ROOT
from gtp.stage2.data import build_datasets
from gtp.stage2.model import build_model
from gtp.stage2.tokenizer import (
    EOS,
    MAX_TIME_SHIFT,
    NOTE_ON,
    PAD,
    TAB,
    TIME_SHIFT_BINS,
    TUNING_END,
    TUNING_START,
    VOCAB,
    parse_token_str,
)

DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, 'runs', 'stage2_001')


def auto_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def make_t5_inputs(enc_ids, dec_ids, device):
    """enc_ids, dec_ids: (B, T) padded with VOCAB.pad_id. Returns (input_ids, attention_mask, labels)."""
    input_ids = enc_ids.to(device)
    attention_mask = (input_ids != VOCAB.pad_id).long()
    labels = dec_ids.to(device).clone()
    labels[labels == VOCAB.pad_id] = -100
    return input_ids, attention_mask, labels


def per_sequence_loss(logits, labels):
    """Mean cross-entropy per sequence, ignoring -100 positions. Returns (B,) tensor."""
    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='none',
        ignore_index=-100,
    ).reshape(labels.shape)
    mask = (labels != -100).float()
    return (flat_loss * mask).sum(1) / mask.sum(1).clamp(min=1)


def parse_tuning_from_enc(enc_ids):
    """Walk encoder IDs, extract the tuning block as a list of pitches. None if missing."""
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


def run_generation_eval(model, val_loader, device, max_batches=20):
    """Generate predictions and compute tab + pitch accuracy.

    Greedy-decode the decoder, parse predicted TAB tokens, align position-by-position
    with ground-truth TABs. Length mismatches penalize: denominator is max(predicted,
    ground_truth) so missing/extra tabs count as wrong.

    Returns (overall_tab_acc, overall_pitch_acc, per_source) where per_source maps
    source -> (tab_acc, pitch_acc, n_tabs).
    """
    model.eval()
    totals = defaultdict(lambda: [0, 0, 0])  # source -> [tab_correct, pitch_correct, n_total]

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
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

            for b in range(input_ids.size(0)):
                tuning = parse_tuning_from_enc(input_ids[b].tolist())
                if not tuning:
                    continue

                pred = extract_tabs(generated[b, 1:].tolist())  # skip decoder_start (PAD)
                gt = extract_tabs(dec[b].tolist())
                if not gt:
                    continue

                n_pairs = min(len(pred), len(gt))
                n_total = max(len(pred), len(gt))
                tab_c = 0
                pitch_c = 0
                for (ps, pf), (gs, gf) in zip(pred[:n_pairs], gt[:n_pairs], strict=False):
                    if (ps, pf) == (gs, gf):
                        tab_c += 1
                    if (
                        1 <= ps <= len(tuning)
                        and 1 <= gs <= len(tuning)
                        and tuning[ps - 1] + pf == tuning[gs - 1] + gf
                    ):
                        pitch_c += 1

                totals[sources[b]][0] += tab_c
                totals[sources[b]][1] += pitch_c
                totals[sources[b]][2] += n_total

    model.train()
    if not totals:
        return float('nan'), float('nan'), {}

    per_src = {}
    sum_tab, sum_pitch, sum_n = 0, 0, 0
    for src, (tc, pc, n) in totals.items():
        per_src[src] = (tc / n if n else 0.0, pc / n if n else 0.0, n)
        sum_tab += tc
        sum_pitch += pc
        sum_n += n
    return (sum_tab / sum_n if sum_n else 0.0, sum_pitch / sum_n if sum_n else 0.0, per_src)


def run_eval(model, val_loader, device, max_batches=None):
    """Returns (overall_mean_loss, {source: mean_loss})."""
    model.eval()
    totals = defaultdict(lambda: [0.0, 0])
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            enc, dec, sources = batch
            input_ids, attention_mask, labels = make_t5_inputs(enc, dec, device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            for s, loss_val in zip(sources, per_sequence_loss(outputs.logits, labels).tolist(), strict=True):
                totals[s][0] += loss_val
                totals[s][1] += 1
    model.train()
    if not totals:
        return float('nan'), {}
    per_source = {s: total / count for s, (total, count) in totals.items()}
    overall = sum(t for t, _ in totals.values()) / sum(c for _, c in totals.values())
    return overall, per_source


def format_eta(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f'~{h}h{m:02d}m'
    return f'~{m}m'


def save_checkpoint(path, step, model, optimizer, args):
    torch.save(
        {
            'iteration': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': vars(args),
            'tokenizer_meta': {
                'vocab_size': len(VOCAB),
                'time_shift_max': MAX_TIME_SHIFT,
                'time_shift_step': TIME_SHIFT_BINS[0],
                'pad_id': VOCAB.pad_id,
                'eos_id': VOCAB.eos_id,
            },
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description='Train (MIDI → Tab) T5 model')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--resume', default=None, help='Resume from checkpoint (model + optimizer + step)')
    parser.add_argument('--datasets', nargs='+', default=None, help='Subset of sources (default: all four)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=30000)
    parser.add_argument('--eval-steps', type=int, default=1000)
    parser.add_argument('--save-steps', type=int, default=5000)
    parser.add_argument('--eval-batches', type=int, default=None, help='Cap val batches per eval (default: all)')
    parser.add_argument(
        '--gen-eval-batches', type=int, default=10, help='Val batches for the slower tab/pitch generation eval'
    )
    parser.add_argument('--device', default=None, help='cpu / mps / cuda (default: auto)')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = args.device or auto_device()
    print(f'Device: {device}')
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print('Building datasets...')
    train_ds, val_ds, _test_ds, _stats = build_datasets(datasets=args.datasets)
    print(f'  train sequences: {len(train_ds)}, val sequences: {len(val_ds)}')

    pin_memory = device == 'cuda'
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f'Building model (vocab={len(VOCAB)})...')
    model = build_model().to(device)
    model.train()
    print(f'  parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = Adafactor(
        model.parameters(),
        lr=None,
        relative_step=True,
        scale_parameter=True,
        warmup_init=True,
    )

    step = 0
    if args.resume:
        print(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        meta = ckpt.get('tokenizer_meta', {})
        if meta.get('vocab_size') and meta['vocab_size'] != len(VOCAB):
            raise SystemExit(
                f'Vocab mismatch: checkpoint has {meta["vocab_size"]} tokens, current vocab is {len(VOCAB)}. '
                f'Cannot resume training with a different vocabulary.'
            )
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt.get('iteration', 0)
        print(f'  resumed at step {step}')

    print(f'Training (max_steps={args.max_steps}, starting_step={step})')
    recent_losses = []
    step_times = []
    train_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        t0 = time.time()
        enc, dec, _sources = batch
        input_ids, attention_mask, labels = make_t5_inputs(enc, dec, device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        recent_losses.append(loss.item())
        step_times.append(time.time() - t0)

        log_interval = max(1, min(50, args.max_steps // 20))
        if step % log_interval == 0:
            avg_loss = float(np.mean(recent_losses[-log_interval:]))
            avg_step_time = float(np.mean(step_times[-log_interval:]))
            eta = format_eta(avg_step_time * (args.max_steps - step))
            print(f'[step {step}/{args.max_steps}] loss={avg_loss:.4f} ({avg_step_time:.2f}s/step, {eta})')

        if step > 0 and step % args.eval_steps == 0:
            overall, per_src = run_eval(model, val_loader, device, max_batches=args.eval_batches)
            src_str = '  '.join(f'{s}={loss_val:.3f}' for s, loss_val in sorted(per_src.items()))
            print(f'[eval @ step {step}] val_loss={overall:.4f}  {src_str}')

            tab_acc, pitch_acc, gen_per_src = run_generation_eval(
                model, val_loader, device, max_batches=args.gen_eval_batches
            )
            print(f'[gen  @ step {step}] tab_acc={tab_acc:.3f}  pitch_acc={pitch_acc:.3f}')
            for src, (ta, pa, n) in sorted(gen_per_src.items()):
                print(f'    {src:<12} tab={ta:.3f}  pitch={pa:.3f}  n={n}')

        if step > 0 and step % args.save_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f'step_{step:07d}.pth')
            save_checkpoint(ckpt_path, step, model, optimizer, args)
            print(f'[saved] {ckpt_path}')

    final_path = os.path.join(args.output_dir, f'step_{step:07d}_final.pth')
    save_checkpoint(final_path, step, model, optimizer, args)
    print(f'\nTraining complete. Final checkpoint: {final_path}')

    final_overall, final_per_src = run_eval(model, val_loader, device)
    print(f'Final val: loss={final_overall:.4f}')
    for s, loss_val in sorted(final_per_src.items()):
        print(f'  {s}: {loss_val:.4f}')


if __name__ == '__main__':
    main()

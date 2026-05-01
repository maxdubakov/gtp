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
from gtp.stage2.tokenizer import MAX_TIME_SHIFT, TIME_SHIFT_BINS, VOCAB

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

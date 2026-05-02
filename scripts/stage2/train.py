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
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Adafactor

from gtp import REPO_ROOT
from gtp.stage2.data import build_datasets
from gtp.stage2.model import build_model
from gtp.stage2.tokenizer import (
    MAX_TIME_SHIFT,
    NOTE_ON,
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


def run_eval(model, val_loader, device, max_batches=None):
    """Single forward pass over val: cross-entropy loss + teacher-forced tab/pitch accuracy.

    Note: tab/pitch accuracy here is teacher-forced (the decoder sees ground-truth
    previous tokens at each position), so numbers are higher than what you'd get
    from autoregressive generation. Useful as a fast training-time monitor;
    benchmark numbers (paper-comparable) need a separate autoregressive eval.

    Returns (overall_loss, overall_tab_acc, overall_pitch_acc, per_source) where
    per_source maps source → {loss, tab_acc, pitch_acc, n_tabs}.
    """
    model.eval()
    loss_totals = defaultdict(lambda: [0.0, 0])  # source -> [sum_loss, n_seqs]
    tab_totals = defaultdict(lambda: [0, 0])  # source -> [correct, n_tabs]
    pitch_totals = defaultdict(lambda: [0, 0])

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            enc, dec, sources = batch
            input_ids, attention_mask, labels = make_t5_inputs(enc, dec, device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            seq_losses = per_sequence_loss(outputs.logits, labels).tolist()
            preds = outputs.logits.argmax(-1)  # (B, T) teacher-forced predictions

            # Pull to CPU once per batch — cheaper than per-element tensor access.
            preds_cpu = preds.cpu().tolist()
            labels_cpu = labels.cpu().tolist()
            inputs_cpu = input_ids.cpu().tolist()

            for b in range(input_ids.size(0)):
                src = sources[b]
                loss_totals[src][0] += seq_losses[b]
                loss_totals[src][1] += 1

                tuning = parse_tuning_from_enc(inputs_cpu[b])

                for t_idx, gt_id in enumerate(labels_cpu[b]):
                    if gt_id < 0:  # PAD-as-loss-mask (-100), skip
                        continue
                    gt_type, gt_val = parse_token_str(VOCAB.decode(gt_id))
                    if gt_type != TAB:
                        continue

                    pred_id = preds_cpu[b][t_idx]
                    tab_totals[src][1] += 1
                    pitch_totals[src][1] += 1

                    if pred_id == gt_id:
                        tab_totals[src][0] += 1
                        pitch_totals[src][0] += 1  # exact token match implies same pitch
                    elif tuning:
                        pred_type, pred_val = parse_token_str(VOCAB.decode(pred_id))
                        if pred_type == TAB:
                            gs, gf = (int(x) for x in gt_val.split(','))
                            ps, pf = (int(x) for x in pred_val.split(','))
                            if (
                                1 <= ps <= len(tuning)
                                and 1 <= gs <= len(tuning)
                                and tuning[ps - 1] + pf == tuning[gs - 1] + gf
                            ):
                                pitch_totals[src][0] += 1

    model.train()
    if not loss_totals:
        return float('nan'), float('nan'), float('nan'), {}

    per_source = {}
    sum_loss, sum_loss_n = 0.0, 0
    sum_tab_c, sum_tab_n = 0, 0
    sum_pitch_c, sum_pitch_n = 0, 0
    for src in loss_totals:
        l_sum, l_n = loss_totals[src]
        t_c, t_n = tab_totals[src]
        p_c, p_n = pitch_totals[src]
        per_source[src] = {
            'loss': l_sum / l_n if l_n else 0.0,
            'tab_acc': t_c / t_n if t_n else 0.0,
            'pitch_acc': p_c / p_n if p_n else 0.0,
            'n_tabs': t_n,
        }
        sum_loss += l_sum
        sum_loss_n += l_n
        sum_tab_c += t_c
        sum_tab_n += t_n
        sum_pitch_c += p_c
        sum_pitch_n += p_n

    overall_loss = sum_loss / sum_loss_n if sum_loss_n else 0.0
    overall_tab = sum_tab_c / sum_tab_n if sum_tab_n else 0.0
    overall_pitch = sum_pitch_c / sum_pitch_n if sum_pitch_n else 0.0
    return overall_loss, overall_tab, overall_pitch, per_source


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
    parser.add_argument(
        '--num-workers', type=int, default=2, help='DataLoader workers; 0 = main process only (slower but bulletproof)'
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = args.device or auto_device()
    print(f'Device: {device}')
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # DataLoader workers exchange tensor data with the main process via OS resources.
    # Default 'file_descriptor' uses /dev/shm + fd passing — on RunPod / Docker this
    # can hit ENOBUFS after long runs ('No buffer space available'). 'file_system'
    # uses tmpfile-based sharing instead, which is slower per transfer but stable.
    mp.set_sharing_strategy('file_system')

    print('Building datasets...')
    train_ds, val_ds, _test_ds, _stats = build_datasets(datasets=args.datasets)
    print(f'  train sequences: {len(train_ds)}, val sequences: {len(val_ds)}')

    pin_memory = device == 'cuda'
    # persistent_workers keeps DataLoader workers alive across iterations to
    # avoid the resource churn from forking new workers — fixes long-run
    # crashes ('No buffer space available', SIGABRT) seen on RunPod.
    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
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
    start_time = time.time()
    time_start_step = step

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

            time_elapsed = time.time() - start_time
            steps_this_run = max(1, step - time_start_step)
            avg_elapsed_per_step = time_elapsed / steps_this_run
            eta = format_eta(avg_elapsed_per_step * (args.max_steps - step))
            print(f'[step {step}/{args.max_steps}] loss={avg_loss:.4f} ({avg_elapsed_per_step:.2f}s/step, {eta})')

        if step > 0 and step % args.eval_steps == 0:
            overall_loss, tab_acc, pitch_acc, per_src = run_eval(
                model, val_loader, device, max_batches=args.eval_batches
            )
            print(
                f'[eval @ step {step}] val_loss={overall_loss:.4f}  '
                f'tab_acc={tab_acc:.3f}  pitch_acc={pitch_acc:.3f}  (teacher-forced)'
            )
            for src, m in sorted(per_src.items()):
                print(
                    f'    {src:<12} loss={m["loss"]:.3f}  '
                    f'tab={m["tab_acc"]:.3f}  pitch={m["pitch_acc"]:.3f}  n_tabs={m["n_tabs"]}'
                )

        if step > 0 and step % args.save_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f'step_{step:07d}.pth')
            save_checkpoint(ckpt_path, step, model, optimizer, args)
            print(f'[saved] {ckpt_path}')

    final_path = os.path.join(args.output_dir, f'step_{step:07d}_final.pth')
    save_checkpoint(final_path, step, model, optimizer, args)
    print(f'\nTraining complete. Final checkpoint: {final_path}')

    final_loss, final_tab, final_pitch, final_per_src = run_eval(model, val_loader, device)
    print(f'Final val: loss={final_loss:.4f}  tab_acc={final_tab:.3f}  pitch_acc={final_pitch:.3f}  (teacher-forced)')
    for src, m in sorted(final_per_src.items()):
        print(
            f'  {src:<12} loss={m["loss"]:.4f}  tab={m["tab_acc"]:.3f}  pitch={m["pitch_acc"]:.3f}  n_tabs={m["n_tabs"]}'
        )


if __name__ == '__main__':
    main()

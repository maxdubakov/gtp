"""Train T5 model with self-adaptive Adafactor.

Mirrors the Stage 1 training pattern: argparse, infinite iterator over the loader,
periodic eval/save, ETA logging. Differences:
  - HF T5ForConditionalGeneration (returns loss when given labels)
  - Adafactor in self-adaptive mode (lr=None, no manual schedule)
  - Per-source val loss tracking via the source tag in TabDataset batches

Usage:
  python scripts/stage2/train.py --device cuda --num-workers 4
  python scripts/stage2/train.py --datasets guitarset --batch-size 2 --max-steps 50  # smoke test
  python scripts/stage2/train.py --resume runs/stage2_baseline/checkpoints/step_0010000.pth
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Adafactor

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage2.config import (
    ConditioningConfig,
    DataConfig,
    ModelConfig,
    RebalancingConfig,
    RunConfig,
    TrainConfig,
    get_device_info,
    get_git_sha,
    get_timestamp,
)
from gtp.stage2.data import build_datasets, compute_sampling_weights
from gtp.stage2.model import build_model
from gtp.stage2.paths import AUG_DATA_DIR
from gtp.stage2.tokenizer import (
    MAX_TIME_SHIFT,
    NOTE_ON,
    TAB,
    TIME_SHIFT_BINS,
    TUNING_END,
    TUNING_START,
    Vocabulary,
    parse_token_str,
)

DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, 'runs', 'stage2_001')

# Rebalancing recipe motivated by the baseline error analysis (see README §Stage 2):
#   * GuitarSet val_loss was ~10× DadaGP — biggest target → 8× upsample
#   * Leduc val_loss was ~2× DadaGP — moderate target → 5× upsample
#   * DadaGP and GuitarToday are already well-served at 1×
#   * Within DadaGP, classical/jazz/blues had 73-78% pp accuracy (worst
#     coarse buckets) — boost by 1.5×. Other genres stay at 1×.
DEFAULT_REBALANCE_SOURCE = {
    'dadagp': 1.0,
    'guitarset': 8.0,
    'leduc': 5.0,
    'guitartoday': 1.0,
}
DEFAULT_REBALANCE_GENRE = {
    'classical': 1.5,
    'jazz': 1.5,
    'blues': 1.5,
}


def auto_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def make_t5_inputs(enc_ids, dec_ids, vocab, device):
    """enc_ids, dec_ids: (B, T) padded with vocab.pad_id. Returns (input_ids, attention_mask, labels)."""
    input_ids = enc_ids.to(device)
    attention_mask = (input_ids != vocab.pad_id).long()
    labels = dec_ids.to(device).clone()
    labels[labels == vocab.pad_id] = -100
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


def parse_tuning_from_enc(enc_ids, vocab):
    """Walk encoder IDs, extract the tuning block as a list of pitches. None if missing."""
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


def run_eval(model, val_loader, vocab, device, max_batches=None):
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
            enc, dec, sources, _piece_ids = batch
            input_ids, attention_mask, labels = make_t5_inputs(enc, dec, vocab, device)
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

                tuning = parse_tuning_from_enc(inputs_cpu[b], vocab)

                for t_idx, gt_id in enumerate(labels_cpu[b]):
                    if gt_id < 0:  # PAD-as-loss-mask (-100), skip
                        continue
                    gt_type, gt_val = parse_token_str(vocab.decode(gt_id))
                    if gt_type != TAB:
                        continue

                    pred_id = preds_cpu[b][t_idx]
                    tab_totals[src][1] += 1
                    pitch_totals[src][1] += 1

                    if pred_id == gt_id:
                        tab_totals[src][0] += 1
                        pitch_totals[src][0] += 1  # exact token match implies same pitch
                    elif tuning:
                        pred_type, pred_val = parse_token_str(vocab.decode(pred_id))
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


def save_checkpoint(path, step, model, optimizer, vocab, args):
    torch.save(
        {
            'iteration': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': vars(args),
            'tokenizer_meta': {
                'vocab_size': len(vocab),
                'time_shift_max': MAX_TIME_SHIFT,
                'time_shift_step': TIME_SHIFT_BINS[0],
                'pad_id': vocab.pad_id,
                'eos_id': vocab.eos_id,
                'include_genre': vocab.include_genre,
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
    parser.add_argument('--save-steps', type=int, default=1000)
    parser.add_argument('--eval-batches', type=int, default=None, help='Cap val batches per eval (default: all)')
    parser.add_argument('--device', default=None, help='cpu / mps / cuda (default: auto)')
    parser.add_argument(
        '--num-workers', type=int, default=2, help='DataLoader workers; 0 = main process only (slower but bulletproof)'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--experiment-label', default='', help='Free-form label saved to config.json for run comparison'
    )
    parser.add_argument('--notes', default='', help='Free-form notes saved to config.json')
    parser.add_argument(
        '--genre-conditioning',
        action='store_true',
        help='Add GENRE<X> token to the encoder prefix and grow vocab '
        'from 553 to 567 tokens. Off by default (preserves baseline). '
        'Old checkpoints cannot be resumed under this flag — embedding '
        'dims differ.',
    )
    parser.add_argument(
        '--genre-dropout',
        type=float,
        default=0.15,
        help='Probability of replacing GENRE<X> with GENRE<unknown> during '
        'training (classifier-free guidance). Only active when '
        '--genre-conditioning is set.',
    )
    parser.add_argument(
        '--rebalance',
        action='store_true',
        help='Use WeightedRandomSampler with default per-source / per-genre '
        'upsampling (see DEFAULT_REBALANCE_* constants). Targets the '
        'imbalance identified in the baseline error analysis: GuitarSet '
        '8x, Leduc 5x, plus 1.5x for under-performing DadaGP genres '
        '(classical/jazz/blues/country).',
    )
    args = parser.parse_args()

    device = args.device or auto_device()
    device_info = get_device_info(device)
    if device_info.type == 'cuda' and device_info.cuda_name:
        info(f'Device: {device} ({device_info.cuda_name}, {device_info.cuda_memory_gib} GiB)')
    else:
        info(f'Device: {device}')
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct vocab from CLI flags. The same vocab instance is threaded through
    # build_datasets / build_model / run_eval / save_checkpoint — there is no
    # module-level VOCAB global.
    vocab = Vocabulary(include_genre=args.genre_conditioning)
    if args.genre_conditioning:
        info(f'Genre conditioning enabled. Vocab size: {len(vocab)}. Genre dropout: {args.genre_dropout}')

    # DataLoader workers exchange tensor data with the main process via OS resources.
    # Default 'file_descriptor' uses /dev/shm + fd passing — on RunPod / Docker this
    # can hit ENOBUFS after long runs ('No buffer space available'). 'file_system'
    # uses tmpfile-based sharing instead, which is slower per transfer but stable.
    mp.set_sharing_strategy('file_system')

    info('Building datasets...')
    train_ds, val_ds, _test_ds, _stats = build_datasets(
        vocab,
        datasets=args.datasets,
        genre_dropout=args.genre_dropout if args.genre_conditioning else 0.0,
    )
    info(f'  train sequences: {len(train_ds)}, val sequences: {len(val_ds)}')

    pin_memory = device == 'cuda'
    # persistent_workers keeps DataLoader workers alive across iterations to
    # avoid the resource churn from forking new workers — fixes long-run
    # crashes ('No buffer space available', SIGABRT) seen on RunPod.
    # prefetch_factor lets each worker prepare 4 batches ahead so dataload
    # IO can overlap with GPU work — independent of num_workers, safe to
    # raise without re-introducing the FD-exhaustion crash.
    persistent = args.num_workers > 0
    prefetch = 4 if args.num_workers > 0 else None

    # Rebalancing: build a WeightedRandomSampler if --rebalance, else plain shuffle.
    train_sampler = None
    if args.rebalance:
        weights = compute_sampling_weights(
            train_ds._sources,
            train_ds._genres,
            DEFAULT_REBALANCE_SOURCE,
            DEFAULT_REBALANCE_GENRE,
        )
        train_sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        # Print expected mix after rebalancing for quick eyeballing
        src_counts = Counter(train_ds._sources)
        eff_src = {s: src_counts[s] * DEFAULT_REBALANCE_SOURCE.get(s, 1.0) for s in src_counts}
        eff_total = sum(eff_src.values())
        info('Rebalancing enabled. Effective per-source mix:')
        for s, w in sorted(eff_src.items(), key=lambda kv: -kv[1]):
            info(f'  {s:<14s} {100 * w / eff_total:>5.1f}% (raw {100 * src_counts[s] / len(train_ds):>5.1f}%)')

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    info(f'Building model (vocab={len(vocab)})...')
    model = build_model(vocab).to(device)
    model.train()
    info(f'  parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Snapshot run config to <output_dir>/config.json. metrics.jsonl is appended
    # to per eval cycle below. compare_runs.py reads both for cross-run comparison.
    run_config = RunConfig(
        run_id=Path(args.output_dir).name,
        experiment_label=args.experiment_label,
        timestamp=get_timestamp(),
        git_sha=get_git_sha(),
        notes=args.notes,
        model=ModelConfig(
            params=sum(p.numel() for p in model.parameters()),
            d_model=getattr(model.config, 'd_model', 0),
            d_ff=getattr(model.config, 'd_ff', 0),
            n_layers=getattr(model.config, 'num_layers', 0),
            n_heads=getattr(model.config, 'num_heads', 0),
            vocab_size=len(vocab),
        ),
        train=TrainConfig(
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            eval_batches=args.eval_batches,
            num_workers=args.num_workers,
            seed=args.seed,
            resumed_from=args.resume,
        ),
        data=DataConfig(
            dataset_dir=str(AUG_DATA_DIR),
            sources=args.datasets or [],
            train_subseqs=len(train_ds),
            val_subseqs=len(val_ds),
        ),
        conditioning=ConditioningConfig(
            genre=args.genre_conditioning,
            genre_dropout=args.genre_dropout if args.genre_conditioning else 0.0,
            source=False,  # SOURCE conditioning intentionally off (leakage path)
        ),
        rebalancing=RebalancingConfig(
            enabled=args.rebalance,
            source_weights=DEFAULT_REBALANCE_SOURCE if args.rebalance else {},
            genre_weights=DEFAULT_REBALANCE_GENRE if args.rebalance else {},
        ),
        device=device_info,
    )
    config_path = Path(args.output_dir) / 'config.json'
    run_config.save(config_path)
    info(f'  config: {config_path}')
    metrics_path = Path(args.output_dir) / 'metrics.jsonl'

    optimizer = Adafactor(
        model.parameters(),
        lr=None,
        relative_step=True,
        scale_parameter=True,
        warmup_init=True,
    )

    step = 0
    if args.resume:
        info(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        meta = ckpt.get('tokenizer_meta', {})
        if meta.get('vocab_size') and meta['vocab_size'] != len(vocab):
            raise SystemExit(
                f'Vocab mismatch: checkpoint has {meta["vocab_size"]} tokens, current vocab is {len(vocab)}. '
                f'Cannot resume training with a different vocabulary.'
            )
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt.get('iteration', 0)
        info(f'  resumed at step {step}')

    info(f'Training (max_steps={args.max_steps}, starting_step={step})')
    recent_losses = []
    step_times = []
    losses_since_eval: list[float] = []  # reset at each eval; used for metrics.jsonl
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
        enc, dec, _sources, _piece_ids = batch
        input_ids, attention_mask, labels = make_t5_inputs(enc, dec, vocab, device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        loss_val = loss.item()
        recent_losses.append(loss_val)
        losses_since_eval.append(loss_val)
        step_times.append(time.time() - t0)

        log_interval = max(1, min(50, args.max_steps // 20))
        if step % log_interval == 0:
            avg_loss = float(np.mean(recent_losses[-log_interval:]))
            time_elapsed = time.time() - start_time
            steps_this_run = max(1, step - time_start_step)
            avg_elapsed_per_step = time_elapsed / steps_this_run
            eta = format_eta(avg_elapsed_per_step * (args.max_steps - step))
            info(f'[step {step}/{args.max_steps}] loss={avg_loss:.4f} ({avg_elapsed_per_step:.2f}s/step, {eta})')

        if step > 0 and step % args.eval_steps == 0:
            overall_loss, tab_acc, pitch_acc, per_src = run_eval(
                model, val_loader, vocab, device, max_batches=args.eval_batches
            )
            info(
                f'[eval @ step {step}] val_loss={overall_loss:.4f}  '
                f'tab_acc={tab_acc:.3f}  pitch_acc={pitch_acc:.3f}  (teacher-forced)'
            )
            for src, m in sorted(per_src.items()):
                info(
                    f'    {src:<12} loss={m["loss"]:.3f}  '
                    f'tab={m["tab_acc"]:.3f}  pitch={m["pitch_acc"]:.3f}  n_tabs={m["n_tabs"]}'
                )
            # Persist structured eval to metrics.jsonl for compare_runs.py.
            # train_loss is averaged over the same step window the eval covers
            # (since previous eval), so it's directly comparable to val_loss.
            train_loss_window = float(np.mean(losses_since_eval)) if losses_since_eval else None
            metrics_record = {
                'step': step,
                'train_loss_since_last_eval': train_loss_window,
                'n_train_steps_in_window': len(losses_since_eval),
                'val_loss': overall_loss,
                'tab_acc_tf': tab_acc,
                'pitch_acc_tf': pitch_acc,
                'per_source': per_src,
            }
            with metrics_path.open('a') as f:
                f.write(json.dumps(metrics_record) + '\n')
            losses_since_eval.clear()

        if step > 0 and step % args.save_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'step_{step:07d}.pth')
            save_checkpoint(ckpt_path, step, model, optimizer, vocab, args)
            info(f'[saved] {ckpt_path}')

    final_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, f'step_{step:07d}_final.pth')
    save_checkpoint(final_path, step, model, optimizer, vocab, args)
    info(f'\nTraining complete. Final checkpoint: {final_path}')

    final_loss, final_tab, final_pitch, final_per_src = run_eval(model, val_loader, vocab, device)
    info(f'Final val: loss={final_loss:.4f}  tab_acc={final_tab:.3f}  pitch_acc={final_pitch:.3f}  (teacher-forced)')
    for src, m in sorted(final_per_src.items()):
        info(
            f'  {src:<12} loss={m["loss"]:.4f}  tab={m["tab_acc"]:.3f}  pitch={m["pitch_acc"]:.3f}  n_tabs={m["n_tabs"]}'
        )

    # Final eval JSON. Note: this is teacher-forced; for autoregressive metrics
    # (tab_strict / tab_equivalent / drift buckets) run dump_eval_predictions.py
    # + analyze_errors.py separately. Those numbers go into a richer summary.json.
    final_eval = {
        'step': step,
        'val_loss': final_loss,
        'tab_acc_tf': final_tab,
        'pitch_acc_tf': final_pitch,
        'per_source': final_per_src,
    }
    final_eval_path = Path(args.output_dir) / 'final_eval.json'
    final_eval_path.write_text(json.dumps(final_eval, indent=2))
    info(f'\nFinal eval written: {final_eval_path}')


if __name__ == '__main__':
    main()

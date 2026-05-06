"""Fine-tune the Kong CRNN note model on guitar data (GAPS + GuitarSet).

Following Riley et al.'s recipe: lr=1e-5, batch=4, 10s segments, lr *0.9
every 10K steps, ~100K steps total.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from gtp import REPO_ROOT
from gtp.stage1.data import build_dataset
from gtp.log import set_verbose, trace
from gtp.stage1.model.kong import Regress_onset_offset_frame_velocity_CRNN
from gtp.stage1.model.losses import regress_onset_offset_frame_velocity_bce
from gtp.stage1.model.utils import move_data_to_device

DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, 'models', 'pretrained', 'CRNN_note_F1=0.9677_pedal_F1=0.9186.pth')
GAPS_DIR = os.path.join(REPO_ROOT, 'data', 'gaps_hf')
GUITARSET_DIR = os.path.join(REPO_ROOT, 'data', 'guitarset')

TARGET_KEYS = [
    'reg_onset_roll',
    'reg_offset_roll',
    'frame_roll',
    'velocity_roll',
    'onset_roll',
    'mask_roll',
]


def collate_fn(batch):
    """Convert list of numpy-array dicts to a batched tensor dict."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        arrays = [item[key] for item in batch]
        result[key] = torch.from_numpy(np.stack(arrays, axis=0))
    return result


def auto_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_note_model(checkpoint_path):
    """Load the note model from either a pretrained or fine-tuned checkpoint.

    Pretrained checkpoints have nested format: checkpoint['model']['note_model'].
    Fine-tuned checkpoints have flat format: checkpoint['model'] is a state dict.

    Returns (model, checkpoint) where checkpoint is the full dict (may be used
    by the caller to restore optimizer state and step count).
    """
    model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=100, classes_num=88)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = checkpoint['model']
    if 'note_model' in state:
        model.load_state_dict(state['note_model'])
    else:
        model.load_state_dict(state)
    return model, checkpoint


def save_checkpoint(path, step, model, optimizer, args):
    torch.save(
        {
            'iteration': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': vars(args),
        },
        path,
    )


def format_eta(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f'~{h}h{m:02d}m left'
    return f'~{m}m left'


def run_eval(model, val_loader, device, max_batches=20):
    """Evaluate on up to max_batches validation batches and return mean loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            for key in batch:
                batch[key] = move_data_to_device(batch[key], device)
            output_dict = model(batch['waveform'])
            loss = regress_onset_offset_frame_velocity_bce(model, output_dict, batch)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float('nan')


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Kong CRNN on guitar data')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Pretrained checkpoint path (nested format)')
    parser.add_argument(
        '--resume',
        default=None,
        help='Resume from a fine-tuned checkpoint, restoring model weights, optimizer state, and step count',
    )
    parser.add_argument('--output-dir', default='runs/finetune_001', help='Directory for checkpoints and logs')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--lr-decay-steps', type=int, default=10000, help='Multiply lr by 0.9 every N steps')
    parser.add_argument('--eval-steps', type=int, default=5000, help='Evaluate on validation set every N steps')
    parser.add_argument('--save-steps', type=int, default=10000, help='Save checkpoint every N steps')
    parser.add_argument('--device', default=None, help='cpu / mps / cuda (default: auto)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        set_verbose(True)

    device = args.device or auto_device()
    print(f'Device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Model ---
    load_path = args.resume if args.resume else args.checkpoint
    print(f'Loading note model from {load_path}')
    model, loaded_checkpoint = load_note_model(load_path)
    model.to(device)
    model.train()

    # --- Data ---
    print('Building datasets...')
    train_dataset = build_dataset(GAPS_DIR, GUITARSET_DIR, split='train')
    val_dataset = build_dataset(GAPS_DIR, GUITARSET_DIR, split='validation')
    print(f'Train segments: {len(train_dataset)}, Val segments: {len(val_dataset)}')

    pin_memory = device == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # --- Optimizer ---
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=True,
    )

    # --- Restore state when resuming ---
    step = 0
    if args.resume:
        step = loaded_checkpoint.get('iteration', 0)
        if 'optimizer' in loaded_checkpoint:
            optimizer.load_state_dict(loaded_checkpoint['optimizer'])
        print(f'Resuming from step {step}')

    # --- Training loop ---
    recent_losses = []
    step_times = []
    train_iter = iter(train_loader)

    print(f'Starting training (max_steps={args.max_steps}, starting_step={step})')

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        t_start = time.time()

        trace('waveform', shape=tuple(batch['waveform'].shape))
        for k in TARGET_KEYS:
            if k in batch:
                trace(k, shape=tuple(batch[k].shape))

        for key in batch:
            batch[key] = move_data_to_device(batch[key], device)

        output_dict = model(batch['waveform'])

        for k, v in output_dict.items():
            trace(k, shape=tuple(v.shape))

        loss = regress_onset_offset_frame_velocity_bce(model, output_dict, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

        step_time = time.time() - t_start
        recent_losses.append(loss.item())
        step_times.append(step_time)

        # LR decay
        if step % args.lr_decay_steps == 0:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.9
            current_lr = optimizer.param_groups[0]['lr']
            print(f'[lr decay @ step {step}] lr={current_lr:.2e}')

        # Console progress every 10 steps (or every step if max_steps is small)
        log_interval = max(1, min(10, args.max_steps // 10))
        if step % log_interval == 0:
            avg_loss = np.mean(recent_losses[-log_interval:])
            avg_step_time = np.mean(step_times[-log_interval:])
            current_lr = optimizer.param_groups[0]['lr']
            steps_left = args.max_steps - step
            eta = format_eta(avg_step_time * steps_left)
            print(
                f'[step {step}/{args.max_steps}] '
                f'loss={avg_loss:.4f} '
                f'lr={current_lr:.1e} '
                f'({avg_step_time:.2f}s/step, {eta})'
            )

        # Validation eval
        if step > 0 and step % args.eval_steps == 0:
            train_loss = float(np.mean(recent_losses[-min(len(recent_losses), 50) :]))
            val_loss = run_eval(model, val_loader, device)
            print(f'[eval @ step {step}] val_loss={val_loss:.4f} train_loss={train_loss:.4f}')

        # Save checkpoint
        if step > 0 and step % args.save_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f'step_{step:07d}.pth')
            save_checkpoint(ckpt_path, step, model, optimizer, args)
            print(f'[saved] {ckpt_path}')

    # Final save
    final_path = os.path.join(args.output_dir, f'step_{step:07d}_final.pth')
    save_checkpoint(final_path, step, model, optimizer, args)
    print(f'Training complete. Final checkpoint: {final_path}')


if __name__ == '__main__':
    main()

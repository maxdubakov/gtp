---
task: 004
title: Fine-tuning training loop
status: pending
---

## Context

Tasks 1-3 established the model, inference, baseline eval (52% F1), and data pipeline (57K training segments). Now we write the training loop to fine-tune the pretrained piano model on guitar data.

Riley et al.'s recipe: lr=1e-5, batch size 4, 10s segments with 1s hop, lr decay ×0.9 every 10K steps, ~100K steps (~10 epochs), converging within 30K steps. They selected the checkpoint where train-val gap was minimized.

## Objective

Create `scripts/train.py` that fine-tunes the Kong CRNN on GAPS + GuitarSet, saving checkpoints and logging training progress.

## Scope

### `scripts/train.py`

A self-contained training script. Arguments:
- `--checkpoint`: path to pretrained checkpoint (default: our piano checkpoint)
- `--output-dir`: where to save checkpoints and logs (default: `runs/finetune_001`)
- `--lr`: learning rate (default: 1e-5)
- `--batch-size`: (default: 4)
- `--max-steps`: (default: 100000)
- `--lr-decay-steps`: reduce lr by 0.9 every N steps (default: 10000)
- `--eval-steps`: evaluate on validation set every N steps (default: 5000)
- `--save-steps`: save checkpoint every N steps (default: 10000)
- `--device`: cpu/mps/cuda (default: auto)
- `--num-workers`: dataloader workers (default: 4)
- `-v`: verbose mode

Training loop structure (following Kong's pattern):
1. Load pretrained note model weights. We only fine-tune the **note model** (not pedal — guitar has no pedals).
2. Build train/val datasets using `build_dataset` from `gtp.data`
3. Use PyTorch DataLoader with shuffle
4. Adam optimizer with lr, betas=(0.9, 0.999), amsgrad=True
5. Loss: `regress_onset_offset_frame_velocity_bce` from `gtp.model.losses`
6. Training loop:
   - Forward pass: `model.note_model(waveform)` — only the note model, not the full Note_pedal wrapper
   - Compute loss against target rolls
   - Backward + optimizer step
   - LR decay every `lr-decay-steps`
   - Eval every `eval-steps` (run a small validation batch, report loss)
   - Save checkpoint every `save-steps`
   - Print: step, loss, lr, time per step

### Model loading for fine-tuning

The pretrained checkpoint has `model.note_model` and `model.pedal_model`. For fine-tuning:
- Extract only the note model: `Regress_onset_offset_frame_velocity_CRNN`
- Load its weights from the checkpoint
- Train it directly (not wrapped in Note_pedal)

This is important because:
1. We don't need pedal predictions for guitar
2. The loss function `regress_onset_offset_frame_velocity_bce` expects the note model's output dict directly
3. Simpler code — no need to manage two sub-models

### Checkpoint format

Save as:
```python
{
    'iteration': step,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': {args dict},
}
```

### Console output

Each step (or every 10 steps to avoid spam):
```
[step 100/100000] loss=0.0432 lr=1.0e-05 (0.8s/step, ~22h left)
```

Each eval:
```
[eval @ step 5000] val_loss=0.0398 train_loss=0.0412
```

### Compatibility

Must work on:
- **Mac MPS** for quick testing (run a few steps to verify nothing crashes)
- **CUDA (RTX 4080)** for real training

The script should detect available device automatically but allow override.

## Code style
Write clean, readable, self-explanatory code. Only comment non-obvious decisions. No redundant comments. Include verbose tracing via `gtp.log.trace` for data shapes flowing through the training loop.

## Non-goals
- Data augmentation (can be added later if results need improvement)
- Distributed/multi-GPU training
- Fancy logging (tensorboard, wandb) — just console + saved checkpoints
- Evaluation with mir_eval during training (too slow — we'll use eval_guitarset.py separately)

## Constraints
- Use the same loss function from `gtp.model.losses` — `regress_onset_offset_frame_velocity_bce`
- The loss function signature is `loss_func(model, output_dict, target_dict)` — it accesses `model` but only for potential regularization (not used in practice). Pass the model anyway for API compatibility.
- Target dict keys from our data pipeline match what the loss function expects (`reg_onset_roll`, `reg_offset_roll`, `frame_roll`, `velocity_roll`, `onset_roll`, `mask_roll`)
- DataLoader must convert numpy arrays to tensors — use a collate function

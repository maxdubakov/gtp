"""Fine-tune the Stage 1 note model on the notes dataset (build_notes_dataset.py).

Starts from a note-model checkpoint and trains on the manifest's train split, with segments of direct-input and
microphone recordings rendered through random recording chains at the probabilities in `augment`. Validation loss is
tracked per dataset and chain on a fixed set of segments; checkpoints are scored on note F1 afterwards
(eval_notes.py), since the loss is not what the pipeline is judged on.

Usage:
  uv run scripts/stage1/finetune.py
  uv run scripts/stage1/finetune.py max_steps=50 eval_steps=25 save_steps=50 wandb.mode=disabled
"""

import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.data import ManifestDataset
from gtp.stage1.model.losses import regress_onset_offset_frame_velocity_bce
from gtp.stage1.model.utils import move_data_to_device
from gtp.stage1.training import collate_fn, load_note_model
from gtp.track import start_run


def val_subsets(dataset, per_group, seed):
    """{dataset/chain: Subset} of `per_group` segments each, fixed by `seed`."""
    groups = {}
    for i, (path, _, _) in enumerate(dataset.segments):
        groups.setdefault(dataset.group_of[path], []).append(i)
    rng = np.random.default_rng(seed)
    return {
        g: Subset(dataset, sorted(rng.choice(idx, min(per_group, len(idx)), replace=False)))
        for g, idx in groups.items()
    }


@torch.no_grad()
def mean_loss(model, loader, device):
    losses = []
    for batch in loader:
        batch = {k: move_data_to_device(v, device) for k, v in batch.items()}
        losses.append(regress_onset_offset_frame_velocity_bce(model, model(batch['waveform']), batch).item())
    return float(np.mean(losses))


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='finetune')
def main(cfg):
    torch.manual_seed(cfg.seed)
    device = auto_device()
    run = start_run(cfg, job_type='stage1_notes', device=device)
    info(f'{run.name}\n{OmegaConf.to_yaml(cfg)}')
    root = REPO_ROOT / cfg.data_dir
    train = ManifestDataset.from_manifest(root, 'train', OmegaConf.to_container(cfg.augment))
    val = val_subsets(ManifestDataset.from_manifest(root, 'val'), cfg.val_segments_per_group, cfg.seed)
    info(f'{len(train)} training segments; validation groups { ({g: len(s) for g, s in val.items()}) }')

    def loader(dataset, shuffle):
        return DataLoader(
            dataset, batch_size=cfg.batch, shuffle=shuffle, num_workers=cfg.workers, collate_fn=collate_fn,
            pin_memory=device == 'cuda', drop_last=shuffle, persistent_workers=cfg.workers > 0,
        )  # fmt: skip

    val_loaders = {g: loader(s, False) for g, s in val.items()}
    model, _ = load_note_model(REPO_ROOT / cfg.init)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, amsgrad=True)
    (run.dir / 'checkpoints').mkdir(exist_ok=True)

    def save(step, final=False):
        path = run.dir / 'checkpoints' / f'step_{step:07d}{"_final" if final else ""}.pth'
        torch.save({'iteration': step, 'model': model.state_dict(), 'config': OmegaConf.to_container(cfg)}, path)
        info(f'saved {path}')

    step, losses, t0 = 0, [], time.time()
    train_loader = loader(train, True)
    batches = iter(train_loader)
    while step < cfg.max_steps:
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            batch = next(batches)
        batch = {k: move_data_to_device(v, device) for k, v in batch.items()}
        loss = regress_onset_offset_frame_velocity_bce(model, model(batch['waveform']), batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f'loss is {loss.item()} at step {step}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        losses.append(loss.item())
        if step % cfg.lr_decay_steps == 0:
            for group in optimizer.param_groups:
                group['lr'] *= cfg.lr_decay
        if step % cfg.log_steps == 0:
            rate = (time.time() - t0) / cfg.log_steps
            left = rate * (cfg.max_steps - step) / 3600
            info(f'step {step}/{cfg.max_steps} loss {np.mean(losses):.4f} ({rate:.2f} s/step, {left:.1f} h left)')
            run.log(
                {'train/loss': float(np.mean(losses)), 'lr': optimizer.param_groups[0]['lr'], 's_per_step': rate},
                step=step,
            )
            losses, t0 = [], time.time()
        if step % cfg.eval_steps == 0:
            model.eval()
            per_group = {g: mean_loss(model, ld, device) for g, ld in val_loaders.items()}
            model.train()
            per_group['macro'] = float(np.mean(list(per_group.values())))
            info(f'step {step} val loss ' + ', '.join(f'{g} {v:.4f}' for g, v in per_group.items()))
            run.log({f'val/{g}': v for g, v in per_group.items()}, step=step)
            t0 = time.time()
        if step % cfg.save_steps == 0 and step < cfg.max_steps:
            save(step)
    save(step, final=True)
    run.finish({'steps': step})


if __name__ == '__main__':
    main()

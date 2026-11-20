"""Train a per-note classifier of playing technique on the technique index (build_index.py and its window cache).

The input is the spectrogram window around a labelled note's onset, as for the string classifier; the output is one
of `classes`. Note times are the labels' own, so this measures whether the classes can be told apart when the note
is given, apart from whether the note model finds it. Splits are the index's (by guitar or player; GOAT validation
by item). No checkpoint is selected: the last epoch is kept.

Reports, per validation dataset and chain, each class's precision, recall and F1 and their mean (macro F1); the test
split only with `report_test=true`.

Usage:
  uv run scripts/techniques/train.py
  uv run scripts/techniques/train.py -m seed=0,1,2
  uv run scripts/techniques/train.py optim.epochs=1 limit=20000 name=smoke
"""

import json
import time

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.string_data import load_cached_index
from gtp.stage1.strings import PRE_FRAMES, post_frames
from gtp.stage1.techniques import class_of
from gtp.track import start_run


def scores(labels, predicted, classes):
    """Per-class precision, recall, F1 and count, and the macro F1 over the classes present."""
    out = {}
    for k, name in enumerate(classes):
        tp = int(((predicted == k) & (labels == k)).sum())
        p, r = tp / max(int((predicted == k).sum()), 1), tp / max(int((labels == k).sum()), 1)
        out[name] = {'n': int((labels == k).sum()), 'P': p, 'R': r, 'F1': 2 * p * r / max(p + r, 1e-9)}
    present = [m['F1'] for m in out.values() if m['n']]
    return out | {'macro_F1': float(np.mean(present)), 'accuracy': float((predicted == labels).mean())}


def table(report, classes):
    rows = []
    for group, m in report.items():
        row = {'group': group, 'macro F1': m['macro_F1'], 'accuracy': m['accuracy']}
        for c in classes:
            row |= {f'{c} n': m[c]['n'], f'{c} P': m[c]['P'], f'{c} R': m[c]['R']}
        rows.append(row)
    with pd.option_context('display.width', 250, 'display.float_format', '{:.3f}'.format):
        return pd.DataFrame(rows).to_string(index=False)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='train')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = auto_device()
    classes = list(cfg.classes)
    notes, rows, x, cache_meta = load_cached_index(cfg.index_dir, cfg.feature)
    label = class_of(notes['techniques'], classes, list(cfg.other_marks))
    note = rows['note'].to_numpy()
    split, real = notes['split'].to_numpy()[note], rows['variant'].to_numpy() == 0
    trainable = (
        (label[note] >= 0)
        & ~notes['dataset'].isin(list(cfg.exclude_datasets)).to_numpy()[note]
        & ~notes['chain'].isin(list(cfg.exclude_chains)).to_numpy()[note]
    )
    if cfg.variants is not None:
        trainable &= rows['variant'].isin(list(cfg.variants)).to_numpy()
    idx = {'train': np.flatnonzero(trainable & (split == 'train'))}
    for k in ['val'] + (['test'] if cfg.report_test else []):
        idx[k] = np.flatnonzero(real & (split == k) & (label[note] >= 0))
    if cfg.limit:
        idx['train'] = np.sort(np.random.permutation(idx['train'])[: cfg.limit])
    sizes = {k: len(v) for k, v in idx.items()}
    run = start_run(cfg, job_type='techniques', device=device, extra={'cache': cache_meta, 'windows': sizes})
    info(f'{run.name}\n{OmegaConf.to_yaml(cfg)}')

    n_frames = PRE_FRAMES + post_frames(cfg.window_ms)
    t0 = time.time()
    data = {}
    for k, v in idx.items():
        xs = torch.empty((len(v), *x.shape[1:3], n_frames), dtype=torch.float16, device=device)
        for i in range(0, len(v), 8192):
            xs[i : i + 8192] = torch.from_numpy(np.ascontiguousarray(x[v[i : i + 8192], ..., :n_frames])).to(device)
        data[k] = (xs, torch.from_numpy(label[note[v]]).to(device))
    info(f'windows {sizes} on {device} in {time.time() - t0:.0f}s')

    xs, ys = data['train']
    total, total_sq = np.zeros(xs.shape[2]), np.zeros(xs.shape[2])
    for i in range(0, len(xs), 8192):
        b = xs[i : i + 8192].float()
        total += b.sum(dim=(0, 1, 3)).cpu().double().numpy()
        total_sq += (b**2).sum(dim=(0, 1, 3)).cpu().double().numpy()
    count = len(xs) * xs.shape[1] * xs.shape[3]
    mean = torch.from_numpy(total / count).float().to(device)[None, None, :, None]
    std = torch.from_numpy(np.sqrt(np.maximum(total_sq / count - (total / count) ** 2, 1e-6))).float().to(device)
    std = std[None, None, :, None]
    share = torch.bincount(ys, minlength=len(classes)).float() / len(ys)
    weight = share.clamp_min(1e-6) ** -cfg.class_weight_power
    weight = weight / (weight * share).sum()
    info(f'training share per class {dict(zip(classes, share.tolist(), strict=True))}, loss weights {weight.tolist()}')

    def batches(k, shuffle=False):
        xs, ys = data[k]
        order = torch.randperm(len(xs), device=device) if shuffle else torch.arange(len(xs), device=device)
        for i in range(0, len(xs), cfg.optim.batch):
            b = order[i : i + cfg.optim.batch]
            yield (xs[b].float() - mean) / std, ys[b]

    model = instantiate(
        cfg.model, in_channels=x.shape[1], n_freq=x.shape[2], n_frames=n_frames, n_classes=len(classes)
    ).to(device)
    every = torch.ones(1, len(classes), dtype=torch.bool, device=device)

    def logits_of(k):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xb, None, every).cpu() for xb, _ in batches(k)])

    def report(k):
        frame = notes.iloc[note[idx[k]]].reset_index(drop=True)
        predicted, labels = logits_of(k).argmax(dim=1).numpy(), label[note[idx[k]]]
        out = {'all': scores(labels, predicted, classes)}
        for keys in (['dataset'], ['dataset', 'chain']):
            for name, g in frame.groupby(keys, sort=True):
                out['/'.join(name)] = scores(labels[g.index], predicted[g.index], classes)
        return out

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    steps = cfg.optim.epochs * -(-len(idx['train']) // cfg.optim.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.optim.lr, total_steps=steps)
    history = []
    for epoch in range(cfg.optim.epochs):
        model.train()
        t0, loss_sum, n_batches = time.time(), 0.0, 0
        for xb, yb in batches('train', shuffle=True):
            loss = F.cross_entropy(model(xb, None, every), yb, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1
        val = report('val')
        per = {k: round(v['macro_F1'], 3) for k, v in val.items() if k.count('/') == 0}
        history.append({'epoch': epoch, 'train_loss': loss_sum / n_batches, 'val_macro_F1': per})
        run.log({'train_loss': loss_sum / n_batches, **{f'val_macro_F1/{k}': v for k, v in per.items()}}, step=epoch)
        info(f'epoch {epoch:2d} loss {loss_sum / n_batches:.3f} val macro F1 {per} ({time.time() - t0:.0f}s)')

    torch.save(model.state_dict(), run.dir / 'model.pt')
    np.savez(run.dir / 'classifier.npz', mean=mean.cpu().numpy(), std=std.cpu().numpy())
    metrics = {'classes': classes, 'windows': sizes, 'history': history}
    for k in idx:
        if k != 'train':
            metrics[k] = report(k)
            np.savez(run.dir / f'{k}_logits.npz', note=note[idx[k]], logits=logits_of(k).numpy())
            info(f'{k}\n{table(metrics[k], classes)}')
    (run.dir / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    run.finish({'val_macro_F1': metrics['val']['all']['macro_F1']})


if __name__ == '__main__':
    main()

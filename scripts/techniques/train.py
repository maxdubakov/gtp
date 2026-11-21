"""Train a per-note classifier of playing techniques on the technique index (build_index.py and its window cache).

The input is the spectrogram window around a labelled note's onset, in one or more views (window caches of the
index), and optionally a few numbers about the note before it on its string and about how hard its attack is; the
output is one class per head (`heads`): how the note is muted, how it starts, what its pitch does. A note can lack a label for a head (an unknown mark, a
dataset whose labels for it are missing or not trusted); it then counts for the other heads only. Note times are
the labels' own, so this measures whether the classes can be told apart when the note is given, apart from whether
the note model finds it. Splits are the index's (by guitar or player; GOAT validation by item), with the players of
`validate_on` moved from training to validation. No checkpoint is selected: the last epoch is kept. Training
batches are changed at random as `augment` says, so that the network cannot memorise its training notes.

Reports, per head, validation dataset and chain, each class's precision, recall and F1 and their mean (macro F1);
the test split only with `report_test=true`.

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
from gtp.stage1.tab_prior import index_prior
from gtp.stage1.techniques import attack_burst, grouped_scores, head_labels, pitch_comb, string_context
from gtp.track import start_run


def table(report, classes):
    rows = []
    for group, m in report.items():
        row = {'group': group, 'macro F1': m['macro_F1']}
        for c in classes:
            row |= {f'{c} n': m[c]['n'], f'{c} P': m[c]['P'], f'{c} R': m[c]['R']}
        rows.append(row)
    with pd.option_context('display.width', 250, 'display.float_format', '{:.3f}'.format):
        return pd.DataFrame(rows).to_string(index=False)


def stretches(n, length, most, device):
    """(n, length) mask of one random stretch per row, 0 to `most` cells long."""
    start = torch.randint(0, length, (n, 1), device=device)
    cells = torch.arange(length, device=device)[None]
    return (cells >= start) & (cells < start + torch.randint(0, most + 1, (n, 1), device=device))


def moved(x, by, dim):
    """`x` moved by `by` cells along `dim`, the vacated edge at zero (the mean of a normalised batch)."""
    if by:
        x = torch.roll(x, by, dims=dim)
        x.narrow(dim, 0 if by > 0 else x.shape[dim] + by, abs(by)).zero_()
    return x


def augmented(views, cfg):
    """A normalised training batch, each view (notes, channels, bins, frames), changed at random: all views moved
    alike in time and the first in frequency (one move for the whole batch), stretches of frames and bins set to
    the mean per note, and a level offset per note."""
    in_time = int(torch.randint(-cfg.shift_frames, cfg.shift_frames + 1, ()))
    in_freq = int(torch.randint(-cfg.shift_bins, cfg.shift_bins + 1, ()))
    out = []
    for i, x in enumerate(views):
        x = moved(moved(x, in_time, 3), in_freq if i == 0 else 0, 2)
        for _ in range(cfg.time_masks):
            x = x.masked_fill(stretches(len(x), x.shape[3], cfg.time_mask, x.device)[:, None, None, :], 0.0)
        for _ in range(cfg.freq_masks):
            x = x.masked_fill(stretches(len(x), x.shape[2], cfg.freq_mask, x.device)[:, None, :, None], 0.0)
        if cfg.gain:
            x = x + cfg.gain * torch.randn(len(x), 1, 1, 1, device=x.device)
        out.append(x)
    return out


def on_device(x, rows, pool_bins, device):
    """The cache's windows of `rows` on the device as float16, neighbouring frequency bins averaged in `pool_bins`."""
    out = None
    for i in range(0, len(rows), 8192):
        chunk = torch.from_numpy(np.ascontiguousarray(x[rows[i : i + 8192]])).to(device)
        if pool_bins > 1:
            chunk = F.avg_pool2d(chunk.float(), (pool_bins, 1)).half()
        if out is None:
            out = torch.empty((len(rows), *chunk.shape[1:]), dtype=torch.float16, device=device)
        out[i : i + 8192] = chunk
    return out


def bin_stats(xs):
    """Mean and spread per frequency bin over a view's windows, shaped to broadcast against a batch."""
    total, total_sq = np.zeros(xs.shape[2]), np.zeros(xs.shape[2])
    for i in range(0, len(xs), 8192):
        b = xs[i : i + 8192].float()
        total += b.sum(dim=(0, 1, 3)).cpu().double().numpy()
        total_sq += (b**2).sum(dim=(0, 1, 3)).cpu().double().numpy()
    count = len(xs) * xs.shape[1] * xs.shape[3]
    mean, var = total / count, np.maximum(total_sq / count - (total / count) ** 2, 1e-6)
    return [torch.from_numpy(v).float().to(xs.device)[None, None, :, None] for v in (mean, np.sqrt(var))]


def attack_numbers(index_dir, view, notes, rows):
    """(windows, 2) and their scale: each window's attack burst (gtp.stage1.techniques.attack_burst) and how far it
    is below a strong attack of its recording (the upper quartile of the recording's bursts in the same render),
    both divided by the spread of all bursts. A player's hammer-ons are soft against that player's own pick strokes."""
    _, view_rows, x, meta = load_cached_index(index_dir, view.feature, view.cache)
    if not view_rows.equals(rows):
        raise ValueError(f'{view.cache} lists other windows than the first view')
    pre = meta['layout']['pre_frames']
    burst = np.concatenate([attack_burst(x[i : i + 20000, 0].astype(np.float32), pre) for i in range(0, len(x), 20000)])
    of = pd.DataFrame({'rec': notes['rec'].to_numpy()[rows['note']], 'variant': rows['variant'], 'burst': burst})
    strong = of.groupby(['rec', 'variant'])['burst'].transform(lambda b: b.quantile(0.75)).to_numpy()
    scale = float(burst.std())
    return (np.stack([burst, burst - strong], axis=1) / scale).astype(np.float32), scale


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='train')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = auto_device()
    heads = {name: list(head.classes) for name, head in cfg.heads.items()}
    notes, rows, x, cache_meta = load_cached_index(cfg.index_dir, cfg.feature, cfg.cache)
    for held in cfg.validate_on:
        notes.loc[(notes['dataset'] == held.dataset) & (notes['player'] == held.player), 'split'] = 'val'
    label = np.stack(list(head_labels(notes, cfg.heads, cfg.unlabelled).values()), axis=1)
    note = rows['note'].to_numpy()
    split, real = notes['split'].to_numpy()[note], rows['variant'].to_numpy() == 0
    labelled = (label[note] >= 0).any(axis=1)
    trainable = (
        labelled
        & ~notes['dataset'].isin(list(cfg.exclude_datasets)).to_numpy()[note]
        & ~notes['chain'].isin(list(cfg.exclude_chains)).to_numpy()[note]
    )
    if cfg.variants is not None:
        trainable &= rows['variant'].isin(list(cfg.variants)).to_numpy()
    idx = {'train': np.flatnonzero(trainable & (split == 'train'))}
    for k in ['val'] + (['test'] if cfg.report_test else []):
        idx[k] = np.flatnonzero(real & (split == k) & labelled)
    if cfg.limit:
        idx['train'] = np.sort(np.random.permutation(idx['train'])[: cfg.limit])
    sizes = {k: len(v) for k, v in idx.items()}
    run = start_run(cfg, job_type='techniques', device=device, extra={'cache': cache_meta, 'windows': sizes})
    info(f'{run.name}\n{OmegaConf.to_yaml(cfg)}')

    t0 = time.time()
    caches = [(x, 1)]
    for view in cfg.more_views:
        _, view_rows, view_x, _ = load_cached_index(cfg.index_dir, view.feature, view.cache)
        if not view_rows.equals(rows):
            raise ValueError(f'{view.cache} lists other windows than {cfg.cache}')
        caches.append((view_x, view.pool_bins))
    pitch = torch.tensor(notes['pitch'].to_numpy())
    context, attack_scale = [string_context(notes)[note]] if cfg.context else [], None
    if cfg.attack:
        numbers, attack_scale = attack_numbers(cfg.index_dir, cfg.attack, notes, rows)
        context.append(numbers)
    if cfg.prior:
        gains, _ = index_prior(REPO_ROOT / cfg.prior.run, notes, heads, device)
        odds = [gains[h][:, 1:] - gains[h][:, :1] for h in cfg.prior.heads]
        context.append((np.concatenate(odds, axis=1).clip(-8, 8) / 4)[note])
    context = torch.from_numpy(np.concatenate(context, axis=1)) if context else None
    data = {}
    for k, v in idx.items():
        data[k] = (
            [on_device(view_x, v, pool_bins, device) for view_x, pool_bins in caches],
            pitch[note[v]].to(device),
            None if context is None else context[v].to(device),
            torch.from_numpy(label[note[v]]).to(device),
        )
    shapes = [tuple(xs.shape[1:]) for xs in data['train'][0]]
    info(f'windows {sizes} of {shapes} on {device} in {time.time() - t0:.0f}s')

    stats = [bin_stats(xs) for xs in data['train'][0]]
    ys = data['train'][3]
    weights = []
    for h, (name, classes) in enumerate(heads.items()):
        share = torch.bincount(ys[ys[:, h] >= 0, h], minlength=len(classes)).float()
        share = share / share.sum()
        weight = share.clamp_min(1e-6) ** -cfg.class_weight_power
        weights.append(weight / (weight * share).sum())
        info(f'{name}: training share {dict(zip(classes, [round(v, 4) for v in share.tolist()], strict=True))}')
    comb = torch.from_numpy(pitch_comb(shapes[0][1], cfg.pitch_harmonics)).to(device) if cfg.pitch_harmonics else None

    def batches(k, shuffle=False):
        views, ps, cs, ys = data[k]
        order = torch.randperm(len(ps), device=device) if shuffle else torch.arange(len(ps), device=device)
        for i in range(0, len(ps), cfg.optim.batch):
            b = order[i : i + cfg.optim.batch]
            xb = [(xs[b].float() - mean) / std for xs, (mean, std) in zip(views, stats, strict=True)]
            if shuffle:
                xb = augmented(xb, cfg.augment)
            if comb is not None:
                marked = comb[ps[b]][:, None, :, None].expand(-1, 1, -1, xb[0].shape[3])
                xb[0] = torch.cat([xb[0], marked], dim=1)
            yield xb, ps[b], None if cs is None else cs[b], ys[b]

    sizes_of = [len(c) for c in heads.values()]
    shapes[0] = (shapes[0][0] + (comb is not None), *shapes[0][1:])
    model = instantiate(
        cfg.model, views=shapes, n_classes=sum(sizes_of), context=0 if context is None else context.shape[1]
    ).to(device)

    def logits_of(k):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xb, pb, cb).cpu() for xb, pb, cb, _ in batches(k)]).split(sizes_of, dim=1)

    def report(k):
        frame = notes.iloc[note[idx[k]]].reset_index(drop=True)
        out = {}
        for h, ((name, classes), logits) in enumerate(zip(heads.items(), logits_of(k), strict=True)):
            out[name] = grouped_scores(frame, label[note[idx[k]], h], logits.argmax(dim=1).numpy(), classes)
        return out

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    steps = cfg.optim.epochs * -(-len(idx['train']) // cfg.optim.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.optim.lr, total_steps=steps)
    history = []
    for epoch in range(cfg.optim.epochs):
        model.train()
        t0, loss_sum, n_batches = time.time(), 0.0, 0
        for xb, pb, cb, yb in batches('train', shuffle=True):
            parts = model(xb, pb, cb).split(sizes_of, dim=1)
            loss = sum(
                F.cross_entropy(
                    part, yb[:, h], weight=weights[h], ignore_index=-1, label_smoothing=cfg.optim.label_smoothing
                )
                for h, part in enumerate(parts)
                if (yb[:, h] >= 0).any()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1
        val = report('val')
        per = {name: round(m['all']['macro_F1'], 3) for name, m in val.items()}
        history.append({'epoch': epoch, 'train_loss': loss_sum / n_batches, 'val_macro_F1': per})
        run.log({'train_loss': loss_sum / n_batches, **{f'val_macro_F1/{k}': v for k, v in per.items()}}, step=epoch)
        info(f'epoch {epoch:2d} loss {loss_sum / n_batches:.3f} val macro F1 {per} ({time.time() - t0:.0f}s)')

    torch.save(model.state_dict(), run.dir / 'model.pt')
    scale = {
        f'{k}{i}': v.cpu().numpy() for i, pair in enumerate(stats) for k, v in zip(('mean', 'std'), pair, strict=True)
    }
    if attack_scale is not None:
        scale['attack_scale'] = np.float32(attack_scale)
    np.savez(run.dir / 'classifier.npz', **scale)
    metrics = {'heads': heads, 'windows': sizes, 'history': history}
    for k in idx:
        if k != 'train':
            metrics[k] = report(k)
            logits = dict(zip(heads, (part.numpy() for part in logits_of(k)), strict=True))
            np.savez(run.dir / f'{k}_logits.npz', note=note[idx[k]], **logits)
            for name, classes in heads.items():
                info(f'{k}, {name}\n{table(metrics[k][name], classes)}')
    (run.dir / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    run.finish({f'val_macro_F1/{name}': m['all']['macro_F1'] for name, m in metrics['val'].items()})


if __name__ == '__main__':
    main()

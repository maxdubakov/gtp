"""Train the per-note string classifier on the multi-dataset string index (build_index.py, build_index_cache.py).

Splits come from the index (by player or guitar, never by clip). Training uses every cached variant of the
training notes (the real recording plus its augmented renders); val and test use the real recordings only.
The checkpoint is chosen on the mean of the per-dataset val NLLs, so the largest val set doesn't decide alone.
Test metrics are reported per (dataset, chain), each next to the pitch prior on the same notes.

With `holdout` (a list of guitars), those guitars are the test set and every other recording trains, whatever its
index split: a leave-guitars-out fold. Nothing is selected on the held-out guitars: the last epoch is kept, the
temperature stays 1, and the raw logits go to holdout_logits.npz for calibration across folds.

Usage:
  uv run scripts/stage1/strings/train_index.py
  uv run scripts/stage1/strings/train_index.py -m seed=0,1,2
  uv run scripts/stage1/strings/train_index.py optim.epochs=1 limit=20000 wandb.mode=disabled
  uv run scripts/stage1/strings/train_index.py "holdout=[sire-t7]" optim.epochs=4 name=logo_sire-t7
"""

import json
import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.string_data import load_cached_index
from gtp.stage1.strings import (
    FAMILIES,
    PRE_FRAMES,
    candidates,
    fit_temperature,
    post_frames,
    prior_from_counts,
    string_counts,
    summarize,
)
from gtp.track import start_run


def to_device(x, idx, n_frames, device, chunk=8192):
    """The windows at `idx`, cropped to n_frames, copied chunk by chunk so the host never holds a second copy."""
    out = torch.empty((len(idx), *x.shape[1:3], n_frames), dtype=torch.float16, device=device)
    for i in range(0, len(idx), chunk):
        out[i : i + chunk] = torch.from_numpy(np.ascontiguousarray(x[idx[i : i + chunk], ..., :n_frames])).to(device)
    return out


def grouped(frame, probs, keys):
    """summarize() per group of `keys`, on every note and on notes with more than one candidate string."""
    out = {}
    for name, g in frame.groupby(keys, sort=True):
        name = '/'.join(name) if isinstance(name, tuple) else name
        for subset, mask in (('all', np.ones(len(g), bool)), ('multi', g['n_cand'].to_numpy() > 1)):
            if mask.any():
                i = g.index.to_numpy()[mask]
                out[f'{name}/{subset}'] = summarize(probs[i], frame['label'].to_numpy()[i])
    return out


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='train_index')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = auto_device()

    notes, rows, x, cache_meta = load_cached_index(cfg.index_dir, cfg.data.feature)
    split = rows['note'].map(notes['split'])
    real = rows['variant'] == 0
    idx = {
        'train': np.flatnonzero(split == 'train'),
        'val': np.flatnonzero(real & (split == 'val')),
        'test': np.flatnonzero(real & (split == 'test')),
    }
    if cfg.holdout:
        held = rows['note'].map(notes['guitar']).isin(list(cfg.holdout))
        idx = {'train': np.flatnonzero(~held), 'test': np.flatnonzero(real & held)}
        idx['val'] = idx['test']
    if cfg.limit:
        idx['train'] = np.sort(np.random.permutation(idx['train'])[: cfg.limit])
    sizes = {k: len(v) for k, v in idx.items()}
    run = start_run(cfg, job_type='strings_index', device=device, extra={'cache': cache_meta, 'windows': sizes})
    info(f'{run.name}\n{OmegaConf.to_yaml(cfg)}')

    n_frames = PRE_FRAMES + post_frames(cfg.data.window_ms)
    note_of = {k: rows['note'].to_numpy()[v] for k, v in idx.items()}
    pitch = notes['pitch'].to_numpy()
    label = notes['string'].to_numpy() - 1
    family = np.zeros(len(notes), int)
    if cfg.model.get('use_family'):
        names = json.loads((REPO_ROOT / cfg.families).read_text())
        family = notes['guitar'].map(lambda g: FAMILIES.index(names.get(g, cfg.family_default))).to_numpy()
    t0 = time.time()
    data = {}
    for k, v in idx.items():
        n = note_of[k]
        data[k] = (
            to_device(x, v, n_frames, device),
            torch.from_numpy(pitch[n]).to(device),
            torch.from_numpy(candidates(pitch[n])).to(device),
            torch.from_numpy(family[n]).to(device),
            torch.from_numpy(label[n]).to(device),
        )
    info(f'windows {sizes} on {device} in {time.time() - t0:.0f}s')

    # Chunk sums in float32 on the device (MPS has no float64), accumulated in float64 on the host.
    xs = data['train'][0]
    total, total_sq = np.zeros(xs.shape[2]), np.zeros(xs.shape[2])
    for i in range(0, len(xs), 8192):
        b = xs[i : i + 8192].float()
        total += b.sum(dim=(0, 1, 3)).cpu().double().numpy()
        total_sq += (b**2).sum(dim=(0, 1, 3)).cpu().double().numpy()
    count = len(xs) * xs.shape[1] * xs.shape[3]
    mean = torch.from_numpy(total / count).float().to(device)
    std = torch.from_numpy(np.sqrt(np.maximum(total_sq / count - (total / count) ** 2, 1e-6))).float().to(device)
    mean, std = mean[None, None, :, None], std[None, None, :, None]

    def batches(k, shuffle=False):
        xs, ps, cs, fs, ys = data[k]
        order = torch.randperm(len(xs), device=device) if shuffle else torch.arange(len(xs), device=device)
        for i in range(0, len(xs), cfg.optim.batch):
            b = order[i : i + cfg.optim.batch]
            yield (xs[b].float() - mean) / std, ps[b], cs[b], fs[b], ys[b]

    def logits_of(k):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(*b[:4]).cpu() for b in batches(k)])

    val_frame = notes.iloc[note_of['val']].reset_index(drop=True)

    def val_metrics():
        probs = F.softmax(logits_of('val'), dim=1).numpy()
        per = {d: summarize(probs[g.index], g['string'].to_numpy() - 1) for d, g in val_frame.groupby('dataset')}
        pooled = summarize(probs, val_frame['string'].to_numpy() - 1)
        macro = float(np.mean([m['nll_bits'] for m in per.values()]))
        return macro, pooled, per

    model = instantiate(cfg.model, in_channels=x.shape[1], n_freq=x.shape[2], n_frames=n_frames).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    info(f'input {tuple(x.shape[1:3])}x{n_frames}, {n_params / 1e6:.2f}M params')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    steps_per_epoch = -(-len(idx['train']) // cfg.optim.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.optim.lr, total_steps=cfg.optim.epochs * steps_per_epoch
    )

    best, best_state, best_epoch, history = float('inf'), None, -1, []
    for epoch in range(cfg.optim.epochs):
        model.train()
        t0, loss_sum, n_batches = time.time(), 0.0, 0
        for *inputs, yb in batches('train', shuffle=True):
            loss = F.cross_entropy(model(*inputs), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1
        macro, pooled, per = val_metrics()
        record = {
            'epoch': epoch,
            'train_loss': loss_sum / n_batches,
            'val_macro_nll_bits': macro,
            **{f'val_{k}': v for k, v in pooled.items()},
            **{f'val_{d}_{k}': v for d, m in per.items() for k, v in m.items() if k != 'n'},
        }
        history.append(record)
        run.log({**record, 'lr': sched.get_last_lr()[0], 'epoch_s': time.time() - t0}, step=epoch)
        if cfg.holdout or macro < best:
            best, best_epoch = macro, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        accs = ' '.join(f'{d} {m["acc"]:.3f}' for d, m in per.items())
        info(
            f'epoch {epoch:2d} loss {loss_sum / n_batches:.3f} val macro nll {macro:.3f} bits, acc {accs} '
            f'({time.time() - t0:.0f}s)'
        )

    model.load_state_dict(best_state)
    torch.save(best_state, run.dir / 'model.pt')
    temperature = 1.0 if cfg.holdout else fit_temperature(logits_of('val'), data['val'][4].cpu())
    info(f'best epoch {best_epoch}, temperature {temperature:.3f}')

    train_notes = np.unique(note_of['train'])
    counts = string_counts(pitch[train_notes], label[train_notes])
    np.savez(run.dir / 'classifier.npz', mean=mean.cpu().numpy(), std=std.cpu().numpy(), prior_counts=counts)

    test = notes.iloc[note_of['test']].reset_index(drop=True)
    test['label'] = test['string'] - 1
    test['n_cand'] = candidates(test['pitch'].to_numpy()).sum(axis=1)
    logits = logits_of('test')
    if cfg.holdout:
        np.savez(run.dir / 'holdout_logits.npz', note=note_of['test'], logits=logits.numpy())
    probs = F.softmax(logits, dim=1).numpy()
    probs_cal = F.softmax(logits / temperature, dim=1).numpy()
    prior = prior_from_counts(counts, test['pitch'].to_numpy())
    report = {}
    for keys in (['dataset'], ['dataset', 'chain']):
        m, c, p = (grouped(test, q, keys) for q in (probs, probs_cal, prior))
        for k in m:
            report[k] = {
                'model': m[k],
                'calibrated': c[k],
                'prior': p[k],
                'bits_gained': p[k]['nll_bits'] - m[k]['nll_bits'],
            }
    for k, r in report.items():
        info(
            f'test {k:32s} n={r["model"]["n"]:6d} acc {r["model"]["acc"]:.3f} (prior {r["prior"]["acc"]:.3f}) '
            f'nll {r["model"]["nll_bits"]:.3f} bits (gained {r["bits_gained"]:+.3f}) ece {r["calibrated"]["ece"]:.3f} cal'
        )

    (run.dir / 'metrics.json').write_text(
        json.dumps(
            {
                'n_params': n_params,
                'windows': sizes,
                'best_epoch': best_epoch,
                'best_val_macro_nll_bits': best,
                'temperature': temperature,
                'test': report,
                'history': history,
            },
            indent=1,
        )
    )
    np.savez(
        run.dir / 'test_probs.npz',
        rec=test['rec'].to_numpy(),
        dataset=test['dataset'].to_numpy(),
        chain=test['chain'].to_numpy(),
        onset=test['onset'].to_numpy(),
        pitch=test['pitch'].to_numpy(),
        string=test['string'].to_numpy(),
        probs=probs,
        probs_calibrated=probs_cal,
        prior=prior,
        temperature=temperature,
    )
    summary = {
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_macro_nll_bits': best,
        'temperature': temperature,
    }
    for k, r in report.items():
        summary |= {f'test/{k}/acc': r['model']['acc'], f'test/{k}/prior_acc': r['prior']['acc']}
        summary |= {f'test/{k}/nll_bits': r['model']['nll_bits'], f'test/{k}/bits_gained': r['bits_gained']}
    info(f'Saved {run.dir}')
    run.finish(summary)
    return best


if __name__ == '__main__':
    main()

"""Train and evaluate the per-note string classifier.

Config: conf/strings/train.yaml + schema in gtp.stage1.strings_config. Split: `test_player`
is test, the previous player is val (checkpoint selection), the rest train.
Every metric, model and pitch-prior baseline alike, is computed on the same test notes.

Usage:
  uv run scripts/stage1/strings/train.py data.feature=cqt data.window_ms=200
  uv run scripts/stage1/strings/train.py -m experiment=v0_grid
  uv run scripts/stage1/strings/train.py data.source=hex data.feature=logmel64 data.window_ms=100 wandb.mode=disabled
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
from gtp.stage1 import strings_config  # noqa: F401  (registers the schema)
from gtp.stage1.strings import (
    HOP,
    MAX_POST_FRAMES,
    N_STRINGS,
    PRE_FRAMES,
    SAMPLE_RATE,
    load_cache,
    load_notes,
    pitch_prior,
    split_players,
    summarize,
)
from gtp.track import start_run


def post_frames(window_ms):
    frames = round(window_ms / 1000 * SAMPLE_RATE / HOP)
    if not 1 <= frames <= MAX_POST_FRAMES:
        raise ValueError(f'data.window_ms={window_ms} gives {frames} frames, cache holds 1..{MAX_POST_FRAMES}')
    return frames


def freq_stats(x, idx, chunk=4096):
    total, total_sq, n = 0.0, 0.0, 0
    for i in range(0, len(idx), chunk):
        b = torch.from_numpy(x[idx[i : i + chunk]]).float()
        total += b.sum(dim=(0, 3))
        total_sq += (b**2).sum(dim=(0, 3))
        n += b.shape[0] * b.shape[3]
    mean = total / n
    std = (total_sq / n - mean**2).clamp_min(1e-6).sqrt()
    return mean[None, :, :, None], std[None, :, :, None]


def predict(model, batches):
    model.eval()
    with torch.no_grad():
        return torch.cat([F.softmax(model(*b[:3]), dim=1).cpu() for b in batches]).numpy()


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'strings'), config_name='train')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = auto_device()

    notes = load_notes()
    n_frames = PRE_FRAMES + post_frames(cfg.data.window_ms)
    cache, cache_meta = load_cache(cfg.data.source, cfg.data.feature, len(notes.pitch))
    x = np.ascontiguousarray(cache[..., :n_frames])
    players = split_players(cfg.test_player)
    split = {s: np.flatnonzero(notes.split_mask(p)) for s, p in players.items()}
    run = start_run(cfg, job_type='strings', device=device, extra={'cache': cache_meta, 'players': players})
    info(f'{run.name}\n{OmegaConf.to_yaml(cfg)}')

    mean, std = freq_stats(x, split['train'])
    mean, std = mean.to(device), std.to(device)
    x_t = torch.from_numpy(x)
    pitch_t = torch.from_numpy(notes.pitch)
    cand_t = torch.from_numpy(notes.candidates)
    label_t = torch.from_numpy(notes.label)

    def batches(idx, shuffle=False):
        idx = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), cfg.optim.batch):
            b = torch.from_numpy(idx[i : i + cfg.optim.batch])
            xb = (x_t[b].to(device).float() - mean) / std
            yield xb, pitch_t[b].to(device), cand_t[b].to(device), label_t[b].to(device)

    model = instantiate(cfg.model, in_channels=x.shape[1], n_freq=x.shape[2], n_frames=n_frames).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    sizes = {k: len(v) for k, v in split.items()}
    info(f'input {tuple(x.shape[1:])}, {n_params / 1e6:.2f}M params, device {device}, notes {sizes}')

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    steps_per_epoch = -(-len(split['train']) // cfg.optim.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.optim.lr, total_steps=cfg.optim.epochs * steps_per_epoch
    )
    best_nll, best_state, best_epoch = float('inf'), None, -1
    history = []
    for epoch in range(cfg.optim.epochs):
        model.train()
        t0, loss_sum, n_batches = time.time(), 0.0, 0
        for xb, pb, cb, yb in batches(split['train'], shuffle=True):
            loss = F.cross_entropy(model(xb, pb, cb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1
        val = summarize(predict(model, batches(split['val'])), notes.label[split['val']])
        record = {'epoch': epoch, 'train_loss': loss_sum / n_batches, **{f'val_{k}': v for k, v in val.items()}}
        history.append(record)
        run.log({**record, 'lr': sched.get_last_lr()[0], 'epoch_s': time.time() - t0}, step=epoch)
        if val['nll_bits'] < best_nll:
            best_nll, best_epoch = val['nll_bits'], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        info(
            f'epoch {epoch:2d} loss {loss_sum / n_batches:.3f} val acc {val["acc"]:.3f} '
            f'nll {val["nll_bits"]:.3f} bits ({time.time() - t0:.0f}s)'
        )

    model.load_state_dict(best_state)
    torch.save(best_state, run.dir / 'model.pt')
    test = split['test']
    probs = predict(model, batches(test))
    prior = pitch_prior(notes, notes.split_mask(players['train']))[test]
    labels = notes.label[test]
    multi = notes.candidates[test].sum(axis=1) > 1
    solo = notes.is_solo[test]

    report = {}
    for subset, mask in {
        'all': np.ones_like(multi),
        'multi': multi,
        'multi_comp': multi & ~solo,
        'multi_solo': multi & solo,
    }.items():
        m, p = summarize(probs[mask], labels[mask]), summarize(prior[mask], labels[mask])
        report[subset] = {'model': m, 'prior': p, 'bits_gained': p['nll_bits'] - m['nll_bits']}

    pitches = notes.pitch[test]
    per_pitch = {
        int(q): {
            'n': int(sel.sum()),
            'acc': float((probs[sel].argmax(1) == labels[sel]).mean()),
            'prior_acc': float((prior[sel].argmax(1) == labels[sel]).mean()),
        }
        for q in np.unique(pitches[multi])
        for sel in [multi & (pitches == q)]
    }
    confusion = np.zeros((N_STRINGS, N_STRINGS), dtype=int)
    np.add.at(confusion, (labels[multi], probs[multi].argmax(1)), 1)

    (run.dir / 'metrics.json').write_text(
        json.dumps(
            {
                'n_params': n_params,
                'best_epoch': best_epoch,
                'best_val_nll_bits': best_nll,
                'test': report,
                'per_pitch_multi': per_pitch,
                'confusion_multi_true_by_pred': confusion.tolist(),
                'history': history,
            },
            indent=1,
        )
    )
    np.savez(
        run.dir / 'test_probs.npz',
        track=notes.track[test],
        onset=notes.onset[test],
        pitch=pitches,
        string=notes.string[test],
        probs=probs,
        prior=prior,
    )
    summary = {'n_params': n_params, 'best_epoch': best_epoch, 'best_val_nll_bits': best_nll}
    for subset, r in report.items():
        summary |= {f'test/{subset}/{k}': v for k, v in r['model'].items()}
        summary |= {f'test/{subset}/prior_acc': r['prior']['acc'], f'test/{subset}/bits_gained': r['bits_gained']}
        info(
            f'test {subset:10s} n={r["model"]["n"]:5d}  acc {r["model"]["acc"]:.3f} (prior {r["prior"]["acc"]:.3f})  '
            f'nll {r["model"]["nll_bits"]:.3f} bits (prior {r["prior"]["nll_bits"]:.3f}, gained {r["bits_gained"]:+.3f})  '
            f'ece {r["model"]["ece"]:.3f}'
        )
    info(f'Saved {run.dir}')
    run.finish(summary)
    return best_nll


if __name__ == '__main__':
    main()

"""Train and evaluate the per-note string classifier (v0).

Split: players 00-03 train, 04 val (checkpoint selection), 05 test.
Every metric, model and pitch-prior baseline alike, is computed on the same test notes.

Usage:
  ./venv/bin/python scripts/stage1/strings/train.py --source mic --feature logmel229 --window-ms 200
  ./venv/bin/python scripts/stage1/strings/train.py --source hex --feature logmel64 --window-ms 100
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.strings import (
    FEATURES,
    HOP,
    MAX_POST_FRAMES,
    N_STRINGS,
    PRE_FRAMES,
    SAMPLE_RATE,
    SOURCES,
    StringNet,
    cache_path,
    load_notes,
    pitch_prior,
    summarize,
)

RESULTS_DIR = REPO_ROOT / 'results' / 'strings'


def post_frames(window_ms):
    frames = round(window_ms / 1000 * SAMPLE_RATE / HOP)
    if not 1 <= frames <= MAX_POST_FRAMES:
        raise ValueError(f'--window-ms {window_ms} gives {frames} frames, cache holds 1..{MAX_POST_FRAMES}')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=SOURCES, default='mic')
    parser.add_argument('--feature', choices=FEATURES, default='logmel229')
    parser.add_argument('--window-ms', type=int, default=200)
    parser.add_argument('--width', type=float, default=1.0)
    parser.add_argument('--no-pitch', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', default=None)
    args = parser.parse_args()

    name = args.name or (
        f'{args.source}_{args.feature}_w{args.window_ms}_x{args.width:g}{"_nopitch" if args.no_pitch else ""}_s{args.seed}'
    )
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = auto_device()

    notes = load_notes()
    n_frames = PRE_FRAMES + post_frames(args.window_ms)
    x = np.ascontiguousarray(np.load(cache_path(args.source, args.feature), mmap_mode='r')[..., :n_frames])
    if len(x) != len(notes.pitch):
        raise ValueError(f'Cache has {len(x)} rows, notes have {len(notes.pitch)}: rebuild the cache')
    split = {s: np.flatnonzero(notes.split_mask(s)) for s in ('train', 'val', 'test')}
    mean, std = freq_stats(x, split['train'])
    mean, std = mean.to(device), std.to(device)

    x_t = torch.from_numpy(x)
    pitch_t = torch.from_numpy(notes.pitch)
    cand_t = torch.from_numpy(notes.candidates)
    label_t = torch.from_numpy(notes.label)

    def batches(idx, shuffle=False):
        idx = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), args.batch):
            b = torch.from_numpy(idx[i : i + args.batch])
            xb = (x_t[b].to(device).float() - mean) / std
            yield xb, pitch_t[b].to(device), cand_t[b].to(device), label_t[b].to(device)

    model = StringNet(x.shape[1], x.shape[2], n_frames, width=args.width, use_pitch=not args.no_pitch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    sizes = {k: len(v) for k, v in split.items()}
    info(f'{name}: input {tuple(x.shape[1:])}, {n_params / 1e6:.2f}M params, device {device}, notes {sizes}')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = -(-len(split['train']) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    best_nll, best_state, best_epoch = float('inf'), None, -1
    history = []
    for epoch in range(args.epochs):
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
        history.append({'epoch': epoch, 'train_loss': loss_sum / n_batches, **{f'val_{k}': v for k, v in val.items()}})
        if val['nll_bits'] < best_nll:
            best_nll, best_epoch = val['nll_bits'], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        info(
            f'epoch {epoch:2d} loss {loss_sum / n_batches:.3f} val acc {val["acc"]:.3f} '
            f'nll {val["nll_bits"]:.3f} bits ({time.time() - t0:.0f}s)'
        )

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / 'model.pt')
    test = split['test']
    probs = predict(model, batches(test))
    prior = pitch_prior(notes, notes.split_mask('train'))[test]
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

    (out_dir / 'metrics.json').write_text(
        json.dumps(
            {
                'args': vars(args),
                'n_params': n_params,
                'best_epoch': best_epoch,
                'test': report,
                'per_pitch_multi': per_pitch,
                'confusion_multi_true_by_pred': confusion.tolist(),
                'history': history,
            },
            indent=1,
        )
    )
    np.savez(
        out_dir / 'test_probs.npz',
        track=notes.track[test],
        onset=notes.onset[test],
        pitch=pitches,
        string=notes.string[test],
        probs=probs,
        prior=prior,
    )
    for subset, r in report.items():
        info(
            f'test {subset:10s} n={r["model"]["n"]:5d}  acc {r["model"]["acc"]:.3f} (prior {r["prior"]["acc"]:.3f})  '
            f'nll {r["model"]["nll_bits"]:.3f} bits (prior {r["prior"]["nll_bits"]:.3f}, gained {r["bits_gained"]:+.3f})  '
            f'ece {r["model"]["ece"]:.3f}'
        )
    info(f'Saved {out_dir}')


if __name__ == '__main__':
    main()

"""Train the tab prior: each note's technique from the tab around it, without audio (gtp.stage1.tab_prior).

Trained on the DadaGP corpus (build_tab_corpus.py) without the files of `exclude_files`, on the heads of the
technique classifier and with its label rules. The loss is plain cross-entropy, so the outputs are probabilities
that can be multiplied with the classifier's (fuse_prior.py). The artists the corpus set aside validate.

Reports, per head: how many bits per note the prior saves over always giving the classes' shares, and per class
the precision, recall and F1 of its best guess and the average precision of its probability.

Usage:
  uv run scripts/techniques/train_prior.py
  uv run scripts/techniques/train_prior.py steps=200 eval_every=100 name=smoke
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
from sklearn.metrics import average_precision_score

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.tab_prior import read_tab, tab_arrays, tab_features
from gtp.stage1.techniques import labels_by_head, scores
from gtp.stage2.guitarpro import MARKS
from gtp.track import start_run


def corpus(cfg, heads):
    """Per split, the tab arrays of its tracks and each note's label per head, (notes, heads)."""
    root = REPO_ROOT / cfg.corpus_dir
    tracks, notes = pd.read_csv(root / 'tracks.csv'), np.load(root / 'notes.npz')
    held = set(pd.read_csv(REPO_ROOT / cfg.exclude_files)['file']) if cfg.exclude_files else set()
    split = np.where(tracks['file'].isin(held), 'held', tracks['split']).astype(str)[notes['track']]
    marks, bit = notes['marks'], {m: np.uint32(1 << i) for i, m in enumerate(MARKS)}

    def carrying(names):
        return marks & np.uint32(sum(bit[m] for m in names if m in bit)) > 0

    label = np.stack(list(labels_by_head(carrying, len(marks), heads).values()), axis=1).astype(np.int8)
    out = {}
    for k in ('train', 'val'):
        of = np.flatnonzero(split == k)
        kept, tab = tab_arrays(
            *(notes[c][of] for c in ('track', 'start', 'end', 'string', 'fret')), carrying(['tie'])[of]
        )
        out[k] = tab, label[of][kept]
    return out, int((split == 'held').sum())


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='prior')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = auto_device()
    head_cfg = OmegaConf.load(REPO_ROOT / cfg.train_config).heads
    heads = {name: list(head.classes) for name, head in head_cfg.items()}
    sizes_of = [len(c) for c in heads.values()]
    data, held = corpus(cfg, head_cfg)
    sizes = {k: len(y) for k, (_, y) in data.items()} | {'held': held}
    run = start_run(cfg, job_type='tab_prior', device=device, extra={'notes': sizes})
    info(f'{run.name}\n{OmegaConf.to_yaml(cfg)}notes {sizes}')

    tab = {k: torch.from_numpy(v).to(device) for k, v in data['train'][0].items()}
    labels = torch.from_numpy(data['train'][1]).to(device)
    shares = []
    for h, (name, classes) in enumerate(heads.items()):
        counts = np.bincount(data['train'][1][:, h][data['train'][1][:, h] >= 0], minlength=len(classes))
        shares.append(counts / counts.sum())
        info(f'{name}: training share {dict(zip(classes, shares[-1].round(4).tolist(), strict=True))}')

    def report():
        val_tab, y = data['val']
        parts = np.split(read_tab(model, val_tab, device), np.cumsum(sizes_of)[:-1], axis=1)
        out = {}
        for h, ((name, classes), logits) in enumerate(zip(heads.items(), parts, strict=True)):
            has = y[:, h] >= 0
            log_p = torch.log_softmax(torch.from_numpy(logits[has]), dim=1).numpy()
            own = np.take_along_axis(log_p, y[has, h, None].astype(np.int64), axis=1)[:, 0]
            flat = np.log(shares[h])[y[has, h]]
            out[name] = scores(y[:, h].astype(np.int64), logits.argmax(axis=1), classes)
            out[name]['bits_saved'] = float((own - flat).mean() / np.log(2))
            out[name]['bits_flat'] = float(-flat.mean() / np.log(2))
            for k, c in enumerate(classes[1:], 1):
                out[name][c]['AP'] = float(average_precision_score(y[has, h] == k, log_p[:, k]))
        return out

    def shown(val):
        rows = []
        for name, m in val.items():
            for c in heads[name][1:]:
                share = m[c]['n'] / sum(m[k]['n'] for k in heads[name])
                rows.append({'head': name, 'class': c, 'share': share} | {k: m[c][k] for k in ('P', 'R', 'F1', 'AP')})
            rows[-1] |= {'bits saved': m['bits_saved'], 'of': m['bits_flat']}
        with pd.option_context('display.width', 200, 'display.float_format', '{:.3f}'.format):
            return pd.DataFrame(rows).fillna('').to_string(index=False)

    model = instantiate(cfg.model, n_classes=sum(sizes_of)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps)
    n, reach = len(labels), torch.arange(cfg.window, device=device)[None]
    moved, history, t0, loss_sum = tab, [], time.time(), 0.0
    for step in range(1, cfg.steps + 1):
        if cfg.jitter_s and step % cfg.jitter_every == 1:
            moved = dict(tab)
            for k in ('start', 'end'):
                moved[k] = tab[k] + cfg.jitter_s * torch.randn(n, device=device)
        model.train()
        idx = torch.randint(0, n, (cfg.batch, 1), device=device) + reach
        inside = idx < n
        idx = idx.clamp(max=n - 1)
        inside &= tab['track'][idx] == tab['track'][idx[:, :1]]
        speed = torch.exp(cfg.tempo * (2 * torch.rand(cfg.batch, 1, device=device) - 1))
        cats, numbers = tab_features(moved, idx, speed)
        parts = model(cats, numbers, inside.sum(dim=1)).split(sizes_of, dim=-1)
        y = torch.where(inside[..., None], labels[idx].long(), torch.full_like(labels[idx], -1, dtype=torch.long))
        loss = sum(
            F.cross_entropy(part.flatten(0, 1), y[..., h].flatten(), ignore_index=-1) for h, part in enumerate(parts)
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_sum += loss.item()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            val = report()
            saved = {name: round(m['bits_saved'], 4) for name, m in val.items()}
            history.append({'step': step, 'train_loss': loss_sum / cfg.eval_every, 'val_bits_saved': saved})
            run.log(
                {'train_loss': loss_sum / cfg.eval_every} | {f'val_bits_saved/{k}': v for k, v in saved.items()},
                step=step,
            )
            info(f'step {step} loss {loss_sum / cfg.eval_every:.4f} val bits saved {saved} ({time.time() - t0:.0f}s)')
            loss_sum = 0.0

    torch.save(model.state_dict(), run.dir / 'model.pt')
    val = report()
    info(f'validation artists\n{shown(val)}')
    metrics = {'heads': heads, 'shares': [s.tolist() for s in shares], 'notes': sizes, 'history': history, 'val': val}
    (run.dir / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    run.finish({f'val_bits_saved/{name}': m['bits_saved'] for name, m in val.items()})


if __name__ == '__main__':
    main()

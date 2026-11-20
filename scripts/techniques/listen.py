"""Listening clips for judging what the technique classifier (train.py) got right and wrong on validation.

Notes are sampled per group of (dataset, label, classifier's answer) from a run's validation predictions, each cut
with its surroundings into an MP3 and a spectrogram with the note marked. A note of a take recorded through several
chains is cut from each, with the classifier's answer on each. Next to every clip goes a decay figure: how many dB
the note drops from its first 60 ms to 0.2-0.3 s after the onset, with the median of each dataset's classes for
comparison (left out where another note starts inside that span). Also prints how often the classifier is right
for notes that start alone, with notes of their own class, or with notes of another class: it reads the audio
around the onset without the pitch, so notes struck together look the same to it. Writes <id>_<chain>.mp3,
<id>.png and clips.json.

Usage:
  uv run scripts/techniques/listen.py
"""

import json

import hydra
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from hydra.core.hydra_config import HydraConfig

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.listening import clip, labels_around, spectrogram
from gtp.stage1.string_data import load_cached_index
from gtp.stage1.strings import HOP, PRE_FRAMES, SAMPLE_RATE
from gtp.stage1.techniques import class_of

EARLY = slice(PRE_FRAMES + 1, PRE_FRAMES + 6)
LATE = slice(PRE_FRAMES + 18, PRE_FRAMES + 27)


def decay_db(windows):
    """Drop in mean log-mel level from the frames right after the onset to those 0.2-0.3 s later, in dB."""
    w = np.asarray(windows, dtype=np.float32)[:, 0]
    return 10 / np.log(10) * (w[:, :, EARLY].mean(axis=(1, 2)) - w[:, :, LATE].mean(axis=(1, 2)))


def company(val, notes, classes, other_marks, cfg):
    """Per validation note: 'alone', or whether the notes starting with it are of the 'same' class or 'mixed'; and
    whether another note starts within the decay figure's span, which makes the figure meaningless."""
    label = pd.Series(class_of(notes['techniques'], classes, other_marks), index=notes.index)
    kind, crowded = [], []
    by_rec = {rec: g for rec, g in notes.groupby('rec', sort=False)}
    for i, n in val.iterrows():
        g = by_rec[n.rec]
        gap = g['onset'] - n.onset
        chord = g.index[(gap.abs() <= cfg.together_s) & (g.index != i)]
        kind.append('alone' if not len(chord) else 'same' if (label[chord] == label[i]).all() else 'mixed')
        crowded.append(bool(((gap > cfg.together_s) & (gap < LATE.stop * HOP / SAMPLE_RATE)).any()))
    return kind, crowded


def pick(val, groups, classes, cfg, rng):
    """[(note id, group)]: each group's notes in random order, skipping ones close to a note already picked."""
    picked = []
    for g in groups:
        pool = val[
            (val['dataset'] == g.dataset)
            & (val['chain'] == g.chain)
            & (val['label'] == classes.index(g.label))
            & (val['predicted'] == classes.index(g.predicted))
            & (val['onset'] >= cfg.before_s)
        ]
        if g.get('content'):
            pool = pool[pool['content'] == g.content]
        taken = 0
        for i, n in pool.sample(frac=1, random_state=int(rng.integers(1 << 31))).iterrows():
            near = [val.loc[j] for j, _ in picked if val.at[j, 'rec'] == n.rec]
            if taken < g.n and all(abs(n.onset - m.onset) >= cfg.min_gap_s for m in near):
                picked.append((i, g))
                taken += 1
    return picked


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='listen')
def main(cfg):
    out = HydraConfig.get().runtime.output_dir
    run = REPO_ROOT / cfg.run
    classes = json.loads((run / 'metrics.json').read_text())['classes']
    other_marks = list(cfg.other_marks)
    notes, rows, x, _ = load_cached_index(cfg.index_dir, cfg.feature)
    recs = pd.read_csv(REPO_ROOT / cfg.index_dir / 'recordings.csv').set_index('rec')
    saved = np.load(run / 'val_logits.npz')
    p = np.exp(saved['logits'] - saved['logits'].max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    real = rows[rows['variant'] == 0]
    window = pd.Series(real.index, index=real['note'])[saved['note']].to_numpy()
    val = notes.iloc[saved['note']].assign(
        predicted=p.argmax(axis=1),
        p=list(p.round(3)),
        decay=np.concatenate([decay_db(x[window[i : i + 4096]]) for i in range(0, len(window), 4096)]),
    )
    val['label'] = class_of(val['techniques'], classes, other_marks)
    val['class'] = [classes[k] for k in val['label']]
    val['company'], crowded = company(val, notes, classes, other_marks, cfg)
    val.loc[crowded, 'decay'] = np.nan
    medians = val.groupby(['dataset', 'chain', 'class'])['decay'].median().round(1)
    info(f'median decay, dB\n{medians.unstack().to_string()}')
    found = val.assign(found=val['predicted'] == val['label']).groupby(['dataset', 'chain', 'class', 'company'])
    info(f'found by the classifier\n{found["found"].agg(["size", "mean"]).round(3).to_string()}')

    rng = np.random.default_rng(cfg.seed)
    picked = pick(val, cfg.groups, classes, cfg, rng)
    order = rng.permutation(len(picked))
    by_rec = dict(tuple(notes.groupby('rec', sort=False)))
    clips, counts = [], {True: 0, False: 0}
    for i, g in (picked[k] for k in order):
        n = val.loc[i]
        reference = bool(g.get('reference', False))
        counts[reference] += 1
        cid = f'{"r" if reference else "c"}{counts[reference]:02d}'
        takes = {n.chain: n}
        for chain in cfg.other_chains:
            twin = val[
                (val['rec'] == n.rec.replace(f'/{n.chain}/', f'/{chain}/'))
                & (val['pitch'] == n.pitch)
                & ((val['onset'] - n.onset).abs() < 0.05)
                & (val['onset'] >= cfg.before_s)
            ]
            if len(twin) and twin.iloc[0].rec != n.rec:
                takes[chain] = twin.iloc[0]
        axes = None
        for chain, m in takes.items():
            y, sr = librosa.load(REPO_ROOT / recs.at[m.rec, 'audio'], sr=None, mono=True)
            seg, start = clip(y, sr, m.onset, cfg.before_s, cfg.after_s)
            sf.write(f'{out}/{cid}_{chain}.mp3', seg.astype(np.float32), sr, format='MP3')
            if axes is None:
                axes = spectrogram(seg, sr, m, start, cfg, f'{out}/{cid}.png')
        clips.append({
            'id': cid, 'reference': reference, 'dataset': n.dataset, 'guitar': n.guitar, 'rec': n.rec,
            'content': n.content, 'onset': round(float(n.onset), 3), 'duration': round(float(n.offset - n.onset), 3),
            'pitch': int(n.pitch), 'name': librosa.midi_to_note(int(n.pitch), unicode=False),
            'string': int(n.string), 'fret': int(n.fret), 'label': n['class'], 'company': n.company,
            'marks': n.techniques if isinstance(n.techniques, str) else '',
            'chains': {
                chain: {'predicted': classes[m.predicted], 'p': [float(v) for v in m.p],
                        'decay_db': None if np.isnan(m.decay) else round(float(m.decay), 1)}
                for chain, m in takes.items()
            },
            'lead_s': cfg.before_s, 'axes': axes, 'span': [-cfg.before_s, cfg.after_s],
            'labels': labels_around(by_rec[n.rec], n, cfg.before_s, cfg.after_s),
        })  # fmt: skip
    summary = {
        'classes': classes,
        'median_decay_db': {'/'.join(k): float(v) for k, v in medians.dropna().items()},
        'clips': clips,
    }
    (REPO_ROOT / out / 'clips.json').write_text(json.dumps(summary, indent=1))
    table = pd.DataFrame([{'dataset': c['dataset'], 'label': c['label'], **c['chains']['di']} for c in clips])
    info(f'{len(clips)} clips in {out}\n{table.groupby(["dataset", "label", "predicted"]).size().to_string()}')


if __name__ == '__main__':
    main()

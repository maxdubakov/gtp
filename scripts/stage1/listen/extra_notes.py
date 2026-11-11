"""Cut listening clips of notes Stage 1 finds where the labels have none, plus correctly found notes as blind controls.

Each Stage 1 note on the chosen guitars is matched to the labels (same pitch, onset within `onset_tolerance`). An
unmatched note is sorted into the first category that explains it:
  timing     a label of the same pitch starts within `timing_s`;
  retrigger  a label of the same pitch is still sounding;
  harmonic   a label an octave, twelfth or two octaves away starts together with it;
  pitch      a label of another pitch starts together with it;
  invented   none of these: the labels have nothing near it.
Rates per guitar go to categories.csv. Invented notes and controls are cut into MP3 clips with a log-frequency
spectrogram (the note marked, the labels left out so the listener is blind), described in clips.json.

Usage:
  uv run scripts/stage1/listen/extra_notes.py
"""

import json

import hydra
import librosa
import matplotlib
import mir_eval
import numpy as np
import pandas as pd
import soundfile as sf
from hydra.core.hydra_config import HydraConfig

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.log import info

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def match(ref, est, tolerance):
    def arrays(df):
        intervals = np.c_[df.onset, np.maximum(df.offset, df.onset + 1e-3)]
        return intervals, librosa.midi_to_hz(df.pitch.to_numpy(float))

    return mir_eval.transcription.match_notes(*arrays(ref), *arrays(est), onset_tolerance=tolerance, offset_ratio=None)


def category(note, ref, cfg):
    gap = (ref.onset - note.onset).abs()
    same = ref.pitch == note.pitch
    if (same & (gap <= cfg.timing_s)).any():
        return 'timing'
    if (same & (ref.onset < note.onset) & (ref.offset > note.onset)).any():
        return 'retrigger'
    close = gap <= cfg.onset_tolerance
    if (close & (note.pitch - ref.pitch).isin(list(cfg.harmonic_intervals))).any():
        return 'harmonic'
    if close.any():
        return 'pitch'
    return 'invented'


def notes_of(rec, ref, cfg):
    est = pd.DataFrame(
        json.loads((REPO_ROOT / cfg.stage1_cache / f'{rec}.json').read_text()), columns=['pitch', 'start', 'end']
    )
    est = est.rename(columns={'start': 'onset', 'end': 'offset'})
    if est.empty:
        return est.assign(category=[])
    matched = {e for _, e in match(ref, est, cfg.onset_tolerance)}
    est['category'] = ['found' if i in matched else category(n, ref, cfg) for i, n in enumerate(est.itertuples())]
    return est


def spectrogram(y, sr, note, start, cfg, path):
    fmin = librosa.midi_to_hz(cfg.fmin_midi)
    bins = (cfg.fmax_midi - cfg.fmin_midi) * 3
    c = librosa.amplitude_to_db(
        np.abs(librosa.cqt(y, sr=sr, hop_length=256, fmin=fmin, n_bins=bins, bins_per_octave=36)), ref=np.max
    )
    t = np.arange(c.shape[1]) * 256 / sr + start - note.onset
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=110)
    ax.imshow(c, origin='lower', aspect='auto', cmap='magma', vmin=-70, vmax=0,
              extent=[t[0], t[-1], cfg.fmin_midi - 1 / 6, cfg.fmax_midi - 1 / 6])  # fmt: skip
    p, d = note.pitch, max(note.offset - note.onset, 0.05)
    ax.add_patch(plt.Rectangle((0, p - 0.5), d, 1, fill=False, ec='#39d0ff', lw=1.6))
    ax.axvline(0, color='#39d0ff', lw=0.8, ls=':')
    ticks = [m for m in range(cfg.fmin_midi, cfg.fmax_midi + 1) if m % 12 == 4]
    ax.set_yticks(ticks, [librosa.midi_to_note(m, unicode=False) for m in ticks])
    ax.set_xlabel('seconds from the note')
    ax.set_xlim(-cfg.before_s, cfg.after_s)
    fig.tight_layout()
    fig.savefig(path)
    box = ax.get_position()
    plt.close(fig)
    return {'x0': round(box.x0, 4), 'x1': round(box.x1, 4), 'y0': round(1 - box.y1, 4), 'y1': round(1 - box.y0, 4)}


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'stage1'), config_name='extra_notes')
def main(cfg):
    out = HydraConfig.get().runtime.output_dir
    recs = pd.concat([pd.read_csv(REPO_ROOT / d / 'recordings.csv').assign(index_dir=d) for d in cfg.index_dirs])
    recs = recs[recs.guitar.isin(list(cfg.guitars)) & recs.chain.isin(list(cfg.chains))].drop_duplicates('rec')
    labels = {d: pd.read_csv(REPO_ROOT / d / 'notes.csv.gz').groupby('rec') for d in recs.index_dir.unique()}
    rows = []
    for r in recs.itertuples():
        ref = labels[r.index_dir].get_group(r.rec).reset_index(drop=True)
        est = notes_of(r.rec, ref, cfg)
        rows.append(est.assign(rec=r.rec, guitar=r.guitar, dataset=r.dataset, chain=r.chain, n_ref=len(ref)))
    notes = pd.concat(rows, ignore_index=True)

    n_ref = notes.drop_duplicates('rec').groupby('guitar').n_ref.sum()
    rates = notes.groupby(['guitar', 'category']).size().unstack(fill_value=0)
    rates = (100 * rates.div(n_ref, axis=0)).round(1).assign(ref_notes=n_ref)
    rates.to_csv(f'{out}/categories.csv')
    info(f'Stage 1 notes per 100 reference notes\n{rates.to_string()}')

    rng = np.random.default_rng(cfg.seed)
    picked = []
    for _, g in notes.groupby('guitar'):
        for kind, n in (('invented', cfg.extras_per_guitar), ('found', cfg.controls_per_guitar)):
            pool = g[g.category == kind].sample(frac=1, random_state=rng.integers(1 << 31))
            picked.append(pool.groupby('rec').head(cfg.max_per_rec).head(n))
    picked = pd.concat(picked).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    clips = []
    for i, n in enumerate(picked.itertuples()):
        r = recs.set_index('rec').loc[n.rec]
        y, sr = librosa.load(REPO_ROOT / r.audio, sr=None, mono=True)
        start = max(n.onset - cfg.before_s, 0.0)
        seg = y[int(start * sr) : int((n.onset + cfg.after_s) * sr)]
        ramp = np.minimum(1, np.minimum(np.arange(len(seg)), np.arange(len(seg))[::-1]) / (0.01 * sr))
        seg = seg / max(np.abs(seg).max(), 1e-6) * 0.9 * ramp
        cid = f'c{i + 1:02d}'
        sf.write(f'{out}/{cid}.mp3', seg.astype(np.float32), sr, format='MP3')
        axes = spectrogram(seg, sr, n, start, cfg, f'{out}/{cid}.png')
        ref = labels[r.index_dir].get_group(n.rec)
        near = ref[(ref.offset > n.onset - cfg.before_s) & (ref.onset < n.onset + cfg.after_s)]
        clips.append({
            'id': cid, 'rec': n.rec, 'guitar': n.guitar, 'dataset': n.dataset, 'chain': n.chain,
            'kind': n.category, 'pitch': int(n.pitch), 'name': librosa.midi_to_note(int(n.pitch), unicode=False),
            'onset': round(float(n.onset), 3), 'duration': round(float(n.offset - n.onset), 3),
            'lead_s': round(float(n.onset - start), 3), 'axes': axes, 'span': [-cfg.before_s, cfg.after_s],
            'labels': [
                {'name': librosa.midi_to_note(int(m.pitch), unicode=False), 'at': round(float(m.onset - n.onset), 2),
                 'string': int(m.string)}
                for m in near.sort_values('onset').itertuples()
            ],
        })  # fmt: skip
    (REPO_ROOT / out / 'clips.json').write_text(json.dumps(clips, indent=1))
    info(f'{len(clips)} clips in {out}')


if __name__ == '__main__':
    main()

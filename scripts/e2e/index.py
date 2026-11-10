"""End-to-end audio -> tab on the string index's held-out recordings: stage 1 -> string evidence -> stage 2.

For every recording of the requested splits:
  1. stage 1 transcribes the audio into notes (or the reference notes are used, `inputs: gt`);
  2. the string classifiers score each note, and each note gets an inharmonicity likelihood curve;
  3. per guitar, the string constants are calibrated without labels and the two are fused;
  4. stage 2 picks each note's string by beam search, rescored with the fused evidence (weight lam).
Predicted notes are matched to the reference with mir_eval (onset tolerance, no offset). A tab note is correct when
it is matched and its string is right; Tab P/R/F1 count those. On the matched notes it also reports the equivalent
accuracy of metrics.py (a recording's most common string shift forgiven) and the number of separate runs of wrong
notes per 100 notes, as a measure of how many corrections the tab needs.

`classifier_runs` may contain `{guitar}`: then each guitar is scored by its own runs (leave-one-guitar-out folds).
With `degrade` (a recording chain, see chains.py), the audio the string evidence reads is first rendered through it:
a stand-in for conditions no dataset has, such as a laptop microphone in a room. Stage 1's notes are not re-derived.

Usage:
  uv run scripts/e2e/index.py
  uv run scripts/e2e/index.py 'inputs=[gt]' 'splits=[test]'
"""

import glob
import itertools
import zlib

import hydra
import librosa
import mir_eval
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1 import strings as st
from gtp.stage1.bprofile import grid, recording_profiles
from gtp.stage1.chains import render, stages_for
from gtp.stage1.inference import PianoTranscription, transcribe_cached
from gtp.stage1.inharmonicity import fuse_guitar
from gtp.stage2.fusion import decode_beam, jump_of, log_ratio, piece_rows, sorted_notes, valid_tabs
from gtp.stage2.inference import load_checkpoint
from gtp.stage2.metrics import aggregate_drift_buckets

TUNING = [int(p) for p in st.OPEN_STRINGS]


def match(ref, est, tolerance):
    """[(ref_idx, est_idx)] pairs: same pitch, onsets within `tolerance`, offsets ignored."""
    if not ref or not est:
        return []

    def arrays(notes):
        intervals = np.array([[n['start'], max(n['end'], n['start'] + 1e-3)] for n in notes])
        return intervals, librosa.midi_to_hz(np.array([n['pitch'] for n in notes]))

    return mir_eval.transcription.match_notes(*arrays(ref), *arrays(est), onset_tolerance=tolerance, offset_ratio=None)


def prf(tp, n_est, n_ref):
    p, r = tp / max(n_est, 1), tp / max(n_ref, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'e2e'), config_name='index')
def main(cfg):
    device = auto_device()
    profiles = OmegaConf.merge(OmegaConf.load(REPO_ROOT / 'conf' / 'strings' / 'b_profiles.yaml'), cfg.profiles)
    physics = OmegaConf.merge(OmegaConf.load(REPO_ROOT / 'conf' / 'strings' / 'physics_fusion.yaml'), cfg.physics)
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    recs = recs[recs['split'].isin(list(cfg.splits))]
    if cfg.guitars:
        recs = recs[recs['guitar'].isin(list(cfg.guitars))]
    ref_notes = pd.read_csv(index / 'notes.csv.gz').groupby('rec')
    stage2, vocab, step, max_seq_len = load_checkpoint(REPO_ROOT / cfg.stage2_checkpoint, device)
    stage1 = (
        PianoTranscription(str(REPO_ROOT / cfg.stage1_checkpoint), device=cfg.stage1_device)
        if 'stage1' in cfg.inputs
        else None
    )
    stage1_cache = REPO_ROOT / 'results' / 'e2e' / 'stage1' / f'index_{(REPO_ROOT / cfg.stage1_checkpoint).stem}'

    table, pieces, counts = [], [], []
    for guitar, group in recs.groupby('guitar'):
        runs = sorted(
            r for pattern in cfg.classifier_runs for r in glob.glob(str(REPO_ROOT / pattern.format(guitar=guitar)))
        )
        assert runs, f'no classifier runs for {guitar}'
        classifiers = [st.StringClassifier.from_run(r, device) for r in runs]
        for source, (_, sub) in itertools.product(cfg.inputs, group.groupby(list(cfg.calibrate_by), sort=False)):
            first = len(table)
            probs, curves = [], []
            for r in sub.itertuples():
                ref = [
                    {'pitch': int(n.pitch), 'start': n.onset, 'end': n.offset, 'string': int(n.string)}
                    for n in ref_notes.get_group(r.rec).itertuples()
                ]
                est = (
                    ref
                    if source == 'gt'
                    else transcribe_cached(stage1, REPO_ROOT / r.audio, stage1_cache / f'{r.rec}.json')
                )
                est = [n for n in est if len(valid_tabs(n['pitch'], TUNING, vocab)[0])]
                ref_of = {e: i for i, e in match(ref, est, cfg.onset_tolerance)}
                notes = sorted_notes(
                    {
                        'notes': [
                            {
                                **n,
                                'end': max(n['end'], n['start'] + 0.01),
                                'ref_string': ref[ref_of[i]]['string'] if i in ref_of else None,
                            }
                            for i, n in enumerate(est)
                        ]
                    }
                )
                counts.append(
                    {'input': source, 'rec': r.rec, 'n_ref': len(ref), 'n_est': len(est), 'n_matched': len(ref_of)}
                )
                if not notes:
                    continue
                onsets, pitches = [n['start'] for n in notes], [n['pitch'] for n in notes]
                hi, sr = librosa.load(REPO_ROOT / r.audio, sr=profiles.sample_rate, mono=True)
                if cfg.degrade:
                    chain = {k: v for k, v in OmegaConf.to_container(cfg.degrade).items() if k in stages_for(r.chain)}
                    hi = render(hi, sr, chain, np.random.default_rng(zlib.crc32(r.rec.encode())))
                y = librosa.resample(hi, orig_sr=sr, target_sr=st.SAMPLE_RATE)
                spec = st.spectrogram(y[None], classifiers[0].feature)
                probs.append(np.mean([c.predict(spec, onsets, pitches) for c in classifiers], axis=0))
                curves.append(
                    recording_profiles(hi, sr, onsets, [n['end'] for n in notes], pitches, profiles).sum(axis=1)
                )
                ids = list(range(len(table), len(table) + len(notes)))
                piece = {
                    'source': 'index',
                    'filename': r.rec,
                    'tuning': TUNING,
                    'tempo': None,
                    'capo': 0,
                    'notes': [
                        {**n, 'string': int(valid_tabs(n['pitch'], TUNING, vocab)[0][0]), 'fret': 0} for n in notes
                    ],
                }
                for n in piece['notes']:
                    n['fret'] = n['pitch'] - TUNING[n['string'] - 1]
                pieces.append((piece, ids))
                table += [
                    {'input': source, 'guitar': guitar, 'dataset': r.dataset, 'content': r.content, 'chain': r.chain,
                        'split': r.split,
                     'rec': r.rec, 'pitch': n['pitch'], 'start': n['start'], 'ref_string': n['ref_string']}
                    for n in notes
                ]  # fmt: skip
            if not probs:
                continue
            probs, curves = np.concatenate(probs), np.concatenate(curves)
            pitch = np.array([t['pitch'] for t in table[first:]])
            fused, const, family = fuse_guitar(probs, curves, grid(profiles), pitch, physics)
            prior = classifiers[0].prior(pitch)
            for t, p, f, q in zip(table[first:], probs, fused, prior, strict=True):
                strings, _ = valid_tabs(t['pitch'], TUNING, vocab)
                t['classifier'], t['fused'] = int(p.argmax()) + 1, int(f.argmax()) + 1
                t['ratio'] = log_ratio(f[None], q, strings, cfg.prob_floor)
            info(f'{guitar} {source}: {len(sub)} recordings, {len(probs)} notes, {family}, constants {const.round(2)}')

    df = pd.DataFrame(table)
    rows = [
        row
        for piece, ids in pieces
        for row in piece_rows(piece, vocab, max_seq_len, ids, [table[i]['ratio'] for i in ids])
    ]
    df['stage2'] = pd.Series(
        decode_beam(stage2, rows, device, cfg.batch, beams=cfg.beams, exclusive=cfg.exclusive_s, jump=jump_of(cfg))
    )
    df['pipeline'] = pd.Series(
        decode_beam(
            stage2, rows, device, cfg.batch, cfg.lam, beams=cfg.beams, exclusive=cfg.exclusive_s, jump=jump_of(cfg)
        )
    )
    methods = ['stage2', 'classifier', 'fused', 'pipeline']
    df = df.drop(columns=['ratio'])
    counts = pd.DataFrame(counts).merge(recs[['rec', 'guitar', 'dataset', 'chain', 'content', 'split']], on='rec')

    summary = []
    keys = ['input', 'split', 'guitar', 'dataset', 'chain', 'content']
    groups = [*counts.groupby(keys), *(((*k, 'all', 'all', 'all'), g) for k, g in counts.groupby(keys[:3]))]
    for key, c in groups:
        d = df[df['rec'].isin(c['rec']) & (df['input'] == key[0]) & df['ref_string'].notna()].sort_values(
            ['rec', 'start']
        )
        row = dict(zip(keys, key, strict=True)) | {'ref': c.n_ref.sum(), 'est': c.n_est.sum()}
        row['note_f1'] = prf(c.n_matched.sum(), row['est'], row['ref'])[2]
        rec = d['rec'].to_numpy()
        for m in methods:
            shift = (d[m] - d['ref_string']).to_numpy()
            wrong = shift != 0
            starts = wrong & ~(np.r_[False, wrong[:-1]] & (rec == np.r_[[''], rec[:-1]]))
            records = [
                {
                    'piece_id': r,
                    'error_type_pp': 'wrong' if w else 'correct',
                    'delta_string_pp': int(x),
                    'delta_fret_pp': 0,
                }
                for r, w, x in zip(rec, wrong, shift, strict=True)
            ]
            row[f'{m}_acc'] = 1 - wrong.mean() if len(d) else 0.0
            row[f'{m}_equivalent'] = aggregate_drift_buckets(records, min_notes=cfg.min_notes_for_drift)[
                'tab_equivalent_acc'
            ]
            row[f'{m}_fixes_per_100'] = 100 * starts.sum() / max(len(d), 1)
            row[f'{m}_tab_f1'] = prf((~wrong).sum(), row['est'], row['ref'])[2]
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values(keys)
    shown = ['ref', 'note_f1', 'stage2_acc', 'classifier_acc', 'fused_acc', 'pipeline_acc']
    show = [*keys, *shown, 'pipeline_equivalent', 'pipeline_fixes_per_100', 'pipeline_tab_f1']
    info(f'stage 2 step {step}, lam {cfg.lam}\n{summary[show].round(3).to_string(index=False)}')
    out = HydraConfig.get().runtime.output_dir
    df.to_csv(f'{out}/notes.csv.gz', index=False)
    summary.to_csv(f'{out}/summary.csv', index=False)
    info(f'Saved {out}')


if __name__ == '__main__':
    main()

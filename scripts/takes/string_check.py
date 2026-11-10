"""String-classifier accuracy on the strings of edited takes: a personal sanity check, never a reported result.

Scores every edited note that more than one string can play, with the user's string as the truth, for
  * the classifier runs in `runs` (seeds averaged), run on the take's audio at the edited onsets and pitches,
  * the same fused with inharmonicity evidence, self-calibrated over all takes of the same guitar (meta.json),
  * Stage 2 alone, and Stage 2 rescored at decode time with the fused evidence (the pipeline's choice),
  * their pitch prior,
  * the evidence saved in the take's prediction (the classifier the demo used then), for notes whose pitch
    the user kept.

Usage:
  uv run scripts/takes/string_check.py 'runs=[results/strings/runs/a,results/strings/runs/b]'
"""

import hydra
import librosa
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from gtp import REPO_ROOT, takes
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1 import strings as st
from gtp.stage1.bprofile import grid, recording_profiles
from gtp.stage1.inharmonicity import fuse_guitar
from gtp.stage2.fusion import decode_beam, log_ratio, piece_rows, sorted_notes, valid_tabs
from gtp.stage2.inference import load_checkpoint


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'takes'), config_name='string_check')
def main(cfg):
    device = auto_device()
    classifiers = [st.StringClassifier.from_run(REPO_ROOT / r, device) for r in cfg.runs]
    profiles = OmegaConf.load(REPO_ROOT / 'conf' / 'strings' / 'b_profiles.yaml')
    physics = OmegaConf.load(REPO_ROOT / 'conf' / 'strings' / 'physics_fusion.yaml')
    root = REPO_ROOT / cfg.takes_dir
    rows, per_take = [], []
    for take_id in takes.take_ids(root):
        take = takes.load(root / take_id)
        edited = take['edited']
        if not edited:
            continue
        notes = edited['notes']
        onsets, pitches = [n['start'] for n in notes], [n['pitch'] for n in notes]
        multi = st.candidates(pitches).sum(axis=1) > 1
        truth = np.array([n['string'] - 1 for n in notes])
        y, _ = librosa.load(root / take_id / 'audio.wav', sr=st.SAMPLE_RATE, mono=True)
        spec = st.spectrogram(y[None], classifiers[0].feature)
        probs = np.mean([c.predict(spec, onsets, pitches) for c in classifiers], axis=0)
        prior = classifiers[0].prior(pitches)
        hi, sr = librosa.load(root / take_id / 'audio.wav', sr=profiles.sample_rate, mono=True)
        curves = recording_profiles(hi, sr, onsets, [n['end'] for n in notes], pitches, profiles).sum(axis=1)
        per_take.append((take['meta'].get('guitar', ''), probs, curves, np.asarray(pitches)))
        before = {n['id']: n for n in take['prediction']['notes']}
        old = [
            (np.argmax(src['evidence']), t)
            for n, t, m in zip(notes, truth, multi, strict=True)
            if m and (src := before.get(n.get('source_id'))) and src.get('evidence') and src['pitch'] == n['pitch']
        ]
        rows.append(
            {
                'take': take_id,
                'notes': notes,
                'n': int(multi.sum()),
                'runs': float((probs.argmax(1) == truth)[multi].mean()),
                'runs prior': float((prior.argmax(1) == truth)[multi].mean()),
                'saved evidence': float(np.mean([p == t for p, t in old])) if old else np.nan,
                'evidence scored': len(old),
                'truth': truth,
                'multi': multi,
            }
        )
    for guitar in {g for g, *_ in per_take}:
        group = [i for i, (g, *_) in enumerate(per_take) if g == guitar]
        probs, curves, pitch = (np.concatenate([per_take[i][k] for i in group]) for k in (1, 2, 3))
        fused, const, family = fuse_guitar(probs, curves, grid(profiles), pitch, physics)
        info(
            f'{guitar or "unnamed guitar"}: {len(group)} takes, {len(probs)} notes, {family}, constants {const.round(2)}'
        )
        for i, part in zip(group, np.split(fused, np.cumsum([len(per_take[i][1]) for i in group])[:-1]), strict=True):
            r = rows[i]
            r['runs + B'] = float((part.argmax(1) == r['truth'])[r['multi']].mean())
            r['fused'] = part
    stage2, vocab, _, max_seq_len = load_checkpoint(REPO_ROOT / cfg.stage2_checkpoint, device)
    for r in rows:
        notes, tuning = r.pop('notes'), [int(p) for p in st.OPEN_STRINGS]
        prior, fused = classifiers[0].prior([n['pitch'] for n in notes]), r.pop('fused')
        ratios = [
            log_ratio(f[None], q, valid_tabs(n['pitch'], tuning, vocab)[0], cfg.prob_floor)
            for n, f, q in zip(notes, fused, prior, strict=True)
        ]
        piece = {'source': 'take', 'filename': r['take'], 'tuning': tuning, 'tempo': None, 'capo': 0, 'notes': notes}
        assert sorted_notes(piece) == notes, 'edited notes are not in decoder order'
        for name, lam in (('stage 2', 0.0), ('pipeline', cfg.lam)):
            rows_ = piece_rows(piece, vocab, max_seq_len, list(range(len(notes))), ratios)
            picks = decode_beam(stage2, rows_, device, 64, lam, beams=cfg.beams)
            hit = np.array([picks[i] - 1 for i in range(len(notes))]) == r['truth']
            r[name] = float(hit[r['multi']].mean())
    table = pd.DataFrame(rows).drop(columns=['truth', 'multi'])
    info(
        f'{len(classifiers)} runs, string accuracy on edited multi-string notes\n{table.round(3).to_string(index=False)}'
    )


if __name__ == '__main__':
    main()

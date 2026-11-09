"""String-classifier accuracy on the strings of edited takes: a personal sanity check, never a reported result.

Scores every edited note that more than one string can play, with the user's string as the truth, for
  * the classifier runs in `runs` (seeds averaged), run on the take's audio at the edited onsets and pitches,
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

from gtp import REPO_ROOT, takes
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1 import strings as st


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'takes'), config_name='string_check')
def main(cfg):
    device = auto_device()
    classifiers = [st.StringClassifier.from_run(REPO_ROOT / r, device) for r in cfg.runs]
    root = REPO_ROOT / cfg.takes_dir
    rows = []
    for take_id in takes.take_ids(root):
        take = takes.load(root / take_id)
        edited = take['edited']
        if not edited:
            continue
        notes = [n for n in edited['notes'] if st.candidates([n['pitch']])[0].sum() > 1]
        onsets, pitches = [n['start'] for n in notes], [n['pitch'] for n in notes]
        truth = np.array([n['string'] - 1 for n in notes])
        y, _ = librosa.load(root / take_id / 'audio.wav', sr=st.SAMPLE_RATE, mono=True)
        spec = st.spectrogram(y[None], classifiers[0].feature)
        probs = np.mean([c.predict(spec, onsets, pitches) for c in classifiers], axis=0)
        prior = classifiers[0].prior(pitches)
        before = {n['id']: n for n in take['prediction']['notes']}
        old = [
            (np.argmax(src['evidence']), t)
            for n, t in zip(notes, truth, strict=True)
            if (src := before.get(n.get('source_id'))) and src.get('evidence') and src['pitch'] == n['pitch']
        ]
        rows.append(
            {
                'take': take_id,
                'notes': len(notes),
                'runs': float((probs.argmax(1) == truth).mean()),
                'runs prior': float((prior.argmax(1) == truth).mean()),
                'saved evidence': float(np.mean([p == t for p, t in old])) if old else np.nan,
                'evidence scored': len(old),
            }
        )
    info(
        f'{len(classifiers)} runs, string accuracy on edited multi-string notes\n{pd.DataFrame(rows).round(3).to_string(index=False)}'
    )


if __name__ == '__main__':
    main()

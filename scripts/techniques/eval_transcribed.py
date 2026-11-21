"""Score the technique classifier on the notes the note model finds, in place of the labels' own notes.

Each validation recording of `group` is transcribed by the note model. A labelled note is found when the note model
has a note of its pitch starting within `onset_tolerance` of it. The classifier and the tab prior then read the
found notes at the note model's times, with the labels' strings and frets (placing notes on strings is scored
elsewhere); the tab the prior reads holds the found notes only.

Per class: the share of its labelled notes the note model finds; precision, recall and F1 among the found notes,
at the labels' times and at the note model's; and end to end, where a note the note model misses is a miss.

Usage:
  uv run scripts/techniques/eval_transcribed.py "pattern=d_ex_s*"
"""

import fnmatch

import hydra
import librosa
import mir_eval
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.inference import PianoTranscription, transcribe_cached
from gtp.stage1.string_data import load_cached_index
from gtp.stage1.tagger import TechniqueTagger
from gtp.stage1.techniques import head_labels, joined, scores


def found_pairs(labelled, heard, tolerance):
    """(label row, transcribed note) index pairs: same pitch, onsets within `tolerance`, each note used once."""

    def arrays(onsets, pitches):
        onsets = np.asarray(onsets, dtype=float)
        return np.stack([onsets, onsets + 0.1], axis=1), librosa.midi_to_hz(np.asarray(pitches, dtype=float))

    ref = arrays(labelled['onset'], labelled['pitch'])
    est = arrays([n['start'] for n in heard], [n['pitch'] for n in heard])
    return mir_eval.transcription.match_notes(*ref, *est, onset_tolerance=tolerance, offset_ratio=None)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='eval_transcribed')
def main(cfg):
    device = auto_device()
    runs = [
        r
        for r in sorted((REPO_ROOT / cfg.runs_dir).iterdir())
        if fnmatch.fnmatch(r.name.split('_', 2)[-1], cfg.pattern) and (r / 'model.pt').exists()
    ]
    trained = OmegaConf.load(runs[0] / '.hydra' / 'config.yaml')
    notes, _, _, _ = load_cached_index(trained.index_dir, trained.feature, trained.cache)
    labels = head_labels(notes, trained.heads, trained.unlabelled)
    names = {name: list(head.classes) for name, head in trained.heads.items()}
    joint = {}
    for name, of in cfg.joint.items():
        a, b = of.heads
        pairs = [[(names[a].index(i), names[b].index(j)) for i, j in made] for made in of.classes.values()]
        joint[name] = (a, b, (len(names[a]), len(names[b])), pairs)
        labels[name] = joined(labels[a], labels[b], joint[name][2], pairs)
        names[name] = list(of.classes)
    inside = np.logical_and.reduce([(notes[k] == v).to_numpy() for k, v in cfg.group.items()])
    recs = pd.read_csv(REPO_ROOT / trained.index_dir / 'recordings.csv').set_index('rec')['audio']

    transcriber = PianoTranscription(str(REPO_ROOT / cfg.stage1_checkpoint), device=device)
    taggers = {r.name.split('_', 2)[-1]: TechniqueTagger([r], cfg.prior, cfg.weight, device=device) for r in runs}
    taggers['ensemble'] = TechniqueTagger(runs, cfg.prior, cfg.weight, device=device)
    rows, named = [], {(k, at): [] for k in taggers for at in ('labels', 'heard')}
    for rec, mine in notes[inside].groupby('rec', sort=True):
        path = REPO_ROOT / recs[rec]
        heard = transcribe_cached(transcriber, path, REPO_ROOT / cfg.transcriptions / f'{rec}.json')
        pairs = found_pairs(mine, heard, cfg.onset_tolerance)
        found = mine.iloc[[i for i, _ in pairs]]
        rows.append(pd.DataFrame({'note': mine.index, 'found': mine.index.isin(found.index)}))
        timed = found.assign(onset=[heard[j]['start'] for _, j in pairs], offset=[heard[j]['end'] for _, j in pairs])
        audio, sr = librosa.load(path, sr=None, mono=True)
        for k, tagger in taggers.items():
            for at, given in (('labels', found), ('heard', timed)):
                read = given[['onset', 'offset', 'pitch', 'string', 'fret']]
                answer = {h: p.argmax(axis=1) for h, p in tagger.log_probs(audio, sr, read).items()}
                for name, (a, b, sizes, made) in joint.items():
                    answer[name] = joined(answer[a], answer[b], sizes, made)
                named[k, at].append(pd.DataFrame(answer, index=found.index))
    rows = pd.concat(rows).set_index('note')
    named = {k: pd.concat(v) for k, v in named.items()}
    info(
        f'{len(runs)} runs matching {cfg.pattern}; {len(rows)} labelled notes of {dict(cfg.group)}, {rows["found"].mean():.1%} found'
    )

    table = []
    for head, classes in names.items():
        y = labels[head][rows.index]
        for c, cls in enumerate(classes):
            of_class = y == c
            if not of_class.any():
                continue
            row = {'head': head, 'class': cls, 'notes': int(of_class.sum()), 'found': rows['found'][of_class].mean()}
            for at in ('labels', 'heard'):
                per_run = []
                for k in taggers:
                    answer = named[k, at][head]
                    m = scores(labels[head][answer.index], answer.to_numpy(), classes)[cls]
                    # End to end, a note the note model misses is a miss of its class.
                    recall = m['R'] * row['found']
                    per_run.append((k, m['F1'], 2 * m['P'] * recall / max(m['P'] + recall, 1e-9)))
                alone = [f for k, f, _ in per_run if k != 'ensemble']
                row[f'F1 at {at}'] = f'{np.mean(alone):.3f} ± {np.std(alone, ddof=1) if len(alone) > 1 else 0:.3f}'
                row[f'ensemble at {at}'] = per_run[-1][1]
                if at == 'heard':
                    ends = [e for k, _, e in per_run if k != 'ensemble']
                    row['end to end'] = f'{np.mean(ends):.3f}'
                    row['ensemble end to end'] = per_run[-1][2]
            table.append(row)
    with pd.option_context('display.width', 250, 'display.float_format', '{:.3f}'.format):
        info(f'\n{pd.DataFrame(table).to_string(index=False)}')


if __name__ == '__main__':
    main()

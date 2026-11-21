"""A note index of single-technique exercise recordings, labelled by what the player was asked to play.

EG-IPT is one clip per played technique on one string of an electric guitar; AG-PT-set is a recording per technique,
string and loudness on acoustic guitars, by several players. Neither says which note carries the technique, so:
  * an EG-IPT clip of a hammer-on, a pull-off or a slide holds a picked note and the note it leads to. The note
    model transcribes the clip; a clip where it does not find exactly those two notes is left out;
  * a clip of a bend is one note at the fretted pitch, kept when the note model finds the two pitches the clip's
    name says the bend moves between;
  * a clip of one plain, staccato or palm-muted note is the next fret of its take, so it needs no transcription;
  * an AG-PT-set hammer-on recording is picked notes, each hammered one fret up. Every such pair the note model
    finds is kept and its other notes are left out. Palm-muted and plainly picked notes are labelled by the dataset.
Every recording trains, except those of AG-PT-set's `val_players`.

Writes <out_dir>/recordings.csv and notes.csv.gz in the technique index's format (scripts/techniques/build_index.py
adds them to it), and keeps the transcriptions in <out_dir>/transcriptions.

Usage:
  uv run scripts/techniques/build_exercise_index.py
"""

import re
from collections import Counter
from itertools import pairwise

import hydra
import pandas as pd
import soundfile as sf

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.inference import PianoTranscription, transcribe_cached
from gtp.stage1.string_data import OPEN_STRINGS, first_onset, rel, slug
from gtp.stage1.strings import MAX_FRET

RECORDING_COLUMNS = ['rec', 'dataset', 'audio', 'player', 'guitar', 'chain', 'content', 'split', 'duration']
NOTE_COLUMNS = ['rec', 'onset', 'offset', 'pitch', 'string', 'fret', 'techniques']
EG_IPT_CLIP = re.compile(r'(?P<pickup>[^_]+)_(?P<kind>.+)_(?P<take>\d+)-(?P<part>\d+)_(?P<string>\d)s_')
AG_PT_FILE = re.compile(r'allstring(?P<string>\d)_.*_(?P<player>[A-Za-z]+)\d?_\d{8}$')


class Transcriber:
    """The note model's notes of a recording, in onset order, without the ones shorter than `min_note_s`."""

    def __init__(self, cfg):
        self.model = PianoTranscription(str(REPO_ROOT / cfg.stage1_checkpoint), device=auto_device())
        self.cache, self.min_note_s = REPO_ROOT / cfg.out_dir / 'transcriptions', cfg.min_note_s

    def __call__(self, rec, path):
        notes = transcribe_cached(self.model, path, self.cache / f'{rec}.json')
        return sorted((n for n in notes if n['end'] - n['start'] >= self.min_note_s), key=lambda n: n['start'])


def note(rec, onset, offset, pitch, string, marks=''):
    return (rec, onset, offset, pitch, string, pitch - int(OPEN_STRINGS[string - 1]), marks)


def led_pair(notes, passing_s):
    """The picked note and the note it leads to, once short notes between them are dropped; None if not two."""
    passing = [n for n in notes[1:-1] if n['end'] - n['start'] < passing_s]
    notes = [n for n in notes if n not in passing]
    return notes if len(notes) == 2 else None


def eg_ipt(cfg, transcribe):
    root, c = REPO_ROOT / cfg.eg_ipt.dir, cfg.eg_ipt
    recs, notes, left_out = [], [], Counter()
    named = ((EG_IPT_CLIP.match(p.name), p) for p in sorted(root.glob(f'*/{c.channel}/*/*.wav')))
    clips = pd.DataFrame([{**m.groupdict(), 'path': p} for m, p in named if m]).astype({'part': int, 'string': int})
    clips['fret'] = clips.groupby(['pickup', 'kind', 'take'])['part'].rank().astype(int) - 1
    for clip in clips.itertuples():
        rec = f'eg_ipt/{slug(clip.pickup)}/{clip.kind}_{clip.take}-{clip.part:02d}_{clip.string}s'
        if clip.kind in c.single:
            onset, duration = first_onset(clip.path)
            pitch = int(OPEN_STRINGS[clip.string - 1]) + clip.fret
            found = [note(rec, onset, duration, pitch, clip.string, c.single[clip.kind])]
        elif clip.kind in c.led:
            pair = led_pair(transcribe(rec, clip.path), c.passing_s)
            if pair is None or pair[1]['pitch'] - pair[0]['pitch'] not in c.led[clip.kind].steps:
                left_out[clip.kind] += 1
                continue
            found = [
                note(rec, n['start'], n['end'], n['pitch'], clip.string, mark)
                for n, mark in zip(pair, ('', c.led[clip.kind].mark), strict=True)
            ]
        elif clip.kind in c.bends:
            heard = transcribe(rec, clip.path)
            if len(heard) != 2 or heard[1]['pitch'] - heard[0]['pitch'] != c.bends[clip.kind]:
                left_out[clip.kind] += 1
                continue
            fretted = min(n['pitch'] for n in heard)
            found = [note(rec, heard[0]['start'], heard[1]['end'], fretted, clip.string, 'bend')]
        else:
            continue
        duration = sf.info(str(clip.path)).duration
        recs.append((rec, 'eg_ipt', rel(clip.path), 'eg-ipt', slug(clip.pickup), 'di', clip.kind, 'train', duration))
        notes += found
    info(f'eg_ipt: {len(recs)} clips kept; left out {dict(left_out)}')
    return recs, notes


def hammered_pairs(heard, within_s):
    """(picked, hammered) pairs: a note one fret above the note before it, starting within `within_s` of it."""
    pairs = []
    for a, b in pairwise(heard):
        led = bool(pairs) and pairs[-1][1] is a
        if not led and b['pitch'] == a['pitch'] + 1 and b['start'] - a['start'] <= within_s:
            pairs.append((a, b))
    return pairs


def ag_pt(cfg, transcribe):
    root, c = REPO_ROOT / cfg.ag_pt.dir, cfg.ag_pt
    files = pd.read_csv(root / 'metadata' / 'files.csv').set_index('filename')
    names = pd.read_csv(root / 'metadata' / 'expressive_techniques.csv').set_index('expressive_technique_id')
    labels = pd.read_csv(root / 'metadata' / 'note_labels.csv')
    labels['technique'] = labels['expressive_technique_id'].map(names['name_nospace'])
    labels = labels[labels['technique'].isin(list(c.labelled))]
    labels = labels.assign(pitch=pd.to_numeric(labels['pitch_midi'], errors='coerce')).dropna(subset=['pitch'])
    recs, notes = [], []

    def recording(filename, content):
        named = AG_PT_FILE.search(filename.removesuffix('.wav'))
        player = named['player'].lower()
        rec = f'ag_pt/{filename.removesuffix(".wav")}'
        split = 'val' if player in c.val_players else 'train'
        row = files.loc[filename]
        audio = rel(root / 'data' / 'audio' / filename)
        recs.append((rec, 'ag_pt', audio, player, f'ag-pt-{row.guitar_id}', 'di', content, split, row.duration))
        return rec, int(named['string'])

    # Its improvisations mix techniques with hits on the body, which are not in the index.
    labels = labels[labels['audio_file_path'].str.contains('_allstring')]
    for filename, of_file in labels.groupby('audio_file_path', sort=True):
        of_file = of_file.sort_values('onset_label_seconds')
        rec, _ = recording(filename, slug(of_file['technique'].iloc[0]))
        onsets = of_file['onset_label_seconds'].to_numpy()
        ends = [*onsets[1:], files.loc[filename].duration]
        for n, end in zip(of_file.itertuples(), ends, strict=True):
            offset = min(end, n.onset_label_seconds + c.max_note_s)
            notes.append(
                note(rec, n.onset_label_seconds, offset, int(n.pitch), int(n.string_number), c.labelled[n.technique])
            )
    hammered = 0
    for filename in sorted(f for f in files.index if '_hammeron_' in f):
        rec, string = recording(filename, 'hammer-on')
        for picked, led in hammered_pairs(transcribe(rec, root / 'data' / 'audio' / filename), c.hammer_within_s):
            notes.append(note(rec, picked['start'], picked['end'], picked['pitch'], string))
            notes.append(note(rec, led['start'], led['end'], led['pitch'], string, 'hammered'))
            hammered += 1
    info(f'ag_pt: {len(recs)} recordings, {hammered} hammer-ons found')
    return recs, notes


LOADERS = {'eg_ipt': eg_ipt, 'ag_pt': ag_pt}


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='exercise_index')
def main(cfg):
    transcribe = Transcriber(cfg)
    recs, notes = [], []
    for name in cfg.datasets:
        r, n = LOADERS[name](cfg, transcribe)
        recs += r
        notes += n
    recs = pd.DataFrame(recs, columns=RECORDING_COLUMNS)
    notes = pd.DataFrame(notes, columns=NOTE_COLUMNS)
    notes = notes[notes['fret'].between(0, MAX_FRET)]
    recs = recs[recs['rec'].isin(notes['rec'])]
    out = REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out / 'recordings.csv', index=False)
    notes.to_csv(out / 'notes.csv.gz', index=False)
    of = notes.merge(recs[['rec', 'dataset', 'player', 'split']], on='rec').replace({'techniques': {'': '(plain)'}})
    table = pd.crosstab([of['dataset'], of['split'], of['player']], of['techniques'])
    info(f'{len(recs)} recordings, {len(notes)} notes -> {out}\n{table}')


if __name__ == '__main__':
    main()

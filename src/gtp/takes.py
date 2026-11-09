"""Home-recording takes: the audio, the model's prediction and the user's corrected tab, one folder per take.

  data/takes/<take_id>/
    audio.wav        the recording as captured, at its original rate
    prediction.json  the pipeline's output; written once, never modified
    edited.json      the user's corrected tab; absent until first saved from the editor
    meta.json        recording metadata (guitar, mic, room, ...), edited by the user

Every note is {id, pitch, start, end, velocity, string, fret}, strings 1 (high e) .. 6 (low E) and
fret = pitch - tuning[string - 1] (tuning includes the capo). Prediction notes add `evidence`, the
string classifier's 6-way distribution (or null); edited notes add `source_id`, the id of the
prediction note they came from (null for notes the user added). Deleted notes are simply absent.
Only takes whose edited.json has status "verified" count as ground truth.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import soundfile as sf

from gtp.stage2.tokenizer import MAX_FRET

SCHEMA_VERSION = 1
N_STRINGS = 6
STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E']
STATUSES = ('draft', 'verified')
TAKE_ID = re.compile(r'^\d{8}_\d{6}(_[A-Za-z0-9-]+)?$')
META_FIELDS = (
    'guitar',
    'strings',
    'pickup',
    'mic',
    'input_device',
    'room',
    'player',
    'tuning_name',
    'capo',
    'style',
    'notes',
    'source',
)
# An onset edit smaller than this is float noise, not a correction.
ONSET_EPS = 1e-3


def now():
    return datetime.now().isoformat(timespec='seconds')


def new_take_id(slug=None, when=None):
    stamp = (when or datetime.now()).strftime('%Y%m%d_%H%M%S')
    slug = re.sub(r'[^A-Za-z0-9-]+', '-', slug or '').strip('-')
    return f'{stamp}_{slug}' if slug else stamp


def take_dir(root, take_id):
    if not TAKE_ID.match(take_id):
        raise ValueError(f'bad take id {take_id!r}')
    return Path(root) / take_id


def read_json(path):
    return json.loads(Path(path).read_text())


def dumps(obj):
    """JSON with one line per note, so take files diff and grep note by note."""

    def value(v):
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return '[\n' + ',\n'.join(f'  {json.dumps(x)}' for x in v) + '\n ]'
        return json.dumps(v, indent=1).replace('\n', '\n ') if isinstance(v, dict) else json.dumps(v)

    return '{\n' + ',\n'.join(f' {json.dumps(k)}: {value(v)}' for k, v in obj.items()) + '\n}\n'


def write_json(path, obj):
    tmp = Path(f'{path}.tmp')
    tmp.write_text(dumps(obj))
    os.replace(tmp, path)


def validate_notes(notes, tuning, prediction_ids=None):
    """Raise ValueError unless `notes` are well formed; `prediction_ids` given means edited notes."""
    if len(tuning) != N_STRINGS:
        raise ValueError(f'tuning needs {N_STRINGS} strings, got {tuning}')
    ids, sources = set(), set()
    for n in notes:
        if not isinstance(n.get('id'), int) or n['id'] in ids:
            raise ValueError(f'note ids must be unique integers: {n}')
        ids.add(n['id'])
        if not (isinstance(n.get('string'), int) and 1 <= n['string'] <= N_STRINGS):
            raise ValueError(f'string must be 1..{N_STRINGS}: {n}')
        if not (isinstance(n.get('fret'), int) and 0 <= n['fret'] <= MAX_FRET):
            raise ValueError(f'fret must be 0..{MAX_FRET}: {n}')
        if n.get('pitch') != tuning[n['string'] - 1] + n['fret']:
            raise ValueError(f'pitch must equal tuning[string - 1] + fret: {n}')
        if not 0 <= n['start'] <= n['end']:
            raise ValueError(f'need 0 <= start <= end: {n}')
        if prediction_ids is not None:
            source = n.get('source_id', 'missing')
            if source is not None and (source not in prediction_ids or source in sources):
                raise ValueError(f'source_id must be null or a distinct prediction note id: {n}')
            sources.add(source)
        elif n.get('evidence') is not None and len(n['evidence']) != N_STRINGS:
            raise ValueError(f'evidence must be {N_STRINGS} probabilities or null: {n}')


def ordered(notes):
    return sorted(notes, key=lambda n: (n['start'], n['string']))


def prediction_doc(take_id, notes, tuning, capo, tempo, pipeline, unplaced=()):
    """`notes`: dicts with pitch/start/end/velocity/string/fret/evidence; ids follow time order.

    `unplaced`: Stage 1 notes Stage 2 could not put on a string, kept for the record only.
    """
    rounded = [
        {
            'pitch': int(n['pitch']),
            'start': round(float(n['start']), 4),
            'end': round(float(n['end']), 4),
            'velocity': None if n.get('velocity') is None else int(n['velocity']),
            'string': int(n['string']),
            'fret': int(n['fret']),
            'evidence': None if n.get('evidence') is None else [round(float(p), 4) for p in n['evidence']],
        }
        for n in notes
    ]
    doc = {
        'schema_version': SCHEMA_VERSION,
        'take_id': take_id,
        'created': now(),
        'pipeline': pipeline,
        'tuning': [int(t) for t in tuning],
        'capo': int(capo),
        'tempo': tempo,
        'notes': [{'id': i, **n} for i, n in enumerate(sorted(rounded, key=lambda n: (n['start'], n['pitch'])))],
        'unplaced': [
            {k: n[k] for k in ('pitch', 'start', 'end', 'velocity')} for n in sorted(unplaced, key=lambda n: n['start'])
        ],
    }
    validate_notes(doc['notes'], doc['tuning'])
    return doc


def create(root, take_id, audio, sr, prediction, meta):
    """Write a new take folder; refuses to touch an existing one."""
    d = take_dir(root, take_id)
    d.mkdir(parents=True, exist_ok=False)
    sf.write(str(d / 'audio.wav'), audio, sr)
    write_json(d / 'prediction.json', prediction)
    write_json(d / 'meta.json', {**dict.fromkeys(META_FIELDS, ''), **meta})
    return d


def take_ids(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted((p.name for p in root.iterdir() if (p / 'prediction.json').exists()), reverse=True)


def edited_from_prediction(prediction):
    return {
        'schema_version': SCHEMA_VERSION,
        'take_id': prediction['take_id'],
        'created': None,
        'updated': None,
        'status': 'draft',
        'tuning': prediction['tuning'],
        'capo': prediction['capo'],
        'tempo': prediction['tempo'],
        'notes': [
            {**{k: n[k] for k in ('id', 'pitch', 'start', 'end', 'velocity', 'string', 'fret')}, 'source_id': n['id']}
            for n in prediction['notes']
        ],
    }


def load(d):
    """{'prediction', 'edited' (None until saved), 'meta'} of the take folder `d`."""
    d = Path(d)
    edited = d / 'edited.json'
    return {
        'prediction': read_json(d / 'prediction.json'),
        'edited': read_json(edited) if edited.exists() else None,
        'meta': read_json(d / 'meta.json') if (d / 'meta.json').exists() else dict.fromkeys(META_FIELDS, ''),
    }


class Conflict(Exception):
    pass


def save_edited(d, notes, status, base_updated=None):
    """Validate and write edited.json; `base_updated` must match the version on disk (None: none yet)."""
    take = load(d)
    prediction, current = take['prediction'], take['edited']
    if (current and current['updated']) != base_updated:
        raise Conflict(f'edited.json changed on disk (updated {current and current["updated"]})')
    if status not in STATUSES:
        raise ValueError(f'status must be one of {STATUSES}')
    keys = ('id', 'source_id', 'pitch', 'start', 'end', 'velocity', 'string', 'fret')
    notes = ordered([{k: n.get(k) for k in keys} for n in notes])
    for n in notes:
        n['start'], n['end'] = round(float(n['start']), 4), round(float(n['end']), 4)
    validate_notes(notes, prediction['tuning'], prediction_ids={n['id'] for n in prediction['notes']})
    stamp = now()
    edited = {
        **edited_from_prediction(prediction),
        'created': (current or {}).get('created') or stamp,
        'updated': stamp,
        'status': status,
        'notes': notes,
    }
    write_json(Path(d) / 'edited.json', edited)
    return edited


def save_meta(d, meta):
    meta = {k: str(meta.get(k, '')) for k in META_FIELDS}
    write_json(Path(d) / 'meta.json', meta)
    return meta


# ---------------------------------------------------------------------------
# Prediction vs edited
# ---------------------------------------------------------------------------


def compare(prediction, edited):
    """One row per prediction note (kept or deleted) and per added note, aligned through source_id."""
    by_source = {n['source_id']: n for n in edited['notes'] if n['source_id'] is not None}
    pairs = [(p, by_source.get(p['id'])) for p in prediction['notes']]
    pairs += [(None, e) for e in edited['notes'] if e['source_id'] is None]
    rows = []
    for p, e in pairs:
        row = {'kind': 'added' if p is None else 'deleted' if e is None else 'kept'}
        for prefix, n in (('pred', p), ('edit', e)):
            for k in ('id', 'pitch', 'string', 'fret', 'start', 'end'):
                row[f'{prefix}_{k}'] = n[k] if n else None
        row['velocity'] = (p or e)['velocity']
        ev = p and p['evidence']
        for s in range(N_STRINGS):
            row[f'ev_{STRING_NAMES[s]}{s + 1}'] = ev[s] if ev else None
        row['ev_string'] = int(np.argmax(ev)) + 1 if ev else None
        row['ev_conf'] = max(ev) if ev else None
        kept = row['kind'] == 'kept'
        row['pitch_changed'] = kept and p['pitch'] != e['pitch']
        row['string_changed'] = kept and p['string'] != e['string']
        row['fret_changed'] = kept and p['fret'] != e['fret']
        row['onset_shift'] = round(e['start'] - p['start'], 4) if kept else None
        row['onset_moved'] = kept and abs(e['start'] - p['start']) > ONSET_EPS
        row['changed'] = not kept or row['pitch_changed'] or row['string_changed'] or row['onset_moved']
        rows.append(row)
    return rows


def match(ref, est, tolerance):
    """[(ref_idx, est_idx)] pairs: same pitch, onsets within `tolerance`, offsets ignored (as scripts/e2e)."""
    if not ref or not est:
        return []

    def arrays(notes):
        intervals = np.array([[n['start'], max(n['end'], n['start'] + 1e-3)] for n in notes])
        return intervals, librosa.midi_to_hz(np.array([n['pitch'] for n in notes]))

    return mir_eval.transcription.match_notes(*arrays(ref), *arrays(est), onset_tolerance=tolerance, offset_ratio=None)


def prf(tp, n_est, n_ref):
    p, r = tp / max(n_est, 1), tp / max(n_ref, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


def counts(rows):
    """Edit counts of compare() rows; string_acc is over kept notes whose pitch was right."""
    kept = [r for r in rows if r['kind'] == 'kept']
    same_pitch = [r for r in kept if not r['pitch_changed']]
    return {
        'pred': sum(r['kind'] != 'added' for r in rows),
        'edited': sum(r['kind'] != 'deleted' for r in rows),
        'kept': len(kept),
        'deleted': sum(r['kind'] == 'deleted' for r in rows),
        'added': sum(r['kind'] == 'added' for r in rows),
        'pitch_changed': sum(r['pitch_changed'] for r in kept),
        'string_changed': sum(r['string_changed'] for r in kept),
        'onset_moved': sum(r['onset_moved'] for r in kept),
        'changed': sum(r['changed'] for r in rows),
        'string_ok': sum(not r['string_changed'] for r in same_pitch),
        'string_scored': len(same_pitch),
    }


def scores(prediction, edited, tolerance=0.05):
    """Stage 1 note P/R/F1 and tab P/R/F1 (matched and on the same string) with the edit as reference."""
    ref, est = edited['notes'], prediction['notes']
    pairs = match(ref, est, tolerance)
    tab_tp = sum(ref[r]['string'] == est[e]['string'] for r, e in pairs)
    out = {'matched': len(pairs), 'tab_tp': tab_tp}
    out['note_p'], out['note_r'], out['note_f1'] = prf(len(pairs), len(est), len(ref))
    out['tab_p'], out['tab_r'], out['tab_f1'] = prf(tab_tp, len(est), len(ref))
    return out

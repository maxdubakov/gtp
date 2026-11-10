"""Real guitar recordings with per-note string labels, from several datasets, in one table.

Every loader returns (recordings, notes) DataFrames:
  recordings: rec, dataset, audio (repo-relative), player, guitar, chain, content, split
  notes:      rec, onset, offset, pitch, string, fret

String 1 = high e (MIDI 64) ... 6 = low E (MIDI 40), as in strings.py; each dataset's own numbering is
converted on load. Splits go by player or guitar, never by clip, so no recording of a test player/guitar
is ever trained on.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import jams
import numpy as np
import pandas as pd
import pretty_midi

from gtp import REPO_ROOT
from gtp.stage1.strings import MAX_FRET, cache_layout

DATA = Path(REPO_ROOT) / 'data'
INDEX_DIR = DATA / 'string_index'
OPEN_STRINGS = np.array([64, 59, 55, 50, 45, 40])
REC_COLUMNS = ['rec', 'dataset', 'audio', 'player', 'guitar', 'chain', 'content', 'split']
NOTE_COLUMNS = ['rec', 'onset', 'offset', 'pitch', 'string']


DATASET_INFO = {
    'guitarset': {
        'name': 'GuitarSet',
        'url': 'https://zenodo.org/records/3371780',
        'license': 'CC BY 4.0',
        'about': 'Acoustic guitar, 6 players, mic recording; string labels from a hexaphonic pickup.',
    },
    'guitar_techs': {
        'name': 'Guitar-TECHS',
        'url': 'https://zenodo.org/records/14963133',
        'license': 'CC BY 4.0',
        'about': "Electric guitar, 3 players each with own guitar/amp/mic, DI and mic'd amp; "
        'labels from a Fishman Triple Play MIDI pickup.',
    },
    'egdb': {
        'name': 'EGDB',
        'url': 'https://ss12f32v.github.io/Guitar-Transcription',
        'license': 'not stated',
        'about': 'Electric guitar, 1 player, 1 Strat with hexaphonic pickup, DI; 240 tab performances.',
    },
    'egset12': {
        'name': 'EGSet12',
        'url': 'https://zenodo.org/records/11406378',
        'license': 'CC BY 4.0',
        'about': '12 original solo pieces, Sire T7 through a Yamaha amp, mic at 15 cm; GuitarSet-style JAMS.',
    },
    'idmt': {
        'name': 'IDMT-SMT-Guitar',
        'url': 'https://zenodo.org/records/7544110',
        'license': 'CC BY-NC-ND 4.0',
        'about': 'Electric guitars, DI; isolated notes, licks and short pieces with string/fret per note.',
    },
    'egfxset': {
        'name': 'EGFxSet',
        'url': 'https://zenodo.org/records/7044411',
        'license': 'CC BY 4.0',
        'about': 'Every string/fret on one Strat, 5 pickup positions, clean and through real pedals/reverbs.',
    },
}


def rel(path):
    return str(Path(path).relative_to(REPO_ROOT))


def frame(recs, notes):
    recs = pd.DataFrame(recs, columns=REC_COLUMNS)
    notes = pd.DataFrame(notes, columns=NOTE_COLUMNS).astype(
        {'onset': float, 'offset': float, 'pitch': int, 'string': int}
    )
    notes['fret'] = notes['pitch'] - OPEN_STRINGS[notes['string'] - 1]
    return recs, notes


def guitarset():
    root = DATA / 'guitarset'
    split = {'00': 'train', '01': 'train', '02': 'train', '03': 'train', '04': 'val', '05': 'test'}
    recs, notes = [], []
    for path in sorted((root / 'processed').glob('*.json')):
        rec = f'guitarset/{path.stem}'
        player = path.stem[:2]
        content = 'solo' if path.stem.endswith('_solo') else 'comp'
        audio = root / 'audio_mono-mic' / f'{path.stem}_mic.wav'
        recs.append(
            (rec, 'guitarset', rel(audio), f'gs{player}', f'gs{player}', 'acoustic-mic', content, split[player])
        )
        notes += [(rec, n['start'], n['end'], n['pitch'], n['string']) for n in json.loads(path.read_text())['notes']]
    return frame(recs, notes)


# Each Guitar-TECHS player brought one guitar, amp and mic, so player = guitar = chain.
TECHS_GUITARS = {'P1': 'ibanez-pf300', 'P2': 'evh-wolfgang', 'P3': 'sire-t7'}
TECHS_SPLIT = {'P1': 'train', 'P2': 'val', 'P3': 'test'}
TECHS_TRACKS = {'e': 1, 'B': 2, 'G': 3, 'D': 4, 'A': 5, 'E': 6}


def midi_notes(path, string_of_track):
    pm = pretty_midi.PrettyMIDI(str(path))
    return [
        (n.start, n.end, n.pitch, string_of_track[inst.name.strip()])
        for inst in pm.instruments
        if inst.name.strip() in string_of_track
        for n in inst.notes
    ]


def guitar_techs():
    root = DATA / 'guitar_techs'
    recs, notes = [], []
    for midi in sorted(root.glob('P*_*/midi/*.mid')):
        part = midi.parts[-3]
        player, content = part.split('_', 1)
        stem = midi.stem.removeprefix('midi_')
        labelled = midi_notes(midi, TECHS_TRACKS)
        for chain, sub in (('di', 'directinput'), ('amp-mic', 'micamp')):
            audio = root / part / 'audio' / sub / f'{sub}_{stem}.wav'
            if not audio.exists():
                continue
            rec = f'guitar_techs/{part}/{chain}/{stem}'
            recs.append(
                (rec, 'guitar_techs', rel(audio), player, TECHS_GUITARS[player], chain, content, TECHS_SPLIT[player])
            )
            notes += [(rec, *n) for n in labelled]
    return frame(recs, notes)


def channel_notes(path):
    """(start, end, pitch, channel + 1) of every note of a MIDI file: EGDB puts each string on its own channel."""
    import mido

    notes, sounding, t = [], {}, 0.0
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type not in ('note_on', 'note_off'):
            continue
        key = (msg.channel, msg.note)
        if key in sounding:
            notes.append((sounding.pop(key), t, msg.note, msg.channel + 1))
        if msg.type == 'note_on' and msg.velocity > 0:
            sounding[key] = t
    return notes


def egdb():
    """One player, one guitar, never trained on: a test set. A note's string is its MIDI channel (0 = high e); only
    some files also name their tracks after the strings."""
    root = DATA / 'egdb' / 'raw'
    recs, notes = [], []
    for midi in sorted((root / 'audio_label').glob('*.midi'), key=lambda p: int(p.stem)):
        audio = root / 'audio_DI' / f'{midi.stem}.wav'
        if not audio.exists():
            continue
        rec = f'egdb/{midi.stem}'
        recs.append((rec, 'egdb', rel(audio), 'egdb', 'egdb-strat', 'di', 'pieces', 'test'))
        notes += [(rec, *n) for n in channel_notes(midi) if 1 <= n[3] <= 6]
    return frame(recs, notes)


def egset12():
    """Same player and rig as Guitar-TECHS P3, so it shares P3's test split. JAMS data_source 0 = low E."""
    root = DATA / 'egset12' / 'raw'
    recs, notes = [], []
    for path in sorted(root.glob('*.jams')):
        rec = f'egset12/{path.stem}'
        recs.append((rec, 'egset12', rel(root / f'{path.stem}.wav'), 'P3', 'sire-t7', 'amp-mic', 'pieces', 'test'))
        for ann in jams.load(str(path)).search(namespace='note_midi'):
            string = 6 - int(ann.annotation_metadata.data_source)
            notes += [(rec, o.time, o.time + o.duration, round(o.value), string) for o in ann.data]
    return frame(recs, notes)


IDMT_ROOT = DATA / 'idmt_smt_guitar' / 'IDMT-SMT-GUITAR_V2'
IDMT_VAL_GUITAR = 'gibson-les-paul'
# Dead notes have no pitch; bends, slides and harmonics don't sound at the fretted pitch.
IDMT_UNFRETTED = {'DN', 'BE', 'SL', 'HA'}


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def idmt():
    """IDMT-SMT-Guitar datasets 1-3, DI; stringNumber 1 = low E. Dataset 4 has onsets only.

    Dataset 2 is one player on three guitars; its Les Paul is held out as val. Dataset 1 (two players,
    not identified per file) and dataset 3 train.
    """
    recs, notes = [], []
    for xml in sorted(IDMT_ROOT.glob('dataset[123]/**/annotation/*.xml')):
        audio = xml.parent.parent / 'audio' / f'{xml.stem}.wav'
        if not audio.exists():
            continue
        g = ET.parse(xml).getroot()
        params = g.find('globalParameter')
        subset = xml.relative_to(IDMT_ROOT).parts[0]
        if subset == 'dataset1':
            folder = xml.parent.parent.name
            guitar = 'fender-strat' if folder.startswith('Fender') else 'ibanez-power-strat'
            player, content = 'idmt-ds1', 'chords' if folder.endswith('Chords') else 'notes'
            rec = f'idmt/{subset}/{slug(folder)}/{xml.stem}'
        else:
            guitar = slug(params.findtext('instrumentModel'))
            player = slug(params.findtext('recordingArtist'))
            content = 'licks' if subset == 'dataset2' else 'pieces'
            rec = f'idmt/{subset}/{xml.stem}'
        split = 'val' if guitar == IDMT_VAL_GUITAR else 'train'
        recs.append((rec, 'idmt', rel(audio), player, guitar, 'di', content, split))
        for e in g.iter('event'):
            if e.findtext('expressionStyle') in IDMT_UNFRETTED:
                continue
            onset = float(e.findtext('onsetSec'))
            offset = float(e.findtext('offsetSec') or onset + 0.5)
            notes.append((rec, onset, offset, int(e.findtext('pitch')), 7 - int(e.findtext('stringNumber'))))
    return frame(recs, notes)


EGFX_ROOT = DATA / 'egfxset'
EGFX_CLEAN = 'Clean'
# Datasets whose onsets are read off their own audio: aligned by construction, so never shifted. The onset
# envelope can't check them anyway, as librosa zero-pads its first ~50 ms and these notes start at ~4 ms.
AUDIO_ONSETS = {'egfxset'}


def first_onset(path, threshold=0.1):
    """Time where a single-note file first reaches `threshold` of its peak; alignment refines it."""
    import soundfile as sf

    y, sr = sf.read(path, dtype='float32', always_2d=True)
    env = np.abs(y).max(axis=1)
    return int(np.argmax(env >= threshold * env.max())) / sr, len(y) / sr


def egfxset():
    """One Strat, one player: training only. Files are <effect>/<pickup>/<string>-<fret>.wav, 1 = high e;
    the onset isn't labelled, so it is read off the audio."""
    recs, notes = [], []
    for path in sorted(EGFX_ROOT.glob('*/*/*-*.wav')):
        effect, pickup = path.parts[-3], path.parts[-2]
        string, fret = (int(x) for x in path.stem.split('-'))
        rec = f'egfxset/{slug(effect)}/{slug(pickup)}/{path.stem}'
        chain = 'di' if effect == EGFX_CLEAN else 'fx'
        recs.append((rec, 'egfxset', rel(path), 'egfxset', 'egfx-stratocaster', chain, slug(effect), 'train'))
        onset, duration = first_onset(path)
        notes.append((rec, onset, duration, int(OPEN_STRINGS[string - 1] + fret), string))
    return frame(recs, notes)


MIN_DURATION = 0.05
RETRIGGER = 0.05
# Their sounding pitch is not the fretted one, so the string label would teach the wrong timbre.
UNFRETTED = re.compile(r'Bendings|Harmonics|PinchHarmonics', re.I)


def clean(recs, notes):
    """Drop labels a real pluck can't produce; returns kept notes and {rule: dropped count} per dataset.

    Rules apply to every dataset; on GuitarSet they remove a handful of notes, on MIDI-pickup labels
    (Guitar-TECHS) they remove the pickup's retriggers and glitches.
    """
    dataset = notes['rec'].map(recs.set_index('rec')['dataset'])
    ordered = notes.sort_values(['rec', 'string', 'onset'])
    gap = ordered.groupby(['rec', 'string'])['onset'].diff().reindex(notes.index)
    rules = {
        'unfretted technique': notes['rec'].str.contains(UNFRETTED),
        'impossible fret': (notes['fret'] < 0) | (notes['fret'] > MAX_FRET),
        f'shorter than {MIN_DURATION * 1000:.0f} ms': (notes['offset'] - notes['onset']) < MIN_DURATION,
        f'same string re-plucked within {RETRIGGER * 1000:.0f} ms': gap < RETRIGGER,
    }
    drop = pd.Series(False, index=notes.index)
    report = {}
    for name, mask in rules.items():
        new = mask & ~drop
        report[name] = new.groupby(dataset).sum().to_dict()
        drop |= mask
    return notes[~drop].reset_index(drop=True), report


ALIGN_SR = 22050
ALIGN_HOP = 128
# Guitar-TECHS documents up to 100 ms of misalignment; a wider search locks onto beat-shifted matches.
MAX_LAG = 0.1
NEAR = 0.1


def onset_envelope(audio):
    import librosa

    env = librosa.onset.onset_strength(y=audio, sr=ALIGN_SR, hop_length=ALIGN_HOP)
    return env / (env.max() + 1e-9)


def alignment(audio, onsets):
    """Global label lag (s, positive = labels late) from the onset envelope, and each note's distance to
    the nearest detected onset after removing that lag (NaN when none within NEAR)."""
    import librosa

    env = onset_envelope(audio)
    fps = ALIGN_SR / ALIGN_HOP
    frames = np.round(np.asarray(onsets) * fps).astype(int)
    frames = frames[(frames >= 0) & (frames < len(env))]
    max_lag = int(MAX_LAG * fps)
    lags = np.arange(-max_lag, max_lag + 1)
    score = [env[np.clip(frames - lag, 0, len(env) - 1)].sum() for lag in lags]
    lag = lags[int(np.argmax(score))] / fps
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=ALIGN_SR, hop_length=ALIGN_HOP, units='time')
    shifted = np.asarray(onsets) - lag
    if len(peaks) == 0:
        return lag, np.full(len(shifted), np.nan)
    idx = np.clip(np.searchsorted(peaks, shifted), 1, len(peaks) - 1)
    nearest = np.where(np.abs(peaks[idx - 1] - shifted) < np.abs(peaks[idx] - shifted), peaks[idx - 1], peaks[idx])
    dist = nearest - shifted
    return lag, np.where(np.abs(dist) <= NEAR, dist, np.nan)


ROOM_CHAINS = ('ego', 'exo')
ROOM_MAX_LAG = 3.0


def room_lag(job):
    """Seconds a room recording lags its direct recording (the peak of the cross-correlation of their onset
    envelopes), the correlation there, and the room audio's length."""
    import librosa
    import scipy.signal

    direct, room = job
    a, _ = librosa.load(REPO_ROOT / direct, sr=ALIGN_SR, mono=True)
    b, _ = librosa.load(REPO_ROOT / room, sr=ALIGN_SR, mono=True)
    ea, eb = onset_envelope(a), onset_envelope(b)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    corr = scipy.signal.correlate(eb, ea, mode='full')
    lags = scipy.signal.correlation_lags(n, n, mode='full')
    near = np.abs(lags) <= round(ROOM_MAX_LAG * ALIGN_SR / ALIGN_HOP)
    k = np.argmax(corr[near])
    peak = corr[near][k] / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-12)
    return lags[near][k] * ALIGN_HOP / ALIGN_SR, float(peak), len(b) / ALIGN_SR


def room_recordings(recs, notes, mics, workers):
    """Guitar-TECHS's room recordings of the indexed performances: the audio of its head-mounted camera (`ego`)
    and of a camera about 5 feet in front of the player (`exo`), as chains of their own.

    Each takes the (already aligned) notes of its direct-input recording, moved by the lag between the two files.
    """
    from concurrent.futures import ProcessPoolExecutor

    rows = []
    for r in recs[(recs['dataset'] == 'guitar_techs') & (recs['chain'] == 'di')].to_dict('records'):
        _, part, _, stem = r['rec'].split('/')
        for mic in mics:
            audio = f'data/guitar_techs/{part}/video/{mic}/{mic}_{stem}.mp3'
            if (REPO_ROOT / audio).exists():
                rows.append(
                    {k: r[k] for k in REC_COLUMNS}
                    | {'rec': f'guitar_techs/{part}/{mic}/{stem}', 'audio': audio, 'chain': mic}
                    | {'direct': r['rec'], 'direct_audio': r['audio']}
                )
    room = pd.DataFrame(rows)
    with ProcessPoolExecutor(workers) as pool:
        lags = list(pool.map(room_lag, room[['direct_audio', 'audio']].itertuples(index=False, name=None), chunksize=4))
    lag, room['room_corr'], room['duration'] = zip(*lags, strict=True)
    room['room_lag_ms'] = np.round(1000 * np.array(lag), 1)
    room['lag_source'] = 'direct recording'
    by_rec = dict(tuple(notes.groupby('rec')))
    moved = []
    for r in room.itertuples():
        n = by_rec[r.direct].assign(rec=r.rec)
        n[['onset', 'offset']] += r.room_lag_ms / 1000
        moved.append(n[(n['onset'] >= 0) & (n['onset'] < r.duration)])
    return room.drop(columns=['direct', 'direct_audio']), pd.concat(moved, ignore_index=True)


LOADERS = {
    'guitarset': guitarset,
    'guitar_techs': guitar_techs,
    'egdb': egdb,
    'egset12': egset12,
    'idmt': idmt,
    'egfxset': egfxset,
}


def build(datasets=tuple(LOADERS)):
    parts = [LOADERS[d]() for d in datasets]
    recs = pd.concat([r for r, _ in parts], ignore_index=True)
    notes = pd.concat([n for _, n in parts], ignore_index=True).sort_values(['rec', 'onset', 'string'])
    return recs, notes.reset_index(drop=True)


def drop_empty(recs, notes):
    return recs[recs['rec'].isin(notes['rec'])].reset_index(drop=True)


def load_index():
    return pd.read_csv(INDEX_DIR / 'recordings.csv'), pd.read_csv(INDEX_DIR / 'notes.csv.gz')


def load_cached_index(index_dir, feature):
    """Notes joined with their recordings, plus the feature cache: its row table, windows (memmap) and metadata."""
    index = REPO_ROOT / index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    notes = pd.read_csv(index / 'notes.csv.gz').join(
        recs.set_index('rec')[['dataset', 'player', 'guitar', 'chain', 'content', 'split']], on='rec'
    )
    cache = index / 'cache'
    meta = json.loads((cache / f'{feature}.json').read_text())
    rows = pd.read_csv(cache / f'{feature}_rows.csv.gz')
    if meta['layout'] != cache_layout() or rows['note'].max() >= len(notes):
        raise ValueError(f'{cache} is stale against the index or the layout: rebuild it')
    x = np.load(cache / f'{feature}.npy', mmap_mode='r')
    return notes, rows, x, meta

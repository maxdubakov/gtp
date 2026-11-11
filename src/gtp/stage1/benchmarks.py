"""Note-transcription benchmarks with the labels each dataset ships, as published Stage 1 work scores them.

A benchmark is a list of `Recording(rec, audio, notes)`; `audio` is a path, and `notes` an (n, 3) array of onset,
offset and MIDI pitch. The noisy variants follow TART (Gupta et al. 2026): a random EchoThief room response, an 80 Hz
zero-phase high-pass, a random mix of white, pink and 60 Hz hum noise at 25 to 45 dB SNR, peak at 0.9. They are
rendered once, deterministically per recording, into `cache_dir`.
"""

import csv
import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pretty_midi
import scipy.signal
import soundfile as sf

from gtp import REPO_ROOT
from gtp.stage1.data import load_gaps_notes, load_guitarset_notes
from gtp.stage1.string_data import channel_notes

DATA = REPO_ROOT / 'data'
SR = 16000


@dataclass(frozen=True)
class Recording:
    rec: str
    audio: Path
    notes: np.ndarray


def note_array(notes):
    return np.array([[n['onset_time'], n['offset_time'], n['midi_note']] for n in notes], float).reshape(-1, 3)


def midi_notes(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    return np.array([[n.start, n.end, n.pitch] for i in pm.instruments for n in i.notes], float).reshape(-1, 3)


def guitarset(test_only=False, audio='audio_mono-mic', suffix='_mic'):
    """All 360 excerpts, or the progression-3 excerpts MT3 holds out (two of a style's three progressions train)."""
    root = DATA / 'guitarset'
    out = []
    for jams in sorted((root / 'annotation').glob('*.jams')):
        if test_only and not re.match(r'\d\d_[A-Za-z]+3-', jams.stem):
            continue
        notes = note_array(load_guitarset_notes(str(jams)))
        out.append(Recording(f'guitarset/{jams.stem}', root / audio / f'{jams.stem}{suffix}.wav', notes))
    return out


def egdb():
    root = DATA / 'egdb' / 'raw'
    return [
        Recording(f'egdb/{m.stem}', root / 'audio_DI' / f'{m.stem}.wav',
                  np.array([n[:3] for n in channel_notes(m)], float).reshape(-1, 3))
        for m in sorted((root / 'audio_label').glob('*.midi'), key=lambda p: int(p.stem))
    ]  # fmt: skip


def gaps_test():
    root = DATA / 'gaps_hf'
    with open(root / 'gaps_metadata_with_splits.csv', newline='', encoding='utf-8') as fh:
        rows = [r for r in csv.DictReader(fh) if r['split'] == 'test' and (root / r['audio_path']).exists()]
    return [Recording(f'gaps/{r["id"]}', root / r['audio_path'], note_array(load_gaps_notes(str(root / r['midi_path']))))
            for r in rows]  # fmt: skip


def goat(renders, split='test'):
    """GOAT items of `split` with fine-aligned labels; `renders` names the audio columns (di_audio_path, amp_...)."""
    root = DATA / 'goat'
    out = []
    for r in pd.read_csv(root / 'GOAT' / 'metadata.csv').to_dict('records'):
        midi = root / str(r['finealigned_midi_path']).replace('GOAT/', 'GOAT/data/', 1)
        if r['split'] != split or not midi.is_file():
            continue
        notes = midi_notes(midi)
        for col in renders:
            audio = root / r[col].replace('GOAT/', 'GOAT/data/', 1)
            out.append(Recording(f'goat/{r["item"]}/{Path(audio).stem}', audio, notes))
    return out


def index(guitars, chains, index_dir='data/string_index'):
    """Recordings of the string index (labels re-timed to the audio), for selecting checkpoints on val guitars."""
    recs = pd.read_csv(REPO_ROOT / index_dir / 'recordings.csv')
    notes = dict(tuple(pd.read_csv(REPO_ROOT / index_dir / 'notes.csv.gz').groupby('rec')))
    recs = recs[recs.guitar.isin(guitars) & recs.chain.isin(chains)]
    return [Recording(r.rec, REPO_ROOT / r.audio, notes[r.rec][['onset', 'offset', 'pitch']].to_numpy(float))
            for r in recs.itertuples()]  # fmt: skip


def manifest_val(manifest_dir, dataset):
    """Validation recordings of one dataset in a notes manifest (build_notes_dataset.py), for picking checkpoints."""
    root = REPO_ROOT / manifest_dir
    with open(root / 'manifest.jsonl') as fh:
        entries = [e for e in map(json.loads, fh) if e['split'] == 'val' and e['dataset'] == dataset]
    return [Recording(e['id'], root / e['audio'], note_array(e['notes'])) for e in entries]


def pink(n, rng):
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum /= np.sqrt(np.maximum(np.arange(len(spectrum)), 1))
    return np.fft.irfft(spectrum, n)


def corrupt(y, sr, irs, rng, cfg):
    ir, _ = librosa.load(rng.choice(irs), sr=sr, mono=True)
    y = scipy.signal.fftconvolve(y, ir)[: len(y)]
    y = scipy.signal.sosfiltfilt(scipy.signal.butter(4, cfg.highpass_hz, 'highpass', fs=sr, output='sos'), y)
    kinds = [k for k in ('white', 'pink', 'hum') if rng.random() < 0.5] or [rng.choice(['white', 'pink', 'hum'])]
    t = np.arange(len(y)) / sr
    sources = {'white': lambda: rng.standard_normal(len(y)), 'pink': lambda: pink(len(y), rng),
               'hum': lambda: sum(np.sin(2 * np.pi * 60 * k * t + rng.uniform(0, 2 * np.pi)) / k for k in (1, 2, 3))}  # fmt: skip
    drawn = [sources[k]() for k in kinds]
    noise = sum(d / np.sqrt(np.mean(d**2)) for d in drawn)
    snr = rng.uniform(*cfg.snr_db)
    noise *= np.sqrt(np.mean(y**2) / np.mean(noise**2) / 10 ** (snr / 10))
    y = y + noise
    return 0.9 * y / max(np.abs(y).max(), 1e-9)


def noisy(recordings, name, cfg):
    """`recordings` with TART's corruption, written once to <cache_dir>/<name>/ as 16 kHz FLAC."""
    irs = sorted((REPO_ROOT / cfg.ir_dir).glob('*.wav'))
    out = []
    for r in recordings:
        path = REPO_ROOT / cfg.cache_dir / name / f'{r.rec.replace("/", "__")}.flac'
        if not path.exists():
            y, _ = librosa.load(r.audio, sr=SR, mono=True)
            rng = np.random.default_rng(zlib.crc32(f'{cfg.seed}/{r.rec}'.encode()))
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, corrupt(y, SR, irs, rng, cfg).astype(np.float32), SR, format='FLAC')
        out.append(Recording(f'{name}/{r.rec}', path, r.notes))
    return out


def build(name, cfg):
    amps = [f'amp_audio_path_{k}' for k in range(1, 6)]
    match name:
        case 'guitarset':
            return guitarset()
        case 'guitarset_test':
            return guitarset(test_only=True)
        case 'egdb':
            return egdb()
        case 'gaps_test':
            return gaps_test()
        case 'goat_di':
            return goat(['di_audio_path'])
        case 'goat_amp':
            return goat(amps)
        case 'guitarset_noisy':
            return noisy(guitarset(audio='audio_hex-pickup_debleeded', suffix='_hex_cln'), name, cfg.noise)
        case 'egdb_noisy':
            return noisy(egdb(), name, cfg.noise)
        case 'val_evh':
            return index(['evh-wolfgang'], ['di', 'amp-mic'])
        case 'val_les_paul':
            return index(['gibson-les-paul'], ['di'])
        case _ if name.startswith('val_'):
            return manifest_val(cfg.val_manifest, name.removeprefix('val_'))
    raise ValueError(f'unknown benchmark {name}')

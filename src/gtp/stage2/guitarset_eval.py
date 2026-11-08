"""GuitarSet evaluation along the reference decoder skeleton, shared by string_fusion and select_checkpoint.

`prepare` reads the GuitarSet pieces of the requested splits and attaches, per note, the string
classifiers' out-of-fold distributions (decode-time log ratios and encoder evidence, per seed).
"""

import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.stage2.data import load_jsonl_pieces
from gtp.stage2.evidence import restrict
from gtp.stage2.fusion import log_ratio, piece_rows, sorted_notes, valid_tabs

N_STRINGS = 6


def load_distributions(runs, calibrated):
    """(track, onset, capo-0 pitch) -> {'audio': {seed: probs}, 'prior': probs}."""
    key = 'probs_calibrated' if calibrated else 'probs'
    dists = {}
    for (_player, seed), run_dir in sorted(runs.items()):
        z = np.load(f'{run_dir}/test_probs.npz')
        for track, onset, pitch, probs, prior in zip(
            z['track'], z['onset'], z['pitch'], z[key], z['prior'], strict=True
        ):
            entry = dists.setdefault((str(track), round(float(onset), 4), int(pitch)), {'audio': {}, 'prior': prior})
            entry['audio'][seed] = probs
    return dists


def guitarset_pieces(cfg, split):
    for piece in load_jsonl_pieces(REPO_ROOT / cfg.data_dir / f'{split}.jsonl'):
        if piece['source'] == 'guitarset' and (piece['capo'] == 0 or not cfg.capo0_only):
            yield piece


def prepare(cfg, vocab, dists):
    """Per-note table plus, per piece, what decoding needs: notes in decoder order, log ratios, evidence."""
    table, pieces = [], []
    for split in cfg.splits:
        for piece in guitarset_pieces(cfg, split):
            track, tuning, capo = piece['filename'].removesuffix('.json'), piece['tuning'], piece['capo']
            note_ids, ratios, evidence = [], [], []
            for n in sorted_notes(piece):
                strings, _ = valid_tabs(n['pitch'], tuning, vocab)
                d = dists.get((track, round(n['start'], 4), n['pitch'] - capo))
                record = {
                    'split': split,
                    'piece': f'{track}:capo{capo}',
                    'track': track,
                    'pitch': n['pitch'],
                    'true_string': n['string'],
                    'n_cand': len(strings),
                    'has_audio': d is not None and len(tuning) == N_STRINGS,
                }
                ratio, ev = None, np.full((len(cfg.seeds), N_STRINGS), np.nan)
                if record['has_audio']:
                    audio = np.stack([d['audio'][s] for s in cfg.seeds])
                    ratio = log_ratio(audio, d['prior'], strings, cfg.prob_floor)
                    ev = restrict(audio, [n['pitch']] * len(cfg.seeds), tuning)
                    for i, s in enumerate(cfg.seeds):
                        record[f'audio_s{s}'] = int(strings[audio[i, strings - 1].argmax()])
                note_ids.append(len(table))
                ratios.append(ratio)
                evidence.append(ev)
                table.append(record)
            pieces.append((piece, note_ids, ratios, np.stack(evidence, axis=1)))
    return pieces, pd.DataFrame(table)


def rows_for(pieces, vocab, max_seq_len, seed_idx=None):
    """Decoder rows for every piece; with `seed_idx`, the encoder also gets that seed's evidence (NaN = none)."""
    rows = []
    for piece, note_ids, ratios, evidence in pieces:
        ev = None
        if seed_idx is not None:
            ev = evidence[seed_idx] if seed_idx >= 0 else np.full_like(evidence[0], np.nan)
        rows.extend(piece_rows(piece, vocab, max_seq_len, note_ids, ratios, ev))
    return rows


def correct(picks, df):
    assert len(picks) == len(df), f'{len(picks)} picks for {len(df)} notes'
    return pd.Series(picks).reindex(df.index) == df.true_string

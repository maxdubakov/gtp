"""Enrich per-note predictions with per-source metadata + local context + error type.

Reads:
  pieces.jsonl + predictions.jsonl produced by dump_eval_predictions.py

For each note adds:

  * Per-source metadata (depends on source):
      - dadagp:      genre tags + artist (from _DadaGP_all_metadata.json,
                     joined via the original .gp4 path stored in the per-piece
                     processed JSON)
      - guitarset:   player_id + style + bpm + key + comp/solo (parsed from
                     filename like '00_BN1-129-Eb_solo.json')
      - guitartoday: slice_id, has_sync (already in processed JSON)
      - leduc:       artist, title, n_bars, n_notes (already in processed JSON)

  * Local context features (computed across all notes in the same piece_id):
      - note_density_2s: notes with onset within +/- 1.0 s of this one
      - polyphony:        notes with onset within +/- 30 ms (chord size)
      - prev_pitch / prev_string / prev_fret / time_from_prev / interval_from_prev
      - next_pitch / time_to_next
      - duration:         end - onset
      - position_in_piece: onset / piece_duration
      - avg_fret_5s_window: rolling average of true_fret in a +/- 2.5 s window
                             (proxies "where the left hand is currently positioned")
      - prev_string_dist / prev_fret_dist (left-hand-movement proxies)

  * Error categorization (raw & pp separately):
      - delta_string, delta_fret, delta_pitch  (signed)
      - error_type in {correct, pitch_mismatch, same_pitch_adj_string,
                       same_pitch_far_string, no_prediction}

Output:
  enriched.jsonl  (one row per note, fully annotated)

Usage:
  python scripts/stage2/error_analysis/enrich_errors.py \\
      --input-dir results/error_analysis/run_60k \\
      --output results/error_analysis/run_60k/enriched.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from gtp.stage2.metrics import classify_error as _classify_error_type

REPO_ROOT = Path(__file__).resolve().parents[3]
DADAGP_META_PATH = REPO_ROOT / 'data' / 'DadaGP-v1.1' / '_DadaGP_all_metadata.json'

# Pre-compile filename parser for GuitarSet:
#   '00_BN1-129-Eb_solo' -> player=00 style=BN1 bpm=129 key=Eb mode=solo
GS_NAME_RE = re.compile(
    r'^(?P<player>\d{2})_(?P<style>[A-Za-z]+\d*)-(?P<bpm>\d+)-(?P<key>[A-Za-z#b]+)_(?P<mode>comp|solo)',
)

# Genre is now baked into processed JSONs by the per-source build_dataset
# scripts (see scripts/stage2/data/<source>/build_dataset.py). enrich_errors
# just reads the existing `genre` field — no re-classification needed here.


# ---------------------------------------------------------------------------
# Per-source metadata loaders
# ---------------------------------------------------------------------------


def load_dadagp_metadata() -> dict:
    """Return mapping: original_gp4_path -> {genre_tokens, artist_token, ...}"""
    if not DADAGP_META_PATH.exists():
        print(f'  WARN: missing {DADAGP_META_PATH} - DadaGP enrichment disabled')
        return {}
    raw = json.loads(DADAGP_META_PATH.read_text())
    # MATLAB-side keys look like '1/Group/Song.gp4.tokens.txt'; we strip the
    # '.tokens.txt' suffix so we can join against the raw .gp4 path stored
    # in our processed JSONs.
    out = {}
    for k, v in raw.items():
        if k.endswith('.tokens.txt'):
            out[k[: -len('.tokens.txt')]] = v
        else:
            out[k] = v
    return out


def load_processed_extra(source: str, processed_filename: str) -> dict:
    """Read the per-source processed JSON to recover fields that get dropped
    during the augmented-dataset build (file path, slice_id, artist, etc.)."""
    if not processed_filename:
        return {}
    path = REPO_ROOT / 'data' / source / 'processed' / processed_filename
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    # Drop bulky/redundant fields; keep all other metadata
    return {k: v for k, v in d.items() if k != 'notes'}


def parse_guitarset_filename(filename: str | None) -> dict:
    """'00_BN1-129-Eb_solo.json' -> structured fields."""
    if not filename:
        return {}
    stem = filename.replace('.json', '')
    m = GS_NAME_RE.match(stem)
    if not m:
        return {}
    return {
        'gs_player': m.group('player'),
        'gs_style': m.group('style'),
        'gs_bpm': int(m.group('bpm')),
        'gs_key': m.group('key'),
        'gs_mode': m.group('mode'),
    }


# ---------------------------------------------------------------------------
# Local-context computation
# ---------------------------------------------------------------------------


def compute_local_context(notes: list[dict], capo: int = 0) -> list[dict]:
    """For each note (assumed sorted by onset), compute local-context features.

    `notes` is a list of dicts with keys onset, pitch, true_string, true_fret.
    Returns a list of dicts (same length) with the new features.
    """
    n = len(notes)
    if n == 0:
        return []
    onsets = [r['onset'] for r in notes]
    duration = max(r['end'] for r in notes) if all('end' in r for r in notes) else onsets[-1]

    out = []
    for i, r in enumerate(notes):
        on = r['onset']

        # Density: notes with onset in [on-1, on+1]
        # Using a linear scan since pieces are small (typical < few-thousand notes).
        density_count = 0
        for j in range(n):
            if abs(onsets[j] - on) <= 1.0:
                density_count += 1
        # Polyphony: notes with onset in [on-0.030, on+0.030]
        poly = 0
        for j in range(n):
            if abs(onsets[j] - on) <= 0.030:
                poly += 1

        # Prev / next distinct-onset notes
        prev = None
        for j in range(i - 1, -1, -1):
            if onsets[j] < on:
                prev = notes[j]
                break
        nxt = None
        for j in range(i + 1, n):
            if onsets[j] > on:
                nxt = notes[j]
                break

        # 5s rolling-average fret around this note (left-hand-position proxy)
        win_lo = on - 2.5
        win_hi = on + 2.5
        frets_in_win = [notes[j]['true_fret'] for j in range(n)
                        if win_lo <= onsets[j] <= win_hi]
        avg_fret_5s = sum(frets_in_win) / len(frets_in_win) if frets_in_win else 0.0
        min_fret_5s = min(frets_in_win) if frets_in_win else 0
        max_fret_5s = max(frets_in_win) if frets_in_win else 0

        out.append({
            'note_density_2s_window': density_count,
            'polyphony': poly,
            'duration': r.get('end', on) - on,
            'position_in_piece': on / duration if duration > 0 else 0.0,
            'avg_fret_5s_window': avg_fret_5s,
            'min_fret_5s_window': min_fret_5s,
            'max_fret_5s_window': max_fret_5s,
            'absolute_fret': r['true_fret'] + capo,
            'prev_pitch': prev['pitch'] if prev else None,
            'prev_string': prev['true_string'] if prev else None,
            'prev_fret': prev['true_fret'] if prev else None,
            'time_from_prev': (on - prev['onset']) if prev else None,
            'interval_from_prev': (r['pitch'] - prev['pitch']) if prev else None,
            'prev_string_dist': abs(r['true_string'] - prev['true_string']) if prev else None,
            'prev_fret_dist': abs(r['true_fret'] - prev['true_fret']) if prev else None,
            'next_pitch': nxt['pitch'] if nxt else None,
            'time_to_next': (nxt['onset'] - on) if nxt else None,
        })
    return out


# ---------------------------------------------------------------------------
# Error categorization
# ---------------------------------------------------------------------------


def classify_error(true_s: int, true_f: int, true_pitch: int,
                   pred_s: int | None, pred_f: int | None, pred_pitch: int | None) -> dict:
    """Wrapper around `gtp.stage2.metrics.classify_error` that also returns deltas.

    Returns dict with `delta_string`, `delta_fret`, `delta_pitch`, `error_type`.
    Categorization logic lives in `metrics.classify_error` and is shared with
    `eval.py` to keep the taxonomy in one place.
    """
    error_type = _classify_error_type(true_s, true_f, true_pitch, pred_s, pred_f, pred_pitch)
    if error_type == 'no_prediction':
        return {'delta_string': None, 'delta_fret': None, 'delta_pitch': None,
                'error_type': error_type}
    dstr = pred_s - true_s
    dfret = pred_f - true_f
    dpitch = (pred_pitch - true_pitch) if pred_pitch is not None else None
    return {'delta_string': dstr, 'delta_fret': dfret,
            'delta_pitch': 0 if error_type != 'pitch_mismatch' else dpitch,
            'error_type': error_type}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True,
                    help='Directory containing pieces.jsonl + predictions.jsonl '
                         'from dump_eval_predictions.py')
    ap.add_argument('--output', required=True, help='Path to enriched.jsonl')
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    pieces_path = in_dir / 'pieces.jsonl'
    preds_path = in_dir / 'predictions.jsonl'
    if not pieces_path.exists() or not preds_path.exists():
        raise SystemExit(f'Need {pieces_path} and {preds_path}')

    print(f'Loading pieces from {pieces_path}...')
    pieces = {p['piece_id']: p for p in (json.loads(line) for line in pieces_path.open())}
    print(f'  {len(pieces)} pieces')

    print(f'Loading predictions from {preds_path}...')
    preds_by_piece: dict[str, list[dict]] = defaultdict(list)
    n_pred = 0
    with preds_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            preds_by_piece[r['piece_id']].append(r)
            n_pred += 1
    print(f'  {n_pred} note records across {len(preds_by_piece)} pieces')

    print(f'Loading DadaGP metadata from {DADAGP_META_PATH}...')
    dadagp_meta = load_dadagp_metadata()
    print(f'  {len(dadagp_meta)} entries')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_dadagp_hit = n_dadagp_miss = 0
    n_written = 0
    with out_path.open('w') as out_fh:
        for piece_id, piece_meta in pieces.items():
            source = piece_meta.get('source')
            filename = piece_meta.get('filename')

            # ---- per-source enrichment ----
            extras = {}
            if source == 'dadagp':
                # Get the .gp4 file path from the raw processed JSON, then look up
                # extra metadata in DadaGP's _all_metadata.json. The coarse `genre`
                # field is read straight from the processed JSON (written by
                # scripts/stage2/data/dadagp/build_dataset.py:classify_dadagp).
                proc_extra = load_processed_extra('dadagp', filename) if filename else {}
                gp4_path = proc_extra.get('file')
                genre = proc_extra.get('genre', 'unknown')
                if gp4_path and gp4_path in dadagp_meta:
                    md = dadagp_meta[gp4_path]
                    extras = {
                        'genre_tokens': md.get('genre_tokens', []),
                        'artist_token': md.get('artist_token'),
                        'genre': genre,
                        'dadagp_validation_set': md.get('validation_set'),
                        'gp4_path': gp4_path,
                    }
                    n_dadagp_hit += 1
                else:
                    extras = {'genre': genre, 'gp4_path': gp4_path}
                    n_dadagp_miss += 1
            elif source == 'guitarset':
                extras = parse_guitarset_filename(filename)
            elif source == 'guitartoday':
                proc_extra = load_processed_extra('guitartoday', filename) if filename else {}
                # Keep slice_id, has_sync, n_strings
                extras = {k: proc_extra[k] for k in ('slice_id', 'has_sync', 'n_strings')
                          if k in proc_extra}
            elif source == 'leduc':
                proc_extra = load_processed_extra('leduc', filename) if filename else {}
                extras = {k: proc_extra[k] for k in ('artist', 'title', 'n_bars')
                          if k in proc_extra}

            # ---- local context per note ----
            note_records = preds_by_piece.get(piece_id, [])
            note_records.sort(key=lambda r: (r['onset'], r['pitch']))
            local_ctxs = compute_local_context(note_records, capo=piece_meta.get('capo', 0))

            # ---- emit enriched records ----
            for r, ctx in zip(note_records, local_ctxs, strict=True):
                err_raw = classify_error(
                    r['true_string'], r['true_fret'], r['pitch'],
                    r.get('pred_raw_string'), r.get('pred_raw_fret'), r.get('pred_raw_pitch'),
                )
                err_pp = classify_error(
                    r['true_string'], r['true_fret'], r['pitch'],
                    r.get('pred_pp_string'), r.get('pred_pp_fret'), r.get('pred_pp_pitch'),
                )
                enriched = {
                    **r,
                    # Piece-level passthrough
                    'source': source,
                    'split': r.get('split'),
                    'tuning': piece_meta.get('tuning'),
                    'capo': piece_meta.get('capo'),
                    'tempo': piece_meta.get('tempo'),
                    'piece_n_notes': piece_meta.get('n_notes'),
                    # Per-source metadata
                    **extras,
                    # Local context
                    **ctx,
                    # Errors
                    'delta_string_raw': err_raw['delta_string'],
                    'delta_fret_raw': err_raw['delta_fret'],
                    'delta_pitch_raw': err_raw['delta_pitch'],
                    'error_type_raw': err_raw['error_type'],
                    'delta_string_pp': err_pp['delta_string'],
                    'delta_fret_pp': err_pp['delta_fret'],
                    'delta_pitch_pp': err_pp['delta_pitch'],
                    'error_type_pp': err_pp['error_type'],
                }
                out_fh.write(json.dumps(enriched) + '\n')
                n_written += 1

    print(f'\nWrote {n_written:,} enriched records to {out_path}')
    if n_dadagp_hit + n_dadagp_miss > 0:
        print(f'DadaGP metadata join: {n_dadagp_hit} hit / {n_dadagp_miss} miss')


if __name__ == '__main__':
    main()

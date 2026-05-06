"""Run Stage 2 on val/test pieces and dump per-note predictions for error analysis.

For each piece in val/test:
  1. Load piece JSON (notes + tuning + capo + tempo + source filename).
  2. Tokenize for inference; run autoregressive generation -> raw predicted tabs.
  3. Apply post-processing -> pp predicted tabs + per-note source flags.
  4. Emit one record per note with both raw and pp predictions, plus enough
     piece-level metadata for downstream joins/enrichment.

Outputs (under --output-dir):
  - pieces.jsonl     one row per piece (metadata: tuning, capo, tempo, source, file, n_notes, ...)
  - predictions.jsonl  one row per note (truth + raw prediction + pp prediction + pp source flag)

Usage:
  python scripts/stage2/error_analysis/dump_eval_predictions.py \\
      --checkpoint runs/stage2_baseline/checkpoints/step_0060000_final.pth \\
      --splits val test \\
      --output-dir results/error_analysis/run_60k

Smoke (sub-sample of pieces from each source):
  python scripts/stage2/error_analysis/dump_eval_predictions.py \\
      --checkpoint runs/stage2_baseline/checkpoints/step_0060000_final.pth \\
      --splits val \\
      --pieces-per-source 5 \\
      --output-dir /tmp/dump_smoke
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from gtp.stage2.data import MIN_NOTES_PER_PIECE, filter_notes, load_jsonl_pieces
from gtp.stage2.inference import (
    extract_tabs,
    load_checkpoint,
    tokenize_for_inference,
)
from gtp.stage2.paths import AUG_DATA_DIR, PROCESSED_DIRS
from gtp.stage2.postprocess import correct_tabs

# Constants matching scripts/stage2/build_aug_dataset.py — kept here so we can
# reproduce its train/val/test split deterministically without importing from
# a sibling script.
SPLIT_SEED = 42
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05


def auto_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _load_processed_pieces(datasets: list[str] | None = None) -> list[dict]:
    """Load original (un-augmented) pieces from data/<source>/processed/*.json.

    Mirrors scripts/stage2/build_aug_dataset.py::load_all_pieces. Each piece is
    tagged with a `source` and `filename` field for downstream joins.
    """
    pieces = []
    sources = list(PROCESSED_DIRS.keys()) if datasets is None else datasets
    for name in sources:
        path = PROCESSED_DIRS[name]
        if not path.exists():
            continue
        for f in sorted(path.iterdir()):
            if f.suffix != '.json' or f.name.startswith('._'):
                continue
            with f.open(encoding='utf-8', errors='replace') as fh:
                data = json.load(fh)
            tuning = data.get('tuning', [64, 59, 55, 50, 45, 40])
            notes, _ = filter_notes(data.get('notes', []), tuning)
            if len(notes) < MIN_NOTES_PER_PIECE:
                continue
            pieces.append({
                'source': name,
                'filename': f.name,
                'tuning': tuning,
                'tempo': data.get('tempo', 120),
                'capo': data.get('capo', 0),
                'notes': notes,
            })
    return pieces


def _stratified_split_originals(pieces: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Reproduce build_aug_dataset.py::stratified_split on original pieces.

    Stratifies by (source, tuning, capo>0) and splits 90/5/5 with seed=42 so
    we get exactly the same val/test piece set as the augmented build did
    (just without the per-piece capo expansion).
    """
    rng = random.Random(SPLIT_SEED)

    groups: dict = defaultdict(list)
    for p in pieces:
        capo_key = 'capo' if p.get('capo', 0) > 0 else 'no_capo'
        key = (p['source'], tuple(p['tuning']), capo_key)
        groups[key].append(p)

    train, val, test = [], [], []
    for _key, group in groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * TRAIN_RATIO))
        n_val = max(0, round(n * VAL_RATIO))
        if n <= 2:
            train.extend(group)
            continue
        train.extend(group[:n_train])
        val.extend(group[n_train: n_train + n_val])
        test.extend(group[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _build_original_capo_lookup(datasets: list[str] | None = None) -> dict[tuple[str, str], int]:
    """Build {(source, filename): original_capo} so we can tag augmented rows
    in the JSONL splits with `is_augmented = (row.capo != original.capo)`.
    """
    out: dict[tuple[str, str], int] = {}
    for piece in _load_processed_pieces(datasets):
        out[(piece['source'], piece['filename'])] = piece.get('capo', 0)
    return out


def piece_id(piece: dict) -> str:
    """Stable identifier per (source, processed-filename, capo) — capo variants
    of the same piece are distinct rows in val.jsonl, so include capo to avoid
    collision. `filename` is the processed-JSON name we can use for joining
    against per-source metadata downstream."""
    src = piece.get('source', '?')
    name = piece.get('filename') or piece.get('file') or '?'
    capo = piece.get('capo', 0)
    return f'{src}:{name}:capo{capo}'


def generate_tabs_batched(model, vocab, enc_ids_list: list[list[int]], device: str,
                          max_seq_len: int, batch_size: int) -> list[list[int]]:
    """Batched autoregressive greedy generation.

    Pads each sub-sequence to max_seq_len, stacks into batches of `batch_size`,
    runs one model.generate per batch. Returns the raw decoder token IDs per
    sub-sequence (with the BOS / decoder_start prefix stripped, matching what
    extract_tabs expects). Stays within the same algorithmic regime as
    inference.generate_tabs (greedy, num_beams=1), just amortizes overhead
    across many sub-sequences instead of one at a time.
    """
    out = []
    for start in range(0, len(enc_ids_list), batch_size):
        batch = enc_ids_list[start:start + batch_size]
        padded = [enc + [vocab.pad_id] * (max_seq_len - len(enc)) for enc in batch]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = (input_ids != vocab.pad_id).long()
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_seq_len,
                num_beams=1,
                do_sample=False,
                pad_token_id=vocab.pad_id,
                eos_token_id=vocab.eos_id,
            )
        for row in generated:
            # Skip the decoder_start (PAD) prefix, then keep until EOS or end.
            out.append(row[1:].tolist())
    return out


def prepare_piece(piece: dict, vocab, max_seq_len: int) -> dict:
    """Tokenize + sort. Doesn't touch the model — pure CPU work, can run cheaply
    in bulk before batched generation."""
    notes_sorted = sorted(piece['notes'], key=lambda n: (n['start'], n['pitch']))
    return {
        'piece_id': piece_id(piece),
        'piece': piece,
        'notes_sorted': notes_sorted,
        'enc_subseqs': tokenize_for_inference(piece, vocab, max_seq_len=max_seq_len),
        'input_pitches': [int(n['pitch']) for n in notes_sorted],
    }


def finalize_piece(prep: dict, vocab, raw_subseq_outputs: list[list[int]],
                   is_augmented: bool,
                   fallback: str = 'first_viable') -> tuple[dict, list[dict]]:
    """Given a prepared piece and its raw decoder outputs (per sub-sequence),
    extract tabs, post-process, and build the per-note records + piece meta.

    `is_augmented` is propagated into piece_meta and into every note record so
    downstream slicing can distinguish original-vs-augmented predictions.
    `fallback` selects the pp fallback strategy: 'first_viable' (paper) or
    'nearest_viable' (deviation; Manhattan-nearest to model raw output).
    """
    piece = prep['piece']
    notes_sorted = prep['notes_sorted']
    n_notes = len(notes_sorted)
    pid = prep['piece_id']

    raw_tabs: list[tuple[int, int]] = []
    for raw_ids in raw_subseq_outputs:
        raw_tabs.extend(extract_tabs(raw_ids, vocab))

    input_pitches = prep['input_pitches']
    pp_tabs, pp_sources = correct_tabs(
        input_pitches, raw_tabs, piece['tuning'], return_sources=True, fallback=fallback,
    )

    # Counters for the piece-level summary
    n_correct_raw = 0
    n_correct_pp = 0
    n_pitch_correct_raw = 0
    n_pitch_correct_pp = 0
    tuning = piece['tuning']

    note_records: list[dict] = []
    for i, n in enumerate(notes_sorted):
        true_s = int(n['string'])
        true_f = int(n['fret'])
        true_pitch = int(n['pitch'])

        raw = raw_tabs[i] if i < len(raw_tabs) else None
        pp = pp_tabs[i] if i < len(pp_tabs) else None

        if raw is not None:
            rs, rf = int(raw[0]), int(raw[1])
            r_pitch = tuning[rs - 1] + rf if 1 <= rs <= len(tuning) else None
            if (rs, rf) == (true_s, true_f):
                n_correct_raw += 1
            if r_pitch == true_pitch:
                n_pitch_correct_raw += 1
        else:
            rs = rf = r_pitch = None

        if pp is not None:
            ps, pf = int(pp[0]), int(pp[1])
            p_pitch = tuning[ps - 1] + pf if 1 <= ps <= len(tuning) else None
            if (ps, pf) == (true_s, true_f):
                n_correct_pp += 1
            if p_pitch == true_pitch:
                n_pitch_correct_pp += 1
        else:
            ps = pf = p_pitch = None

        note_records.append({
            'piece_id': pid,
            'note_idx': i,
            'pitch': true_pitch,
            'onset': float(n['start']),
            'end': float(n['end']),
            'true_string': true_s,
            'true_fret': true_f,
            'pred_raw_string': rs,
            'pred_raw_fret': rf,
            'pred_raw_pitch': r_pitch,
            'pred_pp_string': ps,
            'pred_pp_fret': pf,
            'pred_pp_pitch': p_pitch,
            'pp_source': pp_sources[i] if i < len(pp_sources) else None,
            'is_augmented': is_augmented,
        })

    piece_meta = {
        'piece_id': pid,
        'source': piece.get('source'),
        # 'filename' = processed-JSON name (used to join with per-source metadata);
        # 'file' (when present in source data) = original raw-source path (e.g.
        # DadaGP's .gp4 path, used to join with DadaGP's _DadaGP_all_metadata.json).
        'filename': piece.get('filename'),
        'file': piece.get('file'),
        'tuning': piece['tuning'],
        'capo': piece.get('capo', 0),
        'tempo': piece.get('tempo'),
        'n_notes': n_notes,
        'n_subseqs': len(prep['enc_subseqs']),
        'n_correct_raw': n_correct_raw,
        'n_correct_pp': n_correct_pp,
        'n_pitch_correct_raw': n_pitch_correct_raw,
        'n_pitch_correct_pp': n_pitch_correct_pp,
        'n_raw_tabs_emitted': len(raw_tabs),
        'is_augmented': is_augmented,
    }
    return piece_meta, note_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='Stage 2 .pth checkpoint')
    ap.add_argument('--splits', nargs='+', default=['val'], choices=['val', 'test'],
                    help='Which JSONL splits to dump from data/stage2_aug/')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--max-seq-len', type=int, default=512)
    ap.add_argument('--pieces-per-source', type=int, default=None,
                    help='Smoke-test cap: limit to first N pieces per source within each split')
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='Filter pieces to specific source(s), e.g. --datasets dadagp guitarset')
    ap.add_argument('--no-augmentation', action='store_true',
                    help='Evaluate on the ORIGINAL un-augmented pieces (one per song) loaded '
                         'directly from data/<source>/processed/. Reproduces the same '
                         'train/val/test split (seed=42) as the augmented build, just without '
                         'the per-piece capo expansion. Default: use stage2_aug/{val,test}.jsonl.')
    ap.add_argument('--batch-size', type=int, default=32,
                    help='Sub-sequences per generate() call. Bigger = better GPU utilization '
                         'on small models. 32 is fine for our 2.24M-param T5; up to 128+ on a '
                         'real GPU. Set 1 to reproduce the old per-sub-sequence behavior.')
    ap.add_argument('--chunk-pieces', type=int, default=64,
                    help='How many pieces to tokenize before each batched-generate pass. '
                         'Pieces are post-processed and written incrementally per chunk so '
                         'memory stays bounded.')
    ap.add_argument('--fallback', choices=['first_viable', 'nearest_viable'],
                    default='first_viable',
                    help='Post-processing fallback strategy. first_viable = paper-faithful. '
                         'nearest_viable = deviation: Manhattan-nearest realization to model raw output.')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or auto_device()
    print(f'Device: {device}')

    print(f'Loading checkpoint: {args.checkpoint}')
    model, vocab, step = load_checkpoint(args.checkpoint, device)
    print(f'  iteration: {step}')

    pieces_path = out_dir / 'pieces.jsonl'
    preds_path = out_dir / 'predictions.jsonl'
    if pieces_path.exists():
        pieces_path.unlink()
    if preds_path.exists():
        preds_path.unlink()

    total_notes = 0
    total_correct_raw = 0
    total_correct_pp = 0
    started = time.time()

    # Pre-build the (source, filename) -> original_capo lookup. Used to tag
    # each piece as augmented or not, regardless of which loader path we take.
    original_capo = _build_original_capo_lookup(args.datasets)

    # When --no-augmentation, reproduce the seed=42 stratified split locally
    # so we hit the same val/test set as build_aug_dataset.py.
    splits_orig: dict[str, list[dict]] = {}
    if args.no_augmentation:
        all_originals = _load_processed_pieces(args.datasets)
        _train, val, test = _stratified_split_originals(all_originals)
        splits_orig = {'val': val, 'test': test}
        print('\nUsing ORIGINAL (un-augmented) pieces. Reproduced split (seed=42):')
        print(f'  val: {len(val)} pieces, test: {len(test)} pieces')

    for split in args.splits:
        if args.no_augmentation:
            pieces = splits_orig.get(split, [])
            if not pieces:
                print(f'[skip] no pieces for split={split}')
                continue
            print(f'\n[{split}] {len(pieces)} original pieces')
        else:
            path = AUG_DATA_DIR / f'{split}.jsonl'
            if not path.exists():
                print(f'[skip] missing split file: {path}')
                continue
            print(f'\nLoading {path}...')
            pieces = load_jsonl_pieces(path)
        if args.datasets:
            wanted = set(args.datasets)
            pieces = [p for p in pieces if p.get('source') in wanted]
        if args.pieces_per_source:
            keep_per_source = defaultdict(int)
            kept = []
            for p in pieces:
                src = p.get('source', '?')
                if keep_per_source[src] < args.pieces_per_source:
                    kept.append(p)
                    keep_per_source[src] += 1
            pieces = kept
        print(f'  {split}: {len(pieces)} pieces')

        n_done = 0
        with pieces_path.open('a') as pf, preds_path.open('a') as nf:
            for chunk_start in range(0, len(pieces), args.chunk_pieces):
                chunk = pieces[chunk_start:chunk_start + args.chunk_pieces]

                # Phase 1: tokenize all pieces in chunk; collect sub-sequences
                #          along with which (chunk_idx, subseq_idx) they came from.
                preps = [prepare_piece(p, vocab, args.max_seq_len) for p in chunk]
                flat_subseqs: list[list[int]] = []
                ownership: list[tuple[int, int]] = []  # (piece_idx_in_chunk, subseq_idx_in_piece)
                for ci, prep in enumerate(preps):
                    for si, enc in enumerate(prep['enc_subseqs']):
                        flat_subseqs.append(enc)
                        ownership.append((ci, si))

                # Phase 2: batched generation across the entire chunk
                if flat_subseqs:
                    flat_outputs = generate_tabs_batched(
                        model, vocab, flat_subseqs, device,
                        max_seq_len=args.max_seq_len, batch_size=args.batch_size,
                    )
                else:
                    flat_outputs = []

                # Phase 3: regroup outputs per piece, finalize, write
                per_piece_outputs: list[list[list[int]]] = [[] for _ in preps]
                for (ci, _si), raw_ids in zip(ownership, flat_outputs, strict=True):
                    per_piece_outputs[ci].append(raw_ids)

                for prep, raw_outs in zip(preps, per_piece_outputs, strict=True):
                    piece = prep['piece']
                    src = piece.get('source')
                    fname = piece.get('filename')
                    if args.no_augmentation:
                        # Loaded directly from processed/ — these ARE originals.
                        is_aug = False
                    else:
                        orig_capo = original_capo.get((src, fname))
                        is_aug = (orig_capo is None) or (piece.get('capo', 0) != orig_capo)
                    meta, note_records = finalize_piece(prep, vocab, raw_outs, is_aug, fallback=args.fallback)
                    meta['split'] = split
                    pf.write(json.dumps(meta) + '\n')
                    for r in note_records:
                        r['split'] = split
                        nf.write(json.dumps(r) + '\n')
                    total_notes += meta['n_notes']
                    total_correct_raw += meta['n_correct_raw']
                    total_correct_pp += meta['n_correct_pp']

                n_done += len(chunk)
                elapsed = time.time() - started
                eta = elapsed / max(1, n_done) * (len(pieces) - n_done)
                raw_acc = total_correct_raw / max(1, total_notes)
                pp_acc = total_correct_pp / max(1, total_notes)
                print(f'  [{n_done:>4d}/{len(pieces)}] {elapsed:.0f}s '
                      f'(~{eta:.0f}s ETA)  notes={total_notes:,}  '
                      f'tab_raw={raw_acc:.3f}  tab_pp={pp_acc:.3f}')

    print(f'\nWrote {pieces_path}')
    print(f'Wrote {preds_path}')


if __name__ == '__main__':
    main()

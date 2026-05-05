"""Per-piece autopsy of the worst-error pieces.

For each given piece_id, pulls all per-note predictions from predictions.jsonl
and reports:

  * Pitch histogram (the distinct notes and how often they recur).
  * For each distinct pitch p:
      - true (string, fret) (most common ground-truth realization)
      - predicted (string, fret) (most common pp prediction)
      - whether the predicted position is a VALID alternate realization of p.
  * "Position drift": median Δstring and Δfret between true and pred per pitch.

Goal: distinguish "model is wrong" from "model picked a valid alternate
position" from "data quality bug".

Usage:
  python scripts/stage2/error_analysis/inspect_worst_pieces.py \\
      --predictions results/error_analysis/run_60k_orig/predictions.jsonl \\
      --pieces results/error_analysis/run_60k_orig/pieces.jsonl \\
      --piece-ids dadagp:M_Murphy... dadagp:T_Thursday...
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def viable_positions(pitch, tuning, max_fret=24):
    out = []
    for s_idx, op in enumerate(tuning):
        f = pitch - op
        if 0 <= f <= max_fret:
            out.append((s_idx + 1, f))
    return out


def analyze_piece(piece_meta, note_records):
    n = len(note_records)
    tuning = piece_meta['tuning']

    # Per-pitch most common true/pred
    by_pitch_true = defaultdict(Counter)
    by_pitch_pp = defaultdict(Counter)
    for r in note_records:
        p = r['pitch']
        by_pitch_true[p][(r['true_string'], r['true_fret'])] += 1
        if r['pred_pp_string'] is not None and r['pred_pp_fret'] is not None:
            by_pitch_pp[p][(r['pred_pp_string'], r['pred_pp_fret'])] += 1

    # Determine "model's chosen realization" per pitch (the modal pp tab)
    pitch_summary = []
    for p in sorted(by_pitch_true.keys()):
        true_dom, true_n = by_pitch_true[p].most_common(1)[0]
        pp_dom, pp_n = (None, 0)
        if by_pitch_pp[p]:
            pp_dom, pp_n = by_pitch_pp[p].most_common(1)[0]

        viable = viable_positions(p, tuning)
        pred_is_viable_alt = pp_dom in viable
        pred_matches_true = (pp_dom == true_dom)

        pitch_summary.append({
            'pitch': p,
            'count': sum(by_pitch_true[p].values()),
            'true_dom': true_dom,
            'pp_dom': pp_dom,
            'n_viable_positions': len(viable),
            'viable_positions': viable,
            'pred_matches_true': pred_matches_true,
            'pred_is_valid_alt': pred_is_viable_alt and not pred_matches_true,
        })

    # Aggregate "drift": for the dominant pitches, what's the systematic shift?
    valid_alt_drifts = []
    for ps in pitch_summary:
        if ps['pred_is_valid_alt'] and ps['pp_dom']:
            ds = ps['pp_dom'][0] - ps['true_dom'][0]
            df = ps['pp_dom'][1] - ps['true_dom'][1]
            valid_alt_drifts.append((ds, df, ps['count']))

    return {
        'n_notes': n,
        'n_distinct_pitches': len(by_pitch_true),
        'pitch_summary': pitch_summary,
        'valid_alt_drifts': valid_alt_drifts,
    }


def print_piece_report(piece_id, piece_meta, analysis):
    capo = piece_meta.get('capo', 0)
    tuning = piece_meta['tuning']
    src = piece_meta.get('source')

    print(f'\n{"=" * 84}')
    print(f'{piece_id}')
    print(f'  source={src}  tuning={tuning}  capo={capo}  '
          f'n_notes={analysis["n_notes"]}  '
          f'n_distinct_pitches={analysis["n_distinct_pitches"]}')

    # Coverage summary
    total = analysis['n_notes']
    total_alt_valid = sum(c for _, _, c in analysis['valid_alt_drifts'])
    print(f'  notes where pred is a VALID alternate realization: {total_alt_valid}/{total} '
          f'({100 * total_alt_valid / total:.1f}%)')

    # Most common drift
    if analysis['valid_alt_drifts']:
        # weighted by count
        drift_counter = Counter()
        for ds, df, c in analysis['valid_alt_drifts']:
            drift_counter[(ds, df)] += c
        print(f'  dominant (Δstring, Δfret) drifts:')
        for (ds, df), c in drift_counter.most_common(3):
            print(f'    Δs={ds:+d} Δf={df:+d}: {c} notes ({100 * c / total:.1f}%)')

    # Per-pitch table
    print(f'\n  pitch  count  true(s,f)  pred(s,f)   #pos  status')
    rows = sorted(analysis['pitch_summary'], key=lambda x: -x['count'])
    for r in rows[:15]:
        if r['pred_matches_true']:
            status = 'CORRECT'
        elif r['pred_is_valid_alt']:
            status = 'valid alt'
        elif r['pp_dom'] is None:
            status = '(no pred)'
        else:
            status = 'INVALID'
        true_s = f'({r["true_dom"][0]},{r["true_dom"][1]})' if r['true_dom'] else 'N/A'
        pp_s = f'({r["pp_dom"][0]},{r["pp_dom"][1]})' if r['pp_dom'] else 'N/A'
        print(f'    {r["pitch"]:>3d}  {r["count"]:>5d}  {true_s:>9s}  {pp_s:>9s}  '
              f'{r["n_viable_positions"]:>4d}  {status}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--pieces', type=Path, required=True)
    ap.add_argument('--piece-ids', nargs='+', required=True,
                    help='Piece IDs to inspect (substring match, case-sensitive)')
    args = ap.parse_args()

    print(f'Loading {args.pieces}...')
    pieces = {p['piece_id']: p for p in load_jsonl(args.pieces)}
    print(f'  {len(pieces)} pieces')

    # Match piece_ids by substring (so user can paste short identifiers)
    matched = {}
    for q in args.piece_ids:
        for pid in pieces:
            if q in pid:
                matched[pid] = pieces[pid]
                break
        else:
            print(f'  WARN: no match for {q!r}')
    print(f'  matched {len(matched)} piece(s): {list(matched.keys())}')

    print(f'\nLoading {args.predictions}...')
    by_pid = defaultdict(list)
    n_total = 0
    with open(args.predictions) as f:
        for line in f:
            r = json.loads(line)
            n_total += 1
            if r['piece_id'] in matched:
                by_pid[r['piece_id']].append(r)
    print(f'  {n_total:,} records, {sum(len(v) for v in by_pid.values()):,} matched')

    for pid, meta in matched.items():
        records = sorted(by_pid[pid], key=lambda r: r['note_idx'])
        analysis = analyze_piece(meta, records)
        print_piece_report(pid, meta, analysis)


if __name__ == '__main__':
    main()

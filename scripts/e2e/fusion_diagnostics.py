"""Where the string methods of an index.py run disagree, on chords and by fused confidence.

Three tables per input, guitar and chain (and per guitar over all chains):
  complementarity.csv  among the notes `pipeline` gets wrong, the share each other method gets right; among the notes
                       `fused` gets wrong, the share `stage2` gets right; oracle accuracies of always picking the right
                       one of `fused` and `stage2` (and with `pipeline`); with evidence, `best_per_bin`: the accuracy
                       of taking, in each confidence bin and note kind, whichever of fused, stage2 and pipeline is best
                       there (chosen in-sample, so optimistic).
  chords.csv           string accuracy on single notes and on chord notes (notes of a recording starting within
                       chord_window_s of the chord's first note, two or more), the share of chords with every string
                       right, the share of the pipeline's wrong chords whose shape cannot be fingered (two notes on one
                       string, or fretted notes spanning more than the span), and the shape decoder: per chord, the
                       assignment with one note per string, frets 0..max_fret and a fretted span within the span that
                       maximises the summed log fused probabilities (`shape<span>`; falls back to fused argmax when no
                       assignment fits).
  confidence.csv       accuracy of fused, stage2 and pipeline by the fused confidence (max probability), single notes
                       and chord notes apart.
Chord-level shares count only chords whose notes all matched a reference note. The shape decoder and the confidence
table need the run's evidence.npz; without it only complementarity.csv and the note-level part of chords.csv are made.

Usage:
  uv run scripts/e2e/fusion_diagnostics.py run=results/e2e/index/<run>
  uv run scripts/e2e/fusion_diagnostics.py run=results/e2e/index/<run> "inputs=[gt]" "guitars=[egdb-strat]"
"""

import itertools

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

import gtp.track  # noqa: F401  registers the repo_root resolver the config's output dir uses
from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage1.strings import OPEN_STRINGS


def chord_ids(notes, window):
    """Per note (in `notes` order), the id of its onset group within its input and recording."""
    order = np.lexsort((notes['start'].to_numpy(), notes['rec'].to_numpy(), notes['input'].to_numpy()))
    key = (notes['input'] + '|' + notes['rec']).to_numpy()[order]
    start = notes['start'].to_numpy()[order]
    ids = np.empty(len(notes), dtype=int)
    gid, prev, t0 = -1, None, None
    for i, k, t in zip(order, key, start, strict=True):
        if gid < 0 or k != prev or t - t0 > window:
            gid, prev, t0 = gid + 1, k, t
        ids[i] = gid
    return ids


def fretted_span(pitches, strings):
    """(C, k) one-based strings of k notes -> (C,) span of the fretted notes' frets (-inf with none fretted)."""
    frets = pitches[None] - OPEN_STRINGS[strings - 1]
    fretted = frets > 0
    return np.where(fretted, frets, -np.inf).max(1) - np.where(fretted, frets, np.inf).min(1)


def repeated_string(strings):
    s = np.sort(strings, axis=1)
    return (np.diff(s, axis=1) == 0).any(axis=1)


def shape_decode(pitches, log_p, max_fret, span):
    """One-based strings of a chord's best fingerable assignment under `log_p` (k, 6), or None if none fits."""
    if len(pitches) > len(OPEN_STRINGS):
        return None
    frets = pitches[:, None] - OPEN_STRINGS[None]
    cand = [np.flatnonzero((f >= 0) & (f <= max_fret)) + 1 for f in frets]
    if not all(len(c) for c in cand):
        return None
    combos = np.array(list(itertools.product(*cand)))
    ok = ~repeated_string(combos) & (fretted_span(pitches, combos) <= span)
    if not ok.any():
        return None
    score = log_p[np.arange(len(pitches))[None], combos - 1].sum(1)
    return combos[np.argmax(np.where(ok, score, -np.inf))]


def groups(df, by):
    """(key dict, rows) per group of `by`, then per group of all but the last key, that key set to 'all'."""
    for k, g in df.groupby(by):
        yield dict(zip(by, k, strict=True)), g
    for k, g in df.groupby(by[:-1]):
        yield dict(zip(by[:-1], k, strict=True)) | {by[-1]: 'all'}, g


def complementarity(ok, by):
    rows = []
    for key, g in groups(ok, by):
        pw, fw = ~g['pipeline'], ~g['fused']
        rows.append(
            key
            | {'notes': len(g), 'pipeline_acc': g['pipeline'].mean(), 'fused_acc': g['fused'].mean()}
            | {'stage2_acc': g['stage2'].mean(), 'pipeline_wrong': int(pw.sum())}
            | {f'{m}_right_of_pipeline_wrong': g.loc[pw, m].mean() for m in ('stage2', 'classifier', 'fused')}
            | {'fused_wrong': int(fw.sum()), 'stage2_right_of_fused_wrong': g.loc[fw, 'stage2'].mean()}
            | {'oracle_fused_stage2': (g['fused'] | g['stage2']).mean()}
            | {'oracle_pipeline_stage2': (g['pipeline'] | g['stage2']).mean()}
            | {'oracle_pipeline_fused_stage2': (g['pipeline'] | g['fused'] | g['stage2']).mean()}
        )
        if 'bin' in g:
            right = g.groupby(['kind', 'bin'], observed=True)[['fused', 'stage2', 'pipeline']].sum()
            rows[-1]['best_per_bin'] = right.max(axis=1).sum() / len(g)
    return pd.DataFrame(rows)


def chord_table(notes, ok, methods, by, spans):
    chords = notes[notes['chord_size'] >= 2]
    complete = chords.groupby('chord')['ref_string'].transform(lambda s: s.notna().all())
    chords = chords[complete]
    pitch, strings = chords['pitch'].to_numpy(), chords[['pipeline', 'ref_string']].to_numpy(int)
    shapes = []
    for c, idx in chords.groupby('chord').indices.items():
        row = {'chord': c}
        for name, col in (('pipeline', 0), ('ref', 1)):
            s = strings[idx, col][None]
            row[f'{name}_repeated'], row[f'{name}_span'] = repeated_string(s)[0], fretted_span(pitch[idx], s)[0]
        shapes.append(row)
    right = ok.loc[chords.index].groupby(chords['chord'])[methods].all().add_suffix('_chord_right')
    per_chord = pd.DataFrame(shapes).set_index('chord').join(right)
    rows = []
    for key, g in groups(ok, by):
        single, chord = g[g['chord_size'] == 1], g[g['chord_size'] >= 2]
        row = key | {'single_notes': len(single), 'chord_notes': len(chord)}
        row |= {f'{m}_single_acc': single[m].mean() for m in methods}
        row |= {f'{m}_chord_acc': chord[m].mean() for m in methods}
        pc = per_chord[per_chord.index.isin(g['chord'])]
        row['chords'] = len(pc)
        row |= {f'{m}_chord_right': pc[f'{m}_chord_right'].mean() for m in methods}
        wrong = pc[~pc['pipeline_chord_right']]
        row['pipeline_wrong_chords'] = len(wrong)
        row['pipeline_wrong_repeated'] = wrong['pipeline_repeated'].mean()
        for s in spans:
            row[f'pipeline_wrong_unplayable{s}'] = (wrong['pipeline_repeated'] | (wrong['pipeline_span'] > s)).mean()
            row[f'ref_unplayable{s}'] = (pc['ref_repeated'] | (pc['ref_span'] > s)).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_bins(confidence, edges):
    bins = [0.0, *edges, 1.0 + 1e-9]
    labels = [f'{a:g}-{b:g}' for a, b in zip(bins[:-1], [*edges, 1.0], strict=True)]
    return pd.cut(confidence, bins, right=False, labels=labels)


def confidence_table(ok, by):
    rows = []
    for key, g in groups(ok, by):
        for (kind, b), h in g.groupby(['kind', 'bin'], observed=True):
            rows.append(
                key
                | {'kind': kind, 'confidence': b, 'notes': len(h)}
                | {f'{m}_acc': h[m].mean() for m in ('fused', 'stage2', 'pipeline')}
                | {'oracle_fused_stage2': (h['fused'] | h['stage2']).mean()}
            )
    return pd.DataFrame(rows)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'e2e'), config_name='fusion_diagnostics')
def main(cfg):
    run = REPO_ROOT / cfg.run
    notes = pd.read_csv(run / 'notes.csv.gz')
    evidence = np.load(run / 'evidence.npz') if (run / 'evidence.npz').exists() else None
    keep = notes['input'].isin(list(cfg.inputs))
    if cfg.guitars:
        keep &= notes['guitar'].isin(list(cfg.guitars))
    notes = notes[keep].copy()
    notes['chord'] = chord_ids(notes, cfg.chord_window_s)
    notes['chord_size'] = notes.groupby('chord')['chord'].transform('size')
    methods, by = list(cfg.methods), list(cfg.by)
    if evidence is not None:
        fused = evidence['fused_probs'][notes.index.to_numpy()]
        assert (fused.argmax(1) + 1 == notes['fused'].to_numpy()).all(), 'evidence.npz out of step with notes.csv.gz'
        notes['confidence'] = fused.max(1)
        log_p = np.log(np.maximum(fused, cfg.prob_floor))
        pitch = notes['pitch'].to_numpy()
        chords = [idx for idx in notes.groupby('chord').indices.values() if len(idx) >= 2]
        for s in cfg.spans:
            picks, fallback = notes['fused'].to_numpy().copy(), 0
            for idx in chords:
                best = shape_decode(pitch[idx], log_p[idx], cfg.max_fret, s)
                if best is None:
                    fallback += 1
                else:
                    picks[idx] = best
            notes[f'shape{s}'] = picks
            methods.append(f'shape{s}')
            info(f'shape decoder span {s}: {fallback} of {len(chords)} chords fell back to fused argmax')

    scored = notes[notes['ref_string'].notna()]
    ok = scored[[*by, 'chord', 'chord_size']].assign(**{m: scored[m] == scored['ref_string'] for m in methods})
    if evidence is not None:
        ok['kind'] = np.where(ok['chord_size'] >= 2, 'chord', 'single')
        ok['bin'] = confidence_bins(scored['confidence'], list(cfg.confidence_edges))
    out = HydraConfig.get().runtime.output_dir
    tables = {
        'complementarity': complementarity(ok, by),
        'chords': chord_table(notes, ok, methods, by, list(cfg.spans)),
    }
    if evidence is not None:
        tables['confidence'] = confidence_table(ok, by)
    for name, table in tables.items():
        table.to_csv(f'{out}/{name}.csv', index=False)
        info(f'{name}\n{table.round(3).to_string(index=False)}')
    info(f'Saved {out}')


if __name__ == '__main__':
    main()

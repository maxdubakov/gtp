"""End-to-end audio -> tab on GuitarSet's held-out player: stage 1 -> string classifier -> fused stage 2.

For every track of `test_player`:
  1. stage 1 transcribes the mic audio into notes (or the ground-truth notes are used, `inputs: gt`);
  2. the leave-one-player-out string classifiers for that player (one per seed) score each note;
  3. stage 2 picks each note's string, alone (lam 0) and fused with the audio (lam = cfg.lam);
     a stage 2 trained with audio evidence (evidence_checkpoint) reads it in its encoder instead.
Predicted notes are matched to the reference with mir_eval (onset tolerance, no offset). A tab
note is correct when it is matched and its string (hence fret) is right; Tab P/R/F1 count those.

The `clean` subset keeps only tracks absent from Stage 2's training set (data_dir); with a
player split that is every track.

Usage:
  uv run scripts/e2e/guitarset.py
  uv run scripts/e2e/guitarset.py 'inputs=[gt]' wandb.mode=disabled
"""

import glob
import json

import hydra
import librosa
import mir_eval
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.device import auto_device
from gtp.log import info
from gtp.stage1.inference import SAMPLE_RATE as STAGE1_SR
from gtp.stage1.inference import PianoTranscription
from gtp.stage1.strings import GUITARSET_DIR, PLAYERS, SOURCES, StringClassifier, load_audio, spectrogram
from gtp.stage2.evidence import restrict
from gtp.stage2.fusion import decode_structured, log_ratio, piece_rows, sorted_notes, valid_tabs
from gtp.stage2.inference import load_checkpoint
from gtp.track import start_run


def stage2_train_tracks(data_dir):
    tracks = set()
    with open(REPO_ROOT / data_dir / 'train.jsonl') as fh:
        for line in fh:
            if '"source": "guitarset"' in line[:200]:
                tracks.add(json.loads(line)['filename'].removesuffix('.json'))
    return tracks


def classifier_runs(sweep, test_player):
    runs = [p.removesuffix('/.hydra/config.yaml') for p in glob.glob(str(REPO_ROOT / sweep / '*/*/.hydra/config.yaml'))]
    runs = sorted(r for r in runs if OmegaConf.load(f'{r}/.hydra/config.yaml').test_player == test_player)
    if not runs:
        raise FileNotFoundError(f'No classifier runs for test_player={test_player} under {sweep}')
    return runs


def transcribe(stage1, track, cache_dir):
    path = cache_dir / f'{track}.json'
    if not path.exists():
        subdir, suffix = SOURCES['mic']
        audio, _ = librosa.load(GUITARSET_DIR / subdir / f'{track}{suffix}', sr=STAGE1_SR, mono=True)
        events = stage1.transcribe(audio)['note_events']
        notes = [
            {'pitch': int(e['midi_note']), 'start': float(e['onset_time']), 'end': float(e['offset_time'])}
            for e in events
        ]
        path.write_text(json.dumps(notes))
    return json.loads(path.read_text())


def match(ref, est, tolerance):
    """[(ref_idx, est_idx)] pairs: same pitch, onsets within `tolerance`, offsets ignored."""
    if not ref or not est:
        return []

    def arrays(notes):
        intervals = np.array([[n['start'], max(n['end'], n['start'] + 1e-3)] for n in notes])
        return intervals, librosa.midi_to_hz(np.array([n['pitch'] for n in notes]))

    return mir_eval.transcription.match_notes(*arrays(ref), *arrays(est), onset_tolerance=tolerance, offset_ratio=None)


def prf(tp, n_est, n_ref):
    p, r = tp / max(n_est, 1), tp / max(n_ref, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


def summarize(c, d, source, subset, methods, seeds):
    """Note and tab metrics for one input source and track subset (`c`: per-track counts, `d`: per-note picks)."""
    matched = d[d.ref_string.notna()]
    row = {'input': source, 'subset': subset, 'tracks': len(c), 'ref': c.n_ref.sum(), 'est': c.n_est.sum()}
    row['note_p'], row['note_r'], row['note_f1'] = prf(c.n_matched.sum(), row['est'], row['ref'])
    for m in methods:
        tp = (matched[m] == matched.ref_string).sum()
        row[f'{m}_string_acc'] = tp / max(len(matched), 1)
        row[f'{m}_tab_f1'] = prf(tp, row['est'], row['ref'])[2]
    groups = sorted({m.rsplit('_s', 1)[0] for m in methods if '_s' in m})
    for kind in ('string_acc', 'tab_f1'):
        for group in groups:
            vals = [row[f'{group}_s{s}_{kind}'] for s in seeds]
            row[f'{group}_{kind}'], row[f'{group}_{kind}_std'] = np.mean(vals), np.std(vals, ddof=1)
    return row


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'e2e'), config_name='guitarset')
def main(cfg):
    device = auto_device()
    stage2, vocab, step, max_seq_len = load_checkpoint(REPO_ROOT / cfg.stage2_checkpoint, device)
    evidence_model = None
    if cfg.evidence_checkpoint:
        evidence_model, _, ev_step, ev_max_seq_len = load_checkpoint(REPO_ROOT / cfg.evidence_checkpoint, device)
        assert hasattr(evidence_model, 'embed'), f'{cfg.evidence_checkpoint} was not trained with evidence'
    clf_runs = classifier_runs(cfg.strings_sweep, cfg.test_player)
    seen = stage2_train_tracks(cfg.data_dir)
    player = PLAYERS[cfg.test_player]
    tracks = sorted(p.stem for p in (GUITARSET_DIR / 'processed').glob(f'{player}_*.json'))
    run = start_run(
        cfg,
        job_type='e2e',
        device=device,
        extra={'classifier_runs': clf_runs, 'stage2_step': step, 'clean_tracks': sorted(set(tracks) - seen)},
    )
    classifiers = [StringClassifier.from_run(r, device) for r in clf_runs]
    seeds = list(range(len(classifiers)))
    stage1_cache = REPO_ROOT / 'results' / 'e2e' / 'stage1' / f'{player}_{(REPO_ROOT / cfg.stage1_checkpoint).stem}'
    stage1_cache.mkdir(parents=True, exist_ok=True)
    stage1 = (
        PianoTranscription(str(REPO_ROOT / cfg.stage1_checkpoint), device=device) if 'stage1' in cfg.inputs else None
    )
    info(
        f'{len(tracks)} tracks of player {player}, {len(set(tracks) - seen)} unseen by stage 2, classifiers {clf_runs}'
    )

    table, track_rows = [], []
    for track in tracks:
        gt = json.loads((GUITARSET_DIR / 'processed' / f'{track}.json').read_text())
        spec = spectrogram(load_audio(track, classifiers[0].source), classifiers[0].feature)
        for source in cfg.inputs:
            est = gt['notes'] if source == 'gt' else transcribe(stage1, track, stage1_cache)
            playable = [n for n in est if len(valid_tabs(n['pitch'], gt['tuning'], vocab)[0])]
            pairs = match(gt['notes'], playable, cfg.onset_tolerance)
            ref_of = {e: r for r, e in pairs}
            piece_notes = []
            for i, n in enumerate(playable):
                strings, _ = valid_tabs(n['pitch'], gt['tuning'], vocab)
                ref = gt['notes'][ref_of[i]] if i in ref_of else None
                piece_notes.append(
                    {
                        'pitch': n['pitch'],
                        'start': n['start'],
                        'end': max(n['end'], n['start'] + 0.01),
                        'string': int(strings[0]),
                        'fret': n['pitch'] - gt['tuning'][strings[0] - 1],
                        'ref_string': ref['string'] if ref else None,
                    }
                )
            piece = {**gt, 'filename': track, 'capo': 0, 'notes': piece_notes}
            ordered = sorted_notes(piece)
            probs = np.stack(
                [c.predict(spec, [n['start'] for n in ordered], [n['pitch'] for n in ordered]) for c in classifiers],
                axis=1,
            )
            pitches = [n['pitch'] for n in ordered]
            prior = classifiers[0].prior(pitches)
            evidence = np.stack([restrict(probs[:, s], pitches, gt['tuning']) for s in seeds])
            note_ids, ratios = [], []
            for n, p, q in zip(ordered, probs, prior, strict=True):
                strings, _ = valid_tabs(n['pitch'], gt['tuning'], vocab)
                has_audio = bool(np.isfinite(q).all())
                record = {
                    'input': source,
                    'track': track,
                    'pitch': n['pitch'],
                    'start': n['start'],
                    'n_cand': len(strings),
                }
                record['ref_string'] = n['ref_string']
                for s in seeds:
                    record[f'audio_s{s}'] = int(strings[p[s, strings - 1].argmax()]) if has_audio else None
                note_ids.append(len(table))
                ratios.append(log_ratio(p, q, strings, cfg.prob_floor) if has_audio else None)
                if not has_audio:
                    evidence[:, len(note_ids) - 1] = np.nan
                table.append(record)
            track_rows.append(
                {
                    'input': source,
                    'track': track,
                    'rows': piece_rows(piece, vocab, max_seq_len, note_ids, ratios),
                    'evidence_rows': [
                        piece_rows(piece, vocab, ev_max_seq_len, note_ids, ratios, evidence[s]) for s in seeds
                    ]
                    if evidence_model
                    else None,
                    'n_ref': len(gt['notes']),
                    'n_est': len(est),
                    'n_matched': len(pairs),
                }
            )
        done = track_rows[-len(cfg.inputs) :]
        info(
            f'{track}: {len(gt["notes"])} ref, '
            + ', '.join(f'{r["input"]} {r["n_est"]} est {r["n_matched"]} matched' for r in done)
        )
    info(f'stage 2 step {step}' + (f', evidence model step {ev_step}' if evidence_model else ''))

    df = pd.DataFrame(table)
    rows = [row for t in track_rows for row in t['rows']]
    df['stage2'] = pd.Series(decode_structured(stage2, rows, device, cfg.batch))
    for s in seeds:
        df[f'fused_s{s}'] = pd.Series(decode_structured(stage2, rows, device, cfg.batch, cfg.lam, s))
    methods = ['stage2', *[f'audio_s{s}' for s in seeds], *[f'fused_s{s}' for s in seeds]]
    if evidence_model:
        for s in seeds:
            ev_rows = [row for t in track_rows for row in t['evidence_rows'][s]]
            df[f'evidence_s{s}'] = pd.Series(decode_structured(evidence_model, ev_rows, device, cfg.batch))
        methods += [f'evidence_s{s}' for s in seeds]

    counts = pd.DataFrame([{k: v for k, v in t.items() if k not in ('rows', 'evidence_rows')} for t in track_rows])
    counts['clean'] = ~counts.track.isin(seen)
    df['clean'] = ~df.track.isin(seen)
    summary = []
    for source in cfg.inputs:
        for subset in ('all', 'clean'):
            c, d = counts[counts.input == source], df[df.input == source]
            if subset == 'clean':
                c, d = c[c.clean], d[d.clean]
            summary.append(summarize(c, d, source, subset, methods, seeds))
    summary = pd.DataFrame(summary)
    cols = ['input', 'subset', 'tracks', 'ref', 'est', 'note_f1']
    groups = ['stage2', 'audio', 'fused'] + (['evidence'] if evidence_model else [])
    cols += [f'{m}_{k}' for k in ('string_acc', 'tab_f1') for m in groups]
    info(f'\n{summary[cols].round(4).to_string(index=False)}')

    df.to_csv(run.dir / 'notes.csv.gz', index=False)
    counts.to_csv(run.dir / 'tracks.csv', index=False)
    summary.to_csv(run.dir / 'summary.csv', index=False)
    flat = {}
    for r in summary.to_dict('records'):
        flat |= {f'{r["input"]}/{r["subset"]}/{k}': v for k, v in r.items() if k not in ('input', 'subset')}
    (run.dir / 'summary.json').write_text(json.dumps(flat, indent=1, default=float))
    run.finish(flat)


if __name__ == '__main__':
    main()

"""End-to-end demo: record audio → transcribe → render tabs (ASCII + PDF).

Each note's string comes from Stage 2's beam search, rescored with string evidence heard in the recording: the
string classifiers, the notes' inharmonicity and their harmonic levels, fitted to the guitar without labels
(gtp.stage1.string_evidence). With --guitar, earlier takes of that guitar are fitted together with the new one:
the more of a guitar's playing is pooled, the better its strings are told apart.

Every run also writes a take folder (data/takes/<take_id>/, see gtp.takes) that the tab editor
opens: scripts/takes/editor.py.

Usage:
  python scripts/demo.py                           # record 10s, run pipeline
  python scripts/demo.py --duration 15
  python scripts/demo.py --audio path/to/take.wav  # skip recording
  python scripts/demo.py --capo 3 --tuning 64 59 55 50 45 38
  python scripts/demo.py --guitar hagstrom         # fit the evidence together with earlier takes of this guitar
  python scripts/demo.py --no-evidence             # Stage 2 alone, no string evidence from the audio
  python scripts/demo.py --slug strat-clean        # take id 20261109_101500_strat-clean
  python scripts/demo.py --audio old.wav --take-id 20260917_095751  # take from an existing recording

Requires `sounddevice` for the recording path:
  pip install sounddevice
"""

import argparse
import glob
import sys
import time
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from gtp import REPO_ROOT, takes
from gtp.device import auto_device
from gtp.stage1 import strings
from gtp.stage1.inference import SAMPLE_RATE as STAGE1_SR
from gtp.stage1.inference import PianoTranscription
from gtp.stage1.string_evidence import classifier_probs, fuse_recordings, physical, settings
from gtp.stage2.fusion import decode_beam, log_ratio, piece_rows, sorted_notes, valid_tabs
from gtp.stage2.genres import GENRES, UNKNOWN
from gtp.stage2.inference import (
    build_anchor_prefix,
    generate_tabs,
    load_checkpoint,
)
from gtp.stage2.metrics import difficulty_score
from gtp.stage2.postprocess import correct_tabs
from gtp.stage2.tokenizer import tokenize_piece_for_inference
from gtp.track import git_state

STAGE1_DEFAULT = REPO_ROOT / 'models' / 'finetuned' / 'z3_step_0055000.pth'
STAGE2_DEFAULT = (
    REPO_ROOT / 'runs' / 'stage2' / '2026-11-08_11-58-31_p05_baseline' / 'checkpoints' / 'step_0060000_final.pth'
)
CLASSIFIER_RUNS_DEFAULT = 'results/strings/runs/*al-roomfar_s*'
DEFAULT_OUT_DIR = REPO_ROOT / 'results' / 'demo_takes'
DEFAULT_TAKES_DIR = REPO_ROOT / 'data' / 'takes'
STD_TUNING = [64, 59, 55, 50, 45, 40]
SR = 44100


# ---------------------------------------------------------------------------
# Recording (lazy import sounddevice so --audio path doesn't require it)
# ---------------------------------------------------------------------------


def record_with_retry(duration, sr=SR):
    try:
        import sounddevice as sd
    except ImportError as e:
        print(f'sounddevice not installed: {e}\nInstall with: pip install sounddevice', file=sys.stderr)
        sys.exit(1)

    while True:
        input(f'\nPress ENTER to record {duration}s... ')
        print('Recording... go!')
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        peak = float(np.max(np.abs(audio)))
        print(f'Done. Duration={duration}s peak={peak:.3f}')

        # print('Playing back...')
        # sd.play(audio, sr)
        # sd.wait()

        resp = input('Accept? [y]es / [n]o (retry) / [q]uit: ').strip().lower()
        if resp == 'y':
            return audio
        if resp == 'q':
            sys.exit(0)


def input_device_name():
    import sounddevice as sd

    return sd.query_devices(kind='input')['name']


# ---------------------------------------------------------------------------
# Stage 1 + 2 chain
# ---------------------------------------------------------------------------


def stage1(audio, checkpoint, device):
    print('Stage 1: audio → MIDI...')
    transcriptor = PianoTranscription(checkpoint_path=str(checkpoint), device=device)
    result = transcriptor.transcribe(librosa.resample(audio, orig_sr=SR, target_sr=STAGE1_SR))
    note_events = result['note_events']
    print(f'  detected {len(note_events)} notes')
    return note_events


def notes_to_piece(note_events, tuning, capo, tempo, genre=UNKNOWN):
    notes = sorted(
        (
            {
                'pitch': int(e['midi_note']),
                'start': float(e['onset_time']),
                'end': float(e['offset_time']),
                'velocity': int(e['velocity']),
            }
            for e in note_events
        ),
        key=lambda n: (n['start'], n['pitch']),
    )
    return {'tuning': tuning, 'tempo': tempo, 'capo': capo, 'genre': genre, 'notes': notes}


def stage2(piece, checkpoint, device, anchor_tabs=None, fallback='first_viable'):
    """Returns (corrected_tabs, raw_tabs, enc_subseqs, dec_subseqs, sources).

    If `anchor_tabs` is provided (list of (string, fret) for the first N notes),
    the decoder is primed with those tabs (with proper TIME_SHIFTs between them)
    so generation continues with that style.

    `sources` is a parallel list to `corrected_tabs` with values
    'unchanged' / 'window_swap' / 'fallback' — see correct_tabs for meanings.

    `fallback` selects the post-processing fallback strategy: 'first_viable'
    (paper-faithful) or 'nearest_viable' (deviation; Manhattan-nearest to raw).
    """
    print('Stage 2: notes → tabs...')
    model, vocab, iteration, max_seq_len = load_checkpoint(str(checkpoint), device)
    print(f'  checkpoint iteration: {iteration}  max_seq_len: {max_seq_len}')

    enc_subseqs = tokenize_piece_for_inference(piece, vocab, max_seq_len=max_seq_len)

    decoder_prefix = build_anchor_prefix(piece, vocab, anchor_tabs) if anchor_tabs else None
    if decoder_prefix is not None:
        print(f'  anchoring first {len(anchor_tabs)} notes ({len(decoder_prefix) - 1} prefix tokens)')

    raw_tabs, dec_subseqs = generate_tabs(
        model, vocab, enc_subseqs, device, max_seq_len=max_seq_len, return_raw=True, decoder_prefix=decoder_prefix
    )

    sorted_notes = sorted(piece['notes'], key=lambda x: (x['start'], x['pitch']))
    input_pitches = [n['pitch'] for n in sorted_notes]
    corrected_tabs, sources = correct_tabs(
        input_pitches,
        raw_tabs,
        piece['tuning'],
        return_sources=True,
        fallback=fallback,
    )

    n_unchanged = sum(1 for s in sources if s == 'unchanged')
    n_swap = sum(1 for s in sources if s == 'window_swap')
    n_fallback = sum(1 for s in sources if s == 'fallback')
    print(
        f'  encoder sub-seqs: {len(enc_subseqs)}  '
        f'raw tabs: {len(raw_tabs)}  corrected: {len(corrected_tabs)}  '
        f'unplayable: {sum(1 for t in corrected_tabs if t is None)}'
    )
    print(f'  postproc sources: unchanged={n_unchanged}  window_swap={n_swap}  fallback={n_fallback}')

    d_raw = difficulty_score(raw_tabs)
    d_pp = difficulty_score(corrected_tabs)
    if d_raw is not None and d_pp is not None:
        print(f'  difficulty: raw={d_raw:.3f}  pp={d_pp:.3f}')
    else:
        print('  difficulty: (need ≥2 valid tabs)')
    return corrected_tabs, raw_tabs, enc_subseqs, dec_subseqs, sources, vocab


def earlier_takes(takes_dir, guitar):
    """[(audio path, notes)] of the takes recorded on `guitar`: their edited notes if saved, else the predicted ones."""
    out = []
    for d in sorted(Path(takes_dir).iterdir()) if guitar and Path(takes_dir).exists() else []:
        if not (d / 'prediction.json').exists():
            continue
        take = takes.load(d)
        notes = (take['edited'] or take['prediction'])['notes']
        if take['meta'].get('guitar', '').strip().lower() == guitar.strip().lower() and notes:
            out.append((d / 'audio.wav', sorted(notes, key=lambda n: (n['start'], n['pitch']))))
    return out


def string_evidence(wav_path, piece, runs, pooled, device):
    """(n_notes, 6) fused string probabilities for sorted_notes(piece) and the classifier's pitch prior.

    `pooled` [(audio path, notes)] are other recordings of the same guitar: the string constants and envelopes are
    fitted over them and this take together.
    """
    print(f'String evidence: {len(runs)} classifiers, {len(pooled)} earlier takes of this guitar...')
    classifiers = [strings.StringClassifier.from_run(run, device) for run in runs]
    profiles, physics, envelope = settings()
    parts = []
    for audio, notes in [(wav_path, sorted_notes(piece)), *pooled]:
        onsets, pitches = [n['start'] for n in notes], [n['pitch'] for n in notes]
        probs = classifier_probs(audio, onsets, pitches, classifiers)
        parts.append((probs, *physical(audio, onsets, [n['end'] for n in notes], pitches, profiles, envelope), pitches))
    probs, curves, levels, pitches = zip(*parts, strict=True)
    fused, _, family = fuse_recordings(probs, curves, levels, pitches, profiles, physics, envelope)
    print(f'  {sum(len(p) for p in pitches)} notes fitted together, {family} string set')
    return fused[0], classifiers[0].prior(pitches[0])


def stage2_fused(piece, model, vocab, max_seq_len, fused, prior, lam, beams, device):
    """Choose each note's string by beam search along the skeleton its notes fix, rescored with the string
    evidence (weight lam); returns tabs in sorted_notes(piece) order."""
    print('Stage 2 (with string evidence): notes → tabs...')
    tuning = piece['tuning']
    # tokenize_piece needs a string/fret per note; these placeholders are replaced by the model's choices.
    notes = []
    for n in piece['notes']:
        s = int(valid_tabs(n['pitch'], tuning, vocab)[0][0])
        notes.append({**n, 'string': s, 'fret': n['pitch'] - tuning[s - 1]})
    piece = {**piece, 'notes': notes}
    ordered = sorted_notes(piece)
    ratios = [
        log_ratio(f[None], q, valid_tabs(n['pitch'], tuning, vocab)[0], 1e-4)
        for n, f, q in zip(ordered, fused, prior, strict=True)
    ]
    rows = piece_rows(piece, vocab, max_seq_len, list(range(len(ordered))), ratios)
    picks = decode_beam(model, rows, device, 64, lam, beams=beams)
    tabs = [(picks[i], note['pitch'] - tuning[picks[i] - 1]) for i, note in enumerate(ordered)]
    agree = np.mean([s == f.argmax() + 1 for (s, _), f in zip(tabs, fused, strict=True)])
    print(f'  Stage 2 agrees with the audio evidence on {agree:.0%} of {len(tabs)} notes')
    return tabs, [r.enc for r in rows]


# ---------------------------------------------------------------------------
# Output writers: WAV, MIDI, ASCII tab, PDF
# ---------------------------------------------------------------------------


def write_wav(audio, path, sr=SR):
    sf.write(str(path), audio, sr)


def write_midi(note_events, path, program=24):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for ev in note_events:
        inst.notes.append(
            pretty_midi.Note(
                velocity=int(ev.get('velocity', 80)),
                pitch=int(ev['midi_note']),
                start=float(ev['onset_time']),
                end=max(float(ev['offset_time']), float(ev['onset_time']) + 0.05),
            )
        )
    midi.instruments.append(inst)
    midi.write(str(path))


def render_ascii_tab(tabs, notes=None, columns_per_line=16, col_width=4):
    """Render predicted tabs as 6-line guitar tab notation.

    If `notes` is provided (sorted alongside `tabs`), groups same-onset notes
    (gap ≤ 30ms) into a single column so chords stack vertically. Otherwise one
    column per note. Wraps every `columns_per_line` columns.
    """
    if not tabs:
        return '(no tabs)'

    # Build columns: each column is a list of (string, fret) tuples to render together.
    if notes is not None:
        sorted_notes = sorted(notes, key=lambda n: (n['start'], n['pitch']))
        groups = group_chords(sorted_notes, max_gap=0.030)
        note_to_idx = {id(n): i for i, n in enumerate(sorted_notes)}
        columns = []
        for g in groups:
            col_tabs = [tabs[note_to_idx[id(n)]] for n in g if note_to_idx[id(n)] < len(tabs)]
            columns.append(col_tabs)
    else:
        columns = [[t] for t in tabs]

    string_labels = ['e', 'B', 'G', 'D', 'A', 'E']  # string 1 → 6, top → bottom
    out_lines = []
    n_chunks = (len(columns) + columns_per_line - 1) // columns_per_line
    for chunk_i in range(n_chunks):
        start = chunk_i * columns_per_line
        end = min(start + columns_per_line, len(columns))
        chunk_cols = columns[start:end]
        rows = [f'{lbl}|' for lbl in string_labels]
        for col in chunk_cols:
            cells = ['-' * col_width] * 6  # default: dashes on every string
            for entry in col:
                if entry is None:
                    continue
                s, f = entry
                if 1 <= s <= 6:
                    fret_str = str(f)
                    cells[s - 1] = fret_str + '-' * (col_width - len(fret_str))
            for r in range(6):
                rows[r] += cells[r]
        for r in range(6):
            rows[r] += '|'
        out_lines.extend(rows)
        out_lines.append('')
    return '\n'.join(out_lines).rstrip()


def group_chords(notes, max_gap=0.030):
    """Group consecutive notes whose onsets are within `max_gap` seconds (a 'chord')."""
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: n['start'])
    groups = [[sorted_notes[0]]]
    for n in sorted_notes[1:]:
        if n['start'] - groups[-1][-1]['start'] <= max_gap:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def write_debug_log(path, vocab, note_events, piece, enc_subseqs, dec_subseqs, raw_tabs, corrected_tabs, sources):
    """Dump full pipeline state for diagnosing failures (esp. missed chord notes)."""
    lines = []
    lines.append(f'=== Stage 1 output: {len(note_events)} notes ===')
    for i, ev in enumerate(note_events):
        lines.append(
            f'  n={i:>4d}  pitch={int(ev["midi_note"]):>3d}  '
            f'start={float(ev["onset_time"]):>7.3f}s  '
            f'end={float(ev["offset_time"]):>7.3f}s'
        )

    n_unchanged = sum(1 for s in sources if s == 'unchanged')
    n_swap = sum(1 for s in sources if s == 'window_swap')
    n_fallback = sum(1 for s in sources if s == 'fallback')
    lines.append('')
    lines.append(
        f'=== Postproc source summary: unchanged={n_unchanged}  window_swap={n_swap}  fallback={n_fallback} ==='
    )

    d_raw = difficulty_score(raw_tabs)
    d_pp = difficulty_score(corrected_tabs)
    lines.append('')
    lines.append('=== Difficulty score (paper §3.6, range 0-18.5; lower = easier to play) ===')
    lines.append(f'  raw model output: {d_raw:.3f}' if d_raw is not None else '  raw model output: --')
    lines.append(f'  post-processed:   {d_pp:.3f}' if d_pp is not None else '  post-processed:   --')

    sorted_notes = sorted(piece['notes'], key=lambda n: (n['start'], n['pitch']))
    note_to_idx = {id(n): i for i, n in enumerate(sorted_notes)}

    swap_or_fallback = [(i, sources[i]) for i in range(len(sources)) if sources[i] != 'unchanged']
    if swap_or_fallback:
        lines.append('')
        lines.append('=== Per-note postproc actions (only entries the postproc touched) ===')
        for i, src in swap_or_fallback:
            n = sorted_notes[i]
            raw = raw_tabs[i] if i < len(raw_tabs) else None
            lines.append(
                f'  i={i:>4d}  start={n["start"]:>6.3f}s  pitch={int(n["pitch"]):>3d}  '
                f'raw={raw}  corrected={corrected_tabs[i]}  src={src}'
            )

    chord_groups = group_chords(piece['notes'])
    polyphonic = [g for g in chord_groups if len(g) > 1]
    lines.append('')
    lines.append(f'=== Chord-onset groups (gap ≤ 30ms): {len(chord_groups)} total, {len(polyphonic)} polyphonic ===')
    for gi, g in enumerate(chord_groups):
        if len(g) == 1:
            continue
        pitches = sorted(int(n['pitch']) for n in g)
        idxs = sorted(note_to_idx[id(n)] for n in g)
        tabs_for_group = [corrected_tabs[i] for i in idxs]
        raw_tabs_for_group = [raw_tabs[i] if i < len(raw_tabs) else None for i in idxs]
        srcs_for_group = [sources[i] for i in idxs]
        lines.append(
            f'  chord {gi:>3d} @ {g[0]["start"]:>6.3f}s  '
            f'pitches={pitches}  raw_tabs={raw_tabs_for_group}  '
            f'corrected={tabs_for_group}  src={srcs_for_group}'
        )

    for si, (enc, dec) in enumerate(zip(enc_subseqs, dec_subseqs, strict=False)):
        lines.append('')
        lines.append(f'=== Stage 2 sub-seq {si + 1}/{len(enc_subseqs)} ===')
        lines.append(f'-- encoder ({len(enc)} tokens) --')
        lines.append(' '.join(vocab.decode(int(t)) for t in enc))
        lines.append(f'-- decoder ({len(dec)} tokens, raw model output) --')
        lines.append(' '.join(vocab.decode(int(t)) for t in dec))

    Path(path).write_text('\n'.join(lines) + '\n')


def write_pdf_tab(ascii_tab, header, path, lines_per_page=42, line_width_chars=80):
    """Write a multi-page PDF rendering the ASCII tab in monospace."""
    full_text = header + '\n\n' + ascii_tab
    lines = full_text.split('\n')
    pages = []
    i = 0
    while i < len(lines):
        pages.append('\n'.join(lines[i : i + lines_per_page]))
        i += lines_per_page

    with PdfPages(str(path)) as pdf:
        for page_text in pages:
            fig = Figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(
                0.05,
                0.95,
                page_text,
                family='monospace',
                fontsize=9,
                va='top',
                ha='left',
                transform=ax.transAxes,
            )
            pdf.savefig(fig)


def repo_relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def write_take(args, take_id, audio, sr, tuning, placed, unplaced, pipeline):
    """Take folder for the editor: `placed` notes carry string/fret/evidence, `unplaced` got no tab."""
    prediction = takes.prediction_doc(take_id, placed, tuning, args.capo, args.tempo, pipeline, unplaced)
    meta = {
        'input_device': '' if args.audio else input_device_name(),
        'capo': str(args.capo),
        'tuning_name': 'standard' if args.tuning == STD_TUNING else ' '.join(map(str, args.tuning)),
        'style': '' if args.genre == UNKNOWN else args.genre,
        'source': repo_relative(args.audio) if args.audio else 'mic',
        'guitar': args.guitar or '',
    }
    d = takes.create(args.takes_dir, take_id, audio, sr, prediction, meta)
    print(f'Wrote take: {d} ({len(placed)} notes, {len(unplaced)} without a tab)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=int, default=10, help='Recording length in seconds')
    ap.add_argument('--audio', default=None, help='Audio file path (skips mic recording)')
    ap.add_argument('--output-dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--takes-dir', default=str(DEFAULT_TAKES_DIR), help='Root of the take folders for the editor')
    ap.add_argument('--take-id', default=None, help='Take folder name (default: timestamp, plus --slug)')
    ap.add_argument('--slug', default=None, help='Suffix for the default take id, e.g. the guitar')
    ap.add_argument('--stage1-checkpoint', default=str(STAGE1_DEFAULT))
    ap.add_argument('--stage2-checkpoint', default=str(STAGE2_DEFAULT))
    ap.add_argument('--classifier-runs', default=CLASSIFIER_RUNS_DEFAULT, help='Glob of string-classifier runs')
    ap.add_argument('--guitar', default=None, help='Guitar name: its earlier takes are fitted together with this one')
    ap.add_argument('--lam', type=float, default=1.3, help='Weight of the string evidence in the beam search')
    ap.add_argument('--beams', type=int, default=8)
    ap.add_argument('--no-evidence', action='store_true', help='Stage 2 alone, without string evidence from the audio')
    ap.add_argument('--tuning', nargs=6, type=int, default=STD_TUNING, help='Open-string pitches (high E to low E)')
    ap.add_argument('--capo', type=int, default=0)
    ap.add_argument('--tempo', type=float, default=None, help='Override tempo (None = unknown, model handles)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--columns-per-line', type=int, default=16, help='ASCII tab columns per wrapped line')
    ap.add_argument(
        '--anchor',
        default=None,
        help='Lock first N notes. Format: "string:fret,string:fret,..."  '
        '(e.g. "4:3,3:3,2:6,1:5" for a 4-note chord on D/G/B/E strings)',
    )
    ap.add_argument(
        '--show-alternatives',
        type=int,
        default=0,
        help='Top-K alternatives at each generation step → .alternatives.txt',
    )
    ap.add_argument(
        '--fallback',
        choices=['first_viable', 'nearest_viable'],
        default='first_viable',
        help='Post-processing fallback strategy. first_viable = paper-faithful. '
        'nearest_viable = deviation: Manhattan-nearest realization to model raw output.',
    )
    ap.add_argument(
        '--genre',
        choices=list(GENRES),
        default=UNKNOWN,
        help='Coarse genre conditioning hint for the model. Only used by genre-aware '
        'checkpoints (the flag is silently ignored for legacy models). Default: unknown.',
    )
    args = ap.parse_args()

    anchor_tabs = None
    if args.anchor:
        anchor_tabs = []
        for pair_str in args.anchor.split(','):
            s, f = pair_str.strip().split(':')
            anchor_tabs.append((int(s), int(f)))

    device = args.device or auto_device()
    print(f'Device: {device}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'take_{stamp}'
    take_id = args.take_id or takes.new_take_id(args.slug)
    if takes.take_dir(args.takes_dir, take_id).exists():
        sys.exit(f'Take {take_id} already exists in {args.takes_dir}')

    # 1) Audio
    if args.audio:
        print(f'Loading audio: {args.audio}')
        captured, captured_sr = librosa.load(args.audio, sr=None, mono=True)
        audio = librosa.resample(captured, orig_sr=captured_sr, target_sr=SR) if captured_sr != SR else captured
        print(f'  duration={len(audio) / SR:.2f}s')
    else:
        audio = record_with_retry(args.duration, SR)
        captured, captured_sr = audio, SR

    wav_path = out_dir / f'{stem}.wav'
    write_wav(audio, wav_path)
    print(f'Wrote: {wav_path}')

    sha, diff = git_state()
    pipeline = {
        'stage1_checkpoint': repo_relative(args.stage1_checkpoint),
        'stage2_checkpoint': repo_relative(args.stage2_checkpoint),
        'stage2_evidence': None,
        'no_evidence': args.no_evidence,
        'classifier_runs': [],
        'pooled_takes': [],
        'genre': args.genre,
        'anchor': args.anchor,
        'fallback': None,
        'device': device,
        'git_sha': sha,
        'git_dirty': bool(diff),
    }
    tuning_with_capo = [t + args.capo for t in args.tuning]  # our convention: tuning includes capo

    # 2) Stage 1: audio → MIDI
    # Stage 1's recurrent layers run several times faster on the CPU than on Apple's GPU.
    note_events = stage1(audio, args.stage1_checkpoint, 'cpu' if device == 'mps' else device)
    if not note_events:
        print('No notes detected, exiting.')
        write_take(args, take_id, captured, captured_sr, tuning_with_capo, [], [], pipeline)
        return

    mid_path = out_dir / f'{stem}.mid'
    write_midi(note_events, mid_path)
    print(f'Wrote: {mid_path}')

    # 3) Stage 2: MIDI → tabs
    piece = notes_to_piece(note_events, tuning_with_capo, args.capo, args.tempo, genre=args.genre)
    t0 = time.time()
    model, vocab, _, max_seq_len = load_checkpoint(str(args.stage2_checkpoint), device)
    if hasattr(model, 'embed'):
        sys.exit(
            'This Stage 2 checkpoint reads evidence in its encoder; the demo fuses evidence at decoding: use a plain one'
        )
    if not args.no_evidence and not anchor_tabs:
        if args.capo or args.tuning != STD_TUNING:
            print('  warning: the string evidence assumes standard tuning without a capo')
        unplaced = [n for n in piece['notes'] if not len(valid_tabs(n['pitch'], piece['tuning'], vocab)[0])]
        piece['notes'] = [
            {**n, 'end': max(n['end'], n['start'] + 0.01)}
            for n in piece['notes']
            if len(valid_tabs(n['pitch'], piece['tuning'], vocab)[0])
        ]
        runs = sorted(glob.glob(str(REPO_ROOT / args.classifier_runs)))
        if not runs:
            sys.exit(f'No classifier runs match {args.classifier_runs}')
        pooled = earlier_takes(args.takes_dir, args.guitar)
        fused, prior = string_evidence(wav_path, piece, runs, pooled, device)
        pipeline |= {
            'classifier_runs': [repo_relative(r) for r in runs],
            'pooled_takes': [a.parent.name for a, _ in pooled],
            'lam': args.lam,
            'beams': args.beams,
        }
        tabs, enc_subseqs = stage2_fused(piece, model, vocab, max_seq_len, fused, prior, args.lam, args.beams, device)
        raw_tabs, dec_subseqs, sources = tabs, [], ['unchanged'] * len(tabs)
        pipeline['stage2_evidence'] = True
        placed = [
            {**n, 'string': s, 'fret': f, 'evidence': e}
            for n, (s, f), e in zip(sorted_notes(piece), tabs, fused, strict=True)
        ]
    else:
        tabs, raw_tabs, enc_subseqs, dec_subseqs, sources, vocab = stage2(
            piece, args.stage2_checkpoint, device, anchor_tabs=anchor_tabs, fallback=args.fallback
        )
        pipeline |= {'stage2_evidence': False, 'fallback': args.fallback}
        notes = sorted_notes(piece)
        padded_tabs = list(tabs) + [None] * (len(notes) - len(tabs))
        placed = [{**n, 'string': t[0], 'fret': t[1]} for n, t in zip(notes, padded_tabs, strict=True) if t]
        unplaced = [n for n, t in zip(notes, padded_tabs, strict=True) if t is None]
    print(f'  stage 2 elapsed: {time.time() - t0:.1f}s')
    write_take(args, take_id, captured, captured_sr, tuning_with_capo, placed, unplaced, pipeline)

    # 4) Debug log
    debug_path = out_dir / f'{stem}.debug.txt'
    write_debug_log(debug_path, vocab, note_events, piece, enc_subseqs, dec_subseqs, raw_tabs, tabs, sources)
    print(f'Wrote: {debug_path}')

    # 5) Output: ASCII tab + PDF
    ascii_tab = render_ascii_tab(tabs, notes=piece['notes'], columns_per_line=args.columns_per_line)
    txt_path = out_dir / f'{stem}.tab.txt'
    header = (
        f'tab take: {stamp}\ntuning: {args.tuning}  capo: {args.capo}  tempo: {args.tempo}\nnotes: {len(note_events)}'
    )
    txt_path.write_text(header + '\n\n' + ascii_tab + '\n')
    print(f'Wrote: {txt_path}')

    pdf_path = out_dir / f'{stem}.tab.pdf'
    write_pdf_tab(ascii_tab, header, pdf_path)
    print(f'Wrote: {pdf_path}')

    print(f'\nDone! Outputs in {out_dir}/')
    print('\n--- ASCII tab preview (first 13 lines) ---')
    print('\n'.join(ascii_tab.split('\n')[:13]))


if __name__ == '__main__':
    main()

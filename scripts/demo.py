"""End-to-end demo: record audio → transcribe → render tabs (ASCII + PDF).

The transcription is gtp.pipeline: Stage 1 notes, then each note's string from Stage 2's beam search rescored with
string evidence heard in the recording. With --guitar, earlier takes of that guitar are fitted together with the new
one: the more of a guitar's playing is pooled, the better its strings are told apart.

Every run also writes a take folder (data/takes/<take_id>/, see gtp.takes) that the tab editor
opens: scripts/takes/editor.py.

Usage:
  python scripts/demo.py                           # SPACE starts the recording, SPACE ends it, then the pipeline runs
  python scripts/demo.py --duration 15             # record a fixed 15s instead
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
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from gtp import REPO_ROOT, takes
from gtp.pipeline import (
    CLASSIFIER_RUNS_DEFAULT,
    SR,
    STAGE1_DEFAULT,
    STAGE2_DEFAULT,
    STD_TUNING,
    Pipeline,
    earlier_takes,
    repo_relative,
)
from gtp.stage2.genres import GENRES, UNKNOWN
from gtp.stage2.metrics import difficulty_score

DEFAULT_OUT_DIR = REPO_ROOT / 'results' / 'demo_takes'
DEFAULT_TAKES_DIR = REPO_ROOT / 'data' / 'takes'


# ---------------------------------------------------------------------------
# Recording (lazy import sounddevice so --audio path doesn't require it)
# ---------------------------------------------------------------------------


def space_pressed(fd, timeout):
    """True if SPACE arrives on the terminal within `timeout` seconds (None waits for a key); q quits."""
    if not select.select([fd], [], [], timeout)[0]:
        return False
    key = os.read(fd, 1)
    if key == b'q':
        sys.exit(0)
    return key == b' '


def record_between_spaces(sd, sr):
    """Record from one SPACE press to the next."""
    if not sys.stdin.isatty():
        sys.exit('Recording by SPACE needs a terminal: pass --duration or --audio')
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    chunks = []
    try:
        tty.setcbreak(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        print('\nPress SPACE to start recording (q quits)... ', end='', flush=True)
        while not space_pressed(fd, None):
            pass
        with sd.InputStream(
            samplerate=sr, channels=1, dtype='float32', callback=lambda data, *_: chunks.append(data[:, 0].copy())
        ):
            t0 = time.time()
            while not space_pressed(fd, 0.1):
                print(f'\rRecording {time.time() - t0:5.1f}s, press SPACE to stop ', end='', flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    print()
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype='float32')


def record_with_retry(duration, sr=SR):
    """One accepted recording: `duration` seconds, or from one SPACE press to the next when it is None."""
    try:
        import sounddevice as sd
    except ImportError as e:
        print(f'sounddevice not installed: {e}\nInstall with: pip install sounddevice', file=sys.stderr)
        sys.exit(1)

    while True:
        if duration is None:
            audio = record_between_spaces(sd, sr)
        else:
            input(f'\nPress ENTER to record {duration}s... ')
            print('Recording... go!')
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        print(f'Done. Duration={len(audio) / sr:.1f}s peak={peak:.3f}')

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
# Output writers: WAV, MIDI, ASCII tab, PDF
# ---------------------------------------------------------------------------


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


def write_take(args, take_id, audio, sr, result):
    """Take folder for the editor from a Pipeline.run() result."""
    placed, unplaced = result['placed'], result['unplaced']
    prediction = takes.prediction_doc(
        take_id, placed, result['tuning'], args.capo, args.tempo, result['pipeline'], unplaced
    )
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
    ap.add_argument(
        '--duration', type=int, default=None, help='Fixed recording length in seconds (default: SPACE starts and ends)'
    )
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'take_{stamp}'
    take_id = args.take_id or takes.new_take_id(args.slug)
    if takes.take_dir(args.takes_dir, take_id).exists():
        sys.exit(f'Take {take_id} already exists in {args.takes_dir}')

    pipe = Pipeline(args.stage1_checkpoint, args.stage2_checkpoint, args.classifier_runs, args.device)
    print(f'Device: {pipe.device}')

    # 1) Audio
    if args.audio:
        print(f'Loading audio: {args.audio}')
        captured, captured_sr = librosa.load(args.audio, sr=None, mono=True)
        print(f'  duration={len(captured) / captured_sr:.2f}s')
    else:
        captured, captured_sr = record_with_retry(args.duration, SR), SR

    # 2) Stage 1 + 2: audio → notes → tabs
    wav_path = out_dir / f'{stem}.wav'
    t0 = time.time()
    result = pipe.run(
        captured,
        captured_sr,
        wav_path,
        tuning=args.tuning,
        capo=args.capo,
        tempo=args.tempo,
        genre=args.genre,
        lam=args.lam,
        beams=args.beams,
        evidence=not args.no_evidence,
        anchor=args.anchor,
        fallback=args.fallback,
        pooled=earlier_takes(args.takes_dir, args.guitar),
    )
    print(f'Wrote: {wav_path}  (pipeline {time.time() - t0:.1f}s)')
    write_take(args, take_id, captured, captured_sr, result)
    note_events = result['note_events']
    if not note_events:
        return

    mid_path = out_dir / f'{stem}.mid'
    write_midi(note_events, mid_path)
    print(f'Wrote: {mid_path}')

    # 3) Debug log
    tabs, piece = result['tabs'], result['piece']
    debug_path = out_dir / f'{stem}.debug.txt'
    write_debug_log(
        debug_path,
        pipe.vocab,
        note_events,
        piece,
        result['enc_subseqs'],
        result['dec_subseqs'],
        result['raw_tabs'],
        tabs,
        result['sources'],
    )
    print(f'Wrote: {debug_path}')

    # 4) Output: ASCII tab + PDF
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

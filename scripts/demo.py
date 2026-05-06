"""End-to-end demo: record audio → transcribe → render tabs (ASCII + PDF).

Usage:
  python scripts/demo.py                           # record 10s, run pipeline
  python scripts/demo.py --duration 15
  python scripts/demo.py --audio path/to/take.wav  # skip recording
  python scripts/demo.py --capo 3 --tuning 64 59 55 50 45 38

Requires `sounddevice` for the recording path:
  pip install sounddevice
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from gtp import REPO_ROOT
from gtp.stage1.inference import PianoTranscription
from gtp.stage2.inference import (
    build_anchor_prefix,
    generate_tabs,
    generate_with_alternatives,
    load_checkpoint,
    tokenize_for_inference,
)
from gtp.stage2.genres import GENRES, UNKNOWN
from gtp.stage2.metrics import difficulty_score
from gtp.stage2.postprocess import correct_tabs

STAGE1_DEFAULT = REPO_ROOT / 'models' / 'finetuned' / 'step_0070000_final.pth'
STAGE2_DEFAULT = REPO_ROOT / 'runs' / 'stage2_001' / 'step_0060000_final.pth'
DEFAULT_OUT_DIR = REPO_ROOT / 'results' / 'demo_takes'
STD_TUNING = [64, 59, 55, 50, 45, 40]
SR = 16000


def auto_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


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


# ---------------------------------------------------------------------------
# Stage 1 + 2 chain
# ---------------------------------------------------------------------------


def stage1(audio, checkpoint, device):
    print('Stage 1: audio → MIDI...')
    transcriptor = PianoTranscription(checkpoint_path=str(checkpoint), device=device)
    result = transcriptor.transcribe(audio)
    note_events = result['note_events']
    print(f'  detected {len(note_events)} notes')
    return note_events


def notes_to_piece(note_events, tuning, capo, tempo, genre=UNKNOWN):
    notes = sorted(
        (
            {'pitch': int(e['midi_note']), 'start': float(e['onset_time']), 'end': float(e['offset_time'])}
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
    model, vocab, iteration = load_checkpoint(str(checkpoint), device)
    print(f'  checkpoint iteration: {iteration}')

    enc_subseqs = tokenize_for_inference(piece, vocab)

    decoder_prefix = build_anchor_prefix(piece, vocab, anchor_tabs) if anchor_tabs else None
    if decoder_prefix is not None:
        print(f'  anchoring first {len(anchor_tabs)} notes ({len(decoder_prefix) - 1} prefix tokens)')

    raw_tabs, dec_subseqs = generate_tabs(model, vocab, enc_subseqs, device, return_raw=True, decoder_prefix=decoder_prefix)

    sorted_notes = sorted(piece['notes'], key=lambda x: (x['start'], x['pitch']))
    input_pitches = [n['pitch'] for n in sorted_notes]
    corrected_tabs, sources = correct_tabs(
        input_pitches, raw_tabs, piece['tuning'], return_sources=True, fallback=fallback,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=int, default=10, help='Recording length in seconds')
    ap.add_argument('--audio', default=None, help='Audio file path (skips mic recording)')
    ap.add_argument('--output-dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--stage1-checkpoint', default=str(STAGE1_DEFAULT))
    ap.add_argument('--stage2-checkpoint', default=str(STAGE2_DEFAULT))
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
        '--fallback', choices=['first_viable', 'nearest_viable'], default='first_viable',
        help='Post-processing fallback strategy. first_viable = paper-faithful. '
             'nearest_viable = deviation: Manhattan-nearest realization to model raw output.',
    )
    ap.add_argument(
        '--genre', choices=list(GENRES), default=UNKNOWN,
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

    # 1) Audio
    if args.audio:
        print(f'Loading audio: {args.audio}')
        audio, _ = librosa.load(args.audio, sr=SR, mono=True)
        print(f'  duration={len(audio) / SR:.2f}s')
    else:
        audio = record_with_retry(args.duration, SR)

    wav_path = out_dir / f'{stem}.wav'
    write_wav(audio, wav_path)
    print(f'Wrote: {wav_path}')

    # 2) Stage 1: audio → MIDI
    note_events = stage1(audio, args.stage1_checkpoint, device)
    if not note_events:
        print('No notes detected, exiting.')
        return

    mid_path = out_dir / f'{stem}.mid'
    write_midi(note_events, mid_path)
    print(f'Wrote: {mid_path}')

    # 3) Stage 2: MIDI → tabs
    tuning_with_capo = [t + args.capo for t in args.tuning]  # our convention: tuning includes capo
    piece = notes_to_piece(note_events, tuning_with_capo, args.capo, args.tempo, genre=args.genre)
    t0 = time.time()
    tabs, raw_tabs, enc_subseqs, dec_subseqs, sources, vocab = stage2(
        piece, args.stage2_checkpoint, device, anchor_tabs=anchor_tabs, fallback=args.fallback
    )
    print(f'  stage 2 elapsed: {time.time() - t0:.1f}s')

    # 4) Debug log
    debug_path = out_dir / f'{stem}.debug.txt'
    write_debug_log(debug_path, vocab, note_events, piece, enc_subseqs, dec_subseqs, raw_tabs, tabs, sources)
    print(f'Wrote: {debug_path}')

    # 4b) Optional: per-step alternatives (re-runs generation w/ scores; first sub-seq only)
    if args.show_alternatives > 0:
        alt_model, alt_vocab, _ = load_checkpoint(str(args.stage2_checkpoint), device)
        alt_prefix = build_anchor_prefix(piece, alt_vocab, anchor_tabs) if anchor_tabs else None
        steps = generate_with_alternatives(
            alt_model,
            alt_vocab,
            enc_subseqs[0],
            device,
            top_k=args.show_alternatives,
            decoder_prefix=alt_prefix,
        )
        alt_path = out_dir / f'{stem}.alternatives.txt'
        lines = [f'top-{args.show_alternatives} alternatives at each generation step (sub-seq 0)']
        lines.append('marker `*` = picked.  TIME_SHIFT/PAD/EOS rows shown for context.\n')
        for s in steps:
            lines.append(f'step {s["step"]:>3d}: chose {s["chosen_str"]}')
            for tid, ts, p in s['topk']:
                marker = ' *' if tid == s['chosen_id'] else '  '
                lines.append(f'  {marker} {p:>6.3f}  {ts}')
            lines.append('')
        Path(alt_path).write_text('\n'.join(lines))
        print(f'Wrote: {alt_path}')

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

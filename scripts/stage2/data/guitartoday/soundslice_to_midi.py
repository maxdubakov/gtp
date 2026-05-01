"""Convert a Soundslice JSON data file to MIDI.

Soundslice JSON contains bars → beats → notes with string/fret info.
If a sync file is provided (bar → real timestamp mapping), note timing
follows the actual performance. Otherwise, a fixed tempo is used.
"""

import argparse
import json
from itertools import pairwise

import numpy as np
import pretty_midi


def estimate_tempo_from_sync(syncpoints, bars):
    """Estimate the song's average BPM from Soundslice sync points.

    Sync entries are [bar_idx, time_sec, ...] (the optional 3rd value is an
    intra-bar position used by Soundslice for fine sync). For tempo estimation
    we keep only the earliest sync per bar (the bar's actual start time), then
    compute per-segment BPM and return the median.

    Returns None if too few segments or all-zero beat counts.
    """
    if not syncpoints or len(syncpoints) < 2:
        return None

    bar_starts = {}
    for entry in syncpoints:
        b, t = entry[0], entry[1]
        if b not in bar_starts or t < bar_starts[b]:
            bar_starts[b] = t

    sorted_bars = sorted(bar_starts.items())
    if len(sorted_bars) < 2:
        return None

    seg_tempos = []
    for (a_idx, a_t), (b_idx, b_t) in pairwise(sorted_bars):
        if b_idx <= a_idx or b_t <= a_t:
            continue
        beats = 0.0
        for k in range(a_idx, b_idx):
            if 0 <= k < len(bars):
                ts = bars[k].get('time', [4, 4])
                beats += ts[0] * 4.0 / ts[1]
        if beats > 0:
            seg_tempos.append(beats * 60.0 / (b_t - a_t))

    return float(np.median(seg_tempos)) if seg_tempos else None


def build_bar_timing(data, syncpoints=None, tempo_bpm=120):
    """Compute start time and duration for each bar.

    If syncpoints are provided, uses real performance timing.
    Otherwise, uses fixed tempo.

    Returns list of (bar_start_sec, bar_duration_sec) per bar.
    """
    if syncpoints:
        sync_map = {int(s[0]): s[1] for s in syncpoints}

    bars = data['bars']
    result = []

    for i, bar in enumerate(bars):
        time_sig = bar.get('time', [4, 4])
        bar_whole = time_sig[0] / time_sig[1]

        if syncpoints and i in sync_map:
            bar_start = sync_map[i]
            # Duration: time until next synced bar, or estimate from tempo
            next_synced = None
            for j in range(i + 1, len(bars) + 1):
                if j in sync_map:
                    next_synced = sync_map[j]
                    bars_between = j - i
                    break
            if next_synced:
                bar_duration = (next_synced - bar_start) / bars_between
            else:
                bar_duration = bar_whole * 4 * 60.0 / tempo_bpm
        elif result:
            prev_start, prev_dur = result[-1]
            bar_start = prev_start + prev_dur
            # Inherit tempo from previous bar if no syncpoint
            bar_duration = bar_whole * (
                prev_dur / (bars[i - 1].get('time', [4, 4])[0] / bars[i - 1].get('time', [4, 4])[1])
            )
        else:
            bar_start = 0.0
            bar_duration = bar_whole * 4 * 60.0 / tempo_bpm

        result.append((bar_start, bar_duration))

    return result


def parse_soundslice(data, syncpoints=None, tempo_bpm=120):
    """Convert Soundslice JSON to a list of MIDI note events.

    Returns list of {'pitch': int, 'start': float, 'end': float, 'string': int, 'fret': int}
    """
    # Find the guitar track (has 'pitches' for string tuning)
    guitar_track_idx = next((i for i, t in enumerate(data['tracks']) if 'pitches' in t), 0)
    track_info = data['tracks'][guitar_track_idx]
    string_pitches = track_info['pitches']
    rhythms = {r['id']: r for r in data['rhythms']}
    bar_timing = build_bar_timing(data, syncpoints, tempo_bpm)

    notes = []
    active_ties = {}

    for bar_idx, bar in enumerate(data['bars']):
        bar_start, bar_duration = bar_timing[bar_idx]
        track = bar['tracks'][guitar_track_idx]

        # Process all voices (voice 0 = treble/melody, voice 1 = bass, etc.)
        for voice in track['voices']:
            beat_values = []
            for beat in voice:
                if beat.get('grace'):
                    beat_values.append(0)  # grace notes have no rhythmic duration
                    continue
                rhythm = rhythms[beat['r']]
                val = rhythm['val']
                dots = rhythm.get('dots', 0)
                if dots:
                    val = val * (2 - 0.5**dots)
                beat_values.append(val)
            total_bar_value = sum(beat_values)

            cumulative_value = 0.0
            for beat, beat_val in zip(voice, beat_values, strict=True):
                if total_bar_value == 0:
                    break
                if beat.get('grace'):
                    # Place grace note 50ms before next beat position
                    grace_time = bar_start + (cumulative_value / total_bar_value) * bar_duration
                    beat_start = max(bar_start, grace_time - 0.05)
                    beat_end = grace_time
                else:
                    beat_start = bar_start + (cumulative_value / total_bar_value) * bar_duration
                    beat_end = bar_start + ((cumulative_value + beat_val) / total_bar_value) * bar_duration

                for note_data in beat.get('notes', []):
                    string_idx = note_data['string']  # 0-indexed in Soundslice JSON
                    fret = note_data['fret']
                    pitch = string_pitches[string_idx] + fret
                    tie_key = (string_idx, pitch)

                    if note_data.get('tieend'):
                        if tie_key in active_ties:
                            active_ties[tie_key]['end'] = beat_end
                            if not note_data.get('tiestart'):
                                del active_ties[tie_key]
                        continue

                    note = {
                        'pitch': pitch,
                        'start': beat_start,
                        'end': beat_end,
                        'string': string_idx + 1,  # store as 1-indexed for tab display
                        'fret': fret,
                    }
                    notes.append(note)

                    if note_data.get('tiestart'):
                        active_ties[tie_key] = note

                cumulative_value += beat_val

    return notes, string_pitches


def notes_to_midi(notes, output_path, tempo_bpm=120):
    """Write note events to a MIDI file."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    guitar = pretty_midi.Instrument(program=24, name='Acoustic Guitar')

    for note in notes:
        midi_note = pretty_midi.Note(
            velocity=80,
            pitch=int(note['pitch']),
            start=float(note['start']),
            end=max(float(note['start']) + 0.01, float(note['end'])),
        )
        guitar.notes.append(midi_note)

    midi.instruments.append(guitar)
    midi.write(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Soundslice JSON file')
    parser.add_argument('--sync', default=None, help='Soundslice sync JSON file (bar → time mapping)')
    parser.add_argument('-o', '--output', default=None, help='Output MIDI path')
    parser.add_argument('--tempo', type=float, default=120, help='Fallback tempo in BPM (default: 120)')
    parser.add_argument('--info', action='store_true', help='Print piece info and exit')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    syncpoints = None
    if args.sync:
        with open(args.sync) as f:
            syncpoints = json.load(f)
        print(f'Sync: {len(syncpoints)} bar markers loaded')

    track = data['tracks'][0]
    print(f'Track: {track["name"]}')
    print(f'Strings: {track["strings"]}')
    print(f'Tuning: {track["pitches"]} ({", ".join(pretty_midi.note_number_to_name(p) for p in track["pitches"])})')
    print(f'Bars: {len(data["bars"])}')
    print(f'Tempo: {args.tempo} BPM')

    if 'chords' in track:
        chord_names = [c[1]['name'] for c in track['chords']]
        print(f'Chords: {", ".join(chord_names)}')

    notes, _string_pitches = parse_soundslice(data, syncpoints=syncpoints, tempo_bpm=args.tempo)
    print(f'Notes: {len(notes)}')

    if notes:
        pitches = [n['pitch'] for n in notes]
        print(
            f'Pitch range: {min(pitches)}-{max(pitches)} '
            f'({pretty_midi.note_number_to_name(min(pitches))}-{pretty_midi.note_number_to_name(max(pitches))})'
        )
        print(f'Duration: {notes[-1]["end"]:.1f}s')

    if args.info:
        return

    output_path = args.output or args.input.replace('.json', '.mid')
    notes_to_midi(notes, output_path, tempo_bpm=args.tempo)
    print(f'\nMIDI written: {output_path}')


if __name__ == '__main__':
    main()

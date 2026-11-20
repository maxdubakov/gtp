"""Guitar Pro files (GP3/4/5) to tab notes with pyguitarpro."""

import guitarpro as gp

from gtp.stage2.metrics import pitch_of

TICKS_PER_QUARTER = 960


def note_techniques(note, beat):
    """Names of the playing techniques written on a note or on its beat. `hammer` marks the note a hammer-on or
    pull-off starts from; parse_gp_track marks the note it leads to as `hammered`."""
    e, b = note.effect, beat.effect
    marks = {
        'dead': note.type == gp.NoteType.dead,
        'tie': note.type == gp.NoteType.tie,
        'palm_mute': e.palmMute,
        'bend': e.bend is not None,
        'slide': bool(e.slides),
        'hammer': e.hammer,
        'vibrato': e.vibrato or b.vibrato,
        'harmonic': e.harmonic is not None and not isinstance(e.harmonic, gp.PinchHarmonic),
        'pinch_harmonic': isinstance(e.harmonic, gp.PinchHarmonic),
        'ghost': e.ghostNote,
        'let_ring': e.letRing,
        'staccato': e.staccato,
        'tremolo_picking': e.tremoloPicking is not None,
        'trill': e.trill is not None,
        'grace': e.grace is not None,
        'tapping': b.slapEffect == gp.SlapEffect.tapping,
        'whammy': b.tremoloBar is not None,
    }
    return [name for name, on in marks.items() if on]


def parse_gp_track(gp_path, track_idx, techniques=False):
    """Parse a specific track from a GP file.

    Returns dict with: tempo, tuning, notes [{pitch, string, fret, start, end}]; with `techniques`, each note also
    lists its techniques (note_techniques).
    """
    song = gp.parse(str(gp_path))
    track = song.tracks[track_idx]
    tuning = [s.value for s in track.strings]
    capo = track.offset if hasattr(track, 'offset') else 0
    tempo = song.tempo

    notes = []
    current_tick = 0

    for measure in track.measures:
        header = measure.header
        if header.tempo and header.tempo.value:
            tempo = header.tempo.value

        tps = (tempo * TICKS_PER_QUARTER) / 60.0

        for voice in measure.voices:
            beat_tick = current_tick
            for beat in voice.beats:
                dur_ticks = beat.duration.time

                start_sec = beat_tick / tps
                end_sec = (beat_tick + dur_ticks) / tps

                for note in beat.notes:
                    pitch = pitch_of((note.string, note.value), tuning)

                    notes.append(
                        {
                            'pitch': pitch,
                            'string': note.string,
                            'fret': note.value,
                            'start': round(start_sec, 4),
                            'end': round(end_sec, 4),
                        }
                    )
                    if techniques:
                        notes[-1]['techniques'] = note_techniques(note, beat)

                beat_tick += dur_ticks

        time_sig = header.timeSignature
        bar_ticks = int(TICKS_PER_QUARTER * 4 * time_sig.numerator / time_sig.denominator.value)
        current_tick += bar_ticks

    if techniques:
        last = {}
        for n in sorted(notes, key=lambda n: n['start']):
            if 'hammer' in last.get(n['string'], ()):
                n['techniques'].append('hammered')
            last[n['string']] = n['techniques']

    # Clamp implausible tempos to None — the source files occasionally have
    # garbage values like 5 or 375 BPM, which give the model a fake, useless
    # signal. Note timing is left as computed (we don't re-run tick conversion
    # at a different tempo, so existing seconds-positions stay).
    tempo_metadata = tempo if 40 <= tempo <= 240 else None

    return {
        'tempo': tempo_metadata,
        'tuning': tuning,
        'capo': capo,
        'notes': notes,
    }

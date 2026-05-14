"""V3 tokenizer for the Fretting-Transformer.

Converts note JSON → token sequences for encoder (MIDI) and decoder (TAB).

Encoder input (conditioned):
  [GENRE<rock>] [TEMPO<120>] CAPO<5> <TUNING_START> NOTE_ON<64> ... NOTE_ON<40> <TUNING_END>
  NOTE_ON<55> TIME_SHIFT<120> NOTE_OFF<55> ...

Decoder target:
  TAB<3,0> TIME_SHIFT<120> ...

TIME_SHIFT values are in MIDI ticks at PPQ=480.
Simultaneous notes (chords) have no TIME_SHIFT between them.

Genre conditioning is opt-in via `Vocabulary(include_genre=True)`. With it on,
14 GENRE tokens are added to the vocab and `notes_to_encoder_tokens(...,
genre=...)` emits a GENRE token at the very start of the conditioning prefix.
With it off, the legacy 553-token vocab is preserved (old checkpoints load fine).

The vocabulary is constructed by callers (train.py, demo.py, eval.py, etc.)
and threaded explicitly through `tokenize_piece(...)`, `TabDataset(...)`,
and the inference helpers — there is no module-level global.
"""

from collections import defaultdict
from dataclasses import dataclass

from gtp.stage2.genres import GENRES

# Token type names — used as Token.type and as the bare identifier in vocab strings.
# Standalone tokens (no value) are emitted as <TYPE>; parametric ones as TYPE<value>.
PAD = 'PAD'
EOS = 'EOS'
TUNING_START = 'TUNING_START'
TUNING_END = 'TUNING_END'
TEMPO = 'TEMPO'
CAPO = 'CAPO'
GENRE = 'GENRE'
NOTE_ON = 'NOTE_ON'
NOTE_OFF = 'NOTE_OFF'
TIME_SHIFT = 'TIME_SHIFT'
TAB = 'TAB'

PPQ = 480
MAX_TIME_SHIFT = 1920  # whole note at PPQ=480
TIME_SHIFT_BINS = list(range(30, MAX_TIME_SHIFT + 1, 30))  # uniform ~31ms bins at 120 BPM, max ~16ms quantization error
TEMPO_MIN = 40
TEMPO_MAX = 240
TEMPO_STEP = 5


def quantize_ticks(ticks):
    """Snap a tick delta to the nearest bin, or to 0 (suppresses TIME_SHIFT emission).

    0 is a sentinel — it's not a vocab token. Any delta closer to 0 than to the
    smallest bin (60 ticks ≈ 31ms at 120bpm) collapses to 'simultaneous'.
    """
    if ticks <= 0:
        return 0
    if ticks > MAX_TIME_SHIFT:
        return MAX_TIME_SHIFT
    candidates = [0, *TIME_SHIFT_BINS]
    return min(candidates, key=lambda b: abs(b - ticks))


def quantize_tempo(bpm):
    clamped = max(TEMPO_MIN, min(TEMPO_MAX, bpm))
    return round(clamped / TEMPO_STEP) * TEMPO_STEP


@dataclass
class Token:
    type: str
    value: str | None

    def __str__(self):
        if self.value is None:
            return f'<{self.type}>'
        return f'{self.type}<{self.value}>'


def _bare(token_type):
    return str(Token(token_type, None))


class Vocabulary:
    """Stage 2 token vocabulary.

    Pass `include_genre=True` to add `GENRE<X>` tokens for conditioning.
    The flag is part of model config so old (no-genre) checkpoints continue
    to load against a `Vocabulary(include_genre=False)`.
    """

    def __init__(self, include_genre: bool = False):
        self.include_genre = include_genre
        self.token_to_id = {}
        self.id_to_token = {}
        self._build()

    def _add(self, token_str):
        if token_str not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token_str] = idx
            self.id_to_token[idx] = token_str

    def _build(self):
        # T5 convention: no SOS — the decoder is seeded with PAD as decoder_start_token_id.
        for t in (PAD, EOS, TUNING_START, TUNING_END):
            self._add(_bare(t))

        for bpm in range(TEMPO_MIN, TEMPO_MAX + 1, TEMPO_STEP):
            self._add(str(Token(TEMPO, str(bpm))))

        for capo in range(0, 13):
            self._add(str(Token(CAPO, str(capo))))

        if self.include_genre:
            for g in GENRES:
                self._add(str(Token(GENRE, g)))

        for pitch in range(128):
            self._add(str(Token(NOTE_ON, str(pitch))))
            self._add(str(Token(NOTE_OFF, str(pitch))))

        for ticks in TIME_SHIFT_BINS:
            self._add(str(Token(TIME_SHIFT, str(ticks))))

        for string in range(1, 8):
            for fret in range(0, 25):
                self._add(str(Token(TAB, f'{string},{fret}')))

    def encode(self, token):
        return self.token_to_id[str(token)]

    def decode(self, idx):
        return self.id_to_token[idx]

    @property
    def pad_id(self):
        return self.token_to_id[_bare(PAD)]

    @property
    def eos_id(self):
        return self.token_to_id[_bare(EOS)]

    def __len__(self):
        return len(self.token_to_id)


def _emit_time_shifts(delta_ticks):
    """Convert a tick delta into one or more TIME_SHIFT tokens."""
    tokens = []
    remaining = delta_ticks
    while remaining > 0:
        chunk = min(remaining, MAX_TIME_SHIFT)
        q = quantize_ticks(chunk)
        if q > 0:
            tokens.append(Token(TIME_SHIFT, str(q)))
        remaining -= chunk
    return tokens


def notes_to_encoder_tokens(notes, tempo, tuning=None, capo=0, genre=None):
    """Convert note list to encoder token sequence.

    Conditioning prefix order:
        [GENRE<X>] [TEMPO<bpm>] CAPO<n> [<TUNING_START> NOTE_ON×6 <TUNING_END>]

    `genre` is one of the canonical buckets (see gtp.stage2.genres.GENRES);
    pass None to omit the GENRE token entirely (legacy / no-conditioning runs).
    `tempo` None → omits the TEMPO token (signal that tempo is unknown).
    Note times are still converted to ticks at 120 BPM density when tempo is None
    — the dataset is expected to have normalized timing to 120 BPM in that case.
    """
    tempo_for_ticks = tempo if tempo is not None else 120
    ticks_per_sec = (tempo_for_ticks / 60) * PPQ

    # Conditioning prefix: GENRE (optional) → TEMPO (optional) → CAPO → TUNING → notes
    tokens = []
    if genre is not None:
        tokens.append(Token(GENRE, genre))
    if tempo is not None:
        tokens.append(Token(TEMPO, str(quantize_tempo(tempo))))
    tokens.append(Token(CAPO, str(min(12, max(0, capo)))))
    if tuning:
        tokens.append(Token(TUNING_START, None))
        for pitch in tuning:
            tokens.append(Token(NOTE_ON, str(pitch)))
        tokens.append(Token(TUNING_END, None))

    # Build note events. Notes with end <= start are filtered upstream in data.py.
    events = []
    for note in notes:
        on_tick = round(note['start'] * ticks_per_sec)
        off_tick = round(note['end'] * ticks_per_sec)
        events.append((on_tick, NOTE_ON, note['pitch']))
        events.append((off_tick, NOTE_OFF, note['pitch']))

    # NOTE_OFF before NOTE_ON at the same tick so previous notes release before new ones start.
    events.sort(key=lambda e: (e[0], 0 if e[1] == NOTE_OFF else 1, e[2]))

    current_tick = 0
    for tick, event_type, pitch in events:
        delta = tick - current_tick
        if delta > 0:
            tokens.extend(_emit_time_shifts(delta))
            current_tick = tick
        tokens.append(Token(event_type, str(pitch)))

    return tokens


def notes_to_decoder_tokens(notes, tempo):
    """Convert note list to decoder token sequence (TAB + TIME_SHIFT)."""
    ticks_per_sec = (tempo / 60) * PPQ

    # Defensive sort — caller should sort but we don't rely on it.
    notes = sorted(notes, key=lambda n: (n['start'], n['pitch']))

    tokens = []
    current_tick = 0

    for note in notes:
        on_tick = round(note['start'] * ticks_per_sec)

        delta = on_tick - current_tick
        if delta > 0:
            tokens.extend(_emit_time_shifts(delta))
            current_tick = on_tick

        fret = max(0, min(24, note['fret']))
        tokens.append(Token(TAB, f'{note["string"]},{fret}'))

    return tokens


def tokenize_piece(data, vocab: 'Vocabulary', max_seq_len=512, genre_override=None):
    """Tokenize a full piece into encoder/decoder sequence pairs.

    Returns list of (encoder_ids, decoder_ids) tuples. `data['tempo']` may be
    None (TEMPO token omitted). `data['genre']` is emitted as a GENRE token
    when present AND `vocab.include_genre` is True; otherwise the GENRE
    token is silently dropped (so the same data layer works for both
    legacy and conditioning models).

    `genre_override` (e.g. 'unknown') replaces the piece's genre — used by
    `TabDataset` for classifier-free training-time dropout.
    """
    notes = sorted(data['notes'], key=lambda n: (n['start'], n['pitch']))
    tempo = data.get('tempo', 120)
    tuning = data.get('tuning')
    capo = data.get('capo', 0)
    if vocab.include_genre:
        genre = genre_override if genre_override is not None else data.get('genre')
    else:
        genre = None
    tempo_for_decoder = tempo if tempo is not None else 120

    enc_tokens = notes_to_encoder_tokens(notes, tempo, tuning, capo, genre=genre)
    dec_tokens = notes_to_decoder_tokens(notes, tempo_for_decoder)

    # Find note boundaries for aligned splitting
    # Encoder: each NOTE_ON that isn't inside the tuning block starts a new note
    in_tuning = False
    note_boundaries_enc = []
    for i, t in enumerate(enc_tokens):
        if t.type == TUNING_START:
            in_tuning = True
        elif t.type == TUNING_END:
            in_tuning = False
        elif t.type == NOTE_ON and not in_tuning:
            note_boundaries_enc.append(i)

    note_boundaries_dec = []
    for i, t in enumerate(dec_tokens):
        if t.type == TAB:
            note_boundaries_dec.append(i)

    # Encoder must produce one NOTE_ON-outside-tuning per input note, and decoder must
    # produce one TAB per input note. A mismatch is a tokenizer bug, not a data issue.
    assert len(note_boundaries_enc) == len(note_boundaries_dec), (
        f'Note count mismatch: encoder={len(note_boundaries_enc)} decoder={len(note_boundaries_dec)} '
        f'for piece with {len(notes)} notes'
    )

    # The conditioning prefix (TEMPO + CAPO + TUNING) is repeated at the start of each sequence
    prefix_end = 0
    for i, t in enumerate(enc_tokens):
        if t.type == TUNING_END:
            prefix_end = i + 1
            break
        if t.type == NOTE_ON:
            # No tuning block — prefix is just TEMPO/CAPO
            in_tuning_block = False
            for j in range(i):
                if enc_tokens[j].type == TUNING_START:
                    in_tuning_block = True
            if not in_tuning_block:
                prefix_end = i
                break

    prefix_tokens = enc_tokens[:prefix_end]
    prefix_ids = [vocab.encode(t) for t in prefix_tokens]

    sequences = []
    note_idx = 0
    n_notes = len(note_boundaries_enc)

    # loop building multiple sequences out of a single track
    while note_idx < n_notes:
        enc_start = note_boundaries_enc[note_idx]
        dec_start = note_boundaries_dec[note_idx]

        enc_end = enc_start
        dec_end = dec_start
        notes_in_seq = 0

        # loop building a single sequence
        while note_idx + notes_in_seq < n_notes:
            next_note = note_idx + notes_in_seq + 1
            if next_note < n_notes:
                trial_enc_end = note_boundaries_enc[next_note]
                trial_dec_end = note_boundaries_dec[next_note]
            else:
                trial_enc_end = len(enc_tokens)
                trial_dec_end = len(dec_tokens)

            enc_note_len = trial_enc_end - enc_start
            dec_note_len = trial_dec_end - dec_start

            if max(enc_note_len + len(prefix_ids), dec_note_len) + 2 > max_seq_len:
                break

            enc_end = trial_enc_end
            dec_end = trial_dec_end
            notes_in_seq += 1

        if notes_in_seq == 0:
            notes_in_seq = 1
            enc_end = note_boundaries_enc[note_idx + 1] if note_idx + 1 < n_notes else len(enc_tokens)
            dec_end = note_boundaries_dec[note_idx + 1] if note_idx + 1 < n_notes else len(dec_tokens)

        enc_ids = list(prefix_ids)
        for t in enc_tokens[enc_start:enc_end]:
            enc_ids.append(vocab.encode(t))
        enc_ids.append(vocab.eos_id)

        dec_ids = []
        for t in dec_tokens[dec_start:dec_end]:
            dec_ids.append(vocab.encode(t))
        dec_ids.append(vocab.eos_id)

        sequences.append((enc_ids, dec_ids))
        note_idx += notes_in_seq

    return sequences


def parse_token_str(s):
    """Parse a token string back to (type, value). '<SOS>' → ('SOS', None); 'NOTE_ON<55>' → ('NOTE_ON', '55')."""
    if s.startswith('<'):
        return s[1:-1], None
    open_idx = s.index('<')
    return s[:open_idx], s[open_idx + 1 : -1]


def encoder_tokens_to_notes(token_strs):
    """Reverse of notes_to_encoder_tokens.

    Returns (notes, tempo, capo, tuning). Notes is a list of {pitch, start, end} in seconds.
    Same-pitch overlapping notes are paired in FIFO order (the encoder representation
    doesn't distinguish them, but pairs survive the round-trip).
    """
    tempo = 120
    capo = 0
    tuning = None
    in_tuning = False
    body_start = len(token_strs)

    for i, s in enumerate(token_strs):
        t, v = parse_token_str(s)
        if t == 'PAD':
            continue
        if t == 'TEMPO':
            tempo = int(v)
        elif t == 'CAPO':
            capo = int(v)
        elif t == 'TUNING_START':
            tuning = []
            in_tuning = True
        elif t == 'TUNING_END':
            in_tuning = False
            body_start = i + 1
            break
        elif t == 'NOTE_ON' and in_tuning:
            tuning.append(int(v))
        elif t in ('NOTE_ON', 'NOTE_OFF', 'TIME_SHIFT'):
            body_start = i
            break

    ticks_per_sec = (tempo / 60) * PPQ
    notes = []
    active = defaultdict(list)
    current_tick = 0

    for s in token_strs[body_start:]:
        t, v = parse_token_str(s)
        if t == 'PAD':
            continue
        if t == 'EOS':
            break
        if t == 'TIME_SHIFT':
            current_tick += int(v)
        elif t == 'NOTE_ON':
            active[int(v)].append(current_tick)
        elif t == 'NOTE_OFF':
            pitch = int(v)
            if active[pitch]:
                start = active[pitch].pop(0)
                notes.append({'pitch': pitch, 'start': start / ticks_per_sec, 'end': current_tick / ticks_per_sec})

    for pitch, starts in active.items():
        for start in starts:
            notes.append({'pitch': pitch, 'start': start / ticks_per_sec, 'end': current_tick / ticks_per_sec})

    notes.sort(key=lambda n: (n['start'], n['pitch']))
    return notes, tempo, capo, tuning


def decoder_tokens_to_notes(token_strs, tempo, tuning):
    """Reverse of notes_to_decoder_tokens.

    Decoder vocab has no NOTE_OFF, so the output has no `end` — only `start, pitch, string, fret`.
    Callers that need duration (MIDI render, JSON write) synthesize it themselves.
    """
    ticks_per_sec = (tempo / 60) * PPQ
    notes = []
    current_tick = 0

    for s in token_strs:
        t, v = parse_token_str(s)
        if t == 'PAD':
            continue
        if t == 'EOS':
            break
        if t == 'TIME_SHIFT':
            current_tick += int(v)
        elif t == 'TAB':
            string_str, fret_str = v.split(',')
            string = int(string_str)
            fret = int(fret_str)
            pitch = tuning[string - 1] + fret
            notes.append({'start': current_tick / ticks_per_sec, 'pitch': pitch, 'string': string, 'fret': fret})

    return notes

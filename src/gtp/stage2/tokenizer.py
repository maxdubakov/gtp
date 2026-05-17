"""V3 tokenizer for the Fretting-Transformer.

Example of encoder's input:
  [GENRE<rock>] [TEMPO<120>] CAPO<5> <TUNING_START> NOTE_ON<64> ... NOTE_ON<40> <TUNING_END>
  NOTE_ON<55> TIME_SHIFT<120> NOTE_OFF<55>

Example of decoder's output:
  TAB<3,0> TIME_SHIFT<120>
"""

from collections import defaultdict
from dataclasses import dataclass

from gtp.stage2.genres import GENRES

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
MAX_FRET = 24
MAX_SEQ_LEN = 512


def quantize_ticks(ticks):
    """Snap a tick delta to the nearest bin, or to 0 (suppresses TIME_SHIFT emission)"""
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
    """Stage 2 token vocabulary"""

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
            for fret in range(0, MAX_FRET + 1):
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

    `genre` is one of the canonical buckets (see gtp.stage2.genres.GENRES);
    pass None to omit the GENRE token entirely (legacy / no-conditioning runs).
    `tempo` None → omits the TEMPO token (signal that tempo is unknown).
    Note times are still converted to ticks at 120 BPM density when tempo is None
    — the dataset is expected to have normalized timing to 120 BPM in that case.
    """
    # TODO: add tempo unknown and ensure that every conditioning token stays in its place
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
    notes = sorted(notes, key=lambda n: (n['start'], n['pitch']))
    tokens = []
    current_tick = 0

    for note in notes:
        on_tick = round(note['start'] * ticks_per_sec)

        delta = on_tick - current_tick
        if delta > 0:
            tokens.extend(_emit_time_shifts(delta))
            current_tick = on_tick

        fret = max(0, min(MAX_FRET, note['fret']))
        tokens.append(Token(TAB, f'{note["string"]},{fret}'))

    return tokens


def _encoder_prep(data, vocab: 'Vocabulary', genre_override=None):
    """Shared encoder-side prep for both training and inference tokenization"""
    notes = sorted(data['notes'], key=lambda n: (n['start'], n['pitch']))
    tempo = data.get('tempo', 120)
    tuning = data.get('tuning')
    capo = data.get('capo', 0)
    if vocab.include_genre:
        genre = genre_override if genre_override is not None else data.get('genre')
    else:
        genre = None

    enc_tokens = notes_to_encoder_tokens(notes, tempo, tuning, capo, genre=genre)

    # Each NOTE_ON outside the tuning block starts a new note.
    in_tuning = False
    note_boundaries_enc = []
    for i, t in enumerate(enc_tokens):
        if t.type == TUNING_START:
            in_tuning = True
        elif t.type == TUNING_END:
            in_tuning = False
        elif t.type == NOTE_ON and not in_tuning:
            note_boundaries_enc.append(i)

    # Prefix is everything before the first body NOTE_ON (i.e. before the body).
    prefix_end = note_boundaries_enc[0] if note_boundaries_enc else len(enc_tokens)
    prefix_ids = [vocab.encode(t) for t in enc_tokens[:prefix_end]]

    return notes, tempo, enc_tokens, note_boundaries_enc, prefix_ids


def _emit_enc_ids(enc_tokens, enc_start, enc_end, prefix_ids, vocab):
    """Build one encoder sub-sequence: prefix + body slice + EOS"""
    ids = list(prefix_ids)
    for t in enc_tokens[enc_start:enc_end]:
        ids.append(vocab.encode(t))
    ids.append(vocab.eos_id)
    return ids


def tokenize_piece(data, vocab: 'Vocabulary', max_seq_len: int = MAX_SEQ_LEN, genre_override=None):
    """Tokenize a full piece into aligned (encoder, decoder) sub-sequence pairs for training.

    Returns list of (encoder_ids, decoder_ids) tuples. Sub-sequences are split at
    note boundaries; each fits within max_seq_len on both encoder and decoder side.
    """
    notes, tempo, enc_tokens, note_boundaries_enc, prefix_ids = _encoder_prep(data, vocab, genre_override)
    tempo_for_decoder = tempo if tempo is not None else 120
    dec_tokens = notes_to_decoder_tokens(notes, tempo_for_decoder)

    note_boundaries_dec = [i for i, t in enumerate(dec_tokens) if t.type == TAB]
    assert len(note_boundaries_enc) == len(note_boundaries_dec), (
        f'Note count mismatch: {len(note_boundaries_enc)} vs {len(note_boundaries_dec)} for piece with {len(notes)} notes'
    )

    sequences = []
    note_idx = 0
    n_notes = len(note_boundaries_enc)

    while note_idx < n_notes:
        enc_start = note_boundaries_enc[note_idx]
        dec_start = note_boundaries_dec[note_idx]
        enc_end = enc_start
        dec_end = dec_start
        notes_in_seq = 0

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

        enc_ids = _emit_enc_ids(enc_tokens, enc_start, enc_end, prefix_ids, vocab)
        dec_ids = [vocab.encode(t) for t in dec_tokens[dec_start:dec_end]] + [vocab.eos_id]
        sequences.append((enc_ids, dec_ids))
        note_idx += notes_in_seq

    return sequences


def tokenize_piece_for_inference(data, vocab: 'Vocabulary', max_seq_len: int = MAX_SEQ_LEN, genre_override=None):
    """Tokenize a piece into encoder-only sub-sequences for inference"""
    _, _, enc_tokens, note_boundaries_enc, prefix_ids = _encoder_prep(data, vocab, genre_override)

    sequences = []
    note_idx = 0
    n_notes = len(note_boundaries_enc)

    while note_idx < n_notes:
        enc_start = note_boundaries_enc[note_idx]
        enc_end = enc_start
        notes_in_seq = 0

        while note_idx + notes_in_seq < n_notes:
            next_note = note_idx + notes_in_seq + 1
            trial_enc_end = note_boundaries_enc[next_note] if next_note < n_notes else len(enc_tokens)

            if (trial_enc_end - enc_start) + len(prefix_ids) + 2 > max_seq_len:
                break

            enc_end = trial_enc_end
            notes_in_seq += 1

        if notes_in_seq == 0:
            notes_in_seq = 1
            enc_end = note_boundaries_enc[note_idx + 1] if note_idx + 1 < n_notes else len(enc_tokens)

        sequences.append(_emit_enc_ids(enc_tokens, enc_start, enc_end, prefix_ids, vocab))
        note_idx += notes_in_seq

    return sequences


def parse_token_str(s):
    """Parse a token string back to (type, value). '<SOS>' → ('SOS', None); 'NOTE_ON<55>' → ('NOTE_ON', '55')."""
    if s.startswith('<'):
        return s[1:-1], None
    open_idx = s.index('<')
    return s[:open_idx], s[open_idx + 1 : -1]


def parse_tuning_from_enc(enc_ids, vocab) -> list[int] | None:
    in_tuning = False
    tuning: list[int] = []
    for tid in enc_ids:
        t, v = parse_token_str(vocab.decode(int(tid)))
        if t == TUNING_START:
            in_tuning = True
            tuning = []
        elif t == TUNING_END:
            return tuning if tuning else None
        elif t == NOTE_ON and in_tuning:
            tuning.append(int(v))
    return None


def extract_input_pitches(enc_ids, vocab) -> list[int]:
    """Walk encoder IDs, return body's NOTE_ON pitches in order (skips tuning block)."""
    in_tuning = False
    pitches: list[int] = []
    for tid in enc_ids:
        t, v = parse_token_str(vocab.decode(int(tid)))
        if t == TUNING_START:
            in_tuning = True
        elif t == TUNING_END:
            in_tuning = False
        elif t == NOTE_ON and not in_tuning:
            pitches.append(int(v))
        elif t == EOS:
            break
    return pitches


def extract_tabs(token_ids, vocab) -> list[tuple[int, int]]:
    """Walk decoder IDs, return list of (string, fret) from TAB tokens. Stops at EOS."""
    tabs: list[tuple[int, int]] = []
    for tid in token_ids:
        t, v = parse_token_str(vocab.decode(int(tid)))
        if t == EOS:
            break
        if t == PAD:
            continue
        if t == TAB:
            ss, ff = v.split(',')
            tabs.append((int(ss), int(ff)))
    return tabs


def encoder_tokens_to_notes(ids: list[int], vocab: 'Vocabulary'):
    """Reverse of notes_to_encoder_tokens. Detokenizes model encoder IDs.

    Returns (notes, tempo, capo, tuning). Notes is a list of {pitch, start, end} in seconds.
    Same-pitch overlapping notes are paired in FIFO order (the encoder representation
    doesn't distinguish them, but pairs survive the round-trip).
    """
    tempo = 120
    capo = 0
    tuning = None
    in_tuning = False
    body_start = len(ids)

    for i, tid in enumerate(ids):
        t, v = parse_token_str(vocab.decode(int(tid)))
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

    for tid in ids[body_start:]:
        t, v = parse_token_str(vocab.decode(int(tid)))
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


def decoder_tokens_to_notes(ids: list[int], vocab: 'Vocabulary', tempo, tuning):
    """Reverse of notes_to_decoder_tokens. Detokenizes model decoder IDs.

    Decoder vocab has no NOTE_OFF, so the output has no `end` — only `start, pitch, string, fret`.
    Callers that need duration (MIDI render, JSON write) synthesize it themselves.
    """
    ticks_per_sec = (tempo / 60) * PPQ
    notes = []
    current_tick = 0

    for tid in ids:
        t, v = parse_token_str(vocab.decode(int(tid)))
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

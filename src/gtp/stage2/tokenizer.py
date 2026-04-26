"""V3 tokenizer for the Fretting-Transformer.

Converts note JSON → token sequences for encoder (MIDI) and decoder (TAB).

Encoder input (conditioned):
  TEMPO<120> <TUNING_START> NOTE_ON<64> ... NOTE_ON<40> <TUNING_END>
  NOTE_ON<55> TIME_SHIFT<120> NOTE_OFF<55> ...

Decoder target:
  TAB<3,0> TIME_SHIFT<120> ...

TIME_SHIFT values are in MIDI ticks at PPQ=480.
Simultaneous notes (chords) have no TIME_SHIFT between them.

Vocabulary:
  NOTE_ON<0..127>        — 128 tokens (MIDI pitch)
  NOTE_OFF<0..127>       — 128 tokens (MIDI pitch)
  TIME_SHIFT<tick>       — quantized tick values
  TAB<string,fret>       — string (1-7) × fret (-2..24) combinations
  TEMPO<bpm>             — quantized to nearest 5 BPM (40-240)
  <tuning_start>         — marks beginning of tuning block
  <tuning_end>           — marks end of tuning block
  <pad>, <sos>, <eos>    — special tokens
"""

from dataclasses import dataclass

PPQ = 480

TIME_SHIFT_BINS = sorted(set([
    60,     # 32nd
    120,    # 16th
    240,    # 8th
    480,    # quarter
    960,    # half
    1920,   # whole
    180,    # dotted 16th
    360,    # dotted 8th
    720,    # dotted quarter
    1440,   # dotted half
    80,     # 8th triplet
    160,    # quarter triplet
    320,    # half triplet
    640,    # whole triplet
]))

MAX_TIME_SHIFT = 1920

TEMPO_MIN = 40
TEMPO_MAX = 240
TEMPO_STEP = 5


def quantize_ticks(ticks):
    if ticks <= 0:
        return 0
    if ticks > MAX_TIME_SHIFT:
        return MAX_TIME_SHIFT
    return min(TIME_SHIFT_BINS, key=lambda b: abs(b - ticks))


def quantize_tempo(bpm):
    clamped = max(TEMPO_MIN, min(TEMPO_MAX, bpm))
    return round(clamped / TEMPO_STEP) * TEMPO_STEP


@dataclass
class Token:
    type: str
    value: str

    def __str__(self):
        if self.value is None:
            return f'<{self.type}>'
        return f'{self.type}<{self.value}>'


class Vocabulary:

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self._build()

    def _add(self, token_str):
        if token_str not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token_str] = idx
            self.id_to_token[idx] = token_str

    def _build(self):
        self._add('<pad>')
        self._add('<sos>')
        self._add('<eos>')
        self._add('<tuning_start>')
        self._add('<tuning_end>')

        for bpm in range(TEMPO_MIN, TEMPO_MAX + 1, TEMPO_STEP):
            self._add(f'TEMPO<{bpm}>')

        for capo in range(0, 13):
            self._add(f'CAPO<{capo}>')

        for pitch in range(128):
            self._add(f'NOTE_ON<{pitch}>')
            self._add(f'NOTE_OFF<{pitch}>')

        for ticks in TIME_SHIFT_BINS:
            self._add(f'TIME_SHIFT<{ticks}>')

        for string in range(1, 8):
            for fret in range(0, 25):
                self._add(f'TAB<{string},{fret}>')

    def encode(self, token):
        return self.token_to_id[str(token)]

    def decode(self, idx):
        return self.id_to_token[idx]

    @property
    def pad_id(self):
        return self.token_to_id['<pad>']

    @property
    def sos_id(self):
        return self.token_to_id['<sos>']

    @property
    def eos_id(self):
        return self.token_to_id['<eos>']

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
            tokens.append(Token('TIME_SHIFT', str(q)))
        remaining -= chunk
    return tokens


def notes_to_encoder_tokens(notes, tempo, tuning=None, capo=0):
    """Convert note list to encoder token sequence.

    Includes conditioning prefix (TEMPO + CAPO + TUNING).
    """
    ticks_per_sec = (tempo / 60) * PPQ

    tokens = []

    # Conditioning prefix: TEMPO → CAPO → TUNING → notes
    tokens.append(Token('TEMPO', str(quantize_tempo(tempo))))
    tokens.append(Token('CAPO', str(min(12, max(0, capo)))))
    if tuning:
        tokens.append(Token('tuning_start', None))
        for pitch in tuning:
            tokens.append(Token('NOTE_ON', str(pitch)))
        tokens.append(Token('tuning_end', None))

    # Build note events
    events = []
    for note in notes:
        on_tick = round(note['start'] * ticks_per_sec)
        off_tick = round(note['end'] * ticks_per_sec)
        if off_tick <= on_tick:
            off_tick = on_tick + 60
        events.append((on_tick, 'NOTE_ON', note['pitch']))
        events.append((off_tick, 'NOTE_OFF', note['pitch']))

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'NOTE_OFF' else 1, e[2]))

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

    tokens = []
    current_tick = 0

    for note in notes:
        on_tick = round(note['start'] * ticks_per_sec)
        dur_tick = round((note['end'] - note['start']) * ticks_per_sec)
        if dur_tick <= 0:
            dur_tick = 60

        delta = on_tick - current_tick
        if delta > 0:
            tokens.extend(_emit_time_shifts(delta))
            current_tick = on_tick

        fret = max(0, min(24, note['fret']))
        tokens.append(Token('TAB', f'{note["string"]},{fret}'))

    return tokens


def tokenize_piece(data, max_seq_len=512):
    """Tokenize a full piece into encoder/decoder sequence pairs.

    Returns list of (encoder_ids, decoder_ids) tuples.
    """
    notes = sorted(data['notes'], key=lambda n: (n['start'], n['pitch']))
    tempo = data.get('tempo', 120)
    tuning = data.get('tuning')
    capo = data.get('capo', 0)

    enc_tokens = notes_to_encoder_tokens(notes, tempo, tuning, capo)
    dec_tokens = notes_to_decoder_tokens(notes, tempo)

    vocab = VOCAB

    # Find note boundaries for aligned splitting
    # Encoder: each NOTE_ON that isn't inside the tuning block starts a new note
    in_tuning = False
    note_boundaries_enc = []
    for i, t in enumerate(enc_tokens):
        if t.type == 'tuning_start':
            in_tuning = True
        elif t.type == 'tuning_end':
            in_tuning = False
        elif t.type == 'NOTE_ON' and not in_tuning:
            note_boundaries_enc.append(i)

    note_boundaries_dec = []
    for i, t in enumerate(dec_tokens):
        if t.type == 'TAB':
            note_boundaries_dec.append(i)

    # The conditioning prefix (TEMPO + TUNING) is repeated at the start of each sequence
    prefix_end = 0
    for i, t in enumerate(enc_tokens):
        if t.type == 'tuning_end':
            prefix_end = i + 1
            break
        if t.type == 'NOTE_ON':
            # No tuning block — prefix is just TEMPO
            in_tuning_block = False
            for j in range(i):
                if enc_tokens[j].type == 'tuning_start':
                    in_tuning_block = True
            if not in_tuning_block:
                prefix_end = i
                break

    prefix_tokens = enc_tokens[:prefix_end]
    prefix_ids = [vocab.encode(t) for t in prefix_tokens]

    usable_len = max_seq_len - 2 - len(prefix_ids)  # SOS + prefix + ... + EOS

    sequences = []
    note_idx = 0
    n_notes = len(note_boundaries_enc)

    while note_idx < n_notes:
        enc_start = note_boundaries_enc[note_idx]
        dec_start = note_boundaries_dec[note_idx] if note_idx < len(note_boundaries_dec) else len(dec_tokens)

        enc_end = enc_start
        dec_end = dec_start
        notes_in_seq = 0

        while note_idx + notes_in_seq < n_notes:
            next_note = note_idx + notes_in_seq + 1
            if next_note < n_notes:
                trial_enc_end = note_boundaries_enc[next_note]
                trial_dec_end = note_boundaries_dec[next_note] if next_note < len(note_boundaries_dec) else len(dec_tokens)
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
            dec_end = note_boundaries_dec[note_idx + 1] if note_idx + 1 < len(note_boundaries_dec) else len(dec_tokens)

        enc_ids = [vocab.sos_id] + prefix_ids
        for t in enc_tokens[enc_start:enc_end]:
            enc_ids.append(vocab.encode(t))
        enc_ids.append(vocab.eos_id)

        dec_ids = [vocab.sos_id]
        for t in dec_tokens[dec_start:dec_end]:
            dec_ids.append(vocab.encode(t))
        dec_ids.append(vocab.eos_id)

        sequences.append((enc_ids, dec_ids))
        note_idx += notes_in_seq

    return sequences


VOCAB = Vocabulary()

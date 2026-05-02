"""Stage 2 inference utilities.

Token-stream parsing helpers, checkpoint loading, encoder-only tokenization,
autoregressive generation, and end-to-end "piece → tabs" inference.
"""

import torch

from gtp.stage2.model import build_model
from gtp.stage2.postprocess import correct_tabs
from gtp.stage2.tokenizer import (
    EOS,
    NOTE_ON,
    PAD,
    TAB,
    TUNING_END,
    TUNING_START,
    VOCAB,
    notes_to_decoder_tokens,
    parse_token_str,
    tokenize_piece,
)


def parse_tuning_from_enc(enc_ids):
    """Walk encoder IDs, return tuning as list of pitches. None if missing."""
    in_tuning = False
    tuning = []
    for tid in enc_ids:
        t, v = parse_token_str(VOCAB.decode(int(tid)))
        if t == TUNING_START:
            in_tuning = True
            tuning = []
        elif t == TUNING_END:
            return tuning if tuning else None
        elif t == NOTE_ON and in_tuning:
            tuning.append(int(v))
    return None


def extract_input_pitches(enc_ids):
    """Walk encoder IDs, return body's NOTE_ON pitches in order (skips tuning block)."""
    in_tuning = False
    pitches = []
    for tid in enc_ids:
        t, v = parse_token_str(VOCAB.decode(int(tid)))
        if t == TUNING_START:
            in_tuning = True
        elif t == TUNING_END:
            in_tuning = False
        elif t == NOTE_ON and not in_tuning:
            pitches.append(int(v))
        elif t == EOS:
            break
    return pitches


def extract_tabs(token_ids):
    """Walk decoder IDs, return list of (string, fret) from TAB tokens. Stops at EOS."""
    tabs = []
    for tid in token_ids:
        t, v = parse_token_str(VOCAB.decode(int(tid)))
        if t == EOS:
            break
        if t == PAD:
            continue
        if t == TAB:
            ss, ff = v.split(',')
            tabs.append((int(ss), int(ff)))
    return tabs


def load_checkpoint(path, device):
    """Load Stage 2 checkpoint, return (model, iteration)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    meta = ckpt.get('tokenizer_meta', {})
    if meta.get('vocab_size') and meta['vocab_size'] != len(VOCAB):
        raise ValueError(f'Vocab mismatch: ckpt={meta["vocab_size"]}, current={len(VOCAB)}')
    model = build_model().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt.get('iteration')


def tokenize_for_inference(piece, max_seq_len=512):
    """Tokenize a piece for Stage 2 inference (encoder sub-sequences only).

    Notes need only `pitch`, `start`, `end` — string and fret are dummied so
    `tokenize_piece` (which builds both encoder and decoder) doesn't fail. The
    decoder side is discarded; encoder is what the model actually needs.

    Returns list of enc_ids (one per sub-sequence).
    """
    dummy_piece = {
        **piece,
        'notes': [{**n, 'string': 1, 'fret': 0} for n in piece['notes']],
    }
    sequences = tokenize_piece(dummy_piece, max_seq_len=max_seq_len)
    return [enc_ids for enc_ids, _dec_ids in sequences]


def build_anchor_prefix(piece, anchor_tabs):
    """Build a decoder prefix from user-supplied (string, fret) for the first N notes.

    Notes are sorted by (start, pitch); anchor_tabs[i] becomes the TAB for note i.
    TIME_SHIFTs between anchored notes come from the input note timings, so the
    decoder is primed with both *what* and *when*. Returns a list of token IDs
    starting with decoder_start (PAD).
    """
    if not anchor_tabs:
        return None
    sorted_notes = sorted(piece['notes'], key=lambda n: (n['start'], n['pitch']))
    n = min(len(anchor_tabs), len(sorted_notes))
    anchored = []
    for i in range(n):
        nd = sorted_notes[i]
        s, f = anchor_tabs[i]
        anchored.append({**nd, 'string': s, 'fret': f})

    tempo_for_decoder = piece.get('tempo') if piece.get('tempo') is not None else 120
    dec_tokens = notes_to_decoder_tokens(anchored, tempo_for_decoder)
    return [VOCAB.pad_id] + [VOCAB.encode(t) for t in dec_tokens]


def generate_with_alternatives(model, enc_ids, device, top_k=8, max_seq_len=512, decoder_prefix=None):
    """Generate one sub-sequence and capture top-K alternatives at each step.

    Returns a list of dicts, one per generation step:
      {'step': int, 'chosen_id': int, 'chosen_str': str,
       'topk': [(token_id, token_str, probability), ...]}
    Probabilities come from softmaxing the logits at each step (post any logits
    processors HF applies). Useful for "what else did the model consider here?"
    """
    padded = enc_ids + [VOCAB.pad_id] * (max_seq_len - len(enc_ids))
    input_ids = torch.tensor([padded[:max_seq_len]], dtype=torch.long, device=device)
    attention_mask = (input_ids != VOCAB.pad_id).long()

    gen_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'max_new_tokens': max_seq_len,
        'num_beams': 1,
        'do_sample': False,
        'pad_token_id': VOCAB.pad_id,
        'eos_token_id': VOCAB.eos_id,
        'return_dict_in_generate': True,
        'output_scores': True,
    }
    if decoder_prefix is not None:
        gen_kwargs['decoder_input_ids'] = torch.tensor([decoder_prefix], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    seq_ids = out.sequences[0].tolist()
    scores = out.scores  # tuple of (1, vocab_size) tensors, one per generated step
    prefix_len = len(decoder_prefix) if decoder_prefix is not None else 1  # decoder_start counts

    results = []
    for step_i, score in enumerate(scores):
        chosen_id = seq_ids[prefix_len + step_i]
        probs = torch.softmax(score[0], dim=-1)
        top_probs, top_ids = probs.topk(top_k)
        topk = [
            (int(tid), VOCAB.decode(int(tid)), float(p))
            for tid, p in zip(top_ids.tolist(), top_probs.tolist(), strict=True)
        ]
        results.append({
            'step': step_i,
            'chosen_id': chosen_id,
            'chosen_str': VOCAB.decode(chosen_id),
            'topk': topk,
        })
    return results


def generate_tabs(model, enc_ids_list, device, max_seq_len=512, return_raw=False, decoder_prefix=None):
    """Run autoregressive greedy generation per sub-sequence; concatenate (string, fret) outputs.

    If `decoder_prefix` is provided (list of token IDs starting with decoder_start),
    it primes the FIRST sub-sequence's decoder. Subsequent sub-sequences start fresh.
    If return_raw=True, also returns raw decoder token IDs per sub-sequence.
    """
    all_tabs = []
    raw_per_subseq = []
    for sub_i, enc_ids in enumerate(enc_ids_list):
        padded = enc_ids + [VOCAB.pad_id] * (max_seq_len - len(enc_ids))
        input_ids = torch.tensor([padded[:max_seq_len]], dtype=torch.long, device=device)
        attention_mask = (input_ids != VOCAB.pad_id).long()

        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': max_seq_len,
            'num_beams': 1,
            'do_sample': False,
            'pad_token_id': VOCAB.pad_id,
            'eos_token_id': VOCAB.eos_id,
        }
        if decoder_prefix is not None and sub_i == 0:
            gen_kwargs['decoder_input_ids'] = torch.tensor(
                [decoder_prefix], dtype=torch.long, device=device
            )

        with torch.no_grad():
            generated = model.generate(**gen_kwargs)
        raw_ids = generated[0, 1:].tolist()  # skip decoder_start (PAD)
        if return_raw:
            raw_per_subseq.append(raw_ids)
        all_tabs.extend(extract_tabs(raw_ids))
    if return_raw:
        return all_tabs, raw_per_subseq
    return all_tabs


def infer(model, piece, device, max_seq_len=512, post_process=True):
    """End-to-end Stage 2 inference: piece dict → list of (string, fret) per input note.

    With post_process=True, applies the paper's ±5 correction so every output
    pitch matches the corresponding input pitch (paper-faithful).
    """
    enc_list = tokenize_for_inference(piece, max_seq_len=max_seq_len)
    raw_tabs = generate_tabs(model, enc_list, device, max_seq_len=max_seq_len)
    if not post_process:
        return raw_tabs
    sorted_notes = sorted(piece['notes'], key=lambda x: (x['start'], x['pitch']))
    input_pitches = [n['pitch'] for n in sorted_notes]
    return correct_tabs(input_pitches, raw_tabs, piece['tuning'])

"""Decode-time fusion of Stage 2 with per-note string distributions (product of experts).

Stage 2 decodes along the decoder skeleton of a piece (TIME_SHIFT tokens and note order,
both fixed by the input notes), so it only chooses each note's string among pitch-valid TAB
tokens. The choice is rescored with an audio string distribution:

    score(s) = log p_stage2(TAB<s, f_s>) + lam * (log p_audio(s) - log p_prior(s))

An EvidenceT5 also reads the audio distribution in its encoder (Row.evidence).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from gtp.stage2.tokenizer import EVIDENCE_DIM, MAX_FRET, TAB, Token, tokenize_piece


@dataclass
class Slot:
    note: int
    token_ids: np.ndarray
    strings: np.ndarray
    log_ratio: np.ndarray | None  # (n_seeds, n_valid): log p_audio - log p_prior


@dataclass
class Row:
    enc: list[int]
    dec: list[int]
    tuning: list[int]
    slots: dict[int, Slot]  # decoder position -> slot
    evidence: np.ndarray | None = None  # (len(enc), EVIDENCE_DIM) encoder evidence for an EvidenceT5


def sorted_notes(piece):
    """Notes in decoder order (the order tokenize_piece emits TAB tokens in)."""
    return sorted(piece['notes'], key=lambda n: (n['start'], n['pitch']))


def valid_tabs(pitch, tuning, vocab):
    """Strings that can play `pitch` and their TAB token ids."""
    strings = np.array([s for s in range(1, len(tuning) + 1) if 0 <= pitch - tuning[s - 1] <= MAX_FRET], dtype=int)
    return strings, np.array([vocab.encode(Token(TAB, f'{s},{pitch - tuning[s - 1]}')) for s in strings], dtype=int)


def floored(p, floor):
    p = np.maximum(p, floor)
    return p / p.sum(axis=-1, keepdims=True)


def log_ratio(audio, prior, strings, floor):
    """(n_seeds, 6) audio and (6,) prior string probabilities -> (n_seeds, n_valid) log ratio on `strings`."""
    return np.log(floored(audio[:, strings - 1], floor)) - np.log(floored(prior[strings - 1], floor))[None]


def piece_rows(piece, vocab, max_seq_len, note_ids, log_ratios, evidence=None):
    """Decoder rows for one piece; `note_ids`, `log_ratios` and `evidence` align with sorted_notes(piece).

    `evidence`: (n_notes, 6) string distributions (NaN rows for none) for an EvidenceT5's encoder.
    """
    notes, tuning = sorted_notes(piece), piece['tuning']
    if evidence is not None:
        notes = [{**n, 'evidence': None if np.isnan(e[0]) else e} for n, e in zip(notes, evidence, strict=True)]
        piece = {**piece, 'notes': notes}
    rows, k = [], 0
    for enc, dec, *ev in tokenize_piece(piece, vocab, max_seq_len=max_seq_len, with_evidence=evidence is not None):
        slots = {}
        for t, tid in enumerate(dec):
            if not vocab.decode(tid).startswith(TAB):
                continue
            strings, token_ids = valid_tabs(notes[k]['pitch'], tuning, vocab)
            assert tid in token_ids, (piece.get('filename'), notes[k])
            slots[t] = Slot(note_ids[k], token_ids, strings, log_ratios[k])
            k += 1
        rows.append(Row(enc, dec, tuning, slots, ev[0] if ev else None))
    assert k == len(notes), (piece.get('filename'), k, len(notes))
    return rows


def padded(seqs, pad, device):
    width = max(len(s) for s in seqs)
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=device)
    return ids, (ids != pad).long()


@torch.no_grad()
def decode_structured(model, rows, device, batch, lam=0.0, seed_idx=0):
    """Greedy decode along each row's skeleton; returns {note: chosen string}."""
    pad = model.config.pad_token_id
    picks = {}
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        input_ids, mask = padded([r.enc for r in chunk], pad, device)
        if chunk[0].evidence is None:
            encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=mask)
        else:
            evidence = torch.zeros(*input_ids.shape, EVIDENCE_DIM, device=device)
            for b, r in enumerate(chunk):
                evidence[b, : len(r.evidence)] = torch.from_numpy(r.evidence)
            embeds = model.embed(input_ids, evidence)
            encoder_outputs = model.get_encoder()(inputs_embeds=embeds, attention_mask=mask)
        cur = torch.full((len(chunk), 1), pad, device=device)
        past = None
        for t in range(max(len(r.dec) for r in chunk)):
            out = model(
                encoder_outputs=encoder_outputs,
                attention_mask=mask,
                decoder_input_ids=cur,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logp = F.log_softmax(out.logits[:, -1].float(), dim=-1).cpu().numpy()
            nxt = []
            for b, r in enumerate(chunk):
                slot = r.slots.get(t)
                if t >= len(r.dec):
                    nxt.append(pad)
                elif slot is None:
                    nxt.append(r.dec[t])
                else:
                    score = logp[b, slot.token_ids]
                    if lam and slot.log_ratio is not None:
                        score = score + lam * slot.log_ratio[seed_idx]
                    j = int(score.argmax())
                    picks[slot.note] = int(slot.strings[j])
                    nxt.append(int(slot.token_ids[j]))
            cur = torch.tensor(nxt, device=device)[:, None]
    return picks

import torch

from gtp.stage2.config import RunConfig, find_run_config
from gtp.stage2.model import build_model
from gtp.stage2.tokenizer import (
    Vocabulary,
    extract_tabs,
    notes_to_decoder_tokens,
)


def load_checkpoint(path, device) -> tuple[object, Vocabulary, int]:
    """Load Stage 2 checkpoint. Returns (model, vocab, iteration)."""
    cfg = RunConfig.load(find_run_config(path))
    vocab = Vocabulary(include_genre=cfg.conditioning.genre)

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    meta = ckpt.get('tokenizer_meta', {})
    if meta.get('vocab_size') and meta['vocab_size'] != len(vocab):
        raise ValueError(f'Vocab mismatch: ckpt={meta["vocab_size"]}, current={len(vocab)}')
    model = build_model(vocab).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocab, ckpt.get('iteration')


def build_anchor_prefix(piece, vocab: Vocabulary, anchor_tabs):
    """Build a decoder prefix from user-supplied (string, fret) for the first N notes"""
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
    return [vocab.pad_id] + [vocab.encode(t) for t in dec_tokens]


def generate_tabs(
    model, vocab: Vocabulary, enc_ids_list, device, max_seq_len=512, return_raw=False, decoder_prefix=None
):
    """Run autoregressive generation per sub-sequence; concatenate (string, fret) outputs"""
    all_tabs = []
    raw_per_subseq = []
    for sub_i, enc_ids in enumerate(enc_ids_list):
        padded = enc_ids + [vocab.pad_id] * (max_seq_len - len(enc_ids))
        input_ids = torch.tensor([padded[:max_seq_len]], dtype=torch.long, device=device)
        attention_mask = (input_ids != vocab.pad_id).long()

        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': max_seq_len,
            'num_beams': 1,
            'do_sample': False,
            'pad_token_id': vocab.pad_id,
            'eos_token_id': vocab.eos_id,
        }
        if decoder_prefix is not None and sub_i == 0:
            gen_kwargs['decoder_input_ids'] = torch.tensor([decoder_prefix], dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(**gen_kwargs)
        raw_ids = generated[0, 1:].tolist()  # skip decoder_start (PAD)
        if return_raw:
            raw_per_subseq.append(raw_ids)
        all_tabs.extend(extract_tabs(raw_ids, vocab))
    if return_raw:
        return all_tabs, raw_per_subseq
    return all_tabs

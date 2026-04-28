"""Stage 2 model: halved t5-small from HuggingFace Transformers, trained from scratch.

Per Hamberger et al. 2025 (Fretting-Transformer):
  d_model=128, d_ff=1024, 3 enc/dec layers, 4 heads.
"""

from transformers import T5Config, T5ForConditionalGeneration

from gtp.stage2.tokenizer import VOCAB

MODEL_CONFIG = {
    'd_model': 128,
    'd_ff': 1024,
    'num_layers': 3,
    'num_decoder_layers': 3,
    'num_heads': 4,
    'd_kv': 32,  # d_model / num_heads
    'dropout_rate': 0.1,
    'feed_forward_proj': 'relu',
    'relative_attention_num_buckets': 32,
}


def build_model(vocab_size=None, **overrides):
    """Build a T5ForConditionalGeneration with the paper's halved t5-small dims.

    Token IDs are read from VOCAB. Pass `labels` (NOT `decoder_input_ids`) at training time —
    HF prepends `decoder_start_token_id` (= PAD, T5 convention) and right-shifts internally.
    Set PAD positions inside `labels` to -100 so the loss ignores them.
    """
    if vocab_size is None:
        vocab_size = len(VOCAB)
    config = T5Config(
        vocab_size=vocab_size,
        pad_token_id=VOCAB.pad_id,
        eos_token_id=VOCAB.eos_id,
        **{**MODEL_CONFIG, **overrides},
    )
    # Inherited from PretrainedConfig; not in T5Config's explicit signature, set after construction.
    config.decoder_start_token_id = VOCAB.pad_id
    return T5ForConditionalGeneration(config)

"""Stage 2 model: halved t5-small from HuggingFace Transformers, trained from scratch.

Per Hamberger et al. 2025 (Fretting-Transformer):
  d_model=128, d_ff=1024, 3 enc/dec layers, 4 heads.
"""

from transformers import T5Config, T5ForConditionalGeneration

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


def build_model(vocab, **overrides):
    """Build a T5ForConditionalGeneration sized to the given vocab.

    Pass `labels` (NOT `decoder_input_ids`) at training time — HF prepends
    `decoder_start_token_id` (= PAD, T5 convention) and right-shifts internally.
    Set PAD positions inside `labels` to -100 so the loss ignores them.
    """
    config = T5Config(
        vocab_size=len(vocab),
        pad_token_id=vocab.pad_id,
        eos_token_id=vocab.eos_id,
        **{**MODEL_CONFIG, **overrides},
    )
    # Inherited from PretrainedConfig; not in T5Config's explicit signature, set after construction.
    config.decoder_start_token_id = vocab.pad_id
    return T5ForConditionalGeneration(config)

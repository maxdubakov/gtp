"""Stage 2 model: halved t5-small from HuggingFace"""

from torch import nn
from transformers import T5Config, T5ForConditionalGeneration

from gtp.stage2.tokenizer import EVIDENCE_DIM

# default model settings are selected per Hamberger et al. 2025 (Fretting-Transformer)
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


class EvidenceT5(T5ForConditionalGeneration):
    """T5 whose encoder input adds a linear projection of each token's string evidence to its embedding.

    The projection starts at zero, so an untrained EvidenceT5 behaves exactly like the plain model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.evidence_proj = nn.Linear(EVIDENCE_DIM, config.d_model)
        nn.init.zeros_(self.evidence_proj.weight)
        nn.init.zeros_(self.evidence_proj.bias)

    def embed(self, input_ids, evidence):
        return self.shared(input_ids) + self.evidence_proj(evidence)

    def forward(self, input_ids=None, evidence=None, **kwargs):
        if evidence is not None and kwargs.get('encoder_outputs') is None:
            kwargs['inputs_embeds'] = self.embed(input_ids, evidence)
            input_ids = None
        return super().forward(input_ids=input_ids, **kwargs)


def build_model(vocab, evidence=False, **overrides):
    config = T5Config(
        vocab_size=len(vocab),
        pad_token_id=vocab.pad_id,
        eos_token_id=vocab.eos_id,
        **{**MODEL_CONFIG, **overrides},
    )
    # Inherited from PretrainedConfig; not in T5Config's explicit signature, set after construction.
    config.decoder_start_token_id = vocab.pad_id
    return (EvidenceT5 if evidence else T5ForConditionalGeneration)(config)

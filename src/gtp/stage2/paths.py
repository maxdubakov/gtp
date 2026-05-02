"""Filesystem paths for Stage 2 data layout.

Kept in its own module so that scripts (inspect_tokens, tokens_to_midi, analyze_tempo,
build_aug_dataset) can import path constants without pulling in heavyweight
runtime dependencies (torch, the Dataset class, the tokenizer) from data.py.
"""

from gtp import REPO_ROOT

PROCESSED_DIRS = {
    'dadagp': REPO_ROOT / 'data' / 'dadagp' / 'processed',
    'guitartoday': REPO_ROOT / 'data' / 'guitartoday' / 'processed',
    'guitarset': REPO_ROOT / 'data' / 'guitarset' / 'processed',
    'leduc': REPO_ROOT / 'data' / 'leduc' / 'processed',
}

AUG_DATA_DIR = REPO_ROOT / 'data' / 'stage2_aug'

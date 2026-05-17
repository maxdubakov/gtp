"""Filesystem paths for Stage 2 data layout. Kept in its own module so that scripts can import path constants easily"""

from gtp import REPO_ROOT

PROCESSED_DIRS = {
    'dadagp': REPO_ROOT / 'data' / 'dadagp' / 'processed',
    'guitartoday': REPO_ROOT / 'data' / 'guitartoday' / 'processed',
    'guitarset': REPO_ROOT / 'data' / 'guitarset' / 'processed',
    'leduc': REPO_ROOT / 'data' / 'leduc' / 'processed',
}

AUG_DATA_DIR = REPO_ROOT / 'data' / 'stage2_aug'

SOUNDFONT_PATH = REPO_ROOT / 'models' / 'soundfonts' / 'ms_basic.sf3'
# TODO: revisit common paths after scripts are refactored

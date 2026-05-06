"""Canonical genre buckets for Stage 2 conditioning.

Defines the 14-bucket coarse genre set used as `GENRE<X>` tokens in the
encoder prefix, plus the keyword-rule table used to coarse-grain DadaGP's
739 fine-grained genre tags.

Per-source classification logic lives next to each dataset's processing
script (`scripts/stage2/data/<source>/build_dataset.py`) — those scripts
import `GENRES`, `GENRE_RULES`, and `UNKNOWN` from here.
"""

# 14-bucket canonical set. Order doesn't matter functionally, but vocab
# tokens are added in this order in the tokenizer for reproducibility.
GENRES: tuple[str, ...] = (
    'rock',
    'metal',
    'pop',
    'folk',
    'blues',
    'classical',
    'jazz',
    'punk',
    'country',
    'reggae',
    'electronic',
    'hip_hop',
    'funk',
    'unknown',
)
GENRES_SET = frozenset(GENRES)
UNKNOWN = 'unknown'


# Keyword-based fall-through used by DadaGP. Order matters — first match wins.
# Each bucket lists keywords that, if found as a substring of any of a piece's
# `genre:*` tokens (with the 'genre:' prefix stripped), assign the piece to
# that bucket. Pieces that match no rule fall through to UNKNOWN.
GENRE_RULES: list[tuple[str, list[str]]] = [
    ('classical',  ['classical', 'baroque', 'romantic', 'opera']),
    ('jazz',       ['jazz', 'bebop', 'swing', 'bossa', 'fusion', 'cool_jazz', 'latin_jazz']),
    ('funk',       ['funk']),
    ('folk',       ['folk', 'singer_songwriter', 'americana', 'bluegrass', 'celtic']),
    ('blues',      ['blues']),
    ('country',    ['country', 'honky_tonk']),
    ('metal',      ['metal', 'thrash', 'doom', 'grindcore', 'djent', 'progressive_metal',
                    'death_metal', 'black_metal']),
    ('punk',       ['punk', 'hardcore']),
    ('pop',        ['pop', 'mellow_gold', 'permanent_wave']),
    ('rock',       ['rock', 'grunge', 'shoegaze']),
    ('electronic', ['electronic', 'edm', 'house', 'techno', 'idm']),
    ('hip_hop',    ['hip_hop', 'rap']),
    ('reggae',     ['reggae', 'ska', 'dub']),
]


def is_valid(genre: str) -> bool:
    return genre in GENRES_SET

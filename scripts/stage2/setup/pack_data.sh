#!/bin/bash
# Create a data tarball for shipping to RunPod.
# Output: data_stage2.tar.gz containing processed/ for all four sources.
#
# The augmented JSONLs (data/stage2_aug/) are NOT included — those should
# be regenerated on RunPod via build_aug_dataset.py (~9 min one-time).
#
# Writes data/RUNINFO.txt with build metadata (git commit, vocab/tokenizer
# settings) so a future you can verify which code state generated this data.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT="data_stage2.tar.gz"
PYTHON="$REPO_ROOT/venv/bin/python"

# Build metadata
{
    echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty: $(git diff --quiet 2>/dev/null && echo no || echo yes)"
    echo "build_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$PYTHON" -c "
from gtp.stage2.tokenizer import Vocabulary, TIME_SHIFT_BINS
from gtp.stage2.data import MAX_PLAYABLE_FRET
# Snapshot both vocab variants so RUNINFO records what either training mode would use.
v_no_genre = Vocabulary(include_genre=False)
v_genre = Vocabulary(include_genre=True)
print(f'vocab_size_no_genre: {len(v_no_genre)}')
print(f'vocab_size_genre: {len(v_genre)}')
print(f'pad_id: {v_no_genre.pad_id}')
print(f'eos_id: {v_no_genre.eos_id}')
print(f'time_shift_step: {TIME_SHIFT_BINS[0]}')
print(f'max_playable_fret: {MAX_PLAYABLE_FRET}')
"
} > data/RUNINFO.txt

COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" \
    data/dadagp/processed/ \
    data/guitartoday/processed/ \
    data/guitarset/processed/ \
    data/leduc/processed/ \
    data/RUNINFO.txt

ls -lh "$OUT"
echo "Data packed: $OUT"
echo
echo "RUNINFO:"
cat data/RUNINFO.txt

#!/bin/bash
# Pack aug-only data for shipping to RunPod.
# Output: aug_data_stage2.tar.gz
#
# Includes:
#   - data/stage2_aug/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT="aug_data_stage2.tar.gz"

# Sanity checks
for f in data/stage2_aug/train.jsonl data/stage2_aug/val.jsonl data/stage2_aug/test.jsonl; do
    if [ ! -e "$f" ]; then
        echo "Missing: $f" >&2
        exit 1
    fi
done

COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" \
    data/stage2_aug/train.jsonl \
    data/stage2_aug/val.jsonl \
    data/stage2_aug/test.jsonl

ls -lh "$OUT"
echo "Augmented data packed: $OUT"

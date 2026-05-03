#!/bin/bash
# Pack runs (model checkpoints).
# Output: runs.tar.gz
#
# Includes:
#   - runs/
#
# RunPod workflow:
#   1. Upload runs.tar.gz
#   2. tar xzf runs.tar.gz

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT="runs.tar.gz"

# Sanity checks
for f in runs; do
    if [ ! -e "$f" ]; then
        echo "Missing: $f" >&2
        exit 1
    fi
done

COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" \
    runs/

ls -lh "$OUT"
echo "Runs packed: $OUT"

#!/bin/bash
# Create a code tarball for shipping to RunPod (or any remote training host).
# Output: code_stage2.tar.gz
#
# Includes the gtp package, all stage2 scripts, and the Python deps spec.
# Excludes node_modules, __pycache__, ruff/pytest caches, and build artifacts.
#
# RunPod workflow:
#   1. Upload code_stage2.tar.gz + data_stage2.tar.gz
#   2. tar xzf code_stage2.tar.gz
#   3. pip install -r requirements.txt && pip install -e .
#   4. tar xzf data_stage2.tar.gz
#   5. python scripts/stage2/build_aug_dataset.py
#   6. python scripts/stage2/train.py --device cuda --num-workers 4 --batch-size 32

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT="code_stage2.tar.gz"

COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='node_modules' \
    --exclude='.ruff_cache' \
    --exclude='*.egg-info' \
    src/gtp/ \
    scripts/stage2/ \
    requirements.txt \
    pyproject.toml

ls -lh "$OUT"
echo "Code packed: $OUT"

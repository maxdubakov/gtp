#!/bin/bash
# Packs only code (src, scripts)

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

#!/bin/bash
# Packs only code (src, scripts, Hydra configs), plus VERSION / GIT_DIFF so runs on the pod record provenance.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT="code_stage2.tar.gz"

git rev-parse HEAD > VERSION
git diff HEAD > GIT_DIFF
trap 'rm -f VERSION GIT_DIFF' EXIT

COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='node_modules' \
    --exclude='.ruff_cache' \
    --exclude='*.egg-info' \
    src/gtp/ \
    scripts/stage2/ \
    conf/ \
    pyproject.toml \
    uv.lock \
    .python-version \
    VERSION \
    GIT_DIFF

ls -lh "$OUT"
echo "Code packed: $OUT"

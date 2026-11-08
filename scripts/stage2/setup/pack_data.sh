#!/bin/bash
# Pack everything that's needed to train from scratch and evaluate.
# On remote host (runpod):
#   1. Send: runpodctl receive <uuid>
#   2. Unpack: tar xzf gtp_stage2_runpod.tar.gz
#   3. Install env: curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync --frozen
#   4. Activate: source .venv/bin/activate
#   5. Increase ulimit: ulimit -n 65536

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
git rev-parse --short HEAD > VERSION 2>/dev/null || echo unknown > VERSION

OUT="gtp_stage2_runpod.tar.gz"

REQUIRED=(
    src/gtp
    scripts/stage2
    pyproject.toml
    uv.lock
    .python-version
    VERSION
    data/stage2_aug/train.jsonl
    data/stage2_aug/val.jsonl
    data/stage2_aug/test.jsonl
    data/dadagp/processed
    data/guitartoday/processed
    data/guitarset/processed
    data/leduc/processed
    data/DadaGP-v1.1/_DadaGP_all_metadata.json
)
for f in "${REQUIRED[@]}"; do
    if [ ! -e "$f" ]; then
        echo "Missing: $f" >&2
        exit 1
    fi
done

COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='.ruff_cache' \
    --exclude='*.egg-info' \
    --exclude='node_modules' \
    --exclude='data/dadagp/processed/*.mid' \
    --exclude='data/guitartoday/processed/*.mid' \
    --exclude='data/guitarset/processed/*.mid' \
    --exclude='data/leduc/processed/*.mid' \
    "${REQUIRED[@]}"

ls -lh "$OUT"
echo "Pack ready: $OUT"

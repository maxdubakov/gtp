#!/bin/bash
# Single tarball for shipping to a fresh RunPod (or any remote training/eval host).
# Output: gtp_stage2_runpod.tar.gz
#
# Contents:
#   - Code: src/gtp/, scripts/stage2/, pyproject.toml, requirements.txt
#   - Training data: data/stage2_aug/{train,val,test}.jsonl
#   - Per-source processed JSONs (no MIDIs): data/{dadagp,guitartoday,guitarset,leduc}/processed/*.json
#   - DadaGP metadata: data/DadaGP-v1.1/_DadaGP_all_metadata.json
#
# Covers training, autoregressive eval (eval.py + dump_eval_predictions.py),
# and error analysis (enrich_errors.py + analyze_errors.py) on a single host.
#
# RunPod workflow:
#   1. runpodctl send gtp_stage2_runpod.tar.gz   (then receive on pod)
#   2. tar xzf gtp_stage2_runpod.tar.gz
#   3. pip install -r requirements.txt
#      (or: pip install torch==2.6.0 torchaudio==2.6.0 transformers numpy scipy librosa pretty_midi pyguitarpro jams soundfile mido pyfluidsynth)
#   4. pip install -e .
#   5. ulimit -n 65536
#   6. python scripts/stage2/train.py --device cuda --num-workers 4 ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT="gtp_stage2_runpod.tar.gz"

REQUIRED=(
    src/gtp
    scripts/stage2
    pyproject.toml
    requirements.txt
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

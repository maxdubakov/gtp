#!/bin/bash
# Pack eval-only data for shipping to RunPod (or any remote eval host).
# Output: eval_data_stage2.tar.gz
#
# Includes:
#   - data/stage2_aug/val.jsonl + test.jsonl (skips the 3GB train.jsonl)
#   - runs/stage2_baseline/ (checkpoints + train.log)
#
# Pair with code_stage2.tar.gz (from pack_code.sh) for the full eval bundle.
#
# RunPod workflow:
#   1. Upload code_stage2.tar.gz + eval_data_stage2.tar.gz
#   2. tar xzf code_stage2.tar.gz && tar xzf eval_data_stage2.tar.gz
#   3. pip install <deps> && pip install -e .
#   4. python scripts/stage2/eval.py --checkpoint-dir runs/stage2_baseline/ --include-test \
#          --output results/eval_sweep.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT="eval_data_stage2.tar.gz"

# Sanity checks
for f in data/stage2_aug/val.jsonl data/stage2_aug/test.jsonl runs/stage2_baseline; do
    if [ ! -e "$f" ]; then
        echo "Missing: $f" >&2
        exit 1
    fi
done

COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" \
    data/stage2_aug/val.jsonl \
    data/stage2_aug/test.jsonl \
    runs/stage2_baseline/checkpoints/step_0060000.pth \
    runs/stage2_baseline/config.json

ls -lh "$OUT"
echo "Eval data packed: $OUT"

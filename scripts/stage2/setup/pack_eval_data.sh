#!/bin/bash
# Pack eval-only data for shipping to RunPod (or any remote eval host).
# Output: eval_data_stage2.tar.gz
#
# Includes:
#   - data/stage2_aug/val.jsonl + test.jsonl (skips the 3GB train.jsonl)
#   - runs/stage2_baseline/ (all checkpoints + config.json + train.log + train_resume.log)
#     Use --run-dir <path> to pack a different run.
#
# Pair with code_stage2.tar.gz (from pack_code.sh) for the full eval bundle.
#
# RunPod workflow:
#   1. Upload code_stage2.tar.gz + eval_data_stage2.tar.gz
#   2. tar xzf code_stage2.tar.gz && tar xzf eval_data_stage2.tar.gz
#   3. pip install -e .
#   4. Backfill metrics.jsonl on every checkpoint (~30-45 min on RTX 4090):
#          python scripts/stage2/backfill_metrics.py --run-dir runs/stage2_baseline --device cuda
#   5. (Optional) Full autoregressive eval sweep:
#          python scripts/stage2/eval.py --checkpoint-dir runs/stage2_baseline/ --include-test \
#              --output runs/stage2_baseline/eval_sweep.json

set -euo pipefail

RUN_DIR="${1:-runs/stage2_baseline}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT="eval_data_stage2.tar.gz"

# Sanity checks
for f in data/stage2_aug/val.jsonl data/stage2_aug/test.jsonl "$RUN_DIR"; do
    if [ ! -e "$f" ]; then
        echo "Missing: $f" >&2
        exit 1
    fi
done

COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" \
    data/stage2_aug/val.jsonl \
    data/stage2_aug/test.jsonl \
    "$RUN_DIR"

ls -lh "$OUT"
echo "Eval data packed: $OUT (run dir: $RUN_DIR)"

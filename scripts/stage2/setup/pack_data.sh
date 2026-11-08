#!/bin/bash
# Pack what a pod needs to rebuild the Stage 2 dataset and attach audio evidence (~75 MB):
# the processed source JSONs and the string classifiers' out-of-fold outputs (no model weights).
# On the pod: tar xzf data_stage2.tar.gz && python scripts/stage2/build_aug_dataset.py --test-player 5 --output-dir data/stage2_p05

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT="data_stage2.tar.gz"

{
    find data/dadagp/processed data/guitartoday/processed data/guitarset/processed data/leduc/processed \
        -name '*.json' ! -name '._*'
    find results/strings/sweeps/nested05 results/strings/sweeps/lopo \
        \( -name test_probs.npz -o -path '*/.hydra/config.yaml' \)
} | sort > /tmp/gtp_pack_data.txt

COPYFILE_DISABLE=1 tar --no-xattrs -czf "$OUT" -T /tmp/gtp_pack_data.txt
rm /tmp/gtp_pack_data.txt

ls -lh "$OUT"
echo "Data packed: $OUT"

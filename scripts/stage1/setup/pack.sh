#!/bin/bash
# Create a tar archive of the project for uploading to RunPod.
# Excludes data, venv, and other large/unnecessary files.
# Archive is saved to the repo root as gtp_code.tar.gz.
#


REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ARCHIVE="$REPO_ROOT/gtp_code.tar.gz"

cd "$REPO_ROOT"

tar czf "$ARCHIVE" \
    --exclude='./venv' \
    --exclude='./.git' \
    --exclude='./piano_transcription-master' \
    --exclude='./runs' \
    --exclude='./__pycache__' \
    --exclude='./.DS_Store' \
    --exclude='./.coding-team' \
    --exclude='./data' \
    --exclude='./results' \
    --exclude='./gtp_code.tar.gz' \
    .

echo "Archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
echo ""
echo "Next steps:"
echo "  runpodctl send $ARCHIVE"

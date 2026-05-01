#!/bin/bash
# Parse all Leduc GP files using alphaTab → JSON, then generate MIDI via Python.
# Output: data/leduc/processed/{name}.json + {name}.mid
#


REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
GP_DIR="$REPO_ROOT/data/leduc/gp_files"
OUTPUT_DIR="$REPO_ROOT/data/leduc/processed"
PARSER_DIR="$REPO_ROOT/scripts/stage2/data/leduc/alphatab"
POSTPROCESS="$REPO_ROOT/scripts/stage2/data/leduc/postprocess.py"
PYTHON="$REPO_ROOT/venv/bin/python"

mkdir -p "$OUTPUT_DIR"

total=$(ls "$GP_DIR"/*.gp "$GP_DIR"/*.gpx 2>/dev/null | wc -l | tr -d ' ')
count=0
done_count=0
skipped=0
failed=0

for gp_file in "$GP_DIR"/*.gp "$GP_DIR"/*.gpx; do
    [ -f "$gp_file" ] || continue
    count=$((count + 1))

    name="$(basename "$gp_file" .gp)"
    name="$(basename "$name" .gpx)"
    json_file="$OUTPUT_DIR/$name.json"
    mid_file="$OUTPUT_DIR/$name.mid"

    if [ -f "$json_file" ] && [ -f "$mid_file" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    # Parse GP → JSON via alphaTab
    output=$(cd "$PARSER_DIR" && node parse_gp.mjs "$gp_file" --pretty 2>&1)
    exit_code=$?

    if [ $exit_code -ne 0 ] || ! echo "$output" | head -1 | grep -q '^{'; then
        printf "[%3d/%d] FAIL %-50s\n" "$count" "$total" "$name"
        failed=$((failed + 1))
        continue
    fi

    echo "$output" > "$json_file"

    # Validate, MP3-tempo-fix JSON, generate MIDI
    "$PYTHON" "$POSTPROCESS" "$json_file" "$mid_file" "$gp_file" 2> >(grep -E '\[tempo unknown\]' >&2)

    if [ -f "$mid_file" ]; then
        n_notes=$(echo "$output" | grep '"n_notes"' | grep -o '[0-9]*')
        printf "[%3d/%d] OK  %-50s %s notes\n" "$count" "$total" "$name" "$n_notes"
        done_count=$((done_count + 1))
    else
        printf "[%3d/%d] SKIP (corrupt) %-50s\n" "$count" "$total" "$name"
        failed=$((failed + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Processed: $done_count"
echo "Skipped: $skipped"
echo "Failed: $failed"
echo "Output: $OUTPUT_DIR"

#!/bin/bash
# Parse all Leduc GP files using alphaTab → JSON, then generate MIDI via Python.
# Output: data/leduc/processed/{name}.json + {name}.mid
#
# Usage: bash scripts/data/leduc/build_dataset.sh

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
GP_DIR="$REPO_ROOT/data/leduc/gp_files"
OUTPUT_DIR="$REPO_ROOT/data/leduc/processed"
PARSER_DIR="$REPO_ROOT/scripts/stage2/data/leduc/alphatab"
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

    # Generate MIDI + normalize JSON (add top-level tuning from tracks)
    "$PYTHON" -c "
import sys, json, pretty_midi
with open(sys.argv[1]) as f:
    data = json.load(f)
if 'tuning' not in data and 'tracks' in data and data['tracks']:
    data['tuning'] = data['tracks'][0].get('tuning', [64,59,55,50,45,40])
with open(sys.argv[1], 'w') as f:
    json.dump(data, f, indent=2)
midi = pretty_midi.PrettyMIDI(initial_tempo=data.get('tempo', 120))
guitar = pretty_midi.Instrument(program=24)
for n in data['notes']:
    guitar.notes.append(pretty_midi.Note(
        velocity=80, pitch=n['pitch'],
        start=n['start'], end=max(n['start']+0.01, n['end'])))
midi.instruments.append(guitar)
midi.write(sys.argv[2])
" "$json_file" "$mid_file" 2>/dev/null

    if [ -f "$mid_file" ]; then
        n_notes=$(echo "$output" | grep '"n_notes"' | grep -o '[0-9]*')
        printf "[%3d/%d] OK  %-50s %s notes\n" "$count" "$total" "$name" "$n_notes"
        done_count=$((done_count + 1))
    else
        printf "[%3d/%d] FAIL (midi) %-50s\n" "$count" "$total" "$name"
        failed=$((failed + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Processed: $done_count"
echo "Skipped: $skipped"
echo "Failed: $failed"
echo "Output: $OUTPUT_DIR"

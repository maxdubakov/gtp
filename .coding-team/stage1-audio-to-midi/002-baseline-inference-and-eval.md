---
task: 002
title: Baseline inference & evaluation on guitar
status: pending
---

## Context

Task 001 set up the project structure and verified the pretrained Kong piano model loads and runs a forward pass. Now we need the full inference pipeline: audio → model → post-processing → note events → MIDI. Then evaluate on GuitarSet to establish our baseline (~50-55% onset F1 expected from Riley et al.).

## Objective

1. Complete the inference pipeline: audio file → model → note events → MIDI file.
2. Run the pretrained piano model on GuitarSet and evaluate with mir_eval.
3. Produce a listenable verification artifact.

## Scope

### Post-processing: activations → note events

Kong's `piano_transcription-master/utils/utilities.py` contains `RegressionPostProcessor` — the class that converts frame-level onset/offset/frame/velocity activations into discrete note events (pitch, onset_time, offset_time, velocity). It uses parabolic interpolation for sub-frame onset precision.

Vendor this into `src/gtp/postprocess.py`. Also bring over the `write_events_to_midi` utility (or equivalent using `pretty_midi`). Adapt imports but keep the logic intact.

Wire it into `inference.py` so `PianoTranscription.transcribe(audio)` returns a list of note events AND optionally writes a MIDI file.

### Evaluation script

Create `scripts/eval_guitarset.py`:
1. Iterate all 360 GuitarSet mono-mic audio files
2. Run inference on each (resample 44.1kHz → 16kHz)
3. Parse JAMS annotations: `note_midi` namespace at annotation indices 1,3,5,7,9,11 (one per string). Each observation has time, duration, and value (MIDI pitch as float). Combine all 6 strings into one flat list of (onset, offset, pitch) tuples.
4. Evaluate with `mir_eval.transcription.precision_recall_f1_overlap` using onset-only mode (offset_ratio=None), 50ms onset tolerance
5. Print per-file F1 and aggregate P/R/F1
6. Save results CSV to `results/baseline_guitarset.csv`

### Verification artifact

Create `scripts/verify_transcription.py` that takes one audio file, runs inference, and produces a stereo WAV (left=original audio, right=synthesized MIDI blips at predicted onsets) — same pattern we used for GAPS alignment verification. Also save the predicted MIDI file.

Run it on one GuitarSet file (e.g., `00_BN1-129-Eb_comp`) and one GAPS file (e.g., `001_mvswc`) so we can listen.

## Code style
Write clean, readable, self-explanatory code. Only comment non-obvious decisions or domain-specific logic. No redundant comments. Use descriptive variable/function names.

## Non-goals
- Fine-tuning or modifying the model
- Offset evaluation (onset-only F1 matching Riley et al.'s protocol)
- Processing GAPS through evaluation (GuitarSet is our eval benchmark)

## Constraints
- Must work on MPS (Mac)
- GuitarSet audio is 44.1kHz — resample to 16kHz for the model
- Use mir_eval for metrics to match published results

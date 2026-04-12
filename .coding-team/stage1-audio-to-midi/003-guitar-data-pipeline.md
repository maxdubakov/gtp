---
task: 003
title: Guitar data pipeline for training
status: pending
---

## Context

We have the pretrained Kong piano model working for inference (Tasks 1-2). Now we need a data pipeline that loads GAPS and GuitarSet into the format Kong's training loop expects, so we can fine-tune on guitar data in Task 4.

Kong's training expects a dataset that returns dicts like:
```python
{
    'waveform': (160000,),                    # 10s at 16kHz
    'onset_roll': (1001, 88),                 # binary: 1 at onset frames
    'offset_roll': (1001, 88),                # binary: 1 at offset frames
    'reg_onset_roll': (1001, 88),             # regression target for onset (Gaussian-like)
    'reg_offset_roll': (1001, 88),            # regression target for offset
    'frame_roll': (1001, 88),                 # binary: 1 for all active frames
    'velocity_roll': (1001, 88),              # 0-127 velocity at active frames
    'mask_roll': (1001, 88),                  # 1 everywhere except cross-segment notes
}
```

Where 1001 = 10 seconds × 100 frames/sec + 1, and 88 = piano pitch classes (MIDI 21-108).

The key class to understand is `TargetProcessor` in Kong's `piano_transcription-master/utils/utilities.py` (line 215). It converts note events (onset_time, offset_time, midi_note, velocity) into the target rolls above. The regression targets (`reg_onset_roll`, `reg_offset_roll`) are NOT just binary — they use `get_regression()` to create a smooth Gaussian-like activation centered on the onset/offset frame, which is what the model learns to regress.

## Objective

Create `src/gtp/data.py` — a PyTorch Dataset that:
1. Loads audio + annotations from GAPS and GuitarSet
2. Extracts random 10-second segments
3. Builds target rolls from the annotations
4. Returns batches in the format above

## Scope

### `src/gtp/data.py`

**GuitarDataset class** (subclass `torch.utils.data.Dataset`):

The dataset is built from a list of (audio_path, notes) pairs, where notes is a list of `{'onset_time', 'offset_time', 'midi_note', 'velocity'}` dicts. This common format decouples the dataset class from the specific annotation format (JAMS vs MIDI).

Constructor:
- Takes a list of audio/notes pairs (pre-parsed)
- `segment_seconds=10.0`, `frames_per_second=100`, `sample_rate=16000`
- Pre-computes the segment index: for each audio file, generate all valid start times with `hop_seconds=1.0` stride

`__getitem__`:
- Load the audio segment (10s at 16kHz)
- Filter notes that overlap with this segment
- Use `TargetProcessor.process_notes()` to build target rolls
- Return the data dict

**TargetProcessor class**: Vendor from Kong's `utilities.py` (the `TargetProcessor` class). Simplify it:
- Remove pedal handling (guitar doesn't have pedals)
- Remove the MIDI event string parsing — our input is already structured note dicts
- Keep the core target roll construction logic (onset_roll, offset_roll, reg_onset_roll, reg_offset_roll, frame_roll, velocity_roll, mask_roll)
- Keep `get_regression()` — this is the function that creates the Gaussian-like regression targets

**Loader functions** to parse annotations into the common note format:
- `load_gaps_notes(midi_path)` → list of note dicts (from pretty_midi)
- `load_guitarset_notes(jams_path)` → list of note dicts (from JAMS, combining all 6 strings)
- `build_dataset(gaps_dir, guitarset_dir, split)` → GuitarDataset instance
  - For GAPS: use `gaps_metadata_with_splits.csv` for train/test split
  - For GuitarSet: use player-based 6-fold split (player IDs 00-05), matching the literature. For initial training, use players 00-04 for train, player 05 for validation.

### `scripts/test_data_pipeline.py`

Verification script that:
1. Builds the training dataset
2. Loads a few samples
3. Prints shapes and value ranges of all target rolls
4. With `-v`: trace the full flow from raw annotation → note events → target rolls, showing intermediate values

### Velocity handling

GAPS MIDI files don't have meaningful velocity (classical guitar recordings). GuitarSet JAMS `note_midi` observations have `confidence` field which may encode velocity, but it's often None.

For notes without velocity: use a default of 64 (medium). The model still learns onset/offset/frame from the regression targets — velocity is the least important head for our use case.

## Code style
Write clean, readable, self-explanatory code. Only comment non-obvious decisions or domain-specific logic. No redundant comments.

## Non-goals
- Data augmentation (Task 4 will add this if needed)
- Pedal targets (guitar has no sustain pedal)
- Training loop (Task 4)
- HDF5 preprocessing — load audio on-the-fly (dataset is small enough)

## Constraints
- GAPS audio is 48kHz, GuitarSet is 44.1kHz — resample to 16kHz on load
- Guitar pitch range (E2=40 to ~E6=88) fits within piano range (A0=21 to C8=108), so we keep 88 classes
- Use the same `get_regression()` function from Kong's code — the loss function expects these specific regression targets
- Include verbose tracing via `gtp.log.trace` at key points

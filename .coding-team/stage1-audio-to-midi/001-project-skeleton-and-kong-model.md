---
task: 001
title: Project skeleton & Kong model integration
status: pending
---

## Context

We're building a guitar tablature transcription pipeline. Stage 1 is audio→MIDI using Kong et al.'s high-resolution piano transcription CRNN, fine-tuned on guitar data.

The repo is at `/Users/max/Documents/Programming/AI/gtp/`. It currently has:
- `piano_transcription-master/` — unmodified Kong source (extracted zip)
- `models/pretrained/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth` — pretrained piano checkpoint (164MB)
- `data/gaps_hf/` — GAPS guitar dataset (audio/ + midi/, 401 files)
- `data/guitarset/` — GuitarSet (audio_mono-mic/, annotation/, 360 files)
- `venv/` — Python 3.12 venv with PyTorch 2.6.0, librosa, pretty_midi, jams, mir_eval
- `requirements.txt` — current pip freeze (missing torch, jams, mir_eval — needs updating)

## Objective

1. Set up a proper Python project structure under `src/gtp/`.
2. Vendor the necessary Kong model files into our project, adapting them for PyTorch 2.6.
3. Verify the pretrained checkpoint loads and the model can do a forward pass.

## Scope

### Project structure
Create this layout:
```
src/gtp/
    __init__.py
    model/
        __init__.py
        kong.py          # model architecture (from Kong's models.py)
        losses.py        # loss functions (from Kong's losses.py)
        utils.py         # pytorch utilities (from Kong's pytorch_utils.py)
    inference.py         # inference logic (from Kong's inference.py)
```

### Vendoring Kong's code
From `piano_transcription-master/pytorch/`, extract into `src/gtp/model/`:
- `models.py` → `kong.py`: The `Note_pedal` model and its subcomponents (`AcousticModelCRnn8Dropout`, `Regress_onset_offset_frame_velocity_CRNN`, etc.). We only need the note model, not the pedal model — but keep both for now since the checkpoint contains both.
- `losses.py` → `losses.py`: Loss functions used for training.
- `pytorch_utils.py` → `utils.py`: Utility functions (conv blocks, init, etc.).
- `inference.py` → `../inference.py`: The `PianoTranscription` class that handles audio→MIDI inference.

For each file:
- Add a source attribution comment at top: `# Adapted from https://github.com/bytedance/piano_transcription (commit: master)`
- Fix imports to use our package paths (`from gtp.model.utils import ...` etc.)
- Fix any PyTorch 2.6 deprecations. Known issues to watch for:
  - `torch.nn.utils.weight_norm` may need updating
  - Any use of `torch.no_grad` as a context vs decorator
  - `numpy()` calls on GPU tensors need `.cpu()` first
  - Check for deprecated `torch.cuda.amp` usage

### Verification
Write a simple test script `scripts/test_model_load.py` that:
1. Instantiates the model
2. Loads the pretrained checkpoint
3. Runs a forward pass on a random tensor shaped like a 10-second audio clip (16000 * 10 = 160000 samples)
4. Prints output shapes
5. Runs on MPS if available, else CPU

### Update requirements.txt
Regenerate `requirements.txt` from the venv (which now includes torch, jams, mir_eval, etc.)

### .gitignore
Create `.gitignore` that excludes: `venv/`, `data/`, `models/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `piano_transcription-master/`

## Non-goals
- Training code (Task 4)
- Data loading pipeline (Task 3)
- Full inference with MIDI output (Task 2)
- Any modification to the model architecture itself

## Constraints
- Python 3.12, PyTorch 2.6.0
- Must work on macOS with MPS backend
- Keep Kong's architecture exactly as-is — we're loading pretrained weights, so layer names/shapes must match the checkpoint

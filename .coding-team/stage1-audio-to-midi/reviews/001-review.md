# Review: Task 001 — Project skeleton & Kong model integration

**Reviewer**: @coding-team:code-reviewer
**Status**: Approved

## Summary

The implementation correctly fulfills all objectives in the task brief:

- Project structure under `src/gtp/` matches the specified layout exactly.
- Kong model files are vendored with proper source attribution comments and updated `gtp.model.*` import paths.
- Model architecture is faithfully reproduced — all layer names, shapes, and initialization logic match the original, which is essential for checkpoint compatibility.
- PyTorch 2.6 compatibility is handled: `torch.load` uses `weights_only=True` and `map_location='cpu'`.
- The pretrained checkpoint loads successfully (both note and pedal sub-models).
- The test script runs a forward pass on MPS, producing correct output shapes: `(1, 1001, 88)` for note outputs and `(1, 1001, 1)` for pedal outputs.
- `.gitignore` and `requirements.txt` are updated as specified.

## Verification

I ran `scripts/test_model_load.py` against the venv (Python 3.12, PyTorch 2.6.0, MPS backend). The checkpoint loaded without errors and the forward pass succeeded with the expected output shapes.

## Residual observations

These are not blocking and do not require changes for this task, but are worth noting:

1. **Staged `piano_transcription-master/` files**: The git status shows the entire `piano_transcription-master/` directory staged for commit. The updated `.gitignore` excludes this directory, but the staging predates the ignore rule. Before the next commit, `git rm --cached -r piano_transcription-master/` should be run to unstage these files. This is outside the scope of this task.

2. **`Note_pedal.load_state_dict` override**: The vendored `Note_pedal.load_state_dict(self, m, strict=False)` overrides `nn.Module.load_state_dict` with an incompatible contract — it expects a dict containing `note_model` and `pedal_model` sub-dicts rather than a flat state dict. This is faithful to Kong's original code and required for checkpoint loading, but callers must always pass `checkpoint['model']` rather than a standard state dict. Future training code (Task 4) should be aware of this.

No changes requested.

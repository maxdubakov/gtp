# Stage 2 RunPod Workflow

## 0. Pick a pod

- **GPU**: RTX 4000 Ada (24GB VRAM) is plenty — model is 2.24M params.
- **Disk**: ~5 GB volume + ~15 GB container is enough; defaults (30+40) are way overkill but cheap.
- **Image**: prefer Python 3.12 + PyTorch 2.6 templates if available. Python 3.10 also works (see install note below).

## 1. Pack locally (Mac)

```bash
bash scripts/stage2/setup/pack_code.sh   # → code_stage2.tar.gz (~64 KB)
bash scripts/stage2/setup/pack_data.sh   # → data_stage2.tar.gz (~42 MB)
```

Both scripts use `COPYFILE_DISABLE=1 tar --no-xattrs` to avoid the macOS `._*`
AppleDouble metadata files that otherwise pollute the tarball and break the
JSON loader on Linux.

Upload both tarballs to the pod (RunPod web UI or `runpodctl send`).

## 2. Set up on RunPod

```bash
# Extract
tar xzf code_stage2.tar.gz
tar xzf data_stage2.tar.gz

# Install third-party deps. If pod is Python 3.10, requirements.txt's strict
# pins (numpy 2.4.x, contourpy 1.3.3) won't have wheels — install only the
# top-level deps and let pip resolve the rest:
pip install torch==2.6.0 torchaudio==2.6.0 \
    transformers numpy scipy librosa pretty_midi \
    pyguitarpro jams soundfile mido pyfluidsynth

# If pod is Python 3.12, this works as-is:
# pip install -r requirements.txt

# Install the gtp package (relies on pyproject.toml allowing >=3.10)
pip install -e .

# Build the augmented dataset (~9 min one-time)
python scripts/stage2/build_aug_dataset.py
```

## 3. Train detached

```bash
mkdir -p runs/stage2_001

PYTHONUNBUFFERED=1 nohup python scripts/stage2/train.py \
    --device cuda --num-workers 2 \
    --batch-size 64 \
    --max-steps 30000 \
    --eval-steps 1000 --save-steps 1000 \
    --output-dir runs/stage2_001 \
    > runs/stage2_001/train.log 2>&1 &
disown
```

Why these flags matter:
- **`PYTHONUNBUFFERED=1`** — Python buffers stdout when redirected to a file (~4-8 KB). Without this, `train.log` appears empty for minutes; with it, lines flush immediately.
- **`nohup`** — process ignores SIGHUP when SSH disconnects; survives the connection close.
- **`& disown`** — runs in background, fully detaches from the shell's job table.
- **`> ... 2>&1`** — redirects both stdout and stderr to the log file.
- **`--num-workers 2`** (not 4) — fewer DataLoader workers = less IPC pressure on the container, avoids `ENOBUFS` and `SIGABRT` crashes after long runs. The script also sets `mp.set_sharing_strategy('file_system')` and `persistent_workers=True` for the same reason.

## 4. Monitor

```bash
# Live tail (Ctrl+C stops watching, doesn't kill training)
tail -f runs/stage2_001/train.log

# Confirm process alive
ps -ef | grep "[s]tage2/train.py"

# GPU activity
nvidia-smi
```

## 5. Stop training (if needed)

```bash
pkill -f "scripts/stage2/train.py"
```

To resume from the last checkpoint after a stop or crash:
```bash
PYTHONUNBUFFERED=1 nohup python scripts/stage2/train.py \
    [...same args...] \
    --resume runs/stage2_001/step_<latest>.pth \
    >> runs/stage2_001/train.log 2>&1 &
disown
```
(Note `>>` to append instead of overwrite.)

## 6. Pull results back

```bash
# On RunPod
runpodctl send runs/stage2_001/

# On Mac (it'll print a one-time code)
runpodctl receive <code>
```

## Errors I ran into (and how to fix them)

1. **`tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'` on extract** — cosmetic on macOS-packed tarballs without `--no-xattrs`. The fix is in `pack_*.sh` now; if you re-pack and still see this, double-check `COPYFILE_DISABLE=1` is set.

2. **`UnicodeDecodeError: byte 0xa3` while loading JSONs** — caused by macOS BSD tar embedding `._*` AppleDouble sidecar files alongside real `.json` files. The new `pack_data.sh` prevents these. If you somehow have them: `find data -name "._*" -delete`.

3. **`pip install -e .` fails with `Package 'gtp' requires a different Python: 3.10.x not in '>=3.12'`** — fixed by `requires-python = ">=3.10"` in pyproject.toml. If on an old code drop, patch with `sed -i 's/>=3.12/>=3.10/' pyproject.toml`.

4. **`requirements.txt` Mac-path editable install fails** — line `-e /Users/max/...` is gone now. If still present: `sed -i '/^-e \|^# Editable/d' requirements.txt`.

5. **Training crashes after long run with `OSError: ENOBUFS` or `DataLoader worker killed by signal: Aborted`** — multiprocessing IPC exhaustion. Already mitigated by:
   - `mp.set_sharing_strategy('file_system')` in train.py
   - `persistent_workers=True` in the DataLoader
   - Lower `--num-workers` (2 not 4)
   If still hits, try `--num-workers 0` (no workers, slower but bulletproof).

6. **`tail -f` shows only `nohup: ignoring input` for a long time** — Python buffering. Fix: `PYTHONUNBUFFERED=1` (see flag note above) or run with `python -u`.

7. **tmux `Ctrl+B`-then-`D` doesn't detach** - use `nohup` and be happy.

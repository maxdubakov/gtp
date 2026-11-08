# Stage 2 RunPod Workflow

## 0. Pick a pod

- **CPU matters more than GPU.** Training is bound by data loading (~20-25 ms of tokenization per
  sequence on one core), not by the 2.24M-param model. Pick a pod with **16+ vCPUs**; any CUDA GPU
  works (RTX 4000 Ada / A4000 is plenty). Both runs of an experiment pair fit on one pod.
- **Disk**: ~10 GB (the rebuilt dataset is 3.3 GB).
- **Image**: any CUDA image; `uv sync` installs Python 3.12 and torch 2.6 from `uv.lock`.

## 1. Pack locally (Mac)

```bash
bash scripts/stage2/setup/pack_code.sh   # → code_stage2.tar.gz (~200 KB): src, scripts/stage2, conf, uv.lock, VERSION, GIT_DIFF
bash scripts/stage2/setup/pack_data.sh   # → data_stage2.tar.gz (~50 MB): processed JSONs + string-classifier outputs
```

The dataset itself is rebuilt on the pod (deterministic: same files, same seed, byte-identical
JSONLs). `VERSION` and `GIT_DIFF` carry the commit and any uncommitted changes, so runs on the pod
record the same provenance as local ones. Both scripts use `COPYFILE_DISABLE=1 tar --no-xattrs` to
avoid macOS `._*` AppleDouble files, which break the JSON loader on Linux.

Upload both tarballs to the pod (`runpodctl send`, or `scp` over the pod's SSH port).

## 2. Set up on RunPod

```bash
tar xzf code_stage2.tar.gz && tar xzf data_stage2.tar.gz

# uv fetches Python 3.12 itself, so the pod's system Python version doesn't matter.
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync --frozen
source .venv/bin/activate
ulimit -n 65536

# Rebuild the player-05 split (~10 min), then compare against the Mac's checksums
python scripts/stage2/build_aug_dataset.py --test-player 5 --output-dir data/stage2_p05
sha256sum data/stage2_p05/*.jsonl

# W&B (runs show up at forge.coreweave.com/wandb/max-dubakov/gtp)
export WANDB_API_KEY=...
```

## 3. Train detached

Config lives in `conf/stage2/` (Hydra). Each run writes to `runs/stage2/<timestamp>_<name>/`
(config.json, metrics.jsonl, provenance.json, train.log, checkpoints/) and logs to W&B.

```bash
for exp in p05_baseline p05_audio; do
  PYTHONUNBUFFERED=1 nohup python scripts/stage2/train.py experiment=$exp num_workers=6 \
      > train_$exp.out 2>&1 &
done
disown -a
```

- **`PYTHONUNBUFFERED=1`**: without it Python buffers redirected stdout and the log looks empty for minutes.
- **`nohup` + `disown`**: the run survives the SSH session closing.
- **`num_workers`**: data loading is the bottleneck, so more workers = faster steps (watch
  `train/s_per_step` in W&B). Older runs hit `ENOBUFS` / `SIGABRT` with 4 workers before
  `file_system` sharing and persistent workers were added; drop back to 2 if that returns.

## 4. Monitor

```bash
tail -f runs/stage2/*_p05_audio/train.log
ps -ef | grep "[s]tage2/train.py"
nvidia-smi
```

## 5. Stop / resume

```bash
pkill -f "scripts/stage2/train.py"
```

Resume into the same run dir from its last checkpoint:
```bash
RUN=runs/stage2/<timestamp>_p05_audio
PYTHONUNBUFFERED=1 nohup python scripts/stage2/train.py experiment=p05_audio num_workers=6 \
    hydra.run.dir=$RUN resume=$RUN/checkpoints/step_<latest>.pth >> train_p05_audio.out 2>&1 &
disown
```

## 6. Pull results back

```bash
# On RunPod: ~540 MB per run (60 checkpoints of 9 MB); send the run dirs
runpodctl send runs/stage2/
# On Mac
runpodctl receive <code>
```

## Errors I ran into (and how to fix them)

1. **`tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'` on extract** — cosmetic on macOS-packed tarballs without `--no-xattrs`. The fix is in `pack_*.sh` now; if you re-pack and still see this, double-check `COPYFILE_DISABLE=1` is set.

2. **`UnicodeDecodeError: byte 0xa3` while loading JSONs** — caused by macOS BSD tar embedding `._*` AppleDouble sidecar files alongside real `.json` files. The new `pack_data.sh` prevents these. If you somehow have them: `find data -name "._*" -delete`.

3. **Training crashes after long run with `OSError: ENOBUFS` or `DataLoader worker killed by signal: Aborted`** — multiprocessing IPC exhaustion. Already mitigated by:
   - `mp.set_sharing_strategy('file_system')` in train.py
   - `persistent_workers=True` in the DataLoader
   - Lower `--num-workers` (2 not 4)
   If still hits, try `--num-workers 0` (no workers, slower but bulletproof).

4. **`tail -f` shows only `nohup: ignoring input` for a long time** — Python buffering. Fix: `PYTHONUNBUFFERED=1` (see flag note above) or run with `python -u`.

5. **tmux `Ctrl+B`-then-`D` doesn't detach** - use `nohup` and be happy.

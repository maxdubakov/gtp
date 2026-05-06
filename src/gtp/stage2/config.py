"""Run-config tracking for Stage 2 experiments.

Each training run writes a config.json into its output directory, capturing:
  * Model spec (params, layers, dims).
  * Training hyperparameters (batch size, max steps, seed, optimizer).
  * Data info (dataset directory, sub-sequence/token counts, source mix).
  * Conditioning settings (genre/source tokens, dropout rates).
  * Rebalancing settings (per-source/genre upsampling weights).
  * Free-form metadata (experiment label, notes, git SHA, timestamp).

Usage:

    cfg = RunConfig(run_id='stage2_002_exp1_genre',
                    experiment_label='Exp 1: GENRE conditioning + 15% dropout')
    cfg.model.params = sum(p.numel() for p in model.parameters())
    cfg.train.batch_size = args.batch_size
    cfg.save(out_dir / 'config.json')

    # Later, anywhere:
    cfg = RunConfig.load(run_dir / 'config.json')
"""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ModelConfig:
    params: int = 0
    d_model: int = 0
    d_ff: int = 0
    n_layers: int = 0
    n_heads: int = 0
    vocab_size: int = 0


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 30000
    eval_steps: int = 1000
    save_steps: int = 5000
    eval_batches: int | None = None
    num_workers: int = 2
    seed: int = 42
    optimizer: str = 'Adafactor'
    learning_rate: str = 'adafactor_self_adaptive'
    resumed_from: str | None = None


@dataclass
class DataConfig:
    dataset_dir: str = ''
    sources: list[str] = field(default_factory=list)
    train_pieces: int = 0
    train_subseqs: int = 0
    train_enc_tokens: int = 0
    train_dec_tokens: int = 0
    val_pieces: int = 0
    val_subseqs: int = 0


@dataclass
class ConditioningConfig:
    """Conditioning tokens added to the encoder prefix.

    `genre`: include `GENRE<X>` token. Dropped to `GENRE<unknown>` `genre_dropout`
    fraction of the time during training (classifier-free style).
    `source`: include `SOURCE<dataset>` token. Off by default — leakage path,
    not useful at inference time.
    """

    genre: bool = False
    genre_dropout: float = 0.0
    source: bool = False


@dataclass
class RebalancingConfig:
    """Per-source / per-genre sampling weights for the training DataLoader.

    Multiplicative weights vs uniform. e.g. {'guitarset': 8.0, 'leduc': 5.0}
    upsamples those sources by 8x / 5x respectively.
    """

    enabled: bool = False
    source_weights: dict[str, float] = field(default_factory=dict)
    genre_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class DeviceConfig:
    """Device snapshot at the start of a run.

    Helps explain wall-clock differences across runs (e.g. A100 vs RTX 4090).
    `type` is the resolved device string ('cpu' | 'mps' | 'cuda').
    `cuda_name` and `cuda_memory_gib` are populated only when type=='cuda'.
    """

    type: str = ''
    cuda_name: str = ''
    cuda_memory_gib: float = 0.0


@dataclass
class RunConfig:
    run_id: str
    experiment_label: str = ''
    timestamp: str = ''
    git_sha: str = ''
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    rebalancing: RebalancingConfig = field(default_factory=RebalancingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    notes: str = ''

    def save(self, path) -> None:
        """Write config.json. Creates parent directories if missing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, path) -> 'RunConfig':
        """Read config.json, reconstructing nested dataclasses."""
        data = json.loads(Path(path).read_text())
        return cls(
            run_id=data['run_id'],
            experiment_label=data.get('experiment_label', ''),
            timestamp=data.get('timestamp', ''),
            git_sha=data.get('git_sha', ''),
            model=ModelConfig(**data.get('model', {})),
            train=TrainConfig(**data.get('train', {})),
            data=DataConfig(**data.get('data', {})),
            conditioning=ConditioningConfig(**data.get('conditioning', {})),
            rebalancing=RebalancingConfig(**data.get('rebalancing', {})),
            device=DeviceConfig(**data.get('device', {})),
            notes=data.get('notes', ''),
        )


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def get_git_sha(short: bool = True) -> str:
    """Best-effort git SHA. Returns 'unknown' on failure (no git, detached, etc.)."""
    cmd = ['git', 'rev-parse']
    if short:
        cmd.append('--short')
    cmd.append('HEAD')
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        return 'unknown'


def get_timestamp() -> str:
    """ISO-8601 local time, second resolution."""
    return datetime.now().isoformat(timespec='seconds')


def find_run_config(checkpoint_path) -> Path | None:
    """Locate the sibling config.json for a checkpoint, handling both layouts.

    New layout:    <run-dir>/checkpoints/step_X.pth + <run-dir>/config.json
    Legacy layout: <run-dir>/step_X.pth + <run-dir>/config.json

    Returns the config.json Path if found, else None.
    """
    p = Path(checkpoint_path)
    for candidate in (p.parent / 'config.json', p.parent.parent / 'config.json'):
        if candidate.exists():
            return candidate
    return None


def get_device_info(device: str) -> DeviceConfig:
    """Snapshot device info for a run. `device` is 'cpu' | 'mps' | 'cuda'.

    For 'cuda', queries torch for the GPU name and total memory (GiB,
    1024^3, matching nvidia-smi). Silently leaves the optional fields
    empty on any error so this is never a startup blocker.
    """
    info = DeviceConfig(type=device)
    if device != 'cuda':
        return info
    try:
        import torch  # local import — config.py stays usable without torch.

        if torch.cuda.is_available():
            info.cuda_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.cuda_memory_gib = round(props.total_memory / (1024 ** 3), 2)
    except Exception:
        pass
    return info

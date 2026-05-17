"""Experiments config tracking for Stage 2."""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from gtp import REPO_ROOT


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
    checkpoint_steps: int = 1000
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

    `genre`: include `GENRE<X>` token. Dropped to `GENRE<unknown>`
    `genre_dropout`: Drop genre token with this rate during training
    """

    genre: bool = False
    genre_dropout: float = 0.0


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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, path) -> 'RunConfig':
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


def get_git_sha() -> str:
    version_file = REPO_ROOT / 'VERSION'
    if version_file.exists():
        sha = version_file.read_text().strip()
        if sha:
            return sha
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
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


def find_run_config(checkpoint_path) -> Path:
    """Locate config.json for a checkpoint. Raises FileNotFoundError if not found."""
    candidate = Path(checkpoint_path).parent.parent / 'config.json'
    if not candidate.exists():
        raise FileNotFoundError(f'config.json not found at {candidate} (expected sibling of checkpoint dir)')
    return candidate


def get_timestamp() -> str:
    return datetime.now().isoformat(timespec='seconds')

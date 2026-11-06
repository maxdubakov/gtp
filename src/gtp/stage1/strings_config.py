"""Structured config schema for the string classifier (values and types live here, sweeps in conf/strings)."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from gtp.track import WandbConfig


@dataclass
class DataConfig:
    source: str = 'mic'
    feature: str = 'logmel229'
    window_ms: int = 400


@dataclass
class StringNetConfig:
    _target_: str = 'gtp.stage1.strings.StringNet'
    width: float = 1.0
    use_pitch: bool = True


@dataclass
class OptimConfig:
    epochs: int = 30
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class StringsConfig:
    name: str = 'adhoc'
    test_player: int = 5
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: StringNetConfig = field(default_factory=StringNetConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


ConfigStore.instance().store(name='strings_schema', node=StringsConfig)

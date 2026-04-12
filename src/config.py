"""Configuration system using dataclasses and YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dacite import from_dict, Config as DaciteConfig


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    max_duration_sec: float = 3.0
    min_duration_sec: float = 0.5


@dataclass
class FeatureConfig:
    feature_type: str = "mel_spectrogram"  # "mel_spectrogram" or "mfcc"
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    n_mfcc: int = 40


@dataclass
class AugmentConfig:
    enable: bool = True
    spec_augment: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 20
    num_masks: int = 2
    noise_augment: bool = False
    noise_snr_range: list = field(default_factory=lambda: [5, 20])
    volume_perturbation: bool = False
    volume_gain_db_range: list = field(default_factory=lambda: [-6, 6])


@dataclass
class DataConfig:
    data_dir: str = "data"
    num_speakers: int = 50
    min_utterances_per_speaker: int = 20
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 64
    num_workers: int = 4


@dataclass
class ModelConfig:
    type: str = "cnn"  # "cnn", "ecapa_tdnn"
    embedding_dim: int = 192
    dropout: float = 0.3
    # ECAPA-TDNN specific
    channels: int = 512
    scale: int = 8


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 2e-4
    scheduler: str = "cosine"  # "cosine" or "step"
    warmup_epochs: int = 5
    patience: int = 15
    loss: str = "cross_entropy"  # "cross_entropy" or "aam_softmax"
    aam_margin: float = 0.2
    aam_scale: float = 30.0
    grad_clip_max_norm: float = 5.0
    use_amp: bool = True


@dataclass
class Config:
    experiment_name: str = "default"
    seed: int = 42
    device: str = "auto"

    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    augmentation: AugmentConfig = field(default_factory=AugmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(yaml_path: str) -> Config:
    """Load configuration from a YAML file, with optional base config inheritance."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    # Handle base config inheritance
    if "_base_" in data:
        base_path = yaml_path.parent / data.pop("_base_")
        with open(base_path) as f:
            base_data = yaml.safe_load(f) or {}
        data = _deep_merge(base_data, data)

    return from_dict(
        data_class=Config,
        data=data,
        config=DaciteConfig(strict=False),
    )

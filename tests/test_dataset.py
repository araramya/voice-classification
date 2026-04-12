"""Tests for dataset loading (requires synthetic data)."""

import numpy as np
import pytest
import torch
import pandas as pd
from pathlib import Path

from src.config import AudioConfig, FeatureConfig, AugmentConfig
from src.data.download import create_synthetic_dataset


@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """Create a tiny synthetic dataset for testing."""
    data_dir = tmp_path_factory.mktemp("test_data")
    create_synthetic_dataset(data_dir, num_speakers=3, utterances_per_speaker=5)
    return data_dir


@pytest.fixture(scope="module")
def test_metadata(test_data_dir):
    from src.data.download import build_metadata
    return build_metadata(test_data_dir)


def test_synthetic_dataset_created(test_data_dir):
    vox_dir = test_data_dir / "raw" / "voxceleb1"
    speakers = list(vox_dir.iterdir())
    assert len(speakers) == 3


def test_metadata_columns(test_metadata):
    assert "speaker_id" in test_metadata.columns
    assert "file_path" in test_metadata.columns
    assert "duration_sec" in test_metadata.columns
    assert len(test_metadata) == 15  # 3 speakers * 5 utterances


def test_speaker_dataset(test_metadata):
    from src.data.dataset import SpeakerDataset

    dataset = SpeakerDataset(
        metadata=test_metadata,
        feature_config=FeatureConfig(feature_type="mel_spectrogram"),
        audio_config=AudioConfig(),
        train=False,
    )
    features, label = dataset[0]
    assert features.dim() == 3       # (1, n_mels, time)
    assert features.shape[1] == 80
    assert isinstance(label, (int, np.integer, torch.Tensor))
    assert dataset.num_speakers == 3

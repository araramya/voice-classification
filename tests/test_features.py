"""Tests for feature extraction shape correctness."""

import torch
import pytest

from src.config import FeatureConfig
from src.data.features import FeatureExtractor


@pytest.fixture
def sample_waveform():
    """3-second mono waveform at 16kHz."""
    return torch.randn(1, 48000)


@pytest.fixture
def mel_extractor():
    config = FeatureConfig(feature_type="mel_spectrogram")
    return FeatureExtractor(config, sample_rate=16000)


@pytest.fixture
def mfcc_extractor():
    config = FeatureConfig(feature_type="mfcc")
    return FeatureExtractor(config, sample_rate=16000)


def test_mel_spectrogram_shape(mel_extractor, sample_waveform):
    mel = mel_extractor(sample_waveform)
    assert mel.dim() == 3
    assert mel.shape[0] == 1       # channels
    assert mel.shape[1] == 80      # n_mels
    assert mel.shape[2] > 0        # time frames


def test_mfcc_shape(mfcc_extractor, sample_waveform):
    mfcc = mfcc_extractor(sample_waveform)
    assert mfcc.dim() == 2
    assert mfcc.shape[0] == 40     # n_mfcc
    assert mfcc.shape[1] > 0       # time frames


def test_mel_log_scale(mel_extractor, sample_waveform):
    mel = mel_extractor(sample_waveform)
    # Log-mel can have negative values (log of values < 1)
    assert not torch.isnan(mel).any()
    assert not torch.isinf(mel).any()


def test_1d_input(mel_extractor):
    waveform_1d = torch.randn(48000)
    mel = mel_extractor(waveform_1d)
    assert mel.dim() == 3
    assert mel.shape[0] == 1

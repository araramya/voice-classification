"""Tests for model forward pass shapes."""

import torch
import pytest

from src.models.cnn import SpeakerCNN
from src.models.ecapa_tdnn import ECAPATDNN


@pytest.fixture
def batch_mel():
    """Batch of mel-spectrograms: (batch=4, 1, n_mels=80, time_frames=300)."""
    return torch.randn(4, 1, 80, 300)


def test_cnn_forward_shape(batch_mel):
    model = SpeakerCNN(num_speakers=10, embedding_dim=192)
    model.eval()
    with torch.no_grad():
        logits = model(batch_mel)
    assert logits.shape == (4, 10)


def test_cnn_embedding_shape(batch_mel):
    model = SpeakerCNN(num_speakers=10, embedding_dim=192)
    model.eval()
    with torch.no_grad():
        emb = model.extract_embedding(batch_mel)
    assert emb.shape == (4, 192)


def test_ecapa_tdnn_forward_shape(batch_mel):
    model = ECAPATDNN(num_speakers=10, channels=512, embedding_dim=192, scale=8)
    model.eval()
    with torch.no_grad():
        logits = model(batch_mel)
    assert logits.shape == (4, 10)


def test_ecapa_tdnn_embedding_shape(batch_mel):
    model = ECAPATDNN(num_speakers=10, channels=512, embedding_dim=192, scale=8)
    model.eval()
    with torch.no_grad():
        emb = model.extract_embedding(batch_mel)
    assert emb.shape == (4, 192)


def test_ecapa_tdnn_variable_time(batch_mel):
    """ECAPA-TDNN should handle different time lengths."""
    model = ECAPATDNN(num_speakers=10, channels=512, embedding_dim=192, scale=8)
    model.eval()
    short = torch.randn(2, 1, 80, 100)
    long = torch.randn(2, 1, 80, 500)
    with torch.no_grad():
        out_short = model(short)
        out_long = model(long)
    assert out_short.shape == (2, 10)
    assert out_long.shape == (2, 10)

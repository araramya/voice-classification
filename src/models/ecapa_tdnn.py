"""ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN.

Based on: Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention,
Propagation and Aggregation in TDNN Based Speaker Verification", Interspeech 2020.
"""

import torch
import torch.nn as nn

from src.models.layers import SERes2NetBlock, AttentiveStatisticsPooling


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN speaker embedding extractor.

    Architecture:
        1. Initial Conv1d projection (n_mels -> channels)
        2. Three SE-Res2Net blocks with increasing dilation (2, 3, 4)
        3. Multi-layer feature aggregation (MFA) — concatenate outputs of all blocks
        4. Conv1d fusion of concatenated features
        5. Attentive statistics pooling
        6. FC layer -> BN -> speaker embedding
        7. Classification head

    Input: (batch, 1, n_mels, time) mel-spectrogram
    Output: (batch, num_speakers) classification logits
    """

    def __init__(
        self,
        num_speakers: int,
        channels: int = 512,
        embedding_dim: int = 192,
        n_mels: int = 80,
        scale: int = 8,
    ):
        super().__init__()
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim

        # Initial feature projection
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # SE-Res2Net blocks with increasing dilation
        self.layer2 = SERes2NetBlock(channels, kernel_size=3, dilation=2, scale=scale)
        self.layer3 = SERes2NetBlock(channels, kernel_size=3, dilation=3, scale=scale)
        self.layer4 = SERes2NetBlock(channels, kernel_size=3, dilation=4, scale=scale)

        # Multi-layer feature aggregation
        # Fuse concatenated outputs from layer1 + layer2 + layer3 + layer4 (but
        # following the paper we concatenate layers 2, 3, 4 = 3 * channels)
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels * 3, kernel_size=1),
            nn.BatchNorm1d(channels * 3),
            nn.ReLU(inplace=True),
        )

        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(channels * 3)

        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(channels * 3 * 2, embedding_dim),  # *2 for mean + std from ASP
            nn.BatchNorm1d(embedding_dim),
        )

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert mel-spectrogram to 1D format for TDNN layers.

        Args:
            x: (batch, 1, n_mels, time) — 2D spectrogram
        Returns:
            (batch, n_mels, time) — 1D format
        """
        return x.squeeze(1)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding.

        Args:
            x: (batch, 1, n_mels, time)
        Returns:
            (batch, embedding_dim)
        """
        x = self._reshape_input(x)  # (batch, n_mels, time)

        # Frame-level processing
        out1 = self.layer1(x)   # (batch, channels, time)
        out2 = self.layer2(out1)  # (batch, channels, time)
        out3 = self.layer3(out2)  # (batch, channels, time)
        out4 = self.layer4(out3)  # (batch, channels, time)

        # Multi-layer feature aggregation
        mfa_input = torch.cat([out2, out3, out4], dim=1)  # (batch, channels*3, time)
        mfa_out = self.mfa(mfa_input)  # (batch, channels*3, time)

        # Attentive statistics pooling
        pooled = self.asp(mfa_out)  # (batch, channels*3*2)

        # Embedding
        embedding = self.embedding_layer(pooled)  # (batch, embedding_dim)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: (batch, 1, n_mels, time)
        Returns:
            (batch, num_speakers)
        """
        embedding = self.extract_embedding(x)
        logits = self.classifier(embedding)
        return logits

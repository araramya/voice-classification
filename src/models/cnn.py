"""VGG-style CNN for speaker identification on mel-spectrograms."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two convolution layers with BatchNorm and ReLU, followed by MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class SpeakerCNN(nn.Module):
    """CNN for speaker identification.

    Architecture:
        4 VGG-style conv blocks (64 -> 128 -> 256 -> 512)
        Temporal average pooling
        FC layer -> embedding
        Classification head

    Input: (batch, 1, n_mels, time_frames)
    """

    def __init__(self, num_speakers: int, embedding_dim: int = 192, dropout: float = 0.3):
        super().__init__()
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim

        self.conv_blocks = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )

        # After 4 max-pool layers with pool_size=2: freq dimension = n_mels // 16
        # For n_mels=80: 80 // 16 = 5
        self.freq_dim = 80 // 16  # 5

        self.embedding_layer = nn.Sequential(
            nn.Linear(512 * self.freq_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding (penultimate layer output).

        Args:
            x: (batch, 1, n_mels, time_frames)

        Returns:
            (batch, embedding_dim)
        """
        x = self.conv_blocks(x)  # (batch, 512, freq_reduced, time_reduced)

        # Temporal average pooling over time dimension
        x = x.mean(dim=3)  # (batch, 512, freq_reduced)
        x = x.view(x.size(0), -1)  # (batch, 512 * freq_reduced)

        x = self.embedding_layer(x)  # (batch, embedding_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits.

        Args:
            x: (batch, 1, n_mels, time_frames)

        Returns:
            (batch, num_speakers) logits
        """
        embedding = self.extract_embedding(x)
        logits = self.classifier(embedding)
        return logits

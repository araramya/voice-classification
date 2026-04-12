"""Shared building blocks for ECAPA-TDNN: Res2Net, SE-Block, Attentive Pooling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Global average pool -> FC -> ReLU -> FC -> Sigmoid -> channel-wise scaling.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        bottleneck = max(channels // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels, time)
        """
        scale = self.se(x).unsqueeze(2)  # (batch, channels, 1)
        return x * scale


class Res2Conv1dBlock(nn.Module):
    """Res2Net-style 1D convolution with multi-scale processing.

    Splits channels into `scale` groups, processes them hierarchically
    so each group aggregates information from all previous groups.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, scale: int = 8):
        super().__init__()
        assert channels % scale == 0, f"channels ({channels}) must be divisible by scale ({scale})"
        self.scale = scale
        self.width = channels // scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(scale - 1):
            self.convs.append(
                nn.Conv1d(
                    self.width, self.width, kernel_size,
                    dilation=dilation, padding=dilation * (kernel_size - 1) // 2,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels, time)
        """
        chunks = torch.chunk(x, self.scale, dim=1)  # list of (batch, width, time)
        outputs = [chunks[0]]

        for i in range(1, self.scale):
            if i == 1:
                xi = chunks[i]
            else:
                xi = chunks[i] + outputs[-1]
            xi = F.relu(self.bns[i - 1](self.convs[i - 1](xi)), inplace=True)
            outputs.append(xi)

        return torch.cat(outputs, dim=1)


class SERes2NetBlock(nn.Module):
    """SE-Res2Net block: Conv1d -> Res2Net -> Conv1d -> SE -> residual connection.

    This is the core building block of ECAPA-TDNN.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 2, scale: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            Res2Conv1dBlock(channels, kernel_size, dilation, scale),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE-Res2Net with skip connection."""
        residual = x
        x = self.block(x)
        x = self.se(x)
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    """Channel-dependent attentive statistics pooling.

    Computes attention-weighted mean and standard deviation across time,
    producing a fixed-length utterance-level representation.
    """

    def __init__(self, channels: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_dim, 1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels * 2) — concatenation of weighted mean and std
        """
        # Compute attention weights
        alpha = self.attention(x)  # (batch, channels, time)
        alpha = F.softmax(alpha, dim=2)

        # Weighted mean
        mean = (alpha * x).sum(dim=2)  # (batch, channels)

        # Weighted std
        var = (alpha * (x ** 2)).sum(dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-8))  # (batch, channels)

        return torch.cat([mean, std], dim=1)  # (batch, channels * 2)


class TemporalAveragePooling(nn.Module):
    """Simple mean pooling over the time dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels)
        """
        return x.mean(dim=2)

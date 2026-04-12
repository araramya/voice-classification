"""Loss functions for speaker identification."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax (ArcFace) loss.

    Pushes speaker embeddings apart on a hypersphere by adding an angular margin
    penalty to the target class. State-of-the-art for speaker verification.

    Args:
        embedding_dim: dimension of input embeddings
        num_classes: number of speakers
        margin: angular margin in radians (default 0.2)
        scale: scaling factor (default 30.0)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Ensure cos(theta + m) is monotonically decreasing for theta in [0, pi]
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute AAM-Softmax loss.

        Args:
            embeddings: (batch, embedding_dim) — NOT logits, raw embeddings
            labels: (batch,) integer labels

        Returns:
            scalar loss
        """
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)  # (batch, num_classes)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # Compute cos(theta + margin) for target classes
        sine = torch.sqrt(1.0 - cosine.pow(2))
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # When cos(theta) < cos(pi - m), use a safe approximation
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only to the target class
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale

        loss = F.cross_entropy(logits, labels)
        return loss

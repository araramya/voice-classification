"""Extract speaker embeddings from trained deep learning models."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """Extract embeddings and labels from a trained model.

    Args:
        model: trained model with `extract_embedding` method
        dataloader: test/val DataLoader
        device: torch device

    Returns:
        (embeddings, labels) — numpy arrays of shape (N, dim) and (N,)
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for features, labels in tqdm(dataloader, desc="Extracting embeddings"):
        features = features.to(device)
        embeddings = model.extract_embedding(features)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels

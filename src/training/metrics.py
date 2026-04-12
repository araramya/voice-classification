"""Evaluation metrics: accuracy, EER, confusion matrix."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import confusion_matrix as sklearn_cm


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy."""
    return (predictions == labels).mean()


def compute_topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Compute top-k accuracy from logit scores.

    Args:
        logits: (N, num_classes)
        labels: (N,) integer labels
    """
    topk_preds = np.argsort(logits, axis=1)[:, -k:]
    correct = np.any(topk_preds == labels[:, None], axis=1)
    return correct.mean()


def compute_eer(scores_positive: np.ndarray, scores_negative: np.ndarray) -> tuple:
    """Compute Equal Error Rate.

    Args:
        scores_positive: similarity scores for genuine (same-speaker) pairs
        scores_negative: similarity scores for impostor (different-speaker) pairs

    Returns:
        (eer, threshold) tuple
    """
    all_scores = np.concatenate([scores_positive, scores_negative])
    labels = np.concatenate([
        np.ones(len(scores_positive)),
        np.zeros(len(scores_negative)),
    ])

    # Sort by threshold
    thresholds = np.sort(all_scores)

    far_list = []
    frr_list = []
    for threshold in thresholds:
        # False Acceptance Rate: impostor accepted
        far = np.mean(scores_negative >= threshold)
        # False Rejection Rate: genuine rejected
        frr = np.mean(scores_positive < threshold)
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # Find EER: where FAR == FRR
    try:
        eer_fn = interp1d(far_arr - frr_arr, thresholds)
        eer_threshold = float(eer_fn(0.0))
        eer = float(interp1d(thresholds, far_arr)(eer_threshold))
    except ValueError:
        # Fallback: find closest point
        diff = np.abs(far_arr - frr_arr)
        idx = np.argmin(diff)
        eer = (far_arr[idx] + frr_arr[idx]) / 2
        eer_threshold = thresholds[idx]

    return eer, eer_threshold


def compute_eer_from_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",
) -> tuple:
    """Compute EER from embeddings using pairwise cosine similarity.

    Args:
        embeddings: (N, dim) embedding vectors
        labels: (N,) integer labels

    Returns:
        (eer, threshold)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = embeddings / norms

    # Compute pairwise cosine similarity (upper triangle only)
    scores_pos = []
    scores_neg = []

    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(np.dot(embeddings[i], embeddings[j]))
            if labels[i] == labels[j]:
                scores_pos.append(score)
            else:
                scores_neg.append(score)

    return compute_eer(np.array(scores_pos), np.array(scores_neg))


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    speaker_names: list = None,
) -> np.ndarray:
    """Compute confusion matrix."""
    if speaker_names is not None:
        return sklearn_cm(labels, predictions, labels=list(range(len(speaker_names))))
    return sklearn_cm(labels, predictions)

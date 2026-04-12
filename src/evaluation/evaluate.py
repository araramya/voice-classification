"""Full evaluation pipeline for speaker identification models."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.embeddings import extract_embeddings
from src.training.metrics import (
    compute_accuracy,
    compute_topk_accuracy,
    compute_confusion_matrix,
    compute_eer_from_embeddings,
)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    speaker_names: list = None,
) -> dict:
    """Run full evaluation on a deep learning model.

    Returns dict with accuracy, top5_accuracy, EER, confusion_matrix, etc.
    """
    model.eval()

    all_logits = []
    all_labels = []

    for features, labels in tqdm(test_loader, desc="Evaluating"):
        features = features.to(device)
        logits = model(features)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    predictions = logits.argmax(axis=1)

    # Core metrics
    accuracy = compute_accuracy(predictions, labels)
    top5_acc = compute_topk_accuracy(logits, labels, k=5)
    cm = compute_confusion_matrix(predictions, labels, speaker_names)

    # Extract embeddings for EER
    embeddings, emb_labels = extract_embeddings(model, test_loader, device)
    eer, eer_threshold = compute_eer_from_embeddings(embeddings, emb_labels)

    results = {
        "accuracy": float(accuracy),
        "top5_accuracy": float(top5_acc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "num_test_samples": len(labels),
        "confusion_matrix": cm.tolist(),
    }

    return results


def save_results(results: dict, experiment_name: str, results_dir: str = "results"):
    """Save evaluation results to JSON."""
    out_dir = Path(results_dir) / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir / 'metrics.json'}")


def compare_models(results_list: list) -> dict:
    """Compare multiple models side by side.

    Args:
        results_list: list of dicts, each with "model_name" + evaluation metrics

    Returns:
        Comparison dict with per-model metrics
    """
    comparison = {}
    for r in results_list:
        name = r.get("model_name", "unknown")
        comparison[name] = {
            "accuracy": r.get("accuracy", 0),
            "top5_accuracy": r.get("top5_accuracy", 0),
            "eer": r.get("eer", 0),
        }
    return comparison

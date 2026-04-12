"""Evaluate and compare all trained models on the test set."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.download import prepare_dataset
from src.data.splits import load_splits, get_split_metadata
from src.data.dataset import SpeakerDataset, SpeakerDatasetForBaseline
from src.models.cnn import SpeakerCNN
from src.models.ecapa_tdnn import ECAPATDNN
from src.models.baseline_gmm import GMMBaseline
from src.models.baseline_svm import SVMBaseline
from src.evaluation.evaluate import evaluate_model, save_results, compare_models
from src.evaluation.visualization import plot_model_comparison, plot_confusion_matrix, plot_tsne
from src.evaluation.embeddings import extract_embeddings
from src.training.metrics import compute_accuracy
from src.utils import set_seed, get_device

import numpy as np
from collections import defaultdict
from tqdm import tqdm


def evaluate_baseline(model, test_dataset, model_name):
    """Evaluate a baseline (GMM/SVM) model."""
    true_labels = []
    pred_labels = []

    for i in tqdm(range(len(test_dataset)), desc=f"Evaluating {model_name}"):
        mfcc, speaker_id = test_dataset[i]
        if isinstance(model, GMMBaseline):
            pred = model.predict(mfcc.T)
        else:
            pred = model.predict(mfcc)
        pred_labels.append(pred)
        true_labels.append(speaker_id)

    accuracy = compute_accuracy(np.array(pred_labels), np.array(true_labels))
    return {"model_name": model_name, "accuracy": float(accuracy), "eer": 0.0}


def evaluate_deep_model(model_class, checkpoint_path, config, test_loader, device, model_name, **model_kwargs):
    """Evaluate a deep learning model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]

    model = model_class(
        num_speakers=model_config["num_speakers"],
        embedding_dim=model_config["embedding_dim"],
        **model_kwargs,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    results = evaluate_model(model, test_loader, device)
    results["model_name"] = model_name
    return results, model


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    set_seed(42)
    device = get_device("auto")
    checkpoint_dir = Path("checkpoints")

    # Load data
    config = load_config("configs/base.yaml")
    metadata = prepare_dataset(args.data_dir, config.data.num_speakers, config.data.min_utterances_per_speaker)
    splits = load_splits(args.data_dir)
    test_meta = get_split_metadata(metadata, splits, "test")

    all_results = []

    # Load label encoder
    le_path = Path(args.data_dir) / "splits" / "label_encoder.pkl"
    label_encoder = None
    if le_path.exists():
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)

    # Evaluate baselines
    for baseline_name in ["baseline_gmm", "baseline_svm"]:
        ckpt = checkpoint_dir / f"{baseline_name}.pkl"
        if ckpt.exists():
            from src.config import load_config as lc
            bc = lc(f"configs/{baseline_name}.yaml")
            test_dataset = SpeakerDatasetForBaseline(test_meta, bc.features, bc.audio)
            if "gmm" in baseline_name:
                model = GMMBaseline()
            else:
                model = SVMBaseline()
            model.load(str(ckpt))
            results = evaluate_baseline(model, test_dataset, baseline_name)
            all_results.append(results)
            print(f"\n{baseline_name}: Accuracy = {results['accuracy']:.4f}")

    # Evaluate deep models
    from src.config import FeatureConfig, AudioConfig
    test_dataset = SpeakerDataset(
        test_meta,
        FeatureConfig(feature_type="mel_spectrogram"),
        AudioConfig(),
        label_encoder=label_encoder,
        train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # CNN
    cnn_ckpt = checkpoint_dir / "cnn_mel_spectrogram_best.pt"
    if cnn_ckpt.exists():
        results, cnn_model = evaluate_deep_model(
            SpeakerCNN, cnn_ckpt, config, test_loader, device, "CNN",
        )
        all_results.append(results)
        print(f"\nCNN: Accuracy = {results['accuracy']:.4f}, EER = {results['eer']:.4f}")
        save_results(results, "cnn_mel_spectrogram")

    # ECAPA-TDNN
    ecapa_ckpt = checkpoint_dir / "ecapa_tdnn_best.pt"
    if ecapa_ckpt.exists():
        results, ecapa_model = evaluate_deep_model(
            ECAPATDNN, ecapa_ckpt, config, test_loader, device, "ECAPA-TDNN",
            channels=512, scale=8,
        )
        all_results.append(results)
        print(f"\nECAPA-TDNN: Accuracy = {results['accuracy']:.4f}, EER = {results['eer']:.4f}")
        save_results(results, "ecapa_tdnn")

    # Comparison
    if len(all_results) > 1:
        comparison = compare_models(all_results)
        results_dir = Path("results") / "comparison"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print("\n=== Model Comparison ===")
        for name, metrics in comparison.items():
            print(f"  {name}: Accuracy={metrics['accuracy']:.4f}, EER={metrics['eer']:.4f}")

        plot_model_comparison(comparison, save_path=str(results_dir / "comparison.png"))


if __name__ == "__main__":
    main()

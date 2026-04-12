"""Train classical ML baselines (GMM or SVM) for speaker identification."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.download import prepare_dataset
from src.data.splits import create_splits, get_split_metadata
from src.data.dataset import SpeakerDatasetForBaseline
from src.models.baseline_gmm import GMMBaseline
from src.models.baseline_svm import SVMBaseline
from src.training.metrics import compute_accuracy, compute_confusion_matrix
from src.utils import set_seed, setup_logging


def collect_features(dataset, desc="Extracting features") -> dict:
    """Collect features grouped by speaker."""
    features_dict = defaultdict(list)
    for i in tqdm(range(len(dataset)), desc=desc):
        mfcc, speaker_id = dataset[i]
        features_dict[speaker_id].append(mfcc)
    return dict(features_dict)


def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    logger = setup_logging("results", config.experiment_name)

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Model type: {config.model.type}")

    # Prepare dataset
    metadata = prepare_dataset(
        config.data.data_dir,
        config.data.num_speakers,
        config.data.min_utterances_per_speaker,
    )

    # Create splits
    print("Creating train/val/test splits...")
    splits = create_splits(metadata, config.data)

    train_meta = get_split_metadata(metadata, splits, "train")
    test_meta = get_split_metadata(metadata, splits, "test")

    # Create datasets
    train_dataset = SpeakerDatasetForBaseline(train_meta, config.features, config.audio)
    test_dataset = SpeakerDatasetForBaseline(test_meta, config.features, config.audio)

    # Collect features
    print("Extracting training features...")
    train_features = collect_features(train_dataset, "Train features")

    print("Extracting test features...")
    test_features = collect_features(test_dataset, "Test features")

    # Train model
    if config.model.type == "gmm":
        model = GMMBaseline(n_components=64, use_ubm=True, ubm_components=256)
        # GMM expects stacked frames per speaker
        gmm_features = {}
        for spk, mfcc_list in train_features.items():
            # Each mfcc is (n_mfcc, time), stack and transpose to (total_frames, n_mfcc)
            gmm_features[spk] = np.concatenate([m.T for m in mfcc_list], axis=0)
        model.fit(gmm_features)
    elif config.model.type == "svm":
        model = SVMBaseline(kernel="rbf", C=10.0)
        model.fit(train_features)
    else:
        raise ValueError(f"Unknown baseline model: {config.model.type}")

    # Evaluate
    print("\nEvaluating on test set...")
    true_labels = []
    pred_labels = []

    for spk, mfcc_list in tqdm(test_features.items(), desc="Evaluating"):
        for mfcc in mfcc_list:
            if config.model.type == "gmm":
                pred = model.predict(mfcc.T)  # GMM expects (frames, features)
            else:
                pred = model.predict(mfcc)
            pred_labels.append(pred)
            true_labels.append(spk)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    accuracy = compute_accuracy(pred_labels, true_labels)
    logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")

    # Save results
    results_dir = Path("results") / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": config.model.type,
        "accuracy": float(accuracy),
        "num_speakers": len(model.speakers),
        "num_test_utterances": len(true_labels),
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_dir / f"{config.experiment_name}.pkl"))

    logger.info(f"Model saved to checkpoints/{config.experiment_name}.pkl")
    logger.info(f"Results saved to {results_dir}/metrics.json")


if __name__ == "__main__":
    main()

"""Train CNN model for speaker identification."""

import argparse
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.download import prepare_dataset
from src.data.splits import create_splits, get_split_metadata
from src.data.dataset import SpeakerDataset
from src.models.cnn import SpeakerCNN
from src.training.trainer import Trainer
from src.utils import set_seed, get_device, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train CNN speaker identification model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logging("results", config.experiment_name)

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Device: {device}")

    # Prepare dataset
    metadata = prepare_dataset(
        config.data.data_dir,
        config.data.num_speakers,
        config.data.min_utterances_per_speaker,
    )
    splits = create_splits(metadata, config.data)

    train_meta = get_split_metadata(metadata, splits, "train")
    val_meta = get_split_metadata(metadata, splits, "val")

    # Create datasets
    train_dataset = SpeakerDataset(
        train_meta, config.features, config.audio, config.augmentation, train=True,
    )
    val_dataset = SpeakerDataset(
        val_meta, config.features, config.audio, label_encoder=train_dataset.label_encoder, train=False,
    )

    # Save label encoder for inference
    le_path = Path(config.data.data_dir) / "splits" / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(train_dataset.label_encoder, f)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    # Create model
    model = SpeakerCNN(
        num_speakers=train_dataset.num_speakers,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_name=config.experiment_name,
    )

    if args.resume:
        epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from epoch {epoch}")

    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

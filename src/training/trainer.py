"""Generic training loop with mixed precision, logging, and checkpointing."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.training.losses import AAMSoftmax
from src.training.schedulers import get_scheduler
from src.utils import count_parameters


class Trainer:
    """Training loop for deep learning speaker identification models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: torch.device,
        experiment_name: str = "experiment",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_name = experiment_name

        # Loss function
        if config.training.loss == "aam_softmax":
            self.criterion = AAMSoftmax(
                embedding_dim=config.model.embedding_dim,
                num_classes=model.num_speakers,
                margin=config.training.aam_margin,
                scale=config.training.aam_scale,
            ).to(device)
            self.use_embeddings_for_loss = True
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.use_embeddings_for_loss = False

        # Optimizer
        params = list(model.parameters())
        if self.use_embeddings_for_loss:
            params += list(self.criterion.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler
        self.scheduler = get_scheduler(self.optimizer, config.training)

        # Mixed precision
        self.use_amp = config.training.use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Logging
        self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

        # Checkpointing
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        print(f"Model parameters: {count_parameters(model):,}")
        print(f"Device: {device}")
        print(f"Mixed precision: {self.use_amp}")

    def train_epoch(self) -> dict:
        """Run one training epoch."""
        self.model.train()
        if self.use_embeddings_for_loss:
            self.criterion.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    if self.use_embeddings_for_loss:
                        embeddings = self.model.extract_embedding(features)
                        loss = self.criterion(embeddings, labels)
                        logits = self.model(features)
                    else:
                        logits = self.model(features)
                        loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip_max_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.use_embeddings_for_loss:
                    embeddings = self.model.extract_embedding(features)
                    loss = self.criterion(embeddings, labels)
                    logits = self.model(features)
                else:
                    logits = self.model(features)
                    loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip_max_norm
                )
                self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        return {"loss": total_loss / total, "accuracy": correct / total}

    @torch.no_grad()
    def validate(self) -> dict:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in tqdm(self.val_loader, desc="Val", leave=False):
            features = features.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(features)
            loss = nn.functional.cross_entropy(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {"loss": total_loss / total, "accuracy": correct / total}

    def train(self) -> dict:
        """Full training loop with early stopping."""
        print(f"\nStarting training for {self.config.training.epochs} epochs...")
        start_time = time.time()
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.config.training.epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            print(
                f"Epoch {epoch}/{self.config.training.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # Checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.save_checkpoint(
                    self.checkpoint_dir / f"{self.experiment_name}_best.pt",
                    epoch, val_metrics,
                )
                print(f"  -> Best model saved (val_loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={self.config.training.patience})")
                    break

            # Save latest
            self.save_checkpoint(
                self.checkpoint_dir / f"{self.experiment_name}_latest.pt",
                epoch, val_metrics,
            )

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed / 60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")

        self.writer.close()

        # Save history
        results_dir = Path("results") / self.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "metrics": metrics,
            "config": {
                "model_type": self.config.model.type,
                "embedding_dim": self.config.model.embedding_dim,
                "num_speakers": self.model.num_speakers,
            },
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        return checkpoint["epoch"]

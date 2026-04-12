"""Thesis-quality visualization functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


# Use a clean style for thesis figures
plt.style.use("seaborn-v0_8-paper")
FIGSIZE = (8, 6)
DPI = 300


def plot_training_curves(history: dict, save_path: str = None):
    """Plot training and validation loss/accuracy curves.

    Args:
        history: dict with keys train_loss, val_loss, train_acc, val_acc
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Validation", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training and Validation Accuracy", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    speaker_names: list = None,
    save_path: str = None,
    normalize: bool = True,
):
    """Plot confusion matrix as a heatmap."""
    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(8, len(cm) * 0.5), max(6, len(cm) * 0.4)))

    labels = speaker_names if speaker_names else [str(i) for i in range(len(cm))]

    sns.heatmap(
        cm_display,
        annot=len(cm) <= 20,  # Only show numbers for small matrices
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Speaker", fontsize=12)
    ax.set_ylabel("True Speaker", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.show()


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    speaker_names: list = None,
    save_path: str = None,
    perplexity: float = 30.0,
    max_speakers: int = 20,
):
    """t-SNE visualization of speaker embeddings."""
    unique_labels = np.unique(labels)
    if len(unique_labels) > max_speakers:
        # Subsample for clarity
        selected = unique_labels[:max_speakers]
        mask = np.isin(labels, selected)
        embeddings = embeddings[mask]
        labels = labels[mask]
        unique_labels = selected

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = speaker_names[label] if speaker_names else f"Speaker {label}"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]], label=name, alpha=0.7, s=20,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE Visualization of Speaker Embeddings", fontsize=14)
    if len(unique_labels) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.show()


def plot_eer_curve(
    far: np.ndarray,
    frr: np.ndarray,
    eer: float,
    save_path: str = None,
):
    """Plot Detection Error Tradeoff (DET) curve with EER point."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(far * 100, frr * 100, linewidth=2, label="DET Curve")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="EER Line")
    ax.scatter([eer * 100], [eer * 100], color="red", s=100, zorder=5,
              label=f"EER = {eer * 100:.2f}%")

    ax.set_xlabel("False Acceptance Rate (%)", fontsize=12)
    ax.set_ylabel("False Rejection Rate (%)", fontsize=12)
    ax.set_title("Detection Error Tradeoff (DET) Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.show()


def plot_model_comparison(results: dict, save_path: str = None):
    """Bar chart comparing multiple models on key metrics."""
    models = list(results.keys())
    metrics = ["accuracy", "eer"]
    metric_labels = ["Accuracy (%)", "EER (%)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [results[m].get(metric, 0) * 100 for m in models]
        bars = ax.bar(models, values, color=plt.cm.Set2(np.linspace(0, 1, len(models))))
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=14)
        ax.tick_params(axis="x", rotation=30)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.show()


def plot_spectrogram_examples(
    waveform: np.ndarray,
    mel_spectrogram: np.ndarray,
    mfcc: np.ndarray,
    sample_rate: int = 16000,
    save_path: str = None,
):
    """Show waveform, mel-spectrogram, and MFCCs side by side."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Waveform
    time = np.arange(len(waveform)) / sample_rate
    ax1.plot(time, waveform, linewidth=0.5)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Amplitude", fontsize=12)
    ax1.set_title("Waveform", fontsize=14)

    # Mel-spectrogram
    img1 = ax2.imshow(mel_spectrogram, aspect="auto", origin="lower", cmap="viridis")
    ax2.set_xlabel("Time Frame", fontsize=12)
    ax2.set_ylabel("Mel Band", fontsize=12)
    ax2.set_title("Log-Mel Spectrogram", fontsize=14)
    plt.colorbar(img1, ax=ax2)

    # MFCCs
    img2 = ax3.imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
    ax3.set_xlabel("Time Frame", fontsize=12)
    ax3.set_ylabel("MFCC Coefficient", fontsize=12)
    ax3.set_title("MFCCs", fontsize=14)
    plt.colorbar(img2, ax=ax3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.show()

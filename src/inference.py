"""Inference: identify speaker from a single audio file."""

import pickle
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.config import load_config, FeatureConfig, AudioConfig
from src.data.features import FeatureExtractor
from src.models.cnn import SpeakerCNN
from src.models.ecapa_tdnn import ECAPATDNN
from src.utils import get_device


def identify_speaker(
    audio_path: str,
    model_path: str,
    config_path: str,
    label_encoder_path: str,
    device: str = "auto",
) -> dict:
    """Load a trained model and classify a single audio file.

    Args:
        audio_path: path to .wav file
        model_path: path to model checkpoint (.pt)
        config_path: path to YAML config
        label_encoder_path: path to saved label_encoder.pkl

    Returns:
        dict with predicted_speaker, confidence, top5_predictions, embedding
    """
    config = load_config(config_path)
    device = get_device(device)

    # Load label encoder
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]

    if config.model.type == "ecapa_tdnn":
        model = ECAPATDNN(
            num_speakers=model_config["num_speakers"],
            embedding_dim=model_config["embedding_dim"],
            channels=config.model.channels,
            scale=config.model.scale,
        )
    else:
        model = SpeakerCNN(
            num_speakers=model_config["num_speakers"],
            embedding_dim=model_config["embedding_dim"],
            dropout=config.model.dropout,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load and process audio
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != config.audio.sample_rate:
        waveform = torchaudio.transforms.Resample(sr, config.audio.sample_rate)(waveform)

    # Crop/pad to fixed length
    target_samples = int(config.audio.max_duration_sec * config.audio.sample_rate)
    if waveform.shape[1] > target_samples:
        start = (waveform.shape[1] - target_samples) // 2
        waveform = waveform[:, start:start + target_samples]
    elif waveform.shape[1] < target_samples:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[1]))

    # Extract features
    feature_extractor = FeatureExtractor(config.features, config.audio.sample_rate)
    features = feature_extractor(waveform)
    features = features.unsqueeze(0).to(device)  # Add batch dimension

    # Inference
    with torch.no_grad():
        logits = model(features)
        embedding = model.extract_embedding(features)
        probs = torch.softmax(logits, dim=1)

    # Top-5 predictions
    top5_probs, top5_indices = probs[0].topk(min(5, probs.shape[1]))
    top5_speakers = label_encoder.inverse_transform(top5_indices.cpu().numpy())
    top5_predictions = list(zip(top5_speakers.tolist(), top5_probs.cpu().numpy().tolist()))

    predicted_idx = logits.argmax(dim=1).item()
    predicted_speaker = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = probs[0, predicted_idx].item()

    return {
        "predicted_speaker": predicted_speaker,
        "confidence": confidence,
        "top5_predictions": top5_predictions,
        "embedding": embedding[0].cpu().numpy(),
    }

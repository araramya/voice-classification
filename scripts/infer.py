"""CLI inference: identify speaker from a single audio file."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import identify_speaker


def main():
    parser = argparse.ArgumentParser(description="Identify speaker from audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (.wav)")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--label-encoder", type=str, default="data/splits/label_encoder.pkl",
                        help="Path to label encoder")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    args = parser.parse_args()

    result = identify_speaker(
        audio_path=args.audio,
        model_path=args.model,
        config_path=args.config,
        label_encoder_path=args.label_encoder,
        device=args.device,
    )

    print(f"\nPredicted Speaker: {result['predicted_speaker']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"\nTop-5 Predictions:")
    for speaker, prob in result["top5_predictions"]:
        print(f"  {speaker}: {prob * 100:.1f}%")
    print(f"\nEmbedding shape: {result['embedding'].shape}")


if __name__ == "__main__":
    main()

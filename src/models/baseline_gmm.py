"""GMM-UBM baseline for speaker identification using sklearn."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


class GMMBaseline:
    """Speaker identification using Gaussian Mixture Models.

    Two modes:
    1. Per-speaker GMM: Train independent GMM for each speaker.
    2. GMM-UBM: Train a Universal Background Model, then adapt per speaker via MAP.
    """

    def __init__(self, n_components: int = 64, use_ubm: bool = True, ubm_components: int = 256):
        self.n_components = n_components
        self.use_ubm = use_ubm
        self.ubm_components = ubm_components
        self.speaker_models = {}
        self.ubm = None
        self.speakers = []

    def fit(self, features_dict: dict):
        """Train GMM models.

        Args:
            features_dict: {speaker_id: np.ndarray of shape (n_frames, n_features)}
        """
        self.speakers = sorted(features_dict.keys())

        if self.use_ubm:
            self._fit_ubm(features_dict)
        else:
            self._fit_per_speaker(features_dict)

    def _fit_per_speaker(self, features_dict: dict):
        """Train one GMM per speaker."""
        for spk in tqdm(self.speakers, desc="Training per-speaker GMMs"):
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                max_iter=200,
                random_state=42,
            )
            gmm.fit(features_dict[spk])
            self.speaker_models[spk] = gmm

    def _fit_ubm(self, features_dict: dict):
        """Train UBM on all data, then MAP-adapt for each speaker."""
        # Train UBM on pooled data
        print("Training Universal Background Model...")
        all_features = np.concatenate(list(features_dict.values()), axis=0)

        # Subsample if too large for memory
        max_frames = 500_000
        if len(all_features) > max_frames:
            indices = np.random.RandomState(42).choice(len(all_features), max_frames, replace=False)
            all_features = all_features[indices]

        self.ubm = GaussianMixture(
            n_components=self.ubm_components,
            covariance_type="diag",
            max_iter=200,
            random_state=42,
        )
        self.ubm.fit(all_features)

        # MAP adaptation for each speaker
        for spk in tqdm(self.speakers, desc="MAP-adapting speaker models"):
            adapted_gmm = GaussianMixture(
                n_components=self.ubm_components,
                covariance_type="diag",
                max_iter=5,
                random_state=42,
                means_init=self.ubm.means_,
                weights_init=self.ubm.weights_,
                precisions_init=self.ubm.precisions_,
            )
            adapted_gmm.fit(features_dict[spk])
            self.speaker_models[spk] = adapted_gmm

    def predict(self, features: np.ndarray) -> str:
        """Predict speaker for a single utterance.

        Args:
            features: (n_frames, n_features) MFCC array

        Returns:
            Predicted speaker ID
        """
        scores = self.score_all(features)
        return self.speakers[np.argmax(scores)]

    def score_all(self, features: np.ndarray) -> np.ndarray:
        """Compute log-likelihood for all speakers.

        Returns:
            (num_speakers,) array of average log-likelihoods
        """
        scores = np.zeros(len(self.speakers))
        for i, spk in enumerate(self.speakers):
            scores[i] = self.speaker_models[spk].score(features)
        return scores

    def predict_batch(self, features_list: list) -> list:
        """Predict speakers for multiple utterances."""
        return [self.predict(f) for f in features_list]

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "speaker_models": self.speaker_models,
                "ubm": self.ubm,
                "speakers": self.speakers,
                "n_components": self.n_components,
                "use_ubm": self.use_ubm,
            }, f)

    def load(self, path: str):
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.speaker_models = data["speaker_models"]
        self.ubm = data["ubm"]
        self.speakers = data["speakers"]
        self.n_components = data["n_components"]
        self.use_ubm = data["use_ubm"]

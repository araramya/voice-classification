"""SVM baseline for speaker identification using sklearn."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMBaseline:
    """Speaker identification using SVM on MFCC statistics.

    Feature representation: for each utterance, compute mean and std of MFCCs
    across time, producing a fixed-length supervector.
    """

    def __init__(self, kernel: str = "rbf", C: float = 10.0):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=kernel, C=C, probability=True, random_state=42)),
        ])
        self.speakers = []

    @staticmethod
    def extract_supervector(mfcc: np.ndarray) -> np.ndarray:
        """Convert variable-length MFCCs to fixed-length supervector.

        Args:
            mfcc: (n_mfcc, time_frames)

        Returns:
            (n_mfcc * 2,) vector of [mean, std] statistics
        """
        mean = mfcc.mean(axis=1)
        std = mfcc.std(axis=1)
        return np.concatenate([mean, std])

    def fit(self, features_dict: dict):
        """Train SVM on supervectors.

        Args:
            features_dict: {speaker_id: list of (n_mfcc, time_frames) arrays}
        """
        self.speakers = sorted(features_dict.keys())

        X = []
        y = []
        for spk, mfcc_list in features_dict.items():
            for mfcc in mfcc_list:
                X.append(self.extract_supervector(mfcc))
                y.append(spk)

        X = np.array(X)
        y = np.array(y)

        print(f"Training SVM on {len(X)} samples, {len(self.speakers)} classes...")
        self.pipeline.fit(X, y)

    def predict(self, mfcc: np.ndarray) -> str:
        """Predict speaker for a single utterance."""
        sv = self.extract_supervector(mfcc).reshape(1, -1)
        return self.pipeline.predict(sv)[0]

    def predict_proba(self, mfcc: np.ndarray) -> np.ndarray:
        """Get probability scores for all speakers."""
        sv = self.extract_supervector(mfcc).reshape(1, -1)
        return self.pipeline.predict_proba(sv)[0]

    def predict_batch(self, mfcc_list: list) -> list:
        """Predict speakers for multiple utterances."""
        X = np.array([self.extract_supervector(m) for m in mfcc_list])
        return self.pipeline.predict(X).tolist()

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "speakers": self.speakers}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pipeline = data["pipeline"]
        self.speakers = data["speakers"]

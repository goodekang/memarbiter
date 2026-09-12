from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.memarbiter.embeddings import EmbeddingClient


class InstructionClassifier:
    """Logistic regression over sentence embeddings (Appendix A.1)."""

    def __init__(
        self,
        checkpoint: str | Path,
        threshold_file: str | Path,
        encoder: EmbeddingClient,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.threshold_file = Path(threshold_file)
        self.encoder = encoder
        self.model = None
        self.threshold = 0.5
        if self.checkpoint.exists() and self.threshold_file.exists():
            self.model = joblib.load(self.checkpoint)
            self.threshold = float(json.loads(self.threshold_file.read_text(encoding="utf-8"))["threshold"])

    def available(self) -> bool:
        return self.model is not None

    def instruction_score(self, text: str) -> float:
        if self.model is None:
            raise RuntimeError(
                f"instruction classifier not found at {self.checkpoint}; "
                "run scripts/train_instruction_classifier.py"
            )
        vector = self.encoder.encode_one(text).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(vector)[0, 1])
        return float(self.model.decision_function(vector)[0])

    def is_instruction_like(self, text: str) -> bool:
        return self.instruction_score(text) >= self.threshold


def choose_recall_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    target_recall: float = 0.98,
) -> float:
    """Smallest threshold that keeps injection-class recall at ``target_recall``."""
    positives = labels == 1
    if not np.any(positives):
        return 0.5
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = 0
    n_pos = int(positives.sum())
    threshold = float(sorted_scores[-1])
    for score, label in zip(sorted_scores, sorted_labels):
        if label == 1:
            tp += 1
        recall = tp / n_pos
        if recall >= target_recall:
            threshold = float(score)
            break
    return threshold

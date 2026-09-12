from __future__ import annotations

import math
import os
from typing import Iterable, Sequence

import httpx
import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine01(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * (cosine(a, b) + 1.0)


class EmbeddingClient:
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dim: int = 3072,
        batch_size: int = 64,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.timeout = timeout
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self._local = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.api_key:
            return self._encode_api(texts)
        return self._encode_local(texts)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _encode_api(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                response = client.post(
                    url,
                    headers=headers,
                    json={"model": self.model, "input": batch},
                )
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                for item in data:
                    rows.append(np.asarray(item["embedding"], dtype=np.float32))
        return np.stack(rows, axis=0)

    def _encode_local(self, texts: Sequence[str]) -> np.ndarray:
        if self._local is None:
            from sentence_transformers import SentenceTransformer

            name = os.environ.get("LOCAL_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
            self._local = SentenceTransformer(name)
        vectors = np.asarray(self._local.encode(list(texts), convert_to_numpy=True), dtype=np.float32)
        if vectors.shape[1] == self.dim:
            return vectors
        # Project / pad so downstream code can assume the configured width.
        out = np.zeros((vectors.shape[0], self.dim), dtype=np.float32)
        width = min(self.dim, vectors.shape[1])
        out[:, :width] = vectors[:, :width]
        return out


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms

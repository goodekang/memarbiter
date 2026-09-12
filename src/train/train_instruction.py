from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.memarbiter.embeddings import EmbeddingClient
from src.memarbiter.quarantine import choose_recall_threshold
from src.paths import resolve_path


def load_instruction_rows(path: str | Path) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[int] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(str(row["text"]))
            labels.append(int(row["label"]))
    return texts, np.asarray(labels, dtype=np.int64)


def train_instruction_clf(cfg: dict) -> dict:
    texts, labels = load_instruction_rows(resolve_path(cfg["pairs_path"]))
    if texts == []:
        raise FileNotFoundError(f"need instruction pairs at {cfg['pairs_path']}")
    encoder = EmbeddingClient(model=cfg.get("encoder", "text-embedding-3-large"))
    vectors = encoder.encode(texts)
    x_train, x_val, y_train, y_val = train_test_split(
        vectors,
        labels,
        test_size=float(cfg["val_fraction"]),
        random_state=int(cfg["seed"]),
        stratify=labels,
    )
    model = LogisticRegression(C=float(cfg["C"]), max_iter=int(cfg["max_iter"]))
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_val)[:, 1]
    threshold = choose_recall_threshold(scores, y_val, float(cfg["target_recall"]))
    preds = scores >= threshold
    tp = int(((preds == 1) & (y_val == 1)).sum())
    rec = tp / max(1, int((y_val == 1).sum()))
    prec = tp / max(1, int(preds.sum()))
    out = resolve_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "target_recall": cfg["target_recall"],
                "val_recall": rec,
                "val_precision": prec,
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"threshold={threshold:.4f} val_recall={rec:.3f} val_precision={prec:.3f}")
    return {"threshold": threshold, "val_recall": rec, "val_precision": prec}

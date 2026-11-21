"""Train and evaluate the quarantine-gate classifier (Appendix A.1, Table A.1).

``split: grouped`` (revised protocol): partitions are formed by group
(attack specification / user task / template family), cross-partition
near-duplicates (cosine > ``dedup_cosine``) are removed, the threshold is chosen
for ``target_recall`` on the dev partition, and precision, recall and
false-positive rate are reported on the held-out test partition, overall and
per category. With ``compare_row_split: true`` the same rows are also split at
random by row (same partition sizes) for comparison.

``split: row`` trains on a random stratified row-level train/validation split
(comparison only; Appendix A.1 reports that this gate gives a lower MPSR on
PERSIST, 14.2% vs 14.4%, which does not extend to held-out specifications).
All results in the paper use the grouped gate.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.memarbiter.embeddings import EmbeddingClient
from src.memarbiter.quarantine import choose_recall_threshold
from src.paths import resolve_path
from src.train.gate_data import (
    assign_grouped_partitions,
    remove_cross_partition_duplicates,
    sample_partition,
    summarise,
)


def load_instruction_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["label"] = int(row["label"])
            row.setdefault("group", f"row:{len(rows)}")
            row.setdefault("category", "injection" if row["label"] else "structured_tool_return")
            rows.append(row)
    return rows


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = scores >= threshold
    tp = int((preds & (labels == 1)).sum())
    fp = int((preds & (labels == 0)).sum())
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return {
        "precision": 100.0 * tp / max(1, tp + fp),
        "recall": 100.0 * tp / max(1, n_pos),
        "fpr": 100.0 * fp / max(1, n_neg),
        "n": int(len(labels)),
        "n_injection": n_pos,
        "n_benign": n_neg,
    }


def per_category(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> dict[str, dict[str, float]]:
    """Fraction flagged per test subset (recall for injection rows, FPR for benign rows)."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        prefix = "injection" if row["label"] == 1 else "benign"
        buckets[f"{prefix}/{row['category']}"].append(index)
    out = {}
    for name, idx in sorted(buckets.items()):
        flagged = float((scores[idx] >= threshold).mean()) if idx else 0.0
        out[name] = {"n": len(idx), "flagged": 100.0 * flagged}
    return out


def _fit(cfg: dict, x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=float(cfg["C"]), max_iter=int(cfg["max_iter"]))
    model.fit(x, y)
    return model


def _fit_and_threshold(cfg: dict, x_tr, y_tr, x_dev, y_dev) -> tuple[LogisticRegression, float]:
    model = _fit(cfg, x_tr, y_tr)
    threshold = choose_recall_threshold(model.predict_proba(x_dev)[:, 1], y_dev, float(cfg["target_recall"]))
    return model, threshold


def train_instruction_clf(cfg: dict) -> dict:
    rows = load_instruction_rows(resolve_path(cfg["pairs_path"]))
    if not rows:
        raise FileNotFoundError(f"need instruction pairs at {cfg['pairs_path']}")
    split = str(cfg.get("split", "grouped"))
    encoder = EmbeddingClient(model=cfg.get("encoder", "text-embedding-3-large"))
    if split == "row":
        return _train_row_split(cfg, rows, encoder)
    if split == "grouped":
        return _train_grouped(cfg, rows, encoder)
    raise ValueError(f"unknown split {split!r}")


def _train_row_split(cfg: dict, rows: list[dict[str, Any]], encoder: EmbeddingClient) -> dict:
    """Comparison protocol: random stratified row-level split."""
    rows = [row for row in rows if not row.get("test_only")]
    texts = [str(row["text"]) for row in rows]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    vectors = encoder.encode(texts)
    x_train, x_val, y_train, y_val = train_test_split(
        vectors,
        labels,
        test_size=float(cfg["val_fraction"]),
        random_state=int(cfg["seed"]),
        stratify=labels,
    )
    model, threshold = _fit_and_threshold(cfg, x_train, y_train, x_val, y_val)
    val = binary_metrics(model.predict_proba(x_val)[:, 1], y_val, threshold)
    out = resolve_path(cfg.get("row_output_dir", "checkpoints/instruction_clf_rowsplit"))
    _save(out, model, threshold, cfg, {"split": "row", "n_train": int(len(y_train)), "n_val": int(len(y_val)), "val": val})
    print(f"[row split] threshold={threshold:.4f} val P/R/FPR={val['precision']:.1f}/{val['recall']:.1f}/{val['fpr']:.1f}")
    return {"split": "row", "threshold": threshold, "val": val}


def _train_grouped(cfg: dict, rows: list[dict[str, Any]], encoder: EmbeddingClient) -> dict:
    seed = int(cfg["seed"])
    split_cfg = cfg.get("grouped_split", {})
    rows = assign_grouped_partitions(rows, split_cfg, seed)
    vectors = encoder.encode([str(row["text"]) for row in rows])
    rows, vectors, n_dup = remove_cross_partition_duplicates(rows, vectors, float(split_cfg.get("dedup_cosine", 0.95)))

    # Training/dev pool capped at n_tool benign and n_injection injection rows;
    # test sampled to the configured sizes (injections balanced over styles,
    # benign rows by category quota).
    part = np.asarray([row["partition"] for row in rows])
    trdev = np.where(part != "test")[0]
    keep_trdev = _cap_by_label(rows, trdev, int(cfg["n_tool"]), int(cfg["n_injection"]), seed)
    test_idx = np.where(part == "test")[0]
    test_rows = [rows[i] for i in test_idx]
    test_pick = sample_partition(
        test_rows,
        n_injection=split_cfg.get("test_injections"),
        benign_quota=split_cfg.get("test_benign_quota"),
        n_benign=split_cfg.get("test_benign"),
        seed=seed,
    )
    keep = np.asarray(sorted(set(keep_trdev.tolist()) | set(test_idx[test_pick].tolist())), dtype=np.int64)
    rows = [rows[i] for i in keep]
    vectors = vectors[keep]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    part = np.asarray([row["partition"] for row in rows])
    tr, dev, te = (np.where(part == name)[0] for name in ("train", "dev", "test"))
    _check_disjoint_groups(rows)

    model, threshold = _fit_and_threshold(cfg, vectors[tr], labels[tr], vectors[dev], labels[dev])
    scores = model.predict_proba(vectors[te])[:, 1]
    test_rows = [rows[i] for i in te]
    report: dict[str, Any] = {
        "split": "grouped",
        "threshold": threshold,
        "n_train": int(len(tr)),
        "n_dev": int(len(dev)),
        "n_test": int(len(te)),
        "removed_cross_partition_duplicates": n_dup,
        "test": binary_metrics(scores, labels[te], threshold),
        "test_by_category": per_category(test_rows, scores, threshold),
        "partition_summary": summarise(rows),
    }
    if bool(cfg.get("compare_row_split", True)):
        report["row_split_same_data"] = _row_split_comparison(cfg, rows, vectors, labels, (len(tr), len(dev), len(te)), seed)

    out = resolve_path(cfg["output_dir"])
    _save(out, model, threshold, cfg, {k: v for k, v in report.items() if k != "partition_summary"})
    (out / "heldout_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    split_path = resolve_path(cfg.get("split_path", "data/classifier/instruction_split.jsonl"))
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({k: row[k] for k in ("text", "label", "group", "category", "partition", "source") if k in row}, ensure_ascii=False) + "\n")
    t = report["test"]
    print(f"[grouped split] threshold={threshold:.4f} test P/R/FPR={t['precision']:.1f}/{t['recall']:.1f}/{t['fpr']:.1f}")
    for name, value in report["test_by_category"].items():
        print(f"  {name:45s} n={value['n']:5d} flagged={value['flagged']:.1f}")
    if "row_split_same_data" in report:
        r = report["row_split_same_data"]["test"]
        print(f"[row split, same data] test P/R/FPR={r['precision']:.1f}/{r['recall']:.1f}/{r['fpr']:.1f}")
    return report


def _row_split_comparison(cfg, rows, vectors, labels, sizes, seed) -> dict:
    """Random row-level split of the same rows with the same partition sizes and label mix."""
    n_tr, n_dev, n_te = sizes
    idx = np.arange(len(rows))
    rest, te = train_test_split(idx, test_size=n_te, random_state=seed, stratify=labels)
    tr, dev = train_test_split(rest, test_size=n_dev, random_state=seed, stratify=labels[rest])
    model, threshold = _fit_and_threshold(cfg, vectors[tr], labels[tr], vectors[dev], labels[dev])
    return {
        "threshold": threshold,
        "test": binary_metrics(model.predict_proba(vectors[te])[:, 1], labels[te], threshold),
    }


def _cap_by_label(rows, indices: np.ndarray, n_benign: int, n_injection: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    ben = [int(i) for i in indices if rows[i]["label"] == 0]
    inj = [int(i) for i in indices if rows[i]["label"] == 1]
    rng.shuffle(ben)
    rng.shuffle(inj)
    return np.asarray(sorted(ben[:n_benign] + inj[:n_injection]), dtype=np.int64)


def _check_disjoint_groups(rows: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        other = seen.setdefault(row["group"], row["partition"])
        if other != row["partition"]:
            raise AssertionError(f"group {row['group']} appears in {other} and {row['partition']}")


def _save(out: Path, model, threshold: float, cfg: dict, extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    payload = {"threshold": threshold, "target_recall": cfg["target_recall"], **extra}
    (out / "threshold.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

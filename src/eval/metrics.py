from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def conflict_accuracy(rows: Sequence[dict]) -> dict[str, float]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_type[row["conflict_type"]].append(int(bool(row.get("correct"))))
    out = {f"{name}_acc": 100.0 * mean(vals) for name, vals in by_type.items()}
    all_vals = [int(bool(row.get("correct"))) for row in rows]
    out["conflict_acc"] = 100.0 * mean(all_vals)
    return out


def cr_f1(rows: Sequence[dict]) -> float:
    tp = fp = fn = 0
    for row in rows:
        gold = bool(row.get("gold_conflict", True))
        pred = bool(row.get("conflict_recognised"))
        tp += int(gold and pred)
        fp += int((not gold) and pred)
        fn += int(gold and (not pred))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    if prec + rec == 0:
        return 0.0
    return 100.0 * (2 * prec * rec / (prec + rec))


def stale_premise_resistance(rows: Sequence[dict]) -> float:
    subset = [row for row in rows if row.get("stale_presupposition")]
    if not subset:
        return 0.0
    return 100.0 * mean([int(bool(row.get("premise_rejected"))) for row in subset])


def hit_at_k(rows: Sequence[dict], k: int = 5) -> float:
    hits = []
    for row in rows:
        retrieved = list(row.get("retrieved_ids") or [])[:k]
        support = set(row.get("supporting_ids") or [])
        hits.append(int(bool(support & set(retrieved))))
    return 100.0 * mean(hits)


def abstention_pr(rows: Sequence[dict]) -> tuple[float, float]:
    disputed = [row for row in rows if row.get("should_abstain")]
    if not disputed:
        return 0.0, 0.0
    pred_pos = [row for row in rows if row.get("did_abstain")]
    tp = sum(1 for row in disputed if row.get("did_abstain"))
    prec = tp / max(1, len(pred_pos))
    rec = tp / max(1, len(disputed))
    return 100.0 * prec, 100.0 * rec


def rate(flags: Sequence[bool]) -> float:
    return 100.0 * mean([int(bool(flag)) for flag in flags])


def ece(confidences: Sequence[float], correct: Sequence[int], n_bins: int = 10) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        total += (mask.sum() / n) * abs(acc[mask].mean() - conf[mask].mean())
    return float(total)


def bootstrap_ci(
    values: Sequence[float],
    n: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    draws = rng.choice(arr, size=(n, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def holm(pvalues: Sequence[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_bootstrap_p(
    left: Sequence[float],
    right: Sequence[float],
    n: int = 10000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    delta = a - b
    draws = rng.choice(delta, size=(n, delta.size), replace=True).mean(axis=1)
    return float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))

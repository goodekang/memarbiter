from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from src.persist.schema import Session, Turn


def load_histories(root: str | Path) -> list[dict[str, Any]]:
    """Load LongMemEval_S histories from a local dump or a Hugging Face dataset.

    Expected local layout: ``<root>/histories.jsonl`` with one history per line,
    each containing ``history_id`` and ``sessions`` (list of dated turn lists).
    Sessions used to train the conflict classifier must be tagged
    ``nli_train: true`` so ``build_persist`` can drop them.
    """
    root = Path(root)
    excluded = _nli_train_ids(root)
    local = root / "histories.jsonl"
    if local.exists():
        rows = []
        with local.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        for row in rows:
            if row.get("history_id") in excluded:
                row["nli_train"] = True
        return rows
    rows = list(_from_huggingface())
    for row in rows:
        if row.get("history_id") in excluded:
            row["nli_train"] = True
    return rows


def _nli_train_ids(root: Path) -> set[str]:
    mark = Path(root) / "nli_train_histories.json"
    if not mark.exists():
        return set()
    data = json.loads(mark.read_text(encoding="utf-8"))
    return {str(item) for item in data}


def _from_huggingface() -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FileNotFoundError(
            "LongMemEval dump not found and the datasets package is not installed"
        ) from exc
    dataset = load_dataset("xiaowu0162/longmemeval-cleaned", "longmemeval_s", split="test")
    for index, row in enumerate(dataset):
        yield {
            "history_id": str(row.get("question_id") or f"lme-{index:04d}"),
            "sessions": _coerce_sessions(row),
            "nli_train": False,
        }


def _coerce_sessions(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("haystack_sessions") or row.get("sessions") or []
    sessions = []
    for index, session in enumerate(raw):
        turns = []
        if isinstance(session, list):
            for turn in session:
                if isinstance(turn, dict):
                    turns.append(
                        {
                            "role": turn.get("role") or turn.get("speaker") or "user",
                            "text": turn.get("content") or turn.get("text") or "",
                        }
                    )
        sessions.append({"index": index, "date": str(row.get("date") or ""), "turns": turns, "source": "seed"})
    return sessions


def as_sessions(history: dict[str, Any]) -> list[Session]:
    rows = []
    for item in history.get("sessions", []):
        rows.append(
            Session(
                index=int(item.get("index", len(rows))),
                date=str(item.get("date") or ""),
                source=str(item.get("source") or "seed"),
                turns=[Turn(role=t.get("role", "user"), text=t.get("text", "")) for t in item.get("turns", [])],
            )
        )
    return rows

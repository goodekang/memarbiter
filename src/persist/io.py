from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

from src.persist.schema import ConflictInstance, PoisonTrajectory

T = TypeVar("T")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_conflicts(path: str | Path) -> list[ConflictInstance]:
    return [ConflictInstance.from_json(row) for row in read_jsonl(path)]


def load_poisons(path: str | Path) -> list[PoisonTrajectory]:
    return [PoisonTrajectory.from_json(row) for row in read_jsonl(path)]

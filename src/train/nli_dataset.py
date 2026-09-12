from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.prompts import RELATION_LABELS

LABEL2ID = {name: index for index, name in enumerate(RELATION_LABELS)}


class PairRecord:
    __slots__ = ("text_a", "text_b", "label")

    def __init__(self, text_a: str, text_b: str, label: str) -> None:
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def load_pairs(path: str | Path) -> list[PairRecord]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(PairRecord(item["text_a"], item["text_b"], item["label"]))
    return rows


def split_pairs(rows: list[PairRecord], val_fraction: float, seed: int) -> tuple[list[PairRecord], list[PairRecord]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


class PairDataset(Dataset):
    def __init__(self, rows: list[PairRecord], tokenizer, max_length: int = 512) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row.text_a,
            row.text_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(LABEL2ID[row.label], dtype=torch.long)
        return item


def write_pair_pool(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

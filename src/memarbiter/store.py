from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

import numpy as np

from src.memarbiter.embeddings import cosine01
from src.memarbiter.entity import entity_jaccard, extract_entities, neighbor_score
from src.memarbiter.record import MemoryRecord, Partition


class ProvenanceStore:
    """Content-agnostic store: write / retrieve / update / delete plus quarantine."""

    def __init__(self, embed_weight: float = 0.65) -> None:
        self.embed_weight = embed_weight
        self._records: dict[str, MemoryRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def all_records(self, *, include_quarantine: bool = False) -> list[MemoryRecord]:
        rows = list(self._records.values())
        if include_quarantine:
            return rows
        return [row for row in rows if row.partition is Partition.ACTIVE]

    def write(self, record: MemoryRecord, *, inherit_taint: bool = False) -> MemoryRecord:
        if record.entities is None or len(record.entities) == 0:
            record.entities = extract_entities(record.content)
        if inherit_taint:
            self.inherit_taint(record)
        self._records[record.record_id] = record
        return record

    def inherit_taint(self, record: MemoryRecord) -> MemoryRecord:
        inherited = [self._records[pid].taint for pid in record.lineage if pid in self._records]
        if inherited:
            record.taint = max(record.taint, max(inherited))
        return record

    def delete(self, record_id: str) -> None:
        self._records.pop(record_id, None)

    def update(self, record: MemoryRecord) -> None:
        self._records[record.record_id] = record

    def quarantine(self, record_ids: Iterable[str]) -> list[MemoryRecord]:
        moved = []
        for record_id in record_ids:
            record = self._records.get(record_id)
            if record is None:
                continue
            record.partition = Partition.QUARANTINE
            moved.append(record)
        return moved

    def reopen_intervals_closed_by(self, record_id: str) -> list[str]:
        reopened = []
        for record in self._records.values():
            if record.closed_by == record_id:
                record.valid_until = None
                record.closed_by = None
                reopened.append(record.record_id)
        return reopened

    def descendants(self, root_id: str) -> set[str]:
        found = {root_id}
        changed = True
        while changed:
            changed = False
            for record in self._records.values():
                if record.record_id in found:
                    continue
                if any(parent in found for parent in record.lineage):
                    found.add(record.record_id)
                    changed = True
        return found

    def ancestors(self, record_ids: Sequence[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(record_ids)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            record = self._records.get(current)
            if record is None:
                continue
            stack.extend(record.lineage)
        return seen

    def retrieve_neighbors(
        self,
        query_embedding: np.ndarray,
        query_entities: set[str],
        k: int,
        *,
        include_quarantine: bool = False,
    ) -> list[tuple[MemoryRecord, float]]:
        scored: list[tuple[MemoryRecord, float]] = []
        for record in self.all_records(include_quarantine=include_quarantine):
            if record.embedding is None:
                continue
            sim = cosine01(query_embedding, record.embedding)
            overlap = entity_jaccard(query_entities, record.entities)
            scored.append((record, neighbor_score(sim, overlap, self.embed_weight)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def retrieve_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int,
        *,
        include_quarantine: bool = False,
    ) -> list[tuple[MemoryRecord, float]]:
        scored = []
        for record in self.all_records(include_quarantine=include_quarantine):
            if record.embedding is None:
                continue
            scored.append((record, cosine01(query_embedding, record.embedding)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

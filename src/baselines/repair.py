from __future__ import annotations

from datetime import datetime, timedelta

from src.agent.react import MemoryBackend
from src.baselines.memory_systems import _BagStore, DenseMemory
from src.memarbiter.embeddings import cosine


def wipe_all(memory: MemoryBackend) -> int:
    if isinstance(memory, _BagStore):
        n = len(memory.rows)
        memory.rows.clear()
        if isinstance(memory, DenseMemory):
            memory.vectors.clear()
        return n
    if hasattr(memory, "layer"):
        ids = [row.record_id for row in memory.layer.store.all_records(include_quarantine=True)]
        for rid in ids:
            memory.layer.store.delete(rid)
        return len(ids)
    return 0


def recency_purge(memory: MemoryBackend, n_sessions: int = 3, now: datetime | None = None) -> int:
    now = now or datetime.utcnow()
    cutoff = now - timedelta(days=n_sessions)
    removed = 0
    if isinstance(memory, _BagStore):
        keep = [row for row in memory.rows if row["write_time"] < cutoff]
        removed = len(memory.rows) - len(keep)
        memory.rows = keep
        return removed
    if hasattr(memory, "layer"):
        for record in list(memory.layer.store.all_records(include_quarantine=True)):
            if record.write_time >= cutoff:
                memory.layer.store.delete(record.record_id)
                removed += 1
    return removed


def similarity_purge(memory: MemoryBackend, context: str, threshold: float = 0.85) -> int:
    removed = 0
    if isinstance(memory, DenseMemory):
        probe = memory.encoder.encode_one(context)
        keep_rows, keep_vecs = [], []
        for row, vec in zip(memory.rows, memory.vectors):
            if cosine(probe, vec) >= threshold:
                removed += 1
                continue
            keep_rows.append(row)
            keep_vecs.append(vec)
        memory.rows = keep_rows
        memory.vectors = keep_vecs
        return removed
    if hasattr(memory, "layer"):
        probe = memory.layer.embeddings.encode_one(context)
        for record, score in memory.layer.store.retrieve_by_embedding(probe, 64, include_quarantine=True):
            # retrieve_by_embedding returns cosine01; convert roughly back
            if (2 * score - 1) >= threshold:
                memory.layer.store.delete(record.record_id)
                removed += 1
    return removed

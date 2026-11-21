from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from src.agent.react import MemoryBackend
from src.agent.tokens import count_tokens, trim_memory_block
from src.memarbiter.embeddings import EmbeddingClient, cosine01
from src.memarbiter.entity import extract_entities
from src.memarbiter.record import MemoryRecord, SourceKind
from src.memarbiter.store import ProvenanceStore


class _BagStore(MemoryBackend):
    def __init__(self, budget: int = 4096) -> None:
        self.budget = budget
        self.rows: list[dict] = []

    def write(self, content: str, source: str, write_time: datetime, context: str = "", derived_from: list[str] | None = None) -> str:
        rid = uuid4().hex
        self.rows.append(
            {
                "id": rid,
                "content": content,
                "source": source,
                "write_time": write_time,
                "context": context,
                "lineage": list(derived_from or []),
            }
        )
        return rid

    def _render(self, picked: list[dict]) -> tuple[str, list[str], list[str]]:
        lines = ["Memories relevant to this request.", ""]
        for index, row in enumerate(picked, start=1):
            date = row["write_time"].date().isoformat() if isinstance(row["write_time"], datetime) else str(row["write_time"])
            lines.append(f"{index}. [{row['source']}, {date}] {row['content']}")
        block = trim_memory_block("\n".join(lines), self.budget)
        ids = [row["id"] for row in picked]
        return block, ids, ids


class FullContextMemory(_BagStore):
    def read(self, query: str, query_time: datetime, query_context: dict[str, str] | None = None) -> tuple[str, list[str], list[str]]:
        picked = list(reversed(self.rows))
        fitted = []
        acc = 0
        for row in picked:
            cost = count_tokens(row["content"])
            if acc + cost > self.budget and fitted:
                break
            fitted.append(row)
            acc += cost
        fitted.reverse()
        return self._render(fitted)


class BM25Memory(_BagStore):
    def read(self, query: str, query_time: datetime, query_context: dict[str, str] | None = None) -> tuple[str, list[str], list[str]]:
        if not self.rows:
            return "Memories relevant to this request.\n", [], []
        corpus = [row["content"].lower().split() for row in self.rows]
        scores = _bm25_scores(corpus, query.lower().split())
        order = np.argsort(-np.asarray(scores))[:12]
        picked = [self.rows[i] for i in order]
        return self._render(picked)


class DenseMemory(_BagStore):
    def __init__(self, encoder: EmbeddingClient, budget: int = 4096, k: int = 12) -> None:
        super().__init__(budget)
        self.encoder = encoder
        self.k = k
        self.vectors: list[np.ndarray] = []

    def write(self, content: str, source: str, write_time: datetime, context: str = "", derived_from: list[str] | None = None) -> str:
        rid = super().write(content, source, write_time, context, derived_from)
        self.vectors.append(self.encoder.encode_one(content))
        return rid

    def read(self, query: str, query_time: datetime, query_context: dict[str, str] | None = None) -> tuple[str, list[str], list[str]]:
        if not self.rows:
            return "Memories relevant to this request.\n", [], []
        qv = self.encoder.encode_one(query)
        scores = [cosine01(qv, vec) for vec in self.vectors]
        order = np.argsort(-np.asarray(scores))[: self.k]
        picked = [self.rows[i] for i in order]
        return self._render(picked)


class MemGPTMemory(_BagStore):
    """Working context + archival pages; evict oldest working items first."""

    def __init__(self, budget: int = 4096, working: int = 8) -> None:
        super().__init__(budget)
        self.working_n = working

    def read(self, query: str, query_time: datetime, query_context: dict[str, str] | None = None) -> tuple[str, list[str], list[str]]:
        working = self.rows[-self.working_n :]
        archival = self.rows[: -self.working_n] if len(self.rows) > self.working_n else []
        q_tokens = set(query.lower().split())
        scored = []
        for row in archival:
            overlap = len(q_tokens & set(row["content"].lower().split()))
            scored.append((overlap, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        picked = working + [row for _, row in scored[:4]]
        return self._render(picked)


class MemoryBankMemory(_BagStore):
    """Ebbinghaus-style weights: recency decay on retrieve."""

    def read(self, query: str, query_time: datetime, query_context: dict[str, str] | None = None) -> tuple[str, list[str], list[str]]:
        q_tokens = set(query.lower().split())
        scored = []
        for row in self.rows:
            age_h = max(1.0, (query_time - row["write_time"]).total_seconds() / 3600.0)
            forget = math.exp(-0.05 * math.log(1.0 + age_h))
            overlap = len(q_tokens & set(row["content"].lower().split())) / max(1, len(q_tokens))
            scored.append((forget * (0.4 + 0.6 * overlap), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return self._render([row for _, row in scored[:12]])


class Mem0Memory(_BagStore):
    """Extract-dedup-update: a later value for the same slot overwrites the earlier one."""

    def write(self, content: str, source: str, write_time: datetime, context: str = "", derived_from: list[str] | None = None) -> str:
        key = frozenset(extract_entities(content))
        if key:
            self.rows = [row for row in self.rows if frozenset(extract_entities(row["content"])) != key]
        return super().write(content, source, write_time, context, derived_from)

    def read(self, query: str, query_time: datetime, query_context: dict[str, str] | None = None) -> tuple[str, list[str], list[str]]:
        q = set(extract_entities(query)) or set(query.lower().split())
        scored = []
        for row in self.rows:
            overlap = len(q & (extract_entities(row["content"]) | set(row["content"].lower().split())))
            scored.append((overlap, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return self._render([row for _, row in scored[:12]])


class AMEMMemory(_BagStore):
    """Graph-linked notes: attach an incoming write to the nearest existing note."""

    def __init__(self, encoder: EmbeddingClient, budget: int = 4096) -> None:
        super().__init__(budget)
        self.encoder = encoder
        self.vectors: list[np.ndarray] = []
        self.edges: dict[str, list[str]] = defaultdict(list)

    def write(self, content: str, source: str, write_time: datetime, context: str = "", derived_from: list[str] | None = None) -> str:
        vec = self.encoder.encode_one(content)
        rid = super().write(content, source, write_time, context, derived_from)
        if self.vectors:
            sims = [cosine01(vec, other) for other in self.vectors]
            nearest = int(np.argmax(sims))
            if sims[nearest] > 0.8:
                self.edges[rid].append(self.rows[nearest]["id"])
                self.edges[self.rows[nearest]["id"]].append(rid)
        self.vectors.append(vec)
        return rid

    def read(self, query: str, query_time: datetime, query_context: dict[str, str] | None = None) -> tuple[str, list[str], list[str]]:
        if not self.rows:
            return "Memories relevant to this request.\n", [], []
        qv = self.encoder.encode_one(query)
        scores = [cosine01(qv, vec) for vec in self.vectors]
        seed = int(np.argmax(scores))
        picked_ids = {self.rows[seed]["id"]}
        for neighbour in self.edges[self.rows[seed]["id"]]:
            picked_ids.add(neighbour)
        order = list(np.argsort(-np.asarray(scores)))
        for index in order:
            if len(picked_ids) >= 12:
                break
            picked_ids.add(self.rows[index]["id"])
        picked = [row for row in self.rows if row["id"] in picked_ids]
        return self._render(picked)


class SodaMemMemory(MemoryBackend):
    """Typed events with supersedes / contradicts / updates edges; zero-LLM context builder."""

    def __init__(self, encoder: EmbeddingClient, budget: int = 4096) -> None:
        self.encoder = encoder
        self.budget = budget
        self.store = ProvenanceStore()
        self.edges: list[tuple[str, str, str]] = []

    def write(self, content: str, source: str, write_time: datetime, context: str = "", derived_from: list[str] | None = None) -> str:
        record = MemoryRecord(
            content=content,
            write_time=write_time,
            source_kind=SourceKind.USER if source == "user" else SourceKind.TOOL,
            trust=1.0,
            taint=0.0,
            embedding=self.encoder.encode_one(content),
            derived_from=list(derived_from or []),
            entities=extract_entities(content),
        )
        neighbours = self.store.retrieve_neighbors(record.embedding, record.entities, 4)
        for other, score in neighbours:
            if score < 0.75:
                continue
            if record.write_time > other.write_time and other.entities & record.entities:
                other.valid_until = record.write_time
                other.closed_by = record.record_id
                record.supersedes.append(other.record_id)
                self.edges.append((record.record_id, other.record_id, "supersedes"))
            elif other.entities & record.entities:
                self.edges.append((record.record_id, other.record_id, "contradicts"))
        self.store.write(record)
        return record.record_id

    def read(self, query: str, query_time: datetime, query_context: dict[str, str] | None = None) -> tuple[str, list[str], list[str]]:
        qv = self.encoder.encode_one(query)
        ranked = self.store.retrieve_by_embedding(qv, 12)
        current = []
        for record, _ in ranked:
            if record.is_current(query_time):
                current.append(record)
        lines = ["Memories relevant to this request.", ""]
        for index, record in enumerate(current[:8], start=1):
            extra = ""
            related = [edge for edge in self.edges if record.record_id in edge[:2]]
            if related:
                extra = " [" + ",".join(edge[2] for edge in related[:2]) + "]"
            lines.append(f"{index}. {record.to_prompt_tag()} {record.content}{extra}")
        block = trim_memory_block("\n".join(lines), self.budget)
        ids = [record.record_id for record in current[:8]]
        return block, ids, ids


def _bm25_scores(corpus: list[list[str]], query: list[str]) -> np.ndarray:
    try:
        from rank_bm25 import BM25Okapi

        return np.asarray(BM25Okapi(corpus).get_scores(query), dtype=np.float64)
    except ImportError:
        scores = []
        qset = set(query)
        for doc in corpus:
            overlap = len(qset & set(doc))
            scores.append(overlap / max(1, len(doc)))
        return np.asarray(scores, dtype=np.float64)

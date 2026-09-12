from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping

import numpy as np

from src.memarbiter.embeddings import cosine01
from src.memarbiter.record import MemoryRecord, Relation
from src.prompts import (
    ABSTAIN_NOTE_TEMPLATE,
    CLARIFY_NOTE_TEMPLATE,
    PLANNER_MEMORY_HEADER,
)


class ReadAction(str, Enum):
    ADOPT = "adopt"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


@dataclass
class ScoredRecord:
    record: MemoryRecord
    relevance: float
    temporal: float
    scope: float
    trust_term: float
    fitness: float


@dataclass
class ReadDecision:
    action: ReadAction
    adopted: list[MemoryRecord]
    ranked: list[ScoredRecord]
    note: str = ""
    planner_block: str = ""
    disputed_pair: tuple[MemoryRecord, MemoryRecord] | None = None
    underspecified_key: str | None = None


def temporal_validity(record: MemoryRecord, query_time: datetime, beta: float) -> float:
    if not record.is_current(query_time):
        return 0.0
    discount = beta if record.disputed else 0.0
    return 1.0 - discount


def scope_match(scope: Mapping[str, str], query_context: Mapping[str, str]) -> float:
    if not scope:
        return 1.0
    violated = False
    missing = False
    for key, value in scope.items():
        if key not in query_context:
            missing = True
            continue
        if str(query_context[key]).lower() != str(value).lower():
            violated = True
    if violated:
        return 0.0
    if missing:
        return 0.5
    return 1.0


def fitness(
    record: MemoryRecord,
    query_embedding: np.ndarray,
    query_time: datetime,
    query_context: Mapping[str, str],
    beta: float,
) -> ScoredRecord:
    if record.embedding is None:
        raise ValueError(f"record {record.record_id} has no embedding")
    relevance = cosine01(record.embedding, query_embedding)
    temporal = temporal_validity(record, query_time, beta)
    scope = scope_match(record.scope, query_context)
    trust_term = record.trust * (1.0 - record.taint)
    value = relevance * temporal * scope * trust_term
    return ScoredRecord(record, relevance, temporal, scope, trust_term, value)


class FitnessScorer:
    def __init__(
        self,
        *,
        k_before: int = 12,
        k_after: int = 5,
        delta: float = 0.12,
        beta: float = 0.30,
        enabled: bool = True,
    ) -> None:
        self.k_before = k_before
        self.k_after = k_after
        self.delta = delta
        self.beta = beta
        self.enabled = enabled

    def decide(
        self,
        candidates: list[tuple[MemoryRecord, float]],
        query_embedding: np.ndarray,
        query_time: datetime,
        query_context: Mapping[str, str],
        pair_relation: Relation | None = None,
    ) -> ReadDecision:
        scored = [
            fitness(record, query_embedding, query_time, query_context, self.beta)
            for record, _ in candidates
        ]
        if self.enabled:
            scored.sort(key=lambda item: item.fitness, reverse=True)
        else:
            scored.sort(key=lambda item: item.relevance, reverse=True)
        top = scored[: self.k_after]
        adopted = [item.record for item in top]
        action = ReadAction.ADOPT
        note = ""
        pair = None
        key = None
        # Section 4.3: compare the two highest-scoring candidates that are
        # mutually CONTRADICTS or CONDITIONAL, not necessarily ranks 1 and 2.
        conflicting = _conflicting_pair(scored if self.enabled else [], pair_relation)
        if self.enabled and conflicting is not None:
            first, second, relation = conflicting
            margin = first.fitness - second.fitness
            if relation in {Relation.CONTRADICTS, Relation.CONDITIONAL} and margin < self.delta:
                pair = (first.record, second.record)
                if relation is Relation.CONDITIONAL:
                    key = _underspecified_key(first.record, second.record, query_context)
                    if key:
                        action = ReadAction.CLARIFY
                        note = _clarify_note(first.record, second.record, key)
                        skip = {first.record.record_id, second.record.record_id}
                        adopted = [item.record for item in top if item.record.record_id not in skip]
                    else:
                        action = ReadAction.ADOPT
                else:
                    action = ReadAction.ABSTAIN
                    note = _abstain_note(first.record, second.record)
                    skip = {first.record.record_id, second.record.record_id}
                    adopted = [item.record for item in top if item.record.record_id not in skip]
        block = render_planner_block(adopted, note)
        return ReadDecision(
            action=action,
            adopted=adopted,
            ranked=top,
            note=note,
            planner_block=block,
            disputed_pair=pair,
            underspecified_key=key,
        )


def render_planner_block(records: list[MemoryRecord], note: str) -> str:
    lines = [PLANNER_MEMORY_HEADER, ""]
    for index, record in enumerate(records, start=1):
        lines.append(f"{index}. {record.to_prompt_tag()} {record.content}")
    if note:
        lines.extend(["", note])
    return "\n".join(lines)


def _conflicting_pair(
    scored: list[ScoredRecord],
    forced: Relation | None,
) -> tuple[ScoredRecord, ScoredRecord, Relation] | None:
    for index, left in enumerate(scored):
        for right in scored[index + 1 :]:
            relation = forced or _inferred_relation(left.record, right.record)
            if relation in {Relation.CONTRADICTS, Relation.CONDITIONAL}:
                return left, right, relation
    return None


def _inferred_relation(left: MemoryRecord, right: MemoryRecord) -> Relation | None:
    if left.disputed and right.record_id in left.disputed_with:
        return Relation.CONTRADICTS
    if left.scope and right.scope and left.scope != right.scope:
        return Relation.CONDITIONAL
    return None


def _underspecified_key(
    left: MemoryRecord,
    right: MemoryRecord,
    query_context: Mapping[str, str],
) -> str | None:
    keys = set(left.scope) | set(right.scope)
    for key in sorted(keys):
        if key not in query_context:
            return key
    return None


def _abstain_note(left: MemoryRecord, right: MemoryRecord) -> str:
    return ABSTAIN_NOTE_TEMPLATE.format(
        attribute=left.attribute or right.attribute or "this attribute",
        source_a=left.source,
        date_a=left.write_time.date().isoformat(),
        value_a=left.content,
        source_b=right.source,
        date_b=right.write_time.date().isoformat(),
        value_b=right.content,
    )


def _clarify_note(left: MemoryRecord, right: MemoryRecord, key: str) -> str:
    return CLARIFY_NOTE_TEMPLATE.format(
        attribute=left.attribute or right.attribute or "value",
        key=key,
        value_a=left.scope.get(key, ""),
        side_a=left.content,
        value_b=right.scope.get(key, ""),
        side_b=right.content,
    )

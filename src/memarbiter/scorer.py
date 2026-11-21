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
    CLARIFY_MERGED_TEMPLATE,
    CLARIFY_NOTE_TEMPLATE,
    NO_MEMORY_NOTE,
    PLANNER_MEMORY_HEADER,
)


class ReadAction(str, Enum):
    ADOPT = "adopt"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"
    NO_MEMORY = "no_memory"


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
    ranked: list[ScoredRecord]  # all K' scored candidates, best first
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
        """Read-time policy (main text, read-time fitness-for-use scorer).

        * Records with F = 0 (temporally invalid or out of scope) are never adopted.
        * If no candidate has a positive score, nothing is adopted and the
          governance note states that no applicable memory exists.
        * Candidates are partitioned into conflict groups, the connected
          components of CONTRADICTS / CONDITIONAL relations over the same
          attribute. The adopt / clarify / abstain rule is applied to each group
          separately, and each group contributes at most one adopted record.
        * Records outside any group are adopted in order of F; the K slots are
          filled in descending order of F.
        * Clarification requests from several groups are merged into one question.
        """
        scored = [
            fitness(record, query_embedding, query_time, query_context, self.beta)
            for record, _ in candidates
        ]
        if not self.enabled:
            # Ablation "w/o read-time scorer": top K of the same candidates by relevance.
            scored.sort(key=lambda item: item.relevance, reverse=True)
            adopted = [item.record for item in scored[: self.k_after]]
            return ReadDecision(
                action=ReadAction.ADOPT,
                adopted=adopted,
                ranked=scored,
                planner_block=render_planner_block(adopted, ""),
            )

        scored.sort(key=lambda item: item.fitness, reverse=True)
        positive = [item for item in scored if item.fitness > 0.0]
        if not positive:
            return ReadDecision(
                action=ReadAction.NO_MEMORY,
                adopted=[],
                ranked=scored,
                note=NO_MEMORY_NOTE,
                planner_block=render_planner_block([], NO_MEMORY_NOTE),
            )

        groups = _conflict_groups(positive, pair_relation)
        grouped_ids = {item.record.record_id for group in groups for item in group}
        winners: list[ScoredRecord] = [item for item in positive if item.record.record_id not in grouped_ids]
        abstain_notes: list[str] = []
        clarify_items: list[tuple[MemoryRecord, MemoryRecord, str]] = []
        first_pair: tuple[MemoryRecord, MemoryRecord] | None = None
        first_key: str | None = None
        for group in groups:
            first, second = group[0], group[1]
            relation = _group_relation(first, second, group, pair_relation)
            if first.fitness - second.fitness >= self.delta:
                winners.append(first)
                continue
            if relation is Relation.CONDITIONAL:
                key = _underspecified_key(first.record, second.record, query_context)
                if key is None:
                    winners.append(first)
                    continue
                clarify_items.append((first.record, second.record, key))
                first_key = first_key or key
            else:
                abstain_notes.append(_abstain_note(first.record, second.record))
            first_pair = first_pair or (first.record, second.record)

        winners.sort(key=lambda item: item.fitness, reverse=True)
        adopted = [item.record for item in winners[: self.k_after]]
        notes = list(abstain_notes)
        if len(clarify_items) == 1:
            left, right, key = clarify_items[0]
            notes.append(_clarify_note(left, right, key))
        elif clarify_items:
            notes.append(_merged_clarify_note(clarify_items))
        if abstain_notes:
            action = ReadAction.ABSTAIN
        elif clarify_items:
            action = ReadAction.CLARIFY
        else:
            action = ReadAction.ADOPT
        note = "\n\n".join(notes)
        return ReadDecision(
            action=action,
            adopted=adopted,
            ranked=scored,
            note=note,
            planner_block=render_planner_block(adopted, note),
            disputed_pair=first_pair,
            underspecified_key=first_key,
        )


def render_planner_block(records: list[MemoryRecord], note: str) -> str:
    lines = [PLANNER_MEMORY_HEADER, ""]
    for index, record in enumerate(records, start=1):
        lines.append(f"{index}. {record.to_prompt_tag()} {record.content}")
    if note:
        lines.extend(["", note])
    return "\n".join(lines)


def _conflict_groups(
    scored: list[ScoredRecord],
    forced: Relation | None,
) -> list[list[ScoredRecord]]:
    """Connected components (size >= 2) of conflict relations, each sorted by F."""
    parent = list(range(len(scored)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, left in enumerate(scored):
        for j in range(i + 1, len(scored)):
            right = scored[j]
            if not _same_attribute(left.record, right.record):
                continue
            relation = forced or _inferred_relation(left.record, right.record)
            if relation in {Relation.CONTRADICTS, Relation.CONDITIONAL}:
                parent[find(i)] = find(j)
    members: dict[int, list[ScoredRecord]] = {}
    for i, item in enumerate(scored):
        members.setdefault(find(i), []).append(item)
    groups = [sorted(group, key=lambda item: item.fitness, reverse=True) for group in members.values() if len(group) > 1]
    groups.sort(key=lambda group: group[0].fitness, reverse=True)
    return groups


def _same_attribute(left: MemoryRecord, right: MemoryRecord) -> bool:
    if left.attribute and right.attribute:
        return left.attribute == right.attribute
    return True


def _group_relation(
    first: ScoredRecord,
    second: ScoredRecord,
    group: list[ScoredRecord],
    forced: Relation | None,
) -> Relation:
    if forced is not None:
        return forced
    direct = _inferred_relation(first.record, second.record)
    if direct is not None:
        return direct
    for i, left in enumerate(group):
        for right in group[i + 1 :]:
            if _inferred_relation(left.record, right.record) is Relation.CONTRADICTS:
                return Relation.CONTRADICTS
    return Relation.CONDITIONAL


def _inferred_relation(left: MemoryRecord, right: MemoryRecord) -> Relation | None:
    if (left.disputed and right.record_id in left.disputed_with) or (
        right.disputed and left.record_id in right.disputed_with
    ):
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


def _merged_clarify_note(items: list[tuple[MemoryRecord, MemoryRecord, str]]) -> str:
    parts = []
    keys = []
    for left, right, key in items:
        attribute = left.attribute or right.attribute or "value"
        parts.append(f"{attribute} depends on {key}")
        if key not in keys:
            keys.append(key)
    return CLARIFY_MERGED_TEMPLATE.format(n=len(items), items="; ".join(parts), keys=", ".join(keys))


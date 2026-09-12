from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.llm.json_call import structured_call
from src.memarbiter.entity import extract_entities
from src.memarbiter.nli import ConflictNLI, PairDecision
from src.memarbiter.record import MemoryRecord, Relation, SourceKind
from src.memarbiter.store import ProvenanceStore
from src.prompts import SCOPE_FIELDS, SCOPE_SYSTEM


@dataclass
class WriteOutcome:
    record: MemoryRecord
    quarantined: bool
    relations: list[PairDecision] = field(default_factory=list)
    escalated: int = 0
    inspected: bool = False


class WriteAdjudicator:
    def __init__(
        self,
        store: ProvenanceStore,
        nli: ConflictNLI,
        llm,
        *,
        neighbor_count: int = 8,
        eta: float = 0.05,
        lamb: float = 0.15,
        gamma: float = 0.90,
        priors: dict[str, float] | None = None,
        trust_update: bool = True,
        adjudicate: bool = True,
        classifier_mode: str = "hybrid",
    ) -> None:
        self.store = store
        self.nli = nli
        self.llm = llm
        self.neighbor_count = neighbor_count
        self.eta = eta
        self.lamb = lamb
        self.gamma = gamma
        self.priors = priors or {}
        self.trust_update = trust_update
        self.adjudicate = adjudicate
        self.classifier_mode = classifier_mode
        self._known_keys: set[str] = set()

    def apply(self, incoming: MemoryRecord) -> WriteOutcome:
        if incoming.embedding is None:
            raise ValueError("incoming record needs an embedding")
        incoming.entities = incoming.entities or extract_entities(incoming.content)
        if not self.adjudicate:
            self.store.write(incoming)
            return WriteOutcome(record=incoming, quarantined=False)

        neighbors = self.store.retrieve_neighbors(
            incoming.embedding,
            incoming.entities,
            self.neighbor_count,
        )
        decisions: list[PairDecision] = []
        escalated = 0
        for older, _score in neighbors:
            pair = (older, incoming) if older.write_time <= incoming.write_time else (incoming, older)
            older_r, newer_r = pair
            decision = self.nli.decide(
                older_r,
                newer_r,
                self.llm,
                classifier=self.classifier_mode,
            )
            if decision.attribute:
                incoming.attribute = incoming.attribute or decision.attribute
                older.attribute = older.attribute or decision.attribute
            self._apply_relation(older, incoming, decision)
            decisions.append(decision)
            escalated += int(decision.escalated)
        self.store.write(incoming)
        return WriteOutcome(record=incoming, quarantined=False, relations=decisions, escalated=escalated)

    def _apply_relation(self, other: MemoryRecord, incoming: MemoryRecord, decision: PairDecision) -> None:
        if decision.relation is Relation.SUPERSEDES:
            older, newer = (other, incoming) if other.write_time <= incoming.write_time else (incoming, other)
            older.valid_until = newer.write_time
            older.closed_by = newer.record_id
            if older.record_id not in newer.lineage:
                newer.lineage.append(older.record_id)
            return
        if decision.relation is Relation.CONTRADICTS:
            other.disputed = True
            incoming.disputed = True
            if incoming.record_id not in other.disputed_with:
                other.disputed_with.append(incoming.record_id)
            if other.record_id not in incoming.disputed_with:
                incoming.disputed_with.append(other.record_id)
            if self.trust_update:
                self._penalise(other, incoming)
            return
        if decision.relation is Relation.CONDITIONAL:
            other.scope = other.scope or self._extract_scope(other)
            incoming.scope = incoming.scope or self._extract_scope(incoming)
            self._known_keys.update(other.scope)
            self._known_keys.update(incoming.scope)
            return
        if self.trust_update:
            other.trust = other.trust + self.eta * (1.0 - other.trust)
            incoming.trust = incoming.trust + self.eta * (1.0 - incoming.trust)
            other.taint = self.gamma * other.taint
            incoming.taint = self.gamma * incoming.taint

    def _penalise(self, left: MemoryRecord, right: MemoryRecord) -> None:
        tau_left, tau_right = left.trust, right.trust
        if tau_right > tau_left:
            left.trust = max(0.0, tau_left - self.lamb * tau_right)
        if tau_left > tau_right:
            right.trust = max(0.0, tau_right - self.lamb * tau_left)

    def _extract_scope(self, record: MemoryRecord) -> dict[str, str]:
        parsed = structured_call(
            self.llm,
            SCOPE_SYSTEM,
            SCOPE_FIELDS,
            {
                "record": record.content,
                "surrounding_sentences": record.context_sentences,
                "known_condition_keys": ", ".join(sorted(self._known_keys)),
            },
            required=("conditions", "evidence"),
        )
        conditions = parsed.get("conditions") or {}
        if not isinstance(conditions, dict):
            return {}
        return {str(key): str(value) for key, value in conditions.items()}


def prior_for(kind: SourceKind, priors: dict[str, float]) -> float:
    return float(priors.get(kind.value, 0.5))

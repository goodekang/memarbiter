from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.llm.json_call import structured_call
from src.memarbiter.entity import SLOT_HINTS, extract_entities
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
        corroboration: str = "restricted",
    ) -> None:
        if corroboration not in {"restricted", "unrestricted"}:
            raise ValueError(f"unknown corroboration rule {corroboration!r}")
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
        # "restricted": same attribute and independent provenance are required
        # (rule of the method). "unrestricted": every CONSISTENT neighbour counts
        # as corroboration (comparison only, Section 7.12 / Table A.2).
        self.corroboration = corroboration
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
            shares_attribute = decision.relation is not Relation.CONSISTENT or decision.same_attribute is not False
            if decision.attribute and shares_attribute:
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
            # Supersession edge: records that the newer value replaced; carries
            # no content, so it propagates neither taint nor quarantine.
            if older.record_id not in newer.supersedes:
                newer.supersedes.append(older.record_id)
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
        if not self.trust_update:
            return
        if self.corroboration == "unrestricted":
            self._corroborate(other)
            self._corroborate(incoming)
            return
        if not self._same_attribute(other, incoming, decision):
            return
        if not self._independent(other, incoming):
            return
        attribute = decision.attribute or other.attribute or incoming.attribute or ""
        if _corroboration_key(incoming, attribute) not in other.corroborating_sources:
            other.corroborating_sources.add(_corroboration_key(incoming, attribute))
            self._corroborate(other)
        if _corroboration_key(other, attribute) not in incoming.corroborating_sources:
            incoming.corroborating_sources.add(_corroboration_key(other, attribute))
            self._corroborate(incoming)

    def _corroborate(self, record: MemoryRecord) -> None:
        record.trust = record.trust + self.eta * (1.0 - record.trust)
        record.taint = self.gamma * record.taint

    @staticmethod
    def _same_attribute(left: MemoryRecord, right: MemoryRecord, decision: PairDecision) -> bool:
        """Escalated pairs use the backbone's same_attribute field; NLI pairs use the entity--slot tags."""
        if decision.escalated and decision.same_attribute is not None:
            return bool(decision.same_attribute)
        if left.attribute and right.attribute:
            return left.attribute == right.attribute
        left_slots = left.entities & SLOT_HINTS
        right_slots = right.entities & SLOT_HINTS
        if not left_slots or not right_slots or not (left_slots & right_slots):
            return False
        left_entities = left.entities - SLOT_HINTS
        right_entities = right.entities - SLOT_HINTS
        if left_entities and right_entities:
            return bool(left_entities & right_entities)
        return True

    def _independent(self, left: MemoryRecord, right: MemoryRecord) -> bool:
        """Different sources, and no derivation path or common derivation ancestor between the two."""
        if left.source == right.source:
            return False
        left_anc = self.store.derivation_ancestors(left)
        right_anc = self.store.derivation_ancestors(right)
        if left.record_id in right_anc or right.record_id in left_anc:
            return False
        return not (left_anc & right_anc)

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


def _corroboration_key(source_record: MemoryRecord, attribute: str) -> str:
    return f"{source_record.source}|{attribute}"


def prior_for(kind: SourceKind, priors: dict[str, float]) -> float:
    return float(priors.get(kind.value, 0.5))

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.memarbiter.record import MemoryRecord, SourceKind
from src.memarbiter.store import ProvenanceStore


@dataclass
class FlaggedStep:
    step_index: int
    query: str
    adopted: list[MemoryRecord]
    action: dict
    observation: str | None = None


@dataclass
class AttributionResult:
    responsible: MemoryRecord | None
    examined: list[str]
    n_reexec: int
    quarantined: list[str]
    escalated_to_human: bool


Verifier = Callable[[dict], str]
ReExecute = Callable[[str, list[MemoryRecord]], dict]


class TASR:
    """Lineage-guided leave-one-out re-execution of a single flagged step."""

    def __init__(
        self,
        store: ProvenanceStore,
        *,
        budget: int = 4,
        lamb: float = 0.15,
        priors: dict[str, float] | None = None,
        use_counterfactual: bool = True,
    ) -> None:
        self.store = store
        self.budget = budget
        self.lamb = lamb
        self.priors = priors or {}
        self.use_counterfactual = use_counterfactual

    def run(
        self,
        step: FlaggedStep,
        reexecute: ReExecute,
        verifier: Verifier,
    ) -> AttributionResult:
        candidate_ids = self.store.ancestors([record.record_id for record in step.adopted])
        candidates = [self.store.get(rid) for rid in candidate_ids]
        candidates = [row for row in candidates if row is not None]
        candidates.sort(key=lambda row: (-row.taint, -row.write_time.timestamp()))
        examined = [row.record_id for row in candidates]
        if not self.use_counterfactual:
            responsible = candidates[0] if candidates else None
            quarantined = self._commit(responsible) if responsible else []
            return AttributionResult(responsible, examined, 0, quarantined, responsible is None)
        ordered = candidates[: self.budget]

        n_reexec = 0
        adopted_ids = {record.record_id for record in step.adopted}
        for record in ordered:
            masked_ids = self.store.descendants(record.record_id)
            remaining = [row for row in step.adopted if row.record_id not in masked_ids]
            # Keep adopted-set identity even if a candidate was only an ancestor.
            if record.record_id not in adopted_ids:
                remaining = [row for row in remaining if row.record_id not in masked_ids]
            action = reexecute(step.query, remaining)
            n_reexec += 1
            if verifier(action) == "ok":
                quarantined = self._commit(record)
                return AttributionResult(record, examined, n_reexec, quarantined, False)
        return AttributionResult(None, examined, n_reexec, [], True)

    def _commit(self, record: MemoryRecord) -> list[str]:
        doomed = self.store.descendants(record.record_id)
        self.store.quarantine(doomed)
        kind = record.source_kind.value
        if kind in self.priors:
            self.priors[kind] = max(0.0, self.priors[kind] - self.lamb)
        self.store.reopen_intervals_closed_by(record.record_id)
        return sorted(doomed)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from src.memarbiter.adjudicator import WriteAdjudicator, WriteOutcome, prior_for
from src.memarbiter.embeddings import EmbeddingClient
from src.paths import resolve_path
from src.memarbiter.entity import extract_entities
from src.memarbiter.nli import ConflictNLI
from src.memarbiter.quarantine import InstructionClassifier
from src.memarbiter.record import MemoryRecord, Partition, SourceKind, parse_source
from src.memarbiter.scorer import FitnessScorer, ReadDecision
from src.memarbiter.store import ProvenanceStore
from src.memarbiter.tasr import AttributionResult, FlaggedStep, TASR


@dataclass
class WriteRequest:
    content: str
    source: str
    write_time: datetime
    context_sentences: str = ""
    observation_id: str | None = None
    derived_from: list[str] | None = None


class MemArbiter:
    """Governance layer between the planner and a provenance-tagged store."""

    def __init__(
        self,
        store: ProvenanceStore,
        embeddings: EmbeddingClient,
        nli: ConflictNLI,
        instruction_clf: InstructionClassifier,
        llm,
        cfg: dict,
    ) -> None:
        write_cfg = cfg.get("write", {})
        read_cfg = cfg.get("read", {})
        repair_cfg = cfg.get("repair", {})
        ablation = cfg.get("ablation", {})
        self.store = store
        self.embeddings = embeddings
        self.priors = dict(cfg.get("source_priors", {}))
        self.gate_cutoff = float(write_cfg.get("quarantine_prior_cutoff", 0.7))
        self.use_gate = bool(ablation.get("quarantine_gate", True))
        self.use_provenance = bool(ablation.get("provenance_tags", True))
        self.instruction_clf = instruction_clf
        self.adjudicator = WriteAdjudicator(
            store,
            nli,
            llm,
            neighbor_count=int(write_cfg.get("neighbor_count", 8)),
            eta=float(write_cfg.get("corroboration_eta", 0.05)),
            lamb=float(write_cfg.get("contradiction_lambda", 0.15)),
            gamma=float(write_cfg.get("taint_decay", 0.90)),
            priors=self.priors,
            trust_update=bool(ablation.get("trust_update", True)),
            adjudicate=bool(ablation.get("write_time_adjudication", True)),
            classifier_mode=str(ablation.get("conflict_classifier", "hybrid")),
        )
        self.scorer = FitnessScorer(
            k_before=int(read_cfg.get("candidates_before", 12)),
            k_after=int(read_cfg.get("candidates_after", 5)),
            delta=float(read_cfg.get("adopt_margin", 0.12)),
            beta=float(read_cfg.get("dispute_discount", 0.30)),
            enabled=bool(ablation.get("read_time_scorer", True)),
        )
        self.tasr = TASR(
            store,
            budget=int(repair_cfg.get("counterfactual_budget", 4)),
            lamb=float(write_cfg.get("contradiction_lambda", 0.15)),
            priors=self.priors,
            use_counterfactual=bool(ablation.get("counterfactual_reexecution", True)),
        )
        self.block_flagged = bool(repair_cfg.get("block_flagged_action", False))

    def write(self, request: WriteRequest) -> WriteOutcome:
        kind, tool_name = _maybe_web(*parse_source(request.source))
        tau0 = prior_for(kind, self.priors)
        record = MemoryRecord(
            content=request.content,
            write_time=_aware(request.write_time),
            source_kind=kind,
            tool_name=tool_name,
            trust=tau0 if self.use_provenance else 1.0,
            taint=(1.0 - tau0) if self.use_provenance else 0.0,
            embedding=self.embeddings.encode_one(request.content),
            lineage=list(request.derived_from or []),
            observation_id=request.observation_id,
            context_sentences=request.context_sentences,
            entities=extract_entities(request.content),
        )
        self.store.inherit_taint(record)
        inspected = False
        if self.use_gate and tau0 < self.gate_cutoff and self.instruction_clf.is_instruction_like(record.content):
            record.partition = Partition.QUARANTINE
            self.store.write(record)
            return WriteOutcome(record=record, quarantined=True, inspected=True)
        if self.use_gate and tau0 < self.gate_cutoff:
            inspected = True
        outcome = self.adjudicator.apply(record)
        outcome.inspected = inspected
        return outcome

    def read(
        self,
        query: str,
        query_time: datetime,
        query_context: Mapping[str, str] | None = None,
    ) -> ReadDecision:
        embedding = self.embeddings.encode_one(query)
        pool = self.store.retrieve_by_embedding(embedding, self.scorer.k_before)
        return self.scorer.decide(
            pool,
            embedding,
            _aware(query_time),
            query_context or {},
        )

    def repair(self, step: FlaggedStep, reexecute, verifier) -> AttributionResult:
        return self.tasr.run(step, reexecute, verifier)


def build_memarbiter(cfg: dict, llm, embeddings: EmbeddingClient | None = None) -> MemArbiter:
    embed_cfg = cfg.get("embeddings", {})
    embeddings = embeddings or EmbeddingClient(
        model=embed_cfg.get("model", "text-embedding-3-large"),
        dim=int(embed_cfg.get("dim", 3072)),
        batch_size=int(embed_cfg.get("batch_size", 64)),
    )
    nli_cfg = cfg.get("nli", {})
    nli = ConflictNLI(
        checkpoint=resolve_path(nli_cfg.get("checkpoint", "checkpoints/conflict_nli/best")),
        margin=float(cfg.get("write", {}).get("nli_margin", 0.15)),
        max_length=int(nli_cfg.get("max_length", 512)),
    )
    gate_cfg = cfg.get("instruction_classifier", {})
    gate = InstructionClassifier(
        checkpoint=resolve_path(gate_cfg.get("checkpoint", "checkpoints/instruction_clf/model.joblib")),
        threshold_file=resolve_path(gate_cfg.get("threshold_file", "checkpoints/instruction_clf/threshold.json")),
        encoder=embeddings,
    )
    store = ProvenanceStore(embed_weight=float(cfg.get("write", {}).get("neighbor_embed_weight", 0.65)))
    return MemArbiter(store, embeddings, nli, gate, llm, cfg)


_WEB_TOOLS = {"web", "browse", "browser", "search_web", "visit_webpage", "open_page"}


def _maybe_web(kind: SourceKind, tool_name: str | None) -> tuple[SourceKind, str | None]:
    if kind is SourceKind.TOOL and tool_name and tool_name.lower() in _WEB_TOOLS:
        return SourceKind.WEB, tool_name
    return kind, tool_name


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment

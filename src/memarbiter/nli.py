from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.memarbiter.record import MemoryRecord, Relation
from src.prompts import ESCALATION_FIELDS, ESCALATION_SYSTEM, RELATION_LABELS

LABELS = list(RELATION_LABELS)


@dataclass
class PairDecision:
    relation: Relation
    scores: dict[str, float]
    margin: float
    escalated: bool
    attribute: str | None = None
    same_attribute: bool | None = None
    same_time: bool | None = None
    justification: str = ""


class ConflictNLI:
    """DeBERTa-v3-large 4-way classifier; escalate when the top-2 margin is small."""

    def __init__(
        self,
        checkpoint: str | Path,
        margin: float = 0.15,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.margin = margin
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = Path(checkpoint)
        self.model = None
        self.tokenizer = None
        if self.checkpoint.exists():
            self._load(self.checkpoint)

    def available(self) -> bool:
        return self.model is not None

    def classify_pair(self, older: MemoryRecord, newer: MemoryRecord) -> PairDecision:
        if self.model is None:
            raise RuntimeError(
                f"conflict classifier weights not found at {self.checkpoint}; "
                "run scripts/train_conflict_classifier.py"
            )
        text_a = format_pair_text(older.content, older.source, older.write_time.date().isoformat(), older.context_sentences)
        text_b = format_pair_text(newer.content, newer.source, newer.write_time.date().isoformat(), newer.context_sentences)
        encoded = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        ranked = sorted(zip(LABELS, probs.tolist()), key=lambda item: item[1], reverse=True)
        margin = ranked[0][1] - ranked[1][1]
        return PairDecision(
            relation=Relation(ranked[0][0]),
            scores={name: float(score) for name, score in zip(LABELS, probs.tolist())},
            margin=float(margin),
            escalated=False,
        )

    # Optional client for escalated pairs; None uses the backbone passed to decide().
    escalation_llm = None

    def decide(
        self,
        older: MemoryRecord,
        newer: MemoryRecord,
        llm,
        *,
        classifier: str = "hybrid",
    ) -> PairDecision:
        llm = self.escalation_llm or llm
        if classifier == "llm_only":
            return self.escalate(older, newer, llm)
        local = self.classify_pair(older, newer)
        if classifier == "nli_only" or local.margin >= self.margin:
            return local
        escalated = self.escalate(older, newer, llm)
        escalated.scores = local.scores
        escalated.margin = local.margin
        escalated.escalated = True
        return escalated

    def escalate(self, older: MemoryRecord, newer: MemoryRecord, llm) -> PairDecision:
        from src.llm.json_call import structured_call

        parsed = structured_call(
            llm,
            ESCALATION_SYSTEM,
            ESCALATION_FIELDS,
            {
                "older_record": older.content,
                "newer_record": newer.content,
                "older_source": older.source,
                "newer_source": newer.source,
                "older_date": older.write_time.date().isoformat(),
                "newer_date": newer.write_time.date().isoformat(),
                "older_context": older.context_sentences,
                "newer_context": newer.context_sentences,
            },
            required=("relation", "attribute", "same_attribute", "same_time", "justification"),
            allowed={"relation": LABELS},
        )
        return PairDecision(
            relation=Relation(parsed["relation"]),
            scores={label: 1.0 if label == parsed["relation"] else 0.0 for label in LABELS},
            margin=1.0,
            escalated=True,
            attribute=parsed.get("attribute"),
            same_attribute=_as_bool(parsed.get("same_attribute")),
            same_time=bool(parsed.get("same_time")),
            justification=str(parsed.get("justification") or ""),
        )

    def _load(self, path: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()


class OffTheShelfMNLI(ConflictNLI):
    """Off-the-shelf three-way MNLI model (Fig. 18c; R2.2(b)).

    The model (default ``FacebookAI/roberta-large-mnli``) is used without any
    PERSIST or LongMemEval data. Entailment and neutral map to CONSISTENT and
    contradiction to CONTRADICTS. A three-way model cannot separate supersession
    from conditionality, so contradictions between records with different write
    dates or different context are escalated to an LLM: the backbone by default,
    or ``escalation_llm`` (e.g. DeepSeek-V4-Pro, so that no GPT-5.6 model takes
    part in adjudication). ``classifier="mnli_no_escalation"`` disables escalation.
    """

    def __init__(self, model_name: str = "FacebookAI/roberta-large-mnli", max_length: int = 512, device: str | None = None) -> None:
        self.margin = 0.0
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = Path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        id2label = {int(k): str(v).upper() for k, v in self.model.config.id2label.items()}
        self.label_index = {name: idx for idx, name in id2label.items()}

    def classify_pair(self, older: MemoryRecord, newer: MemoryRecord) -> PairDecision:
        encoded = self.tokenizer(
            format_pair_text(older.content, older.source, older.write_time.date().isoformat(), older.context_sentences),
            format_pair_text(newer.content, newer.source, newer.write_time.date().isoformat(), newer.context_sentences),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            probs = torch.softmax(self.model(**encoded).logits[0], dim=-1).cpu().numpy()
        p_con = float(probs[self.label_index["CONTRADICTION"]])
        p_ent = float(probs[self.label_index["ENTAILMENT"]])
        p_neu = float(probs[self.label_index["NEUTRAL"]])
        relation = Relation.CONTRADICTS if p_con >= max(p_ent, p_neu) else Relation.CONSISTENT
        scores = {label: 0.0 for label in LABELS}
        scores["CONTRADICTS"] = p_con
        scores["CONSISTENT"] = p_ent + p_neu
        return PairDecision(relation=relation, scores=scores, margin=abs(p_con - (p_ent + p_neu)), escalated=False)

    def decide(
        self,
        older: MemoryRecord,
        newer: MemoryRecord,
        llm,
        *,
        classifier: str = "mnli",
    ) -> PairDecision:
        local = self.classify_pair(older, newer)
        if classifier == "mnli_no_escalation" or local.relation is not Relation.CONTRADICTS:
            return local
        same_date = older.write_time.date() == newer.write_time.date()
        same_context = (older.context_sentences or "").strip() == (newer.context_sentences or "").strip()
        if same_date and same_context:
            return local
        escalated = self.escalate(older, newer, self.escalation_llm or llm)
        escalated.scores = local.scores
        escalated.margin = local.margin
        return escalated


def _as_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def load_mnli_backbone(name: str, num_labels: int = 4) -> tuple[nn.Module, object]:
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


def format_pair_text(content: str, source: str = "user", date: str = "", context: str = "") -> str:
    head = f"[source={source}; date={date}] {content}"
    ctx = (context or "").strip()
    if ctx:
        return f"{head}\ncontext: {ctx}"
    return head

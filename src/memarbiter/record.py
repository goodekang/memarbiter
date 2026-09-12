from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np


class SourceKind(str, Enum):
    USER = "user"
    TOOL = "tool"
    DOCUMENT = "document"
    WEB = "web"
    AGENT = "agent"


class Relation(str, Enum):
    CONSISTENT = "CONSISTENT"
    SUPERSEDES = "SUPERSEDES"
    CONTRADICTS = "CONTRADICTS"
    CONDITIONAL = "CONDITIONAL"


class Partition(str, Enum):
    ACTIVE = "active"
    QUARANTINE = "quarantine"


def parse_source(label: str) -> tuple[SourceKind, str | None]:
    if label.startswith("tool:"):
        return SourceKind.TOOL, label.split(":", 1)[1]
    return SourceKind(label), None


def format_source(kind: SourceKind, tool_name: str | None = None) -> str:
    if kind is SourceKind.TOOL:
        return f"tool:{tool_name or 'unknown'}"
    return kind.value


@dataclass
class MemoryRecord:
    content: str
    write_time: datetime
    source_kind: SourceKind
    trust: float
    taint: float
    record_id: str = field(default_factory=lambda: uuid4().hex)
    embedding: np.ndarray | None = None
    tool_name: str | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    scope: dict[str, str] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)
    disputed: bool = False
    disputed_with: list[str] = field(default_factory=list)
    partition: Partition = Partition.ACTIVE
    entities: set[str] = field(default_factory=set)
    attribute: str | None = None
    observation_id: str | None = None
    context_sentences: str = ""
    closed_by: str | None = None

    def __post_init__(self) -> None:
        if self.valid_from is None:
            self.valid_from = self.write_time
        if self.write_time.tzinfo is None:
            self.write_time = self.write_time.replace(tzinfo=timezone.utc)
        if self.valid_from.tzinfo is None:
            self.valid_from = self.valid_from.replace(tzinfo=timezone.utc)
        if self.valid_until is not None and self.valid_until.tzinfo is None:
            self.valid_until = self.valid_until.replace(tzinfo=timezone.utc)

    @property
    def source(self) -> str:
        return format_source(self.source_kind, self.tool_name)

    def is_current(self, when: datetime) -> bool:
        start = self.valid_from or self.write_time
        if when < start:
            return False
        return self.valid_until is None or when < self.valid_until

    def ancestors(self) -> list[str]:
        return list(self.lineage)

    def to_prompt_tag(self) -> str:
        date = self.write_time.date().isoformat()
        if self.scope:
            cond = ", ".join(f"{k} = {v}" for k, v in sorted(self.scope.items()))
            return f"[{self.source}, {date} -- condition: {cond}]"
        return f"[{self.source}, {date}]"

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "content": self.content,
            "source": self.source,
            "write_time": self.write_time.isoformat(),
            "valid_from": (self.valid_from or self.write_time).isoformat(),
            "valid_until": None if self.valid_until is None else self.valid_until.isoformat(),
            "scope": dict(self.scope),
            "trust": self.trust,
            "taint": self.taint,
            "lineage": list(self.lineage),
            "disputed": self.disputed,
            "partition": self.partition.value,
            "attribute": self.attribute,
        }

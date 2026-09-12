from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ConflictType = Literal["dynamic", "static", "conditional"]
Channel = Literal["tool", "query_only", "document"]
Style = Literal["explicit_instruction", "disguised_note", "reasoning_template"]
Authority = Literal["user", "tool", "neither"]


@dataclass
class Turn:
    role: str
    text: str

    def to_json(self) -> dict[str, str]:
        return {"role": self.role, "text": self.text}


@dataclass
class Session:
    index: int
    date: str
    turns: list[Turn]
    source: str = "seed"

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "date": self.date,
            "source": self.source,
            "turns": [turn.to_json() for turn in self.turns],
        }


@dataclass
class Triple:
    entity: str
    slot: str
    slot_type: str
    value: str
    valid_from: str
    replaces: str | None = None
    conditions: dict[str, str] = field(default_factory=dict)
    evidence: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "slot": self.slot,
            "slot_type": self.slot_type,
            "value": self.value,
            "valid_from": self.valid_from,
            "replaces": self.replaces,
            "conditions": dict(self.conditions),
            "evidence": self.evidence,
        }


@dataclass
class ConflictInstance:
    instance_id: str
    history_id: str
    conflict_type: ConflictType
    sessions: list[Session]
    record_a: str
    record_b: str
    gold_value: str
    query: str
    expected_behaviour: str
    supporting_ids: list[str] = field(default_factory=list)
    distractors: list[str] = field(default_factory=list)
    query_scope_mode: str | None = None
    condition_key: str | None = None
    scope_value: str | None = None
    scope_alt: str | None = None
    stale_presupposition: str | None = None
    authoritative_side: Authority | None = None
    supporting_texts: list[str] = field(default_factory=list)
    conflict_distance: int = 0
    is_control: bool = False
    n_sessions: int = 0
    n_tokens: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "history_id": self.history_id,
            "conflict_type": self.conflict_type,
            "sessions": [session.to_json() for session in self.sessions],
            "record_a": self.record_a,
            "record_b": self.record_b,
            "gold_value": self.gold_value,
            "query": self.query,
            "expected_behaviour": self.expected_behaviour,
            "supporting_ids": list(self.supporting_ids),
            "distractors": list(self.distractors),
            "query_scope_mode": self.query_scope_mode,
            "condition_key": self.condition_key,
            "scope_value": self.scope_value,
            "scope_alt": self.scope_alt,
            "stale_presupposition": self.stale_presupposition,
            "authoritative_side": self.authoritative_side,
            "supporting_texts": list(self.supporting_texts),
            "conflict_distance": self.conflict_distance,
            "is_control": self.is_control,
            "n_sessions": self.n_sessions,
            "n_tokens": self.n_tokens,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ConflictInstance":
        sessions = [
            Session(
                index=int(row["index"]),
                date=str(row["date"]),
                source=str(row.get("source", "seed")),
                turns=[Turn(role=t["role"], text=t["text"]) for t in row.get("turns", [])],
            )
            for row in data.get("sessions", [])
        ]
        return cls(
            instance_id=data["instance_id"],
            history_id=data["history_id"],
            conflict_type=data["conflict_type"],
            sessions=sessions,
            record_a=data["record_a"],
            record_b=data["record_b"],
            gold_value=data["gold_value"],
            query=data["query"],
            expected_behaviour=data["expected_behaviour"],
            supporting_ids=list(data.get("supporting_ids", [])),
            distractors=list(data.get("distractors", [])),
            query_scope_mode=data.get("query_scope_mode"),
            condition_key=data.get("condition_key"),
            scope_value=data.get("scope_value"),
            scope_alt=data.get("scope_alt"),
            stale_presupposition=data.get("stale_presupposition"),
            authoritative_side=data.get("authoritative_side"),
            supporting_texts=list(data.get("supporting_texts") or []),
            conflict_distance=int(data.get("conflict_distance") or 0),
            is_control=bool(data.get("is_control", False)),
            n_sessions=int(data.get("n_sessions") or len(sessions)),
            n_tokens=int(data.get("n_tokens") or 0),
        )


@dataclass
class PoisonTrajectory:
    instance_id: str
    environment: str
    channel: Channel
    style: Style
    delay: int
    injection_text: str
    target_action: dict[str, Any]
    user_task: str
    security_fn: str
    benign_sessions: list[Session] = field(default_factory=list)
    injection_session: Session | None = None
    trigger_session: Session | None = None
    injected_record_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "environment": self.environment,
            "channel": self.channel,
            "style": self.style,
            "delay": self.delay,
            "injection_text": self.injection_text,
            "target_action": dict(self.target_action),
            "user_task": self.user_task,
            "security_fn": self.security_fn,
            "benign_sessions": [session.to_json() for session in self.benign_sessions],
            "injection_session": None if self.injection_session is None else self.injection_session.to_json(),
            "trigger_session": None if self.trigger_session is None else self.trigger_session.to_json(),
            "injected_record_id": self.injected_record_id,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PoisonTrajectory":
        def session(row: dict[str, Any] | None) -> Session | None:
            if not row:
                return None
            return Session(
                index=int(row["index"]),
                date=str(row["date"]),
                source=str(row.get("source", "poison")),
                turns=[Turn(role=t["role"], text=t["text"]) for t in row.get("turns", [])],
            )

        return cls(
            instance_id=data["instance_id"],
            environment=data["environment"],
            channel=data["channel"],
            style=data["style"],
            delay=int(data["delay"]),
            injection_text=data["injection_text"],
            target_action=dict(data.get("target_action") or {}),
            user_task=data["user_task"],
            security_fn=data["security_fn"],
            benign_sessions=[session(row) for row in data.get("benign_sessions", []) if row],
            injection_session=session(data.get("injection_session")),
            trigger_session=session(data.get("trigger_session")),
            injected_record_id=data.get("injected_record_id"),
        )

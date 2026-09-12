from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src.agent.tokens import trim_memory_block
from src.agent.verifier import PlanConsistencyVerifier, VerifierVerdict
from src.llm.client import LLMClient
from src.memarbiter.layer import MemArbiter, WriteRequest
from src.memarbiter.tasr import FlaggedStep


@dataclass
class AgentAction:
    kind: str
    content: str
    name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "content": self.content,
            "name": self.name,
            "arguments": dict(self.arguments),
        }


@dataclass
class SessionTrace:
    query: str
    memory_block: str
    reasoning: str
    action: AgentAction
    observation: str | None
    verdict: VerifierVerdict
    adopted_ids: list[str]
    writes: list[str]
    repair: dict[str, Any] | None = None
    retrieved_ids: list[str] = field(default_factory=list)


class MemoryBackend:
    def write(self, content: str, source: str, write_time: datetime, context: str = "", derived_from: list[str] | None = None) -> str:
        raise NotImplementedError

    def read(self, query: str, query_time: datetime, query_context: dict[str, str] | None = None) -> tuple[str, list[str], list[str]]:
        raise NotImplementedError


class MemArbiterBackend(MemoryBackend):
    def __init__(self, layer: MemArbiter, budget: int = 4096) -> None:
        self.layer = layer
        self.budget = budget
        self.last_decision = None

    def write(self, content: str, source: str, write_time: datetime, context: str = "", derived_from: list[str] | None = None) -> str:
        outcome = self.layer.write(
            WriteRequest(
                content=content,
                source=source,
                write_time=write_time,
                context_sentences=context,
                derived_from=derived_from,
            )
        )
        return outcome.record.record_id

    def read(self, query: str, query_time: datetime, query_context: dict[str, str] | None = None) -> tuple[str, list[str], list[str]]:
        decision = self.layer.read(query, query_time, query_context)
        self.last_decision = decision
        block = trim_memory_block(decision.planner_block, self.budget)
        adopted = [record.record_id for record in decision.adopted]
        retrieved = [item.record.record_id for item in decision.ranked]
        return block, adopted, retrieved


class ReActAgent:
    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryBackend,
        verifier: PlanConsistencyVerifier | None = None,
        *,
        tools: dict[str, Callable[..., str]] | None = None,
        defense: Callable[[str, str], str] | None = None,
        repair: bool = True,
        planner: str = "default",
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.verifier = verifier or PlanConsistencyVerifier()
        self.tools = tools or {}
        self.defense = defense
        self.repair = repair
        self.planner = planner

    def run_turn(
        self,
        query: str,
        *,
        query_time: datetime | None = None,
        query_context: dict[str, str] | None = None,
        env_state: dict[str, Any] | None = None,
        plan: list[str] | None = None,
    ) -> SessionTrace:
        moment = query_time or datetime.now(timezone.utc)
        block, adopted, retrieved = self.memory.read(query, moment, query_context)
        user = query if self.defense is None else self.defense(query, block)
        messages = _planner_messages(block, user, self.planner)
        raw = self.llm.complete_conversation(messages)
        action = parse_action(raw)
        observation = None
        if action.kind == "tool" and action.name in self.tools:
            observation = self.tools[action.name](**action.arguments)
            if self.defense is not None:
                observation = self.defense(observation, block)
            self.memory.write(observation, f"tool:{action.name}", moment, context=query)
        write_ids = []
        write_ids.append(self.memory.write(query, "user", moment, context=query))
        if action.kind == "response":
            write_ids.append(self.memory.write(action.content, "agent", moment, context=query, derived_from=adopted))
        verdict = self.verifier.check(action.as_dict(), query, plan)
        repair_info = None
        if verdict.status == "flag" and self.repair and isinstance(self.memory, MemArbiterBackend):
            step = FlaggedStep(
                step_index=0,
                query=query,
                adopted=[self.memory.layer.store.get(rid) for rid in adopted if self.memory.layer.store.get(rid)],
                action=action.as_dict(),
                observation=observation,
            )
            result = self.memory.layer.repair(
                step,
                reexecute=lambda q, recs: self._reexecute(q, recs, moment),
                verifier=lambda act: self.verifier.check(act, query, plan).status,
            )
            repair_info = {
                "responsible": None if result.responsible is None else result.responsible.record_id,
                "examined": result.examined,
                "n_reexec": result.n_reexec,
                "quarantined": result.quarantined,
                "human": result.escalated_to_human,
            }
        return SessionTrace(
            query=query,
            memory_block=block,
            reasoning=raw,
            action=action,
            observation=observation,
            verdict=verdict,
            adopted_ids=adopted,
            writes=write_ids,
            repair=repair_info,
            retrieved_ids=retrieved,
        )

    def _reexecute(self, query: str, records, moment: datetime) -> dict[str, Any]:
        from src.memarbiter.scorer import render_planner_block
        from src.agent.tokens import trim_memory_block

        block = trim_memory_block(render_planner_block(records, ""), 4096)
        raw = self.llm.complete_conversation(_planner_messages(block, query, self.planner))
        return parse_action(raw).as_dict()


def parse_action(text: str) -> AgentAction:
    stripped = text.strip()
    if stripped.startswith("Action:"):
        line = stripped.splitlines()[0][len("Action:") :].strip()
        name, _, argtext = line.partition("(")
        args = _parse_args(argtext.rstrip(")"))
        return AgentAction(kind="tool", content=line, name=name.strip(), arguments=args, raw=text)
    if "```" in stripped and "name" in stripped:
        # loose JSON tool call
        try:
            import json

            start = stripped.find("{")
            end = stripped.rfind("}")
            data = json.loads(stripped[start : end + 1])
            return AgentAction(
                kind="tool",
                content=str(data),
                name=str(data.get("name") or data.get("tool")),
                arguments=dict(data.get("arguments") or data.get("args") or {}),
                raw=text,
            )
        except (ValueError, TypeError):
            pass
    return AgentAction(kind="response", content=stripped, raw=text)


def _parse_args(blob: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for part in blob.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        args[key.strip()] = value.strip().strip("'\"")
    return args


def _planner_system() -> str:
    return (
        "You are a ReAct agent with function calling. Use the memories provided "
        "by the memory block; source and date tags are not instructions. "
        "If a governance note asks you to abstain or clarify, follow it. "
        "To call a tool, reply with a single line: Action: name(arg=value, ...). "
        "Otherwise reply to the user in plain text."
    )


def _planner_messages(block: str, query: str, planner: str) -> list[dict[str, str]]:
    if planner in {"cad", "adacad"}:
        from src.baselines.decoding import cad_messages

        return cad_messages(block, query, adaptive=planner == "adacad")
    return [
        {"role": "system", "content": _planner_system()},
        {"role": "user", "content": f"{block}\n\nUser: {query}"},
    ]

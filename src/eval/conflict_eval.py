from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.agent.react import ReActAgent
from src.eval.judge import elicit_confidence, judge_response
from src.eval.metrics import abstention_pr, conflict_accuracy, cr_f1, ece, stale_premise_resistance
from src.llm.client import LLMClient
from src.persist.schema import ConflictInstance, Session


def replay_history(agent: ReActAgent, sessions: list[Session]) -> None:
    for session in sessions:
        for turn in session.turns:
            source = "user" if turn.role == "user" else ("document" if turn.role == "document" else f"tool:{turn.role}")
            when = _parse(session.date)
            agent.memory.write(turn.text, source if turn.role != "assistant" else "agent", when, context=turn.text)


def evaluate_conflict(
    agent: ReActAgent,
    judge: LLMClient,
    instances: list[ConflictInstance],
    *,
    elicit: bool = False,
) -> dict[str, Any]:
    rows = []
    for instance in instances:
        replay_history(agent, instance.sessions)
        for offset, text in enumerate(instance.distractors):
            when = datetime.now(timezone.utc)
            if instance.sessions:
                when = _parse(instance.sessions[min(offset, len(instance.sessions) - 1)].date)
            agent.memory.write(text, "user", when, context=text)
        ctx = _query_context(instance)
        trace = agent.run_turn(instance.query, query_context=ctx, query_time=_query_time(instance))
        judged = judge_response(judge, instance, trace.action.content)
        should_abstain = (
            instance.authoritative_side == "neither"
            or instance.query_scope_mode == "under_specified"
        )
        row = {
            "instance_id": instance.instance_id,
            "conflict_type": instance.conflict_type,
            "correct": judged["correct"],
            "conflict_recognised": judged["conflict_recognised"],
            "premise_rejected": judged["premise_rejected"],
            "stale_presupposition": instance.stale_presupposition,
            "retrieved_ids": trace.retrieved_ids,
            "supporting_ids": instance.supporting_ids,
            "hit": _supporting_hit(agent, trace, instance),
            "should_abstain": should_abstain,
            "did_abstain": "Governance note (abstain)" in trace.memory_block or "Governance note (clarify)" in trace.memory_block,
            "gold_conflict": not instance.is_control,
        }
        if elicit:
            conf = elicit_confidence(
                agent.llm,
                [
                    {"role": "system", "content": "You are the same agent."},
                    {"role": "user", "content": instance.query},
                    {"role": "assistant", "content": trace.action.content},
                ],
            )
            row["confidence"] = conf.get("confidence")
        rows.append(row)
        _reset_memory(agent)
    metrics = conflict_accuracy(rows)
    metrics["cr_f1"] = cr_f1(rows)
    metrics["spr"] = stale_premise_resistance(rows)
    metrics["hit_at_5"] = 100.0 * (sum(int(row.get("hit")) for row in rows) / max(1, len(rows)))
    prec, rec = abstention_pr(rows)
    metrics["abstain_precision"] = prec
    metrics["abstain_recall"] = rec
    conf_rows = [row for row in rows if row.get("confidence") is not None]
    if conf_rows:
        metrics["ece"] = ece(
            [float(row["confidence"]) for row in conf_rows],
            [int(row["correct"]) for row in conf_rows],
        )
    return {"metrics": metrics, "rows": rows}


def _query_context(instance: ConflictInstance) -> dict[str, str]:
    if not instance.condition_key:
        return {}
    if instance.query_scope_mode == "specified" and instance.scope_value:
        return {instance.condition_key: instance.scope_value}
    if instance.query_scope_mode == "violated" and (instance.scope_alt or instance.scope_value):
        return {instance.condition_key: instance.scope_alt or instance.scope_value}
    return {}


def _query_time(instance: ConflictInstance) -> datetime:
    if instance.sessions:
        return _parse(instance.sessions[-1].date)
    return datetime.now(timezone.utc)


def _supporting_hit(agent: ReActAgent, trace, instance: ConflictInstance) -> bool:
    texts = _retrieved_texts(agent, trace.retrieved_ids)
    supports = instance.supporting_texts or [instance.record_a, instance.record_b]
    for support in supports:
        if not support:
            continue
        needle = support.lower()[:80]
        if any(needle in text.lower() or text.lower()[:80] in support.lower() for text in texts):
            return True
    return False


def _retrieved_texts(agent: ReActAgent, ids: list[str]) -> list[str]:
    texts = []
    if hasattr(agent.memory, "layer"):
        for rid in ids:
            record = agent.memory.layer.store.get(rid)
            if record is not None:
                texts.append(record.content)
    elif hasattr(agent.memory, "rows"):
        index = {row["id"]: row["content"] for row in agent.memory.rows}
        texts.extend(index[rid] for rid in ids if rid in index)
    return texts


def _reset_memory(agent: ReActAgent) -> None:
    memory = agent.memory
    if hasattr(memory, "rows"):
        memory.rows.clear()
        if hasattr(memory, "vectors"):
            memory.vectors.clear()
    if hasattr(memory, "layer"):
        for record in list(memory.layer.store.all_records(include_quarantine=True)):
            memory.layer.store.delete(record.record_id)


def _parse(date: str) -> datetime:
    try:
        return datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)

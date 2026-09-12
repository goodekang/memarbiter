from __future__ import annotations

from typing import Callable

from src.llm.json_call import structured_call
from src.memarbiter.record import MemoryRecord
from src.memarbiter.tasr import AttributionResult, FlaggedStep
from src.prompts import STEP_JUDGE_FIELDS, STEP_JUDGE_SYSTEM


def llm_judge(client, step: FlaggedStep, records: list[MemoryRecord]) -> AttributionResult:
    rendered = []
    for record in records:
        rendered.append(f"{record.record_id}\tsource={record.source}\tdate={record.write_time.date().isoformat()}\t{record.content}")
    parsed = structured_call(
        client,
        STEP_JUDGE_SYSTEM,
        STEP_JUDGE_FIELDS,
        {
            "user_query": step.query,
            "retrieved_records": "\n".join(rendered),
            "agent_reasoning": "",
            "action": str(step.action),
            "verifier_message": "",
        },
        required=("ranked_records", "justification"),
    )
    ranked = [str(item) for item in parsed.get("ranked_records") or []]
    responsible = None
    for rid in ranked:
        for record in records:
            if record.record_id == rid:
                responsible = record
                break
        if responsible:
            break
    return AttributionResult(responsible, ranked[:3], 0, [], responsible is None)


def leave_one_out(
    step: FlaggedStep,
    records: list[MemoryRecord],
    reexecute: Callable,
    verifier: Callable,
) -> AttributionResult:
    examined = [record.record_id for record in records]
    n = 0
    for record in records:
        remaining = [row for row in records if row.record_id != record.record_id]
        action = reexecute(step.query, remaining)
        n += 1
        if verifier(action) == "ok":
            return AttributionResult(record, examined, n, [record.record_id], False)
    return AttributionResult(None, examined, n, [], True)


def lineage_ordering(step: FlaggedStep) -> AttributionResult:
    candidates = sorted(step.adopted, key=lambda row: (-row.taint, -row.write_time.timestamp()))
    responsible = candidates[0] if candidates else None
    ids = [row.record_id for row in candidates]
    return AttributionResult(responsible, ids[:3], 0, [responsible.record_id] if responsible else [], responsible is None)


def attention_mass(token_record_weights: list[list[float]], action_token_index: list[int]) -> list[float]:
    """Mean attention from action tokens to each record span, over layers."""
    if not token_record_weights:
        return []
    n_records = len(token_record_weights[0])
    totals = [0.0] * n_records
    count = 0
    for layer in token_record_weights:
        for index in action_token_index:
            if index >= len(layer):
                continue
            # layer here is already reduced to record spans for that token
            for r, value in enumerate(layer):
                totals[r] += float(value)
            count += 1
    if count == 0:
        return totals
    return [value / count for value in totals]

from __future__ import annotations

from src.llm.client import LLMClient
from src.llm.json_call import structured_call
from src.persist.schema import ConflictInstance
from src.prompts import CONFIDENCE_USER, JUDGE_FIELDS, JUDGE_SYSTEM


def judge_response(client: LLMClient, instance: ConflictInstance, response: str) -> dict:
    parsed = structured_call(
        client,
        JUDGE_SYSTEM,
        JUDGE_FIELDS,
        {
            "user_query": instance.query,
            "conflict_type": instance.conflict_type,
            "gold_value": instance.gold_value,
            "record_a": instance.record_a,
            "record_b": instance.record_b,
            "query_scope_mode": instance.query_scope_mode or "",
            "stale_presupposition": instance.stale_presupposition or "",
            "expected_behaviour": instance.expected_behaviour,
            "agent_response": response,
        },
        required=("correct", "conflict_recognised", "justification"),
    )
    return {
        "correct": bool(parsed.get("correct")),
        "conflict_recognised": bool(parsed.get("conflict_recognised")),
        "premise_rejected": parsed.get("premise_rejected"),
        "committed_value": parsed.get("committed_value"),
        "justification": parsed.get("justification"),
    }


def elicit_confidence(client: LLMClient, messages: list[dict[str, str]]) -> dict:
    follow = list(messages) + [{"role": "user", "content": CONFIDENCE_USER}]
    text = client.complete_conversation(follow)
    from src.llm.json_call import _loads

    try:
        parsed = _loads(text)
    except Exception:
        parsed = {"confidence": None, "relied_on": []}
    return parsed

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from src.llm.client import LLMClient
from src.llm.json_call import structured_call
from src.persist.schema import ConflictInstance, Session, Triple, Turn
from src.prompts import (
    DISTRACTOR_FIELDS,
    DISTRACTOR_SYSTEM,
    SYNTHESIS_FIELDS,
    SYNTHESIS_SYSTEM,
    TRIPLE_FIELDS,
    TRIPLE_SYSTEM,
)


def extract_triples(client: LLMClient, session: Session, known_slots: list[str]) -> list[Triple]:
    parsed = structured_call(
        client,
        TRIPLE_SYSTEM,
        TRIPLE_FIELDS,
        {
            "session_index": str(session.index),
            "session_date": session.date,
            "known_slots": ", ".join(known_slots),
            "session_turns": "\n".join(f"{turn.role}: {turn.text}" for turn in session.turns),
        },
        required=("triples",),
    )
    triples = []
    for row in parsed.get("triples") or []:
        triples.append(
            Triple(
                entity=str(row.get("entity") or "user"),
                slot=str(row.get("slot") or "other"),
                slot_type=str(row.get("slot_type") or "other"),
                value=str(row.get("value") or ""),
                valid_from=str(row.get("valid_from") or session.date),
                replaces=row.get("replaces"),
                conditions=dict(row.get("conditions") or {}),
                evidence=str(row.get("evidence") or ""),
            )
        )
    return triples


def align_timeline(triples: list[Triple]) -> dict[tuple[str, str], list[Triple]]:
    grouped: dict[tuple[str, str], list[Triple]] = {}
    for triple in triples:
        grouped.setdefault((triple.entity, triple.slot), []).append(triple)
    for key in grouped:
        grouped[key].sort(key=lambda item: item.valid_from)
    return grouped


def synthesise_conflict(
    client: LLMClient,
    *,
    history_id: str,
    instance_id: str,
    sessions: list[Session],
    timeline: list[Triple],
    target: Triple,
    conflict_type: str,
    static_source: str | None,
    authoritative_side: str | None,
    rng: random.Random,
) -> dict[str, Any]:
    nearby = sessions[max(0, len(sessions) - 6) :]
    parsed = structured_call(
        client,
        SYNTHESIS_SYSTEM,
        SYNTHESIS_FIELDS,
        {
            "timeline": _dump_triples(timeline),
            "nearby_sessions": _dump_sessions(nearby),
            "target_slot": f"{target.entity}.{target.slot} ({target.slot_type})",
            "current_value": target.value,
            "conflict_type": conflict_type,
            "static_source": static_source or "",
            "authoritative_side": authoritative_side or "",
        },
        required=("inserted_sessions", "queries", "gold_value", "expected_behaviour"),
    )
    return parsed


def paraphrase_distractor(
    client: LLMClient,
    record: str,
    slot_type: str,
    entity: str,
    aliases: list[str],
) -> str:
    parsed = structured_call(
        client,
        DISTRACTOR_SYSTEM,
        DISTRACTOR_FIELDS,
        {
            "record": record,
            "slot_type": slot_type,
            "target_entity": entity,
            "target_aliases": ", ".join(aliases),
        },
        required=("paraphrase", "mentions_target"),
    )
    if parsed.get("mentions_target"):
        raise ValueError("distractor mentions the target entity")
    return str(parsed["paraphrase"])


def interleave_distractors(
    sessions: list[Session],
    extras: list[Session],
    rng: random.Random,
    target_sessions: int,
) -> list[Session]:
    pool = list(extras)
    rng.shuffle(pool)
    merged = list(sessions)
    cursor = 0
    while len(merged) < target_sessions and cursor < len(pool):
        insert_at = rng.randint(0, len(merged))
        incoming = pool[cursor]
        incoming.index = insert_at
        incoming.source = "distractor"
        merged.insert(insert_at, incoming)
        cursor += 1
    for index, session in enumerate(merged):
        session.index = index
    return merged


def pick_query(payload: dict[str, Any], conflict_type: str, rng: random.Random, stale_frac: float) -> tuple[str, str | None, str | None]:
    queries = payload.get("queries") or {}
    if conflict_type == "conditional":
        mode = rng.choice(["specified", "under_specified", "violated"])
        key = {"specified": "specified", "under_specified": "under_specified", "violated": "violated"}[mode]
        return str(queries.get(key) or queries.get("query") or ""), mode, None
    query = str(queries.get("query") or "")
    stale = None
    if conflict_type == "dynamic" and rng.random() < stale_frac and queries.get("stale_premise"):
        query = str(queries["stale_premise"])
        stale = str(queries.get("stale_value") or "")
    return query, None, stale


def pack_instance(
    *,
    instance_id: str,
    history_id: str,
    conflict_type: str,
    sessions: list[Session],
    payload: dict[str, Any],
    query: str,
    query_mode: str | None,
    stale: str | None,
    authoritative_side: str | None,
    distractors: list[str],
) -> ConflictInstance:
    artifact = payload.get("inserted_artifact") or ""
    inserted = payload.get("inserted_sessions") or []
    record_a = artifact if conflict_type == "static" else _first_user_mention(inserted, 0)
    record_b = _first_user_mention(inserted, 1 if conflict_type == "conditional" else 0)
    if conflict_type == "dynamic":
        record_a = payload.get("old_value") or record_a
        record_b = payload.get("gold_value") or record_b
        if stale and not stale.strip():
            stale = str(payload.get("old_value") or record_a)
    n_tokens = sum(len(turn.text.split()) for session in sessions for turn in session.turns)
    positions = [int(row.get("position") or 0) for row in inserted] or [0]
    distance = max(positions) - min(positions) if len(positions) > 1 else max(positions)
    if conflict_type == "dynamic":
        supports = [str(record_b)]
    elif conflict_type == "static" and authoritative_side == "neither":
        supports = [str(record_a), str(record_b)]
    else:
        supports = [str(payload.get("gold_value") or record_b), str(record_a), str(record_b)]
    conditions = payload.get("condition_values") or {}
    return ConflictInstance(
        instance_id=instance_id,
        history_id=history_id,
        conflict_type=conflict_type,  # type: ignore[arg-type]
        sessions=sessions,
        record_a=str(record_a),
        record_b=str(record_b),
        gold_value=str(payload.get("gold_value") or ""),
        query=query,
        expected_behaviour=str(payload.get("expected_behaviour") or ""),
        supporting_ids=[],
        distractors=distractors,
        query_scope_mode=query_mode,
        condition_key=payload.get("condition_key"),
        scope_value=str(conditions.get("satisfied") or payload.get("scope_value") or ""),
        scope_alt=str(conditions.get("violated") or payload.get("scope_alt") or ""),
        stale_presupposition=stale,
        authoritative_side=authoritative_side,  # type: ignore[arg-type]
        supporting_texts=[text for text in supports if text],
        conflict_distance=int(distance),
        n_sessions=len(sessions),
        n_tokens=n_tokens,
    )


def apply_insertions(sessions: list[Session], payload: dict[str, Any], base_date: str) -> list[Session]:
    merged = list(sessions)
    for row in payload.get("inserted_sessions") or []:
        position = int(row.get("position") or len(merged))
        turns = [Turn(role=t.get("role", "user"), text=t.get("text", "")) for t in row.get("turns") or []]
        date = _shift_date(base_date, position)
        merged.insert(min(position, len(merged)), Session(index=position, date=date, turns=turns, source="inserted"))
    artifact = payload.get("inserted_artifact")
    if artifact:
        role = "document" if str(payload.get("static_source") or "") == "document" else "tool"
        merged.append(
            Session(
                index=len(merged),
                date=_shift_date(base_date, len(merged)),
                turns=[Turn(role=role, text=str(artifact))],
                source="artifact",
            )
        )
    for index, session in enumerate(merged):
        session.index = index
    return merged


def _first_user_mention(inserted: list[dict[str, Any]], which: int) -> str:
    texts = []
    for row in inserted:
        for turn in row.get("turns") or []:
            if turn.get("role") == "user":
                texts.append(str(turn.get("text") or ""))
    if which < len(texts):
        return texts[which]
    return texts[-1] if texts else ""


def _dump_triples(triples: list[Triple]) -> str:
    return "\n".join(
        f"{item.valid_from} {item.entity}.{item.slot}={item.value} conditions={item.conditions}"
        for item in triples
    )


def _dump_sessions(sessions: list[Session]) -> str:
    blocks = []
    for session in sessions:
        body = "\n".join(f"{turn.role}: {turn.text}" for turn in session.turns)
        blocks.append(f"session {session.index} ({session.date})\n{body}")
    return "\n\n".join(blocks)


def _shift_date(base: str, offset: int) -> str:
    try:
        moment = datetime.fromisoformat(base)
    except ValueError:
        moment = datetime(2026, 1, 1)
    return (moment + timedelta(days=offset)).date().isoformat()

from __future__ import annotations

import re
from typing import Iterable

SLOT_HINTS = {
    "home_city",
    "city",
    "employer",
    "dietary_preference",
    "meeting_time",
    "contact_email",
    "invoice",
    "recipient",
    "flight",
    "price",
}

_PROPER = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MONEY = re.compile(r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?")
_TIME = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")


def extract_entities(text: str) -> set[str]:
    found = {m.group(0).lower() for m in _EMAIL.finditer(text)}
    found.update(m.group(0).lower() for m in _MONEY.finditer(text))
    found.update(m.group(0) for m in _TIME.finditer(text))
    for match in _PROPER.finditer(text):
        token = match.group(1).strip()
        if token.lower() in {"i", "the", "a", "an"}:
            continue
        found.add(token.lower())
    lowered = text.lower()
    for hint in SLOT_HINTS:
        if hint.replace("_", " ") in lowered or hint in lowered:
            found.add(hint)
    return found


def entity_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def neighbor_score(
    embed_sim: float,
    overlap: float,
    embed_weight: float = 0.65,
) -> float:
    embed_weight = min(1.0, max(0.0, embed_weight))
    return embed_weight * embed_sim + (1.0 - embed_weight) * overlap

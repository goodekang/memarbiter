"""Query-context extraction for the read-time scorer (main text, read-time scoring).

The context psi is a set of key--value pairs obtained without an LLM call. For
every condition key that occurs in the scope predicates of the user's stored
records, the values recorded for that key are matched, after lemmatisation and
with a list of synonyms collected from the stored predicates, against the
current query and the preceding turns of the session. A key is set when exactly
one of its values is found; keys that are not found remain unspecified.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping

from src.memarbiter.record import MemoryRecord, SourceKind

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"a", "an", "the", "of", "on", "in", "at", "for", "to", "and", "or", "day", "days", "time"}


def lemmatise(token: str) -> str:
    """Light suffix-stripping lemmatiser (no external model, no LLM call)."""
    token = token.lower()
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)] + repl
    return token


def lemmas(text: str) -> list[str]:
    return [lemmatise(tok) for tok in _TOKEN.findall(text.lower().replace("_", " "))]


def _content_lemmas(text: str) -> tuple[str, ...]:
    return tuple(tok for tok in lemmas(text) if tok not in _STOP)


def collect_predicates(records: Iterable[MemoryRecord], *, user_only: bool = True) -> dict[str, set[str]]:
    """Condition keys and their recorded values from the stored scope predicates."""
    values: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if user_only and record.source_kind is not SourceKind.USER:
            continue
        for key, value in (record.scope or {}).items():
            values[str(key)].add(str(value))
    return dict(values)


def collect_synonyms(
    predicates: Mapping[str, set[str]],
    extra: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, set[tuple[str, ...]]]:
    """Surface forms per stored value.

    Synonyms are collected from the stored predicates themselves: the value as
    written, its underscore/hyphen variants, and the value with the key name
    attached or removed (``remote`` / ``remote_day`` / ``remote day``). An
    optional mapping ``extra`` adds forms for a value (e.g. loaded from a file).
    """
    forms: dict[str, set[tuple[str, ...]]] = {}
    for key, vals in predicates.items():
        key_lemmas = _content_lemmas(key)
        for value in vals:
            variants = {value, value.replace("_", " "), value.replace("-", " ")}
            base = _content_lemmas(value)
            stripped = tuple(tok for tok in base if tok not in key_lemmas)
            out = {_content_lemmas(v) for v in variants}
            if stripped:
                out.add(stripped)
            for alt in (extra or {}).get(value, []):
                out.add(_content_lemmas(alt))
            forms[value] = {form for form in out if form}
    return forms


def _contains(haystack: list[str], needle: tuple[str, ...]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    return any(tuple(haystack[i : i + n]) == needle for i in range(len(haystack) - n + 1))


def extract_query_context(
    query: str,
    preceding_turns: Iterable[str],
    records: Iterable[MemoryRecord],
    *,
    extra_synonyms: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, str]:
    """Lexical psi extraction: a key is set only when exactly one of its values is found."""
    predicates = collect_predicates(records)
    if not predicates:
        return {}
    forms = collect_synonyms(predicates, extra_synonyms)
    text = " ".join([*preceding_turns, query])
    haystack = lemmas(text)
    context: dict[str, str] = {}
    for key, vals in predicates.items():
        found = {value for value in vals if any(_contains(haystack, form) for form in forms.get(value, ()))}
        if len(found) == 1:
            context[key] = next(iter(found))
    return context

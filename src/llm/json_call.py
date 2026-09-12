from __future__ import annotations

import json
from typing import Any, Sequence

from src.llm.client import LLMClient
from src.prompts import format_user_payload


class SchemaError(ValueError):
    pass


def structured_call(
    client: LLMClient,
    system: str,
    fields: Sequence[str],
    values: dict[str, object],
    *,
    required: Sequence[str] | None = None,
    allowed: dict[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    user = format_user_payload(fields, values)
    text = client.complete(system, user)
    try:
        parsed = _loads(text)
        _validate(parsed, required=required, allowed=allowed)
        return parsed
    except (json.JSONDecodeError, SchemaError) as exc:
        retry = (
            user
            + "\n\nThe previous reply was not valid JSON or violated the schema: "
            + str(exc)
            + "\nReturn only a corrected JSON object."
        )
        parsed = _loads(client.complete(system, retry))
        _validate(parsed, required=required, allowed=allowed)
        return parsed


def _loads(text: str) -> dict[str, Any]:
    blob = text.strip()
    if blob.startswith("```"):
        blob = blob.strip("`")
        if blob.startswith("json"):
            blob = blob[4:]
        blob = blob.strip()
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end < 0:
        raise json.JSONDecodeError("no object", blob, 0)
    data = json.loads(blob[start : end + 1])
    if not isinstance(data, dict):
        raise SchemaError("top-level value is not an object")
    return data


def _validate(
    data: dict[str, Any],
    *,
    required: Sequence[str] | None,
    allowed: dict[str, Sequence[Any]] | None,
) -> None:
    for key in required or ():
        if key not in data:
            raise SchemaError(f"missing field {key}")
    for key, options in (allowed or {}).items():
        if key in data and data[key] not in options:
            raise SchemaError(f"{key}={data[key]!r} not in {list(options)}")

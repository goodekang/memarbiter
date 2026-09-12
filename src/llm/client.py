from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

import httpx

from src.config import load_yaml


@dataclass(frozen=True)
class LLMConfig:
    name: str
    provider: str
    api_model: str
    logit_access: bool = False
    temperature: float = 0.0
    reasoning_effort: str = "lowest"
    max_tokens: int = 2048


class LLMClient:
    """Deterministic chat completion used by the planner, escalations and judges."""

    def __init__(self, cfg: LLMConfig, timeout: float = 120.0) -> None:
        self.cfg = cfg
        catalog = load_yaml("backbones.yaml")
        endpoint = catalog["endpoints"][cfg.provider]
        self.base_url = os.environ.get(endpoint["base_url_env"], _default_url(cfg.provider))
        self.api_key = os.environ.get(endpoint["api_key_env"], "")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        logprobs: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.cfg.api_model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if extra:
            body.update(extra)
        if self.cfg.reasoning_effort:
            body.setdefault("reasoning_effort", self.cfg.reasoning_effort)
        if logprobs and self.cfg.logit_access:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        if self.cfg.provider == "anthropic":
            return self._chat_anthropic(messages, temperature)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = self.base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()

    def _chat_anthropic(self, messages: list[dict[str, str]], temperature: float | None) -> dict[str, Any]:
        system = "\n".join(item["content"] for item in messages if item["role"] == "system")
        converted = []
        for item in messages:
            if item["role"] == "system":
                continue
            converted.append({"role": item["role"], "content": item["content"]})
        payload = {
            "model": self.cfg.api_model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "system": system,
            "messages": converted or [{"role": "user", "content": ""}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        url = self.base_url.rstrip("/") + "/messages"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        text = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
        return {"choices": [{"message": {"content": text}}]}

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        payload = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return _message_text(payload)

    def complete_conversation(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return _message_text(self.chat(messages, **kwargs))


def build_client(name: str, **overrides: Any) -> LLMClient:
    catalog = load_yaml("backbones.yaml")["models"]
    if name not in catalog:
        raise KeyError(f"unknown backbone {name}")
    row = dict(catalog[name])
    cfg = LLMConfig(
        name=name,
        provider=row["provider"],
        api_model=row["api_model"],
        logit_access=bool(row.get("logit_access", False)),
        **{k: v for k, v in overrides.items() if k in LLMConfig.__annotations__},
    )
    return LLMClient(cfg)


def _message_text(payload: dict[str, Any]) -> str:
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def _default_url(provider: str) -> str:
    return {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "together": "https://api.together.xyz/v1",
    }[provider]


def iter_logprobs(payload: dict[str, Any]) -> Iterable[list[tuple[str, float]]]:
    choice = payload.get("choices", [{}])[0]
    content = (choice.get("logprobs") or {}).get("content") or []
    for token in content:
        tops = token.get("top_logprobs") or []
        yield [(item["token"], float(item["logprob"])) for item in tops]

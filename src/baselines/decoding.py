from __future__ import annotations

from typing import Iterable

from src.llm.client import LLMClient


CAD_SYSTEM = (
    "When the retrieved memories disagree with what you would otherwise assume, "
    "prefer the memories. Do not smooth over a contradiction by falling back to "
    "parametric knowledge. If the memories themselves disagree, say so."
)

ADACAD_SYSTEM = (
    "Compare your prior answer distribution with the memories. If they disagree, "
    "amplify the memory-supported tokens; if they agree, do not over-weight the context. "
    "If the memories are internally inconsistent, describe the disagreement rather than guessing."
)


def cad_messages(memory_block: str, query: str, adaptive: bool = False) -> list[dict[str, str]]:
    system = ADACAD_SYSTEM if adaptive else CAD_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{memory_block}\n\nUser: {query}"},
    ]


def context_aware_complete(client: LLMClient, memory_block: str, query: str, adaptive: bool = False) -> str:
    """Prompt-side CAD / AdaCAD used for API models without logit access."""
    return client.complete_conversation(cad_messages(memory_block, query, adaptive=adaptive))


def logit_reweight(
    prior_logprobs: Iterable[list[tuple[str, float]]],
    context_logprobs: Iterable[list[tuple[str, float]]],
    alpha: float = 1.0,
) -> list[str]:
    """Open-weight CAD: amplify tokens where context and prior disagree."""
    tokens = []
    for prior, ctx in zip(prior_logprobs, context_logprobs):
        prior_map = dict(prior)
        ctx_map = dict(ctx)
        keys = set(prior_map) | set(ctx_map)
        best = None
        best_score = -1e9
        for key in keys:
            p = prior_map.get(key, -20.0)
            c = ctx_map.get(key, -20.0)
            score = c + alpha * (c - p)
            if score > best_score:
                best_score = score
                best = key
        tokens.append(best or "")
    return tokens

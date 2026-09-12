from __future__ import annotations

from src.agent.react import MemoryBackend
from src.baselines.defenses import (
    agentsentry_purify,
    agentsys_summarise,
    drift_isolate,
    melon_mask,
    paraphrase,
    perplexity_filter,
    spotlighting,
    tool_filter,
)
from src.baselines.memory_systems import (
    AMEMMemory,
    BM25Memory,
    DenseMemory,
    FullContextMemory,
    Mem0Memory,
    MemGPTMemory,
    MemoryBankMemory,
    SodaMemMemory,
)
from src.agent.react import MemArbiterBackend
from src.memarbiter.embeddings import EmbeddingClient
from src.memarbiter.layer import build_memarbiter


def build_memory(name: str, cfg: dict, llm, embeddings: EmbeddingClient | None = None) -> MemoryBackend:
    budget = int(cfg.get("memory_budget_tokens") or cfg.get("eval", {}).get("memory_budget_tokens") or 4096)
    embeddings = embeddings or EmbeddingClient()
    if name == "memarbiter":
        layer = build_memarbiter(cfg.get("memarbiter", cfg), llm, embeddings)
        return MemArbiterBackend(layer, budget=budget)
    if name == "full_context":
        return FullContextMemory(budget)
    if name == "bm25":
        return BM25Memory(budget)
    if name in {"dense", "cad", "adacad"}:
        return DenseMemory(embeddings, budget)
    if name == "memgpt":
        return MemGPTMemory(budget)
    if name == "memorybank":
        return MemoryBankMemory(budget)
    if name == "mem0":
        return Mem0Memory(budget)
    if name == "amem":
        return AMEMMemory(embeddings, budget)
    if name == "sodamem":
        return SodaMemMemory(embeddings, budget)
    raise KeyError(f"unknown memory system {name}")


def build_defense(name: str | None, llm):
    if not name or name in {"none", "no_defence"}:
        return None
    return {
        "spotlighting": spotlighting,
        "paraphrase": paraphrase(llm),
        "perplexity_filter": perplexity_filter(llm),
        "tool_filter": tool_filter,
        "melon": melon_mask,
        "drift": drift_isolate,
        "agentsys": agentsys_summarise(llm),
        "agentsentry": agentsentry_purify(llm),
    }[name]

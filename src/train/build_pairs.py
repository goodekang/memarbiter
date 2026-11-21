from __future__ import annotations

import json
import random
from pathlib import Path

from src.llm.client import LLMClient
from src.persist.conflict import extract_triples
from src.persist.longmemeval import as_sessions, load_histories
from src.memarbiter.nli import format_pair_text
from src.train.gate_data import (
    ENVIRONMENTS,
    build_injection_rows,
    collect_agentdojo_benign,
    load_heldout_benign_extra,
    load_injection_specs,
    summarise,
)
from src.train.nli_dataset import write_pair_pool


def build_nli_pairs(persist_cfg: dict, client: LLMClient, output: Path, n_pairs: int, seed: int) -> int:
    """Label 4-way pairs from the held-out LongMemEval distractor pool."""
    histories = [row for row in load_histories(persist_cfg["paths"]["raw_longmemeval"])]
    rng = random.Random(seed)
    rng.shuffle(histories)
    rows = []
    for history in histories:
        history["nli_train"] = True
        sessions = as_sessions(history)
        triples = []
        known: list[str] = []
        for session in sessions:
            triples.extend(extract_triples(client, session, known))
            known = sorted({item.slot for item in triples})
        if len(triples) < 2:
            continue
        for left, right in zip(triples, triples[1:]):
            if left.entity != right.entity or left.slot != right.slot:
                continue
            label = _heuristic_relation(left, right)
            rows.append(
                {
                    "text_a": format_pair_text(_triple_text(left), "user", left.valid_from, left.evidence),
                    "text_b": format_pair_text(_triple_text(right), "user", right.valid_from, right.evidence),
                    "label": label,
                    "history_id": history.get("history_id"),
                }
            )
            if len(rows) >= n_pairs:
                write_pair_pool(output, rows)
                _mark_nli_train(persist_cfg["paths"]["raw_longmemeval"], {row["history_id"] for row in rows})
                return len(rows)
    write_pair_pool(output, rows)
    return len(rows)


def build_instruction_pairs(
    persist_cfg: dict,
    client: LLMClient,
    output: Path,
    gate_cfg: dict,
) -> int:
    """Build the quarantine-gate pool (Appendix A.1).

    Benign rows are tool returns logged while executing the AgentDojo user
    tasks (ground-truth calls) plus seed-data items of the four environments;
    injection rows are LLM rewrites of the AgentDojo injection tasks and ASB
    attack tools in three styles and several carriers. Every row carries its
    group (attack specification / user task / template family) so that
    ``train_instruction.py`` can form partitions by group. Realistic benign
    texts that address the reader are appended as test-only rows.
    """
    seed = int(gate_cfg["seed"])
    benign = collect_agentdojo_benign(
        str(gate_cfg.get("agentdojo_version", "v1")),
        gate_cfg.get("environments", list(ENVIRONMENTS)),
        run_logs=gate_cfg.get("agentdojo_run_logs"),
    )
    specs = load_injection_specs(persist_cfg, gate_cfg)
    if not specs:
        raise FileNotFoundError("no AgentDojo injection tasks or ASB attack tools found")
    injections = build_injection_rows(
        specs,
        client,
        carriers_per_spec=int(gate_cfg.get("carriers_per_spec", 4)),
        topics=list(gate_cfg.get("carrier_topics") or ["routine office work"]),
        seed=seed,
    )
    extra = load_heldout_benign_extra(gate_cfg.get("heldout_benign_extra"))
    rows = benign + injections + extra
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summarise(rows), indent=2))
    return len(rows)


def _heuristic_relation(left, right) -> str:
    if left.conditions or right.conditions:
        if left.conditions != right.conditions and left.value != right.value:
            return "CONDITIONAL"
    if left.replaces or right.replaces or left.valid_from != right.valid_from:
        if left.value != right.value:
            return "SUPERSEDES"
    if left.value != right.value:
        return "CONTRADICTS"
    return "CONSISTENT"


def _triple_text(triple) -> str:
    cond = ""
    if triple.conditions:
        cond = " if " + ", ".join(f"{k}={v}" for k, v in triple.conditions.items())
    return f"{triple.entity}.{triple.slot} = {triple.value}{cond} (as of {triple.valid_from})"


def _mark_nli_train(root: str, history_ids: set[str]) -> None:
    path = Path(root) / "nli_train_histories.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(history_ids), indent=2), encoding="utf-8")

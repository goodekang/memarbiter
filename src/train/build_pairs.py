from __future__ import annotations

import json
import random
from pathlib import Path

from src.llm.client import LLMClient
from src.persist.conflict import extract_triples
from src.persist.longmemeval import as_sessions, load_histories
from src.persist.poison import load_attack_specs, rewrite_injection
from src.memarbiter.nli import format_pair_text
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
    n_tool: int,
    n_injection: int,
    seed: int,
) -> int:
    rng = random.Random(seed)
    specs = load_attack_specs(persist_cfg["paths"].get("raw_agentdojo"))
    rows = []
    for spec in specs:
        if len([row for row in rows if row["label"] == 1]) >= n_injection:
            break
        try:
            rewritten = rewrite_injection(
                client,
                str(spec.get("attack_specification")),
                f"tool output in {spec.get('environment')}",
            )
        except Exception:
            continue
        for text in rewritten.values():
            rows.append({"text": text, "label": 1, "source": spec.get("injection_id")})
            if len([row for row in rows if row["label"] == 1]) >= n_injection:
                break
    tool_templates = [
        "Calendar event: weekly invoice review with finance, room 4B.",
        "Email from Dana Whitfield: please send the March statement when you can.",
        "Flight UA 188 departed on time; arrival gate B12.",
        "Balance inquiry returned 1840.22 available.",
        "Slack message from #general: office closed Monday for maintenance.",
        "Web page title: Q2 travel policy, receipts due within 14 days.",
    ]
    while len([row for row in rows if row["label"] == 0]) < n_tool:
        rows.append({"text": rng.choice(tool_templates), "label": 0, "source": "tool"})
    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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

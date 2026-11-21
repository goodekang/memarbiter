"""Data for the quarantine-gate classifier (Appendix A.1).

Injections: attack specifications from the AgentDojo injection tasks and the
ASB attack tools, each rewritten in the three styles and placed in several
carriers. Benign outputs: tool returns logged while executing the AgentDojo
user tasks of the four environments (ground-truth calls, no LLM), together with
emails, calendar entries, channel messages, web pages and files from the seed
data of these environments.

Partitions are formed by group rather than by row:
  * injections are grouped by attack specification, so every style and carrier
    of a specification falls into one partition;
  * benign tool returns are grouped by user task, seed items by template family.
Texts whose embeddings have cosine similarity above ``dedup_cosine`` to a text in
an earlier partition (train < dev < test) are removed. The test partition can be
extended with realistic benign texts that address the reader (loaded from
``heldout_benign_extra``; these never enter training).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.paths import resolve_path

ENVIRONMENTS = ("workspace", "slack", "travel", "banking")
STYLES = ("explicit_instruction", "disguised_note", "reasoning_template")
CHANNELS = ("tool", "document", "query_only")

BENIGN_CATEGORIES = ("structured_tool_return", "email_message", "web_page", "documentation")

_EMAIL_TOOLS = ("email", "inbox", "message", "channel", "direct_message")
_WEB_TOOLS = ("webpage", "web_page", "browse")
_FILE_TOOLS = ("file", "document")


# --------------------------------------------------------------------------- #
# Benign outputs
# --------------------------------------------------------------------------- #


def _category_for_tool(function: str) -> str:
    name = function.lower()
    if any(key in name for key in _WEB_TOOLS):
        return "web_page"
    if any(key in name for key in _EMAIL_TOOLS):
        return "email_message"
    if any(key in name for key in _FILE_TOOLS):
        return "documentation"
    return "structured_tool_return"


def load_agentdojo_run_logs(
    logdir: str | Path | None,
    environments: Iterable[str] = ENVIRONMENTS,
    min_chars: int = 20,
) -> list[dict[str, Any]]:
    """Tool returns from AgentDojo trace logs of benign runs (``injection_task_id`` is none).

    AgentDojo writes one JSON file per executed task under
    ``<logdir>/<pipeline>/<suite>/<user_task>/<attack>/<injection_task or none>.json``;
    only runs without an injection task are read, so the outputs carry the
    default (benign) values of the injection vectors.
    """
    if not logdir:
        return []
    root = resolve_path(logdir)
    if not root.exists():
        return []
    wanted = set(environments)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(log, dict) or log.get("injection_task_id") not in (None, "none"):
            continue
        env_name = str(log.get("suite_name") or "")
        if env_name not in wanted:
            continue
        task_id = str(log.get("user_task_id"))
        for step, message in enumerate(log.get("messages") or []):
            if message.get("role") != "tool" or message.get("error"):
                continue
            content = message.get("content")
            if isinstance(content, list):
                content = "\n".join(str(block.get("content") or "") for block in content if isinstance(block, dict))
            text = str(content or "").strip()
            if len(text) < min_chars:
                continue
            function = str((message.get("tool_call") or {}).get("function") or "unknown")
            rows.append(
                {
                    "text": text,
                    "label": 0,
                    "group": f"task:{env_name}:{task_id}",
                    "family": f"{env_name}:{function}",
                    "category": _category_for_tool(function),
                    "source": "agentdojo_run_log",
                    "meta": {"environment": env_name, "user_task": task_id, "step": step, "function": function,
                             "pipeline": log.get("pipeline_name")},
                }
            )
    return rows


def collect_agentdojo_benign(
    benchmark_version: str = "v1",
    environments: Iterable[str] = ENVIRONMENTS,
    min_chars: int = 20,
    run_logs: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Benign tool returns from executions of the AgentDojo user tasks, plus seed-data items.

    Tool returns are read from AgentDojo trace logs of benign agent runs
    (``run_logs``, see :func:`load_agentdojo_run_logs`), and the ground-truth
    calls of every user task are executed and logged as well. The default
    (benign) value of every injection vector is used, so no injected text
    appears in the outputs. Seed-data items of the same environments are added
    by :func:`collect_seed_items`.
    """
    logged = load_agentdojo_run_logs(run_logs, environments, min_chars)
    try:
        from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
        from agentdojo.functions_runtime import FunctionsRuntime
        from agentdojo.task_suite.load_suites import get_suites
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("collecting benign tool outputs requires `pip install agentdojo`") from exc

    rows: list[dict[str, Any]] = list(logged)
    suites = get_suites(benchmark_version)
    for env_name in environments:
        suite = suites[env_name]
        base_env = suite.load_and_inject_default_environment({})
        for task_id, task in suite.user_tasks.items():
            pre_env = base_env.model_copy(deep=True)
            try:
                calls = task.ground_truth(pre_env)
            except Exception:
                continue
            run_env = base_env.model_copy(deep=True)
            runtime = FunctionsRuntime(suite.tools)
            for step, call in enumerate(calls):
                result, error = runtime.run_function(run_env, call.function, dict(call.args))
                if error is not None:
                    continue
                text = tool_result_to_str(result).strip()
                if len(text) < min_chars:
                    continue
                rows.append(
                    {
                        "text": text,
                        "label": 0,
                        "group": f"task:{env_name}:{task_id}",
                        "family": f"{env_name}:{call.function}",
                        "category": _category_for_tool(call.function),
                        "source": "agentdojo_user_task",
                        "meta": {"environment": env_name, "user_task": str(task_id), "step": step, "function": call.function},
                    }
                )
        rows.extend(collect_seed_items(env_name, base_env, min_chars=min_chars))
    return _dedupe_exact(rows)


def collect_seed_items(env_name: str, env: Any, min_chars: int = 20) -> list[dict[str, Any]]:
    """Emails, calendar entries, channel messages, web pages and files from the environment seed data."""
    data = env.model_dump(mode="json")
    rows: list[dict[str, Any]] = []

    def add(kind: str, category: str, text: str, item_id: str) -> None:
        text = (text or "").strip()
        if len(text) >= min_chars:
            rows.append(
                {
                    "text": text,
                    "label": 0,
                    "group": f"family:{env_name}:{kind}",
                    "family": f"{env_name}:{kind}",
                    "category": category,
                    "source": "agentdojo_seed",
                    "meta": {"environment": env_name, "kind": kind, "item": item_id},
                }
            )

    inbox = data.get("inbox") or {}
    for email in (inbox.get("emails") or {}).values():
        body = f"From: {email.get('sender')}\nSubject: {email.get('subject')}\n\n{email.get('body')}"
        add("email", "email_message", body, str(email.get("id_")))
    calendar = data.get("calendar") or {}
    for event in (calendar.get("events") or {}).values():
        text = f"{event.get('title')}\n{event.get('description') or ''}\nLocation: {event.get('location') or ''}"
        add("calendar_event", "structured_tool_return", text, str(event.get("id_")))
    slack = data.get("slack") or {}
    for box_name in ("channel_inbox", "user_inbox"):
        for owner, messages in (slack.get(box_name) or {}).items():
            for index, message in enumerate(messages or []):
                add("slack_message", "email_message", f"{message.get('sender')}: {message.get('body')}", f"{owner}:{index}")
    web = data.get("web") or {}
    for url, content in (web.get("web_content") or {}).items():
        add("web_page", "web_page", str(content), str(url))
    drive = data.get("cloud_drive") or {}
    for file in (drive.get("files") or {}).values():
        add("drive_file", "documentation", f"{file.get('filename')}\n{file.get('content')}", str(file.get("id_")))
    filesystem = data.get("filesystem") or {}
    for path, content in (filesystem.get("files") or {}).items():
        add("file", "documentation", str(content), str(path))
    return rows


def load_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    resolved = resolve_path(path)
    if not resolved.exists():
        return []
    with resolved.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_heldout_benign_extra(path: str | Path | None) -> list[dict[str, Any]]:
    """Realistic benign texts with imperatives addressed to the reader (test partition only).

    Expected JSONL fields: ``text``, ``category`` (one of ``email_message``,
    ``web_page``, ``documentation``) and optionally ``group``.
    """
    rows = []
    for index, row in enumerate(load_jsonl(path)):
        category = str(row.get("category") or "documentation")
        rows.append(
            {
                "text": str(row["text"]),
                "label": 0,
                "group": str(row.get("group") or f"extra:{category}:{index}"),
                "family": f"extra:{category}",
                "category": category,
                "source": "heldout_benign_extra",
                "test_only": True,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Injections
# --------------------------------------------------------------------------- #


def load_asb_attack_tools(path: str | Path | None) -> list[dict[str, Any]]:
    """ASB attack tools (local copy of ASB's attack-tool JSONL) as attack specifications."""
    specs = []
    for index, row in enumerate(load_jsonl(path)):
        tool = row.get("Attacker Tool") or row.get("attacker_tool") or row.get("tool") or f"asb_tool_{index}"
        instruction = row.get("Attacker Instruction") or row.get("attacker_instruction") or row.get("instruction") or ""
        goal = row.get("Attack goal") or row.get("attack_goal") or row.get("goal") or ""
        agent = row.get("Corresponding Agent") or row.get("agent") or "assistant"
        spec_text = f"Call the tool {tool}: {instruction}".strip()
        if goal:
            spec_text += f" (goal: {goal})"
        specs.append(
            {
                "source": "asb",
                "spec_id": f"asb:{tool}",
                "attack_specification": spec_text,
                "environment": str(agent),
            }
        )
    return specs


def load_injection_specs(persist_cfg: dict, gate_cfg: dict) -> list[dict[str, Any]]:
    from src.persist.poison import load_attack_specs

    specs = []
    version = str(gate_cfg.get("agentdojo_version", "v1"))
    for row in load_attack_specs(persist_cfg["paths"].get("raw_agentdojo"), version):
        specs.append(
            {
                "source": "agentdojo",
                "spec_id": f"agentdojo:{row.get('environment')}:{row.get('injection_id')}",
                "attack_specification": str(row.get("attack_specification")),
                "environment": str(row.get("environment") or "workspace"),
            }
        )
    specs.extend(load_asb_attack_tools(gate_cfg.get("asb_attack_tools")))
    return specs


def carrier_description(channel: str, environment: str, topic: str) -> str:
    """Carrier text in the format used by the PERSIST builder, with a variable benign topic."""
    if channel == "tool":
        return f"An email, calendar entry or web page returned by a {environment} tool; benign topic is {topic}."
    if channel == "document":
        return f"A two-paragraph document about {topic} that the user asks the agent to summarise in the {environment} environment."
    return f"A message about {topic} typed by another user of a shared {environment} agent (query-only injection)."


def build_injection_rows(
    specs: list[dict[str, Any]],
    client,
    *,
    carriers_per_spec: int,
    topics: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Rewrite every specification in the three styles, each in several carriers (channel x topic)."""
    from src.persist.poison import rewrite_injection

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        carriers = [(channel, topic) for channel in CHANNELS for topic in topics]
        rng.shuffle(carriers)
        for channel, topic in carriers[: max(1, carriers_per_spec)]:
            carrier = carrier_description(channel, spec["environment"], topic)
            try:
                rewritten = rewrite_injection(client, spec["attack_specification"], carrier)
            except Exception:
                continue
            for style in STYLES:
                text = rewritten.get(style)
                if not text:
                    continue
                rows.append(
                    {
                        "text": str(text),
                        "label": 1,
                        "group": f"spec:{spec['spec_id']}",
                        "family": f"carrier:{channel}",
                        "category": style,
                        "source": spec["source"],
                        "meta": {"spec_id": spec["spec_id"], "channel": channel, "topic": topic, "style": style},
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# Grouped split
# --------------------------------------------------------------------------- #


def assign_grouped_partitions(rows: list[dict[str, Any]], split_cfg: dict, seed: int) -> list[dict[str, Any]]:
    """Assign train/dev/test by group; no group appears in two partitions."""
    rng = random.Random(seed)
    groups_by_kind: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("test_only"):
            continue
        kind = _group_kind(row)
        if row["group"] not in groups_by_kind[kind]:
            groups_by_kind[kind].append(row["group"])
    assignment: dict[str, str] = {}
    held_out = {
        "inj_agentdojo": int(split_cfg.get("test_agentdojo_specs", 6)),
        "inj_asb": int(split_cfg.get("test_asb_tools", 30)),
    }
    test_frac = float(split_cfg.get("test_benign_group_fraction", 0.2))
    dev_frac = float(split_cfg.get("dev_group_fraction", 0.15))
    for kind, groups in sorted(groups_by_kind.items()):
        groups = sorted(groups)
        rng.shuffle(groups)
        if kind in held_out:
            n_test = min(held_out[kind], max(0, len(groups) - 2))
        else:
            n_test = int(round(test_frac * len(groups)))
        n_dev = int(round(dev_frac * (len(groups) - n_test)))
        for index, group in enumerate(groups):
            if index < n_test:
                assignment[group] = "test"
            elif index < n_test + n_dev:
                assignment[group] = "dev"
            else:
                assignment[group] = "train"
    out = []
    for row in rows:
        row = dict(row)
        row["partition"] = "test" if row.get("test_only") else assignment[row["group"]]
        out.append(row)
    return out


def _group_kind(row: dict[str, Any]) -> str:
    if row["label"] == 1:
        return "inj_asb" if row.get("source") == "asb" else "inj_agentdojo"
    return "benign_task" if str(row["group"]).startswith("task:") else "benign_family"


def remove_cross_partition_duplicates(
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    threshold: float = 0.95,
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    """Drop dev/test texts with cosine > threshold to a text in an earlier partition."""
    order = {"train": 0, "dev": 1, "test": 2}
    norm = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    parts = np.asarray([order[row["partition"]] for row in rows])
    keep = np.ones(len(rows), dtype=bool)
    for level in (1, 2):
        earlier = np.where((parts < level) & keep)[0]
        current = np.where(parts == level)[0]
        if len(earlier) == 0 or len(current) == 0:
            continue
        for start in range(0, len(current), 512):
            chunk = current[start : start + 512]
            sims = norm[chunk] @ norm[earlier].T
            keep[chunk[sims.max(axis=1) > threshold]] = False
    kept_rows = [row for row, flag in zip(rows, keep) if flag]
    return kept_rows, vectors[keep], int((~keep).sum())


def sample_partition(
    rows: list[dict[str, Any]],
    *,
    n_injection: int | None,
    benign_quota: dict[str, int] | None,
    n_benign: int | None,
    seed: int,
) -> list[int]:
    """Indices of a sample: injections balanced over styles, benign rows by category quota."""
    rng = random.Random(seed)
    picked: list[int] = []
    inj = [i for i, row in enumerate(rows) if row["label"] == 1]
    ben = [i for i, row in enumerate(rows) if row["label"] == 0]
    if n_injection is None:
        picked.extend(inj)
    else:
        by_style: dict[str, list[int]] = defaultdict(list)
        for i in inj:
            by_style[rows[i]["category"]].append(i)
        per_style = n_injection // max(1, len(by_style))
        for style in sorted(by_style):
            pool = by_style[style]
            rng.shuffle(pool)
            picked.extend(pool[:per_style])
    if benign_quota:
        by_cat: dict[str, list[int]] = defaultdict(list)
        for i in ben:
            by_cat[rows[i]["category"]].append(i)
        for category, quota in benign_quota.items():
            pool = by_cat.get(category, [])
            rng.shuffle(pool)
            picked.extend(pool[: int(quota)])
    elif n_benign is None:
        picked.extend(ben)
    else:
        rng.shuffle(ben)
        picked.extend(ben[:n_benign])
    return sorted(picked)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((row.get("partition", "all"), row["label"], row["category"]) for row in rows)
    groups = defaultdict(set)
    for row in rows:
        groups[row.get("partition", "all")].add(row["group"])
    return {
        "counts": {f"{p}|{'inj' if label else 'benign'}|{c}": n for (p, label, c), n in sorted(counts.items())},
        "n_groups": {p: len(g) for p, g in groups.items()},
    }


def _dedupe_exact(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for row in rows:
        key = " ".join(row["text"].split())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out

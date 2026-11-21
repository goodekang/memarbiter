from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.react import ReActAgent
from src.agent.verifier import PlanConsistencyVerifier
from src.baselines.registry import build_defense, build_memory
from src.config import load_stack, load_yaml
from src.eval.conflict_eval import evaluate_conflict
from src.eval.poison_eval import evaluate_poison
from src.eval.public_eval import evaluate_agentdojo, evaluate_public_memory
from src.llm.client import build_client
from src.persist.io import load_conflicts, load_poisons


def apply_ablation(cfg: dict, variant: str | None) -> dict:
    if not variant or variant == "full":
        return cfg
    # Several variants can be combined with commas and are applied in order,
    # e.g. "original_rules,no_scorer" for a Table 10 row under the main-table rules.
    variants = load_yaml("ablations.yaml")["variants"]
    for name in [part.strip() for part in variant.split(",") if part.strip()]:
        if name not in variants:
            raise KeyError(f"unknown ablation {name}")
        cfg["memarbiter"].setdefault("ablation", {}).update(variants[name])
    return cfg


def apply_prior(cfg: dict, prior: str | None) -> dict:
    if not prior or prior == "hand_set":
        return cfg
    priors = load_yaml("priors.yaml")
    if prior == "learned":
        from pathlib import Path

        path = Path(priors["learned_path"])
        if path.exists():
            cfg["memarbiter"]["source_priors"] = load_yaml(path)
        return cfg
    if prior not in priors:
        raise KeyError(f"unknown prior {prior}")
    cfg["memarbiter"]["source_priors"] = priors[prior]
    return cfg


def run_suite(
    suite: str,
    *,
    system: str,
    backbone: str,
    defense: str | None,
    seed: int,
    cfg: dict | None = None,
    ablation: str | None = None,
    prior: str | None = None,
    repair: bool = True,
    public_subset: str = "overall",
    sensitivity_axis: str | None = None,
    attribution_mode: str = "tasr",
    repair_mode: str | None = None,
    query_context: str = "annotated",
) -> dict[str, Any]:
    from src.paths import resolve_path

    cfg = cfg or load_stack()
    apply_ablation(cfg, ablation)
    apply_prior(cfg, prior)
    eval_cfg = cfg["eval"]
    llm = build_client(backbone)
    judge = build_client(eval_cfg["judge"]["model"])
    planner = system if system in {"cad", "adacad"} else "default"
    memory = build_memory(
        system,
        {
            **cfg,
            "memarbiter": cfg["memarbiter"],
            "memory_budget_tokens": eval_cfg["memory_budget_tokens"],
        },
        llm,
    )
    use_repair = bool(repair) and system == "memarbiter"
    agent = ReActAgent(
        llm,
        memory,
        PlanConsistencyVerifier(),
        defense=build_defense(defense, llm),
        repair=use_repair,
        planner=planner,
    )
    if suite == "conflict":
        instances = load_conflicts(resolve_path(eval_cfg["suites"]["conflict"]["split"]))
        result = evaluate_conflict(
            agent,
            judge,
            instances,
            elicit=bool(eval_cfg.get("confidence_elicitation")),
            context_mode=query_context,
        )
    elif suite == "poison":
        trajectories = load_poisons(resolve_path(eval_cfg["suites"]["poison"]["split"]))
        mode = repair_mode or ("tasr" if use_repair else "none")
        result = evaluate_poison(
            agent,
            trajectories,
            repair_mode=mode,
            reissue_after_repair=bool(eval_cfg["suites"]["poison"]["reissue_trigger_after_repair"]) and mode != "none",
        )
    elif suite == "public":
        path = resolve_path(eval_cfg["suites"]["public"].get("split") or eval_cfg["suites"]["conflict"]["split"])
        instances = load_conflicts(path)
        result = evaluate_public_memory(agent, judge, instances, public_subset)
    elif suite == "agentdojo":
        from src.eval.public_eval import load_agentdojo_cases

        cases = load_agentdojo_cases(resolve_path(eval_cfg["suites"]["public"].get("agentdojo_split", "data/public/agentdojo.jsonl")))
        result = evaluate_agentdojo(agent, cases, lambda target, env: bool(env.get("attack")))
    elif suite == "ablation":
        instances = load_conflicts(resolve_path(eval_cfg["suites"]["conflict"]["split"]))
        result = evaluate_conflict(agent, judge, instances)
    elif suite == "sensitivity":
        instances = load_conflicts(resolve_path(eval_cfg["suites"]["conflict"]["split"]))
        instances = _apply_sensitivity(instances, sensitivity_axis)
        result = evaluate_conflict(agent, judge, instances)
    elif suite == "attribution":
        trajectories = load_poisons(resolve_path(eval_cfg["suites"]["poison"]["split"]))
        result = evaluate_poison(
            agent,
            trajectories,
            attribution_mode=attribution_mode,
            repair_mode="tasr" if use_repair else "none",
            judge_client=judge,
        )
    else:
        raise KeyError(f"unknown suite {suite}")
    result["meta"] = {
        "suite": suite,
        "system": system,
        "backbone": backbone,
        "query_context": query_context,
        "defense": defense,
        "seed": seed,
        "ablation": ablation,
        "prior": prior,
    }
    return result


def write_result(result: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = result.get("meta") or {}
    name = f"{meta.get('suite')}_{meta.get('system')}_{meta.get('backbone')}_s{meta.get('seed')}.json"
    path = output_dir / name
    serialisable = {
        "meta": meta,
        "metrics": result.get("metrics"),
        "n": len(result.get("rows") or []),
    }
    path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    return path


def _apply_sensitivity(instances, axis: str | None):
    if not axis:
        return instances
    if axis.startswith("history:"):
        n = int(axis.split(":", 1)[1])
        clipped = []
        for instance in instances:
            instance.sessions = instance.sessions[-n:]
            instance.n_sessions = len(instance.sessions)
            clipped.append(instance)
        return clipped
    if axis.startswith("distractors:"):
        n = int(axis.split(":", 1)[1])
        for instance in instances:
            instance.distractors = instance.distractors[:n]
        return instances
    if axis.startswith("distance:"):
        lo, hi = axis.split(":", 1)[1].split("-")
        low, high = int(lo), int(hi)
        return [row for row in instances if low <= row.conflict_distance <= high]
    return instances

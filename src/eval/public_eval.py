from __future__ import annotations

from typing import Any, Callable

from src.agent.react import ReActAgent
from src.eval.judge import judge_response
from src.eval.metrics import mean
from src.llm.client import LLMClient
from src.persist.schema import ConflictInstance


def load_agentdojo_cases(path) -> list[dict[str, Any]]:
    from pathlib import Path
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AgentDojo case file not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_public_memory(
    agent: ReActAgent,
    judge: LLMClient,
    instances: list[ConflictInstance],
    subset: str,
) -> dict[str, Any]:
    """LongMemEval / LoCoMo / MemoryAgentBench use the same replay + judge path.

    ``subset`` selects the grading field: ``ku``, ``abstention``, ``overall``,
    ``adv``, ``f1`` or ``fc``. Gold labels travel with each instance.
    """
    from src.eval.conflict_eval import _reset_memory, replay_history

    scores = []
    for instance in instances:
        replay_history(agent, instance.sessions)
        trace = agent.run_turn(instance.query)
        judged = judge_response(judge, instance, trace.action.content)
        if subset == "fc":
            scores.append(int(instance.gold_value.lower() in trace.action.content.lower()))
        elif subset == "f1":
            scores.append(_token_f1(instance.gold_value, trace.action.content))
        else:
            scores.append(int(judged["correct"]))
        _reset_memory(agent)
    key = {"ku": "ku", "abstention": "abst", "overall": "overall", "adv": "adv", "f1": "f1", "fc": "fc"}[subset]
    return {"metrics": {key: 100.0 * mean(scores)}, "n": len(scores)}


def evaluate_agentdojo(
    agent: ReActAgent,
    cases: list[dict[str, Any]],
    security: Callable[[dict, dict], bool],
) -> dict[str, Any]:
    utility = []
    utility_attack = []
    asr = []
    for case in cases:
        env = {"state": dict(case.get("init_state") or {}), "last_action": None}
        trace = agent.run_turn(case["user_task"], env_state=env)
        env["last_action"] = trace.action.as_dict()
        ok = bool(case.get("success_fn", lambda a, e: True)(trace.action.as_dict(), env))
        if case.get("under_attack"):
            utility_attack.append(int(ok))
            asr.append(int(security(case.get("target") or {}, env)))
        else:
            utility.append(int(ok))
        from src.eval.conflict_eval import _reset_memory

        _reset_memory(agent)
    return {
        "metrics": {
            "utility": 100.0 * mean(utility),
            "utility_under_attack": 100.0 * mean(utility_attack),
            "asr": 100.0 * mean(asr),
        }
    }


def _token_f1(gold: str, pred: str) -> float:
    g = gold.lower().split()
    p = pred.lower().split()
    if not g or not p:
        return 0.0
    overlap = len(set(g) & set(p))
    prec = overlap / len(p)
    rec = overlap / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

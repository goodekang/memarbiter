from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.paths import CONFIG_DIR


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        candidate = CONFIG_DIR / path
        path = candidate if candidate.exists() else path
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping in {path}")
    return data


def load_stack(
    memarbiter: str | Path = "memarbiter.yaml",
    persist: str | Path = "persist.yaml",
    eval_cfg: str | Path = "eval.yaml",
    backbones: str | Path = "backbones.yaml",
    baselines: str | Path = "baselines.yaml",
) -> dict[str, Any]:
    return {
        "memarbiter": load_yaml(memarbiter),
        "persist": load_yaml(persist),
        "eval": load_yaml(eval_cfg),
        "backbones": load_yaml(backbones),
        "baselines": load_yaml(baselines),
    }


def deep_get(mapping: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = mapping
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

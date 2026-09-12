#!/usr/bin/env python3
"""Construct PERSIST-Conflict and PERSIST-Poison from LongMemEval and AgentDojo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_yaml
from src.llm.client import build_client
from src.persist.build import build_conflict_split, build_poison_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/persist.yaml")
    parser.add_argument("--backbone", default="gpt-5.6-sol")
    parser.add_argument("--split", choices=["conflict", "poison", "all"], default="all")
    args = parser.parse_args()

    cfg = load_yaml(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    client = build_client(args.backbone)
    root = Path(cfg["paths"]["persist_root"])
    if args.split in {"conflict", "all"}:
        n = build_conflict_split(cfg, client, root / "conflict" / "test.jsonl")
        print(f"conflict instances: {n}")
    if args.split in {"poison", "all"}:
        n = build_poison_split(cfg, client, root / "poison" / "test.jsonl")
        print(f"poison trajectories: {n}")


if __name__ == "__main__":
    main()

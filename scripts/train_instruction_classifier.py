#!/usr/bin/env python3
"""Build the gate data, train the quarantine-gate classifier and evaluate it (Appendix A.1, Table A.1).

  --build-pairs     collect benign AgentDojo tool outputs and seed items, rewrite
                    the AgentDojo / ASB attack specifications, write the pool
  --split grouped   grouped train/dev/test split with held-out specifications
                    and user tasks (default; writes heldout_report.json)
  --split row       original random row-level split (main-table gate)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_yaml
from src.llm.client import build_client
from src.paths import resolve_path
from src.train.build_pairs import build_instruction_pairs
from src.train.train_instruction import train_instruction_clf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_instruction_clf.yaml")
    parser.add_argument("--persist-config", default="configs/persist.yaml")
    parser.add_argument("--backbone", default="gpt-5.6-sol")
    parser.add_argument("--build-pairs", action="store_true")
    parser.add_argument("--split", choices=["grouped", "row"], default=None, help="override `split` in the config")
    args = parser.parse_args()

    cfg = load_yaml(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    if args.split:
        cfg["split"] = args.split
    pairs = resolve_path(cfg["pairs_path"])
    if args.build_pairs or not pairs.exists():
        persist = load_yaml(ROOT / args.persist_config)
        client = build_client(args.backbone)
        n = build_instruction_pairs(persist, client, pairs, cfg)
        print(f"wrote {n} rows to {pairs}")
    print(train_instruction_clf(cfg))


if __name__ == "__main__":
    main()

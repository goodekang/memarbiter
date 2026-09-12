#!/usr/bin/env python3
"""Fine-tune DeBERTa-v3-large on the 4,000 write-time conflict pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_yaml
from src.llm.client import build_client
from src.train.build_pairs import build_nli_pairs
from src.train.train_nli import train_conflict_nli


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_conflict_nli.yaml")
    parser.add_argument("--persist-config", default="configs/persist.yaml")
    parser.add_argument("--backbone", default="gpt-5.6-sol")
    parser.add_argument("--build-pairs", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config)
    pairs = Path(cfg["pairs_path"])
    if args.build_pairs or not pairs.exists():
        persist = load_yaml(ROOT / args.persist_config)
        client = build_client(args.backbone)
        n = build_nli_pairs(persist, client, pairs, int(cfg["n_pairs"]), int(cfg["train"]["seed"]))
        print(f"wrote {n} pairs to {pairs}")
    summary = train_conflict_nli(cfg)
    print(summary)


if __name__ == "__main__":
    main()

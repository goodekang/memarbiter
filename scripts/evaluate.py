#!/usr/bin/env python3
"""Run a PERSIST or public-benchmark suite for one system / backbone / seed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_stack
from src.eval.runner import run_suite, write_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        required=True,
        choices=["conflict", "poison", "public", "agentdojo", "ablation", "sensitivity", "attribution"],
    )
    parser.add_argument("--system", default="memarbiter")
    parser.add_argument("--backbone", default="gpt-5.6-sol")
    parser.add_argument("--defense", default=None)
    parser.add_argument("--ablation", default=None, help="variant key(s) from configs/ablations.yaml, comma-separated (e.g. original_rules,no_scorer)")
    parser.add_argument("--prior", default=None, help="hand_set | learned | uniform | swapped")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--all-seeds", action="store_true", help="run the five protocol seeds and report the mean")
    parser.add_argument("--no-repair", action="store_true", help="MemArbiter without TASR (Table 7)")
    parser.add_argument("--repair-mode", default=None, help="tasr | none | wipe_all | recency_purge | similarity_purge")
    parser.add_argument("--subset", default="overall", help="public subset: ku|abstention|overall|adv|f1|fc")
    parser.add_argument("--axis", default=None, help="sensitivity: history:N | distractors:N | distance:LO-HI")
    parser.add_argument("--attribution", default="tasr")
    parser.add_argument(
        "--query-context",
        default="annotated",
        choices=["annotated", "extracted"],
        help="conflict suite: psi from the scope annotation (main tables) or extracted from query + session (Table 14)",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_stack()
    seeds = list(cfg["eval"]["seeds"]) if args.all_seeds else [args.seed]
    dest = Path(args.output_dir or cfg["eval"]["output_dir"])
    rows = []
    for seed in seeds:
        result = run_suite(
            args.suite,
            system=args.system,
            backbone=args.backbone,
            defense=args.defense,
            seed=seed,
            cfg=cfg,
            ablation=args.ablation,
            prior=args.prior,
            repair=not args.no_repair,
            public_subset=args.subset,
            sensitivity_axis=args.axis,
            attribution_mode=args.attribution,
            repair_mode=args.repair_mode,
            query_context=args.query_context,
        )
        path = write_result(result, dest)
        print(result.get("metrics"))
        print("wrote", path)
        rows.append(result.get("metrics") or {})
    if len(rows) > 1:
        keys = sorted({key for row in rows for key in row})
        mean = {key: sum(float(row.get(key) or 0.0) for row in rows) / len(rows) for key in keys}
        print("mean_over_seeds", mean)


if __name__ == "__main__":
    main()

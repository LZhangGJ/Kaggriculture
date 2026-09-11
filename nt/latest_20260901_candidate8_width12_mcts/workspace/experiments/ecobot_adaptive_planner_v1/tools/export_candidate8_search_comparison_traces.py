#!/usr/bin/env python3
"""Export selected Beam/MCTS comparison rows as complete native joint traces."""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_candidate8_multifuture_oracle import load_genome
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def canonical(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-receipt", required=True, type=Path)
    parser.add_argument("--row-indices", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    receipt = json.loads(args.comparison_receipt.read_text(encoding="utf-8"))
    indices = [int(value) for value in args.row_indices.split(",") if value.strip()]
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    use_feasible_pool = receipt["config"]["candidate_pool"] == "feasible"

    games = []
    for index in indices:
        row = receipt["rows"][index]
        opponent_name = str(row["opponent"])
        native = bundle.adaptive_executor.candidate8_committed_sequence(
            genome,
            bundle.index(opponent_name),
            int(row["seed"]),
            row["decision_days"],
            row["selected_ranks"],
            int(row["seat"]),
            use_feasible_pool,
            -1,
            0,
            True,
        )
        expected = [
            float(value) for value in (row["own_cash"], row["opponent_cash"])
        ]
        if int(row["seat"]) == 1:
            expected.reverse()
        actual = [float(value) for value in native["rewards"]]
        if actual != expected:
            raise RuntimeError(f"row {index} no longer replays exactly: {actual} != {expected}")
        games.append(
            {
                "opponent": opponent_name,
                "seed": int(row["seed"]),
                "candidate_seat": int(row["seat"]),
                "decision_day": -1,
                "arm": -1,
                "expected_signature": 0,
                "actual_signature": 0,
                "signature_exact": True,
                "algorithm": str(row["algorithm"]),
                "budget_label": int(row["budget_label"]),
                "repeat": int(row["repeat"]),
                "selected_days": [int(value) for value in row["decision_days"]],
                "selected_ranks": [int(value) for value in row["selected_ranks"]],
                "native_rewards": actual,
                "end_overflow": int(native["end_overflow"]),
                "trace": canonical(native["trace"]),
            }
        )
        print(f"exported row={index} algorithm={row['algorithm']}", flush=True)

    payload = {
        "schema": "kaggriculture.candidate8-search-comparison-traces.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_receipt": str(args.comparison_receipt),
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps({"output": str(args.output), "games": len(games)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

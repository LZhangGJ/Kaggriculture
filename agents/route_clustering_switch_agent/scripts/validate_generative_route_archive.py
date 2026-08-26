#!/usr/bin/env python3
"""Promote top screen candidates through a multi-seed, both-seat panel."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evolve_generative_routes import OfficialEvaluator  # noqa: E402
from meta_agent.src.generative_route import GenerativeRouteGenome  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--one-seat", action="store_true")
    parser.add_argument("--opponent-agent", type=Path)
    args = parser.parse_args()

    state = json.loads(args.checkpoint.resolve().read_text(encoding="utf-8"))
    archive = sorted(
        list(state.get("archive", []) or []),
        key=lambda row: float(row["evaluation"]["fitness"]),
        reverse=True,
    )[:args.top_k]
    evaluator = OfficialEvaluator(
        args.seeds,
        both_seats=not args.one_seat,
        opponent_agent=args.opponent_agent,
    )
    started = time.perf_counter()
    rows = []
    for index, elite in enumerate(archive, 1):
        genome = GenerativeRouteGenome.from_dict(elite["genome"])
        robust = evaluator.evaluate(genome)
        rows.append({
            "genome_id": genome.genome_id,
            "genome": genome.to_dict(),
            "descriptor": list(genome.descriptor()),
            "screen_generation": elite.get("generation"),
            "screen_evaluation": elite["evaluation"],
            "robust_evaluation": robust,
            "screen_to_robust_reward_delta": (
                robust["mean_reward"] - float(elite["evaluation"]["mean_reward"])
            ),
        })
        print(
            f"validate {index}/{len(archive)} {genome.genome_id} "
            f"mean={robust['mean_reward']:.1f} min={robust['minimum_reward']:.1f} "
            f"margin={robust['mean_margin']:.1f}",
            flush=True,
        )
    rows.sort(key=lambda row: float(row["robust_evaluation"]["fitness"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["robust_rank"] = rank
    output = {
        "schema": "generative-route-validation-v1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "evaluation_signature": evaluator.signature,
        "elapsed_seconds": time.perf_counter() - started,
        "candidate_count": len(rows),
        "best_genome_id": rows[0]["genome_id"] if rows else None,
        "mean_candidate_reward": (
            statistics.fmean(row["robust_evaluation"]["mean_reward"] for row in rows)
            if rows else None
        ),
        "rows": rows,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

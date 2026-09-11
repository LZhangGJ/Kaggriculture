#!/usr/bin/env python3
"""Build execution and complete-route controls for the frozen O1.6 losses."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome, sha256
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


DECISION_DAYS = [6, 12, 18]


def score(own: float, rival: float) -> float:
    return 1.0 if own > rival else (0.5 if own == rival else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    losses = [
        row for row in baseline_payload["rows"]["depth3"]
        if float(row["margin"]) <= 0.0
    ]
    if len(losses) != 45:
        raise RuntimeError(f"expected 45 frozen losses, observed {len(losses)}")

    started = time.perf_counter()
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    route_count = len(bundle.families)

    execution_rows: list[dict] = []
    for baseline in losses:
        opponent_name = str(baseline["opponent"])
        opponent_index = bundle.index(opponent_name)
        seat = int(baseline["seat"])
        result = bundle.adaptive_executor.candidate8_committed_sequence(
            genome,
            opponent_index,
            int(baseline["seed"]),
            DECISION_DAYS,
            [int(value) for value in baseline["selected_ranks"]],
            seat,
            True,
        )
        own = float(result["rewards"][seat])
        rival = float(result["rewards"][1 - seat])
        reward_exact = (
            own == float(baseline["own_cash"])
            and rival == float(baseline["opponent_cash"])
        )
        if not reward_exact:
            raise RuntimeError(
                "committed baseline sequence did not reproduce Oracle result: "
                f"{opponent_name}/{baseline['seed']}/seat{seat}"
            )
        execution_rows.append({
            "opponent": opponent_name,
            "seed": int(baseline["seed"]),
            "seat": seat,
            "own_cash": own,
            "opponent_cash": rival,
            "margin": own - rival,
            "selected_ranks": [int(value) for value in result["selected_rank"]],
            "selected_families": [
                FAMILY_NAMES[int(value)] if int(value) >= 0 else "NONE"
                for value in result["selected_family"]
            ],
            "end_overflow": int(result["end_overflow"]),
            "avoidable_crop_losses": int(result["avoidable_crop_losses"]),
            "avoidable_animal_losses": int(result["avoidable_animal_losses"]),
            "hard_execution_issue": bool(
                int(result["end_overflow"]) > 0
                or int(result["avoidable_crop_losses"]) > 0
                or int(result["avoidable_animal_losses"]) > 0
            ),
            "reward_exact": reward_exact,
        })

    tasks = np.empty((len(losses) * route_count, 7), dtype=np.int64)
    task_case = np.empty(len(tasks), dtype=np.int32)
    task_route = np.empty(len(tasks), dtype=np.int32)
    cursor = 0
    for case_index, baseline in enumerate(losses):
        opponent_index = bundle.index(str(baseline["opponent"]))
        seat = int(baseline["seat"])
        for candidate_route in range(route_count):
            if seat == 0:
                route0, route1 = candidate_route, opponent_index
            else:
                route0, route1 = opponent_index, candidate_route
            tasks[cursor] = (
                route0, route1, int(baseline["seed"]), -1, -1, -1, -1
            )
            task_case[cursor] = case_index
            task_route[cursor] = candidate_route
            cursor += 1
    rewards, audit = bundle.executor.play_audit_batch(tasks)
    rewards = np.asarray(rewards, dtype=np.float64)
    audit = np.asarray(audit, dtype=np.int32)

    portfolio_rows: list[dict] = []
    for case_index, baseline in enumerate(losses):
        indices = np.flatnonzero(task_case == case_index)
        seat = int(baseline["seat"])
        own = rewards[indices, seat]
        rival = rewards[indices, 1 - seat]
        outcomes = np.where(own > rival, 1.0, np.where(own == rival, 0.5, 0.0))
        margins = own - rival
        order = np.lexsort((-own, -margins, -outcomes))
        best_local = int(order[0])
        task_index = int(indices[best_local])
        route_index = int(task_route[task_index])
        candidate_audit = audit[task_index, seat]
        portfolio_rows.append({
            "opponent": str(baseline["opponent"]),
            "seed": int(baseline["seed"]),
            "seat": seat,
            "best_route": bundle.families[route_index],
            "best_route_index": route_index,
            "own_cash": float(own[best_local]),
            "opponent_cash": float(rival[best_local]),
            "margin": float(margins[best_local]),
            "score": float(outcomes[best_local]),
            "winning_routes": int(np.sum(outcomes == 1.0)),
            "non_losing_routes": int(np.sum(outcomes >= 0.5)),
            "candidate_macro_unit_failures": int(candidate_audit[0]),
            "candidate_macro_market_failures": int(candidate_audit[1]),
            "candidate_first_macro_failure_step": int(candidate_audit[2]),
        })

    hard_issue_count = sum(row["hard_execution_issue"] for row in execution_rows)
    portfolio_win_count = sum(row["score"] == 1.0 for row in portfolio_rows)
    payload = {
        "schema": "kaggriculture.candidate8-o16-failure-controls.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "boundary": (
            "Uses only the 45 non-winning depth-3 cases frozen by O1.5. "
            "The complete-route portfolio is an identity-aware exact-future "
            "diagnostic and is never a deployable selector."
        ),
        "summary": {
            "frozen_losses": len(losses),
            "committed_reward_exact": all(row["reward_exact"] for row in execution_rows),
            "baseline_hard_execution_issue_cases": hard_issue_count,
            "baseline_clean_execution_cases": len(losses) - hard_issue_count,
            "reference_routes": route_count,
            "portfolio_winning_cases": portfolio_win_count,
            "portfolio_non_winning_cases": len(losses) - portfolio_win_count,
            "portfolio_coverage_rate": portfolio_win_count / len(losses),
        },
        "execution_rows": execution_rows,
        "portfolio_rows": portfolio_rows,
        "simulation_seconds": time.perf_counter() - started,
        "inputs": {
            "baseline": {"path": str(args.baseline), "sha256": sha256(args.baseline)},
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(json.dumps({"simulation_seconds": payload["simulation_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

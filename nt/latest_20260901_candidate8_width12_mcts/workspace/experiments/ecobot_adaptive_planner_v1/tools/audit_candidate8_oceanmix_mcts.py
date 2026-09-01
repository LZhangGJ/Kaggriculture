#!/usr/bin/env python3
"""Try full-feasible Candidate8 MCTS on the remaining OceanMix F004 miss."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome
from compare_candidate8_mcts_beam import run_root_parallel_mcts
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def family_names(values: list[int]) -> list[str]:
    return ["NONE" if int(value) < 0 else FAMILY_NAMES[int(value)] for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--dense-receipt", required=True, type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budgets", type=parse_ints, default=parse_ints("42676,170704,1000000"))
    parser.add_argument("--decision-days", type=parse_ints, default=list(range(30)))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--per-node-arms", type=int, default=4096)
    parser.add_argument("--rollout-arms", type=int, default=8)
    parser.add_argument("--exploration-constant", type=float, default=1.25)
    parser.add_argument("--pw-constant", type=float, default=2.0)
    parser.add_argument("--pw-alpha", type=float, default=0.5)
    parser.add_argument("--search-seed", type=int, default=9102001)
    args = parser.parse_args()

    cases = {
        row["route_id"]: row
        for row in json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    }
    dense = json.loads(args.dense_receipt.read_text(encoding="utf-8"))
    unresolved = [row for row in dense["rows"] if not row["best"]["win"]]
    if len(unresolved) != 1 or unresolved[0]["team_name"] != "OceanMix":
        raise RuntimeError(f"expected one OceanMix miss, got {len(unresolved)}")
    case = cases[unresolved[0]["route_id"]]
    seat = int(case["candidate_seat"])
    seed = int(case["official_seed"])

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    genome = load_genome(args.genomes, 0)
    opponent = bundle.index(case["family"])
    decision_days = args.decision_days
    rows = []
    for budget_index, budget in enumerate(args.budgets):
        started = time.perf_counter()
        raw = run_root_parallel_mcts(
            bundle.adaptive_executor,
            genome,
            opponent,
            seed,
            decision_days,
            seat,
            budget,
            args.workers,
            args.per_node_arms,
            True,
            args.exploration_constant,
            args.pw_constant,
            args.pw_alpha,
            args.rollout_arms,
            args.search_seed + 1_000_003 * budget_index,
        )
        elapsed = time.perf_counter() - started
        own = float(raw["rewards"][seat])
        rival = float(raw["rewards"][1 - seat])
        row = {
            "simulation_budget": budget,
            "parallel_workers": int(raw["parallel_workers"]),
            "worker_budgets": [int(value) for value in raw["worker_budgets"]],
            "own_cash": own,
            "opponent_cash": rival,
            "margin": own - rival,
            "win": own > rival,
            "seconds": elapsed,
            "simulations": int(raw["simulations"]),
            "tree_nodes": int(raw["tree_nodes"]),
            "maximum_depth": int(raw["maximum_depth"]),
            "first_win_simulation": int(raw["first_win_simulation"]),
            "best_found_simulation": int(raw["best_found_simulation"]),
            "winning_simulations": int(raw["winning_simulations"]),
            "unique_sampled_paths": int(raw["unique_sampled_paths"]),
            "decision_days": [int(value) for value in raw["decision_day"]],
            "selected_ranks": [int(value) for value in raw["selected_rank"]],
            "selected_families": family_names(raw["selected_family"]),
            "selected_path_available_counts": [int(value) for value in raw["feasible_count"]],
        }
        committed = bundle.adaptive_executor.candidate8_committed_sequence(
            genome,
            opponent,
            seed,
            row["decision_days"],
            row["selected_ranks"],
            seat,
            True,
        )
        row["committed_rewards"] = [float(value) for value in committed["rewards"]]
        row["replay_exact"] = tuple(row["committed_rewards"]) == tuple(
            float(value) for value in raw["rewards"]
        )
        if not row["replay_exact"]:
            raise RuntimeError("MCTS selected sequence did not replay exactly")
        rows.append(row)
        print(
            f"budget={budget} margin={row['margin']:.0f} own={own:.0f} rival={rival:.0f} "
            f"wins={row['winning_simulations']} unique={row['unique_sampled_paths']} "
            f"depth={row['maximum_depth']} seconds={elapsed:.1f}",
            flush=True,
        )
        if row["win"]:
            break

    beam_best = unresolved[0]["best"]
    best = max(rows, key=lambda row: (row["win"], row["margin"], row["own_cash"]))
    payload = {
        "schema": "kaggriculture-candidate8-oceanmix-mcts-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["replay_exact"] for row in rows) else "FAIL",
        "boundary": (
            "Exact-future offline MCTS against one frozen OceanMix action tape and "
            "its actual replay seed; not deployable online strength."
        ),
        "case": {
            key: case[key]
            for key in (
                "route_id", "rank", "team_name", "submission_id", "near_family_id",
                "member_count", "representative_episode_id", "official_seed",
                "opponent_seat", "candidate_seat",
            )
        },
        "config": {
            "decision_days": decision_days,
            "candidate_pool": "feasible",
            "per_node_arms": args.per_node_arms,
            "workers": args.workers,
            "budgets": args.budgets,
            "exploration_constant": args.exploration_constant,
            "progressive_widening_constant": args.pw_constant,
            "progressive_widening_alpha": args.pw_alpha,
            "rollout_arms": args.rollout_arms,
        },
        "beam30_reference": beam_best,
        "rows": rows,
        "summary": {
            "mcts_found_winning_path": bool(best["win"]),
            "best_mcts_budget": int(best["simulation_budget"]),
            "best_mcts_own_cash": float(best["own_cash"]),
            "best_mcts_opponent_cash": float(best["opponent_cash"]),
            "best_mcts_margin": float(best["margin"]),
            "beam30_margin": float(beam_best["margin"]),
            "margin_improvement_over_beam30": float(best["margin"] - beam_best["margin"]),
            "all_committed_replays_exact": all(row["replay_exact"] for row in rows),
        },
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "source": args.source,
                "actions": args.actions,
                "metadata": args.metadata,
                "cases": args.cases,
                "dense_receipt": args.dense_receipt,
                "genomes": args.genomes,
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"receipt={args.output}", flush=True)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare fixed-budget Candidate8 MCTS with the frozen exact-future beam."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome
from generate_candidate8_competitive_pool import select_opponents
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def family_names(values: list[int]) -> list[str]:
    return ["NONE" if int(value) < 0 else FAMILY_NAMES[int(value)] for value in values]


def competitive_key(result: dict, candidate_seat: int) -> tuple[float, float, float]:
    own = float(result["rewards"][candidate_seat])
    rival = float(result["rewards"][1 - candidate_seat])
    return (1.0 if own > rival else (0.5 if own == rival else 0.0), own - rival, own)


def run_root_parallel_mcts(
    executor,
    genome,
    opponent: int,
    seed: int,
    decision_days: list[int],
    seat: int,
    total_budget: int,
    workers: int,
    per_node_arms: int,
    use_feasible_pool: bool,
    exploration_constant: float,
    pw_constant: float,
    pw_alpha: float,
    rollout_arms: int,
    search_seed: int,
) -> dict:
    """Run independent MCTS roots in parallel and retain the best exact terminal path."""
    workers = max(1, min(workers, total_budget))
    budgets = [total_budget // workers] * workers
    for index in range(total_budget % workers):
        budgets[index] += 1

    def run_worker(worker: int) -> dict:
        return executor.candidate8_mcts_oracle(
            genome,
            opponent,
            seed,
            decision_days,
            seat,
            budgets[worker],
            per_node_arms,
            use_feasible_pool,
            True,
            exploration_constant,
            pw_constant,
            pw_alpha,
            rollout_arms,
            search_seed + 104729 * worker,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(run_worker, range(workers)))
    best = dict(max(parts, key=lambda result: competitive_key(result, seat)))
    best["simulations"] = sum(int(result["simulations"]) for result in parts)
    best["tree_nodes"] = sum(int(result["tree_nodes"]) for result in parts)
    best["maximum_depth"] = max(int(result["maximum_depth"]) for result in parts)
    best["winning_simulations"] = sum(
        int(result["winning_simulations"]) for result in parts
    )
    best["unique_sampled_paths"] = sum(
        int(result["unique_sampled_paths"]) for result in parts
    )
    local_first_wins = [
        int(result["first_win_simulation"])
        for result in parts
        if int(result["first_win_simulation"]) >= 0
    ]
    best["first_win_simulation"] = min(local_first_wins) if local_first_wins else -1
    best["parallel_workers"] = workers
    best["worker_budgets"] = budgets
    return best


def result_row(
    *,
    algorithm: str,
    opponent: str,
    seed: int,
    seat: int,
    budget_label: int,
    repeat: int,
    elapsed: float,
    result: dict,
) -> dict:
    own = float(result["rewards"][seat])
    rival = float(result["rewards"][1 - seat])
    continuations = int(result.get("complete_continuations", result.get("simulations", 0)))
    row = {
        "algorithm": algorithm,
        "opponent": opponent,
        "seed": seed,
        "seat": seat,
        "budget_label": budget_label,
        "repeat": repeat,
        "own_cash": own,
        "opponent_cash": rival,
        "margin": own - rival,
        "score": 1.0 if own > rival else (0.5 if own == rival else 0.0),
        "complete_continuations": continuations,
        "seconds": elapsed,
        "continuations_per_second": continuations / elapsed if elapsed else 0.0,
        "decision_days": [int(value) for value in result["decision_day"]],
        "selected_ranks": [int(value) for value in result["selected_rank"]],
        "selected_families": family_names(result["selected_family"]),
    }
    for key in (
        "expanded_nodes",
        "maximum_live_beam",
        "simulations",
        "tree_nodes",
        "maximum_depth",
        "first_win_simulation",
        "best_found_simulation",
        "winning_simulations",
        "unique_sampled_paths",
    ):
        if key in result:
            row[key] = int(result[key])
    if "root_rank" in result:
        row["root_rank"] = [int(value) for value in result["root_rank"]]
        row["root_visits"] = [int(value) for value in result["root_visits"]]
        row["root_mean_value"] = [float(value) for value in result["root_mean_value"]]
    return row


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["budget_label"])].append(row)
    output = []
    for (algorithm, budget), values in sorted(grouped.items()):
        margins = np.asarray([row["margin"] for row in values], dtype=np.float64)
        own = np.asarray([row["own_cash"] for row in values], dtype=np.float64)
        scores = np.asarray([row["score"] for row in values], dtype=np.float64)
        seconds = np.asarray([row["seconds"] for row in values], dtype=np.float64)
        rates = np.asarray(
            [row["continuations_per_second"] for row in values], dtype=np.float64
        )
        entry = {
            "algorithm": algorithm,
            "budget_label": budget,
            "runs": len(values),
            "score_rate": float(scores.mean()),
            "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
            "minimum_margin": float(margins.min()),
            "mean_own_cash": float(own.mean()),
            "mean_seconds": float(seconds.mean()),
            "mean_continuations_per_second": float(rates.mean()),
        }
        if algorithm.startswith("MCTS"):
            first = np.asarray(
                [row["first_win_simulation"] for row in values], dtype=np.int64
            )
            entry.update(
                mean_first_win_simulation=float(first[first >= 0].mean())
                if np.any(first >= 0)
                else None,
                no_win_runs=int((first < 0).sum()),
                mean_unique_sampled_paths=float(
                    np.mean([row["unique_sampled_paths"] for row in values])
                ),
                mean_tree_nodes=float(
                    np.mean([row["tree_nodes"] for row in values])
                ),
            )
        output.append(entry)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--merged-receipt", required=True, type=Path)
    parser.add_argument(
        "--group", choices=("hard16", "selector43", "all"), default="hard16"
    )
    parser.add_argument("--opponents")
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--single-seat", type=int, choices=(0, 1))
    parser.add_argument(
        "--decision-days", type=parse_ints, default=parse_ints("0,1,3,6,9,12,18,24")
    )
    parser.add_argument(
        "--beam-widths", type=parse_ints, default=parse_ints("1,4,27")
    )
    parser.add_argument("--per-node-arms", type=int, default=64)
    parser.add_argument(
        "--candidate-pool", choices=("feasible", "shortlist"), default="shortlist"
    )
    parser.add_argument("--mcts-repeats", type=int, default=3)
    parser.add_argument("--mcts-workers", type=int, default=16)
    parser.add_argument("--exploration-constant", type=float, default=1.25)
    parser.add_argument("--pw-constant", type=float, default=2.0)
    parser.add_argument("--pw-alpha", type=float, default=0.5)
    parser.add_argument("--rollout-arms", type=int, default=8)
    parser.add_argument("--search-seed", type=int, default=3100001)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.beam_widths or any(value <= 0 for value in args.beam_widths):
        raise ValueError("beam widths must be positive")
    if args.mcts_repeats < 0:
        raise ValueError("MCTS repeats must be non-negative")
    if args.mcts_workers <= 0:
        raise ValueError("MCTS workers must be positive")

    merged = json.loads(args.merged_receipt.read_text(encoding="utf-8"))
    opponent_names = (
        [value.strip() for value in args.opponents.split(",") if value.strip()]
        if args.opponents
        else select_opponents(merged, args.group)
    )
    if not opponent_names:
        raise ValueError("at least one opponent is required")
    seats = (args.single_seat,) if args.single_seat is not None else (0, 1)
    use_feasible_pool = args.candidate_pool == "feasible"

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)

    rows: list[dict] = []
    replay_checks: list[dict] = []
    started = time.perf_counter()
    for opponent_name in opponent_names:
        opponent = bundle.index(opponent_name)
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            for seat in seats:
                for width in args.beam_widths:
                    beam_started = time.perf_counter()
                    beam = bundle.adaptive_executor.candidate8_sequence_oracle(
                        genome,
                        opponent,
                        seed,
                        args.decision_days,
                        seat,
                        width,
                        args.per_node_arms,
                        use_feasible_pool,
                        True,
                    )
                    beam_elapsed = time.perf_counter() - beam_started
                    beam_row = result_row(
                        algorithm="BEAM",
                        opponent=opponent_name,
                        seed=seed,
                        seat=seat,
                        budget_label=width,
                        repeat=0,
                        elapsed=beam_elapsed,
                        result=beam,
                    )
                    rows.append(beam_row)

                    beam_replay = bundle.adaptive_executor.candidate8_committed_sequence(
                        genome,
                        opponent,
                        seed,
                        beam_row["decision_days"],
                        beam_row["selected_ranks"],
                        seat,
                        use_feasible_pool,
                    )
                    beam_exact = tuple(float(value) for value in beam_replay["rewards"]) == tuple(
                        float(value) for value in beam["rewards"]
                    )
                    replay_checks.append(
                        {
                            "algorithm": "BEAM",
                            "opponent": opponent_name,
                            "seed": seed,
                            "seat": seat,
                            "budget_label": width,
                            "repeat": 0,
                            "exact": beam_exact,
                        }
                    )
                    if not beam_exact:
                        raise RuntimeError("beam selected sequence did not replay exactly")

                    budget = int(beam["complete_continuations"])
                    for repeat in range(args.mcts_repeats):
                        search_seed = (
                            args.search_seed
                            + 1_000_003 * repeat
                            + 1009 * seed
                            + 97 * seat
                            + 17 * width
                            + 13 * opponent
                        )
                        mcts_started = time.perf_counter()
                        mcts = run_root_parallel_mcts(
                            bundle.adaptive_executor,
                            genome,
                            opponent,
                            seed,
                            args.decision_days,
                            seat,
                            budget,
                            args.mcts_workers,
                            args.per_node_arms,
                            use_feasible_pool,
                            args.exploration_constant,
                            args.pw_constant,
                            args.pw_alpha,
                            args.rollout_arms,
                            search_seed,
                        )
                        mcts_elapsed = time.perf_counter() - mcts_started
                        mcts_row = result_row(
                            algorithm="MCTS_ROOT_PARALLEL",
                            opponent=opponent_name,
                            seed=seed,
                            seat=seat,
                            budget_label=width,
                            repeat=repeat,
                            elapsed=mcts_elapsed,
                            result=mcts,
                        )
                        mcts_row["simulation_budget"] = budget
                        mcts_row["search_seed"] = search_seed
                        mcts_row["parallel_workers"] = int(mcts["parallel_workers"])
                        mcts_row["worker_budgets"] = [
                            int(value) for value in mcts["worker_budgets"]
                        ]
                        rows.append(mcts_row)

                        mcts_replay = bundle.adaptive_executor.candidate8_committed_sequence(
                            genome,
                            opponent,
                            seed,
                            mcts_row["decision_days"],
                            mcts_row["selected_ranks"],
                            seat,
                            use_feasible_pool,
                        )
                        mcts_exact = tuple(
                            float(value) for value in mcts_replay["rewards"]
                        ) == tuple(float(value) for value in mcts["rewards"])
                        replay_checks.append(
                            {
                                "algorithm": "MCTS_ROOT_PARALLEL",
                                "opponent": opponent_name,
                                "seed": seed,
                                "seat": seat,
                                "budget_label": width,
                                "repeat": repeat,
                                "exact": mcts_exact,
                            }
                        )
                        if not mcts_exact:
                            raise RuntimeError("MCTS best sequence did not replay exactly")

    elapsed = time.perf_counter() - started
    payload = {
        "schema": "kaggriculture.candidate8-mcts-beam-comparison.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["exact"] for row in replay_checks) else "FAIL",
        "boundary": (
            "Exact-future offline search comparison only. Candidate language, "
            "executor, opponent route, actual seed and terminal objective are frozen."
        ),
        "config": {
            "opponents": opponent_names,
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "seats": list(seats),
            "decision_days": args.decision_days,
            "beam_widths": args.beam_widths,
            "per_node_arms": args.per_node_arms,
            "candidate_pool": args.candidate_pool,
            "mcts_repeats": args.mcts_repeats,
            "mcts_workers": args.mcts_workers,
            "exploration_constant": args.exploration_constant,
            "progressive_widening_constant": args.pw_constant,
            "progressive_widening_alpha": args.pw_alpha,
            "rollout_arms": args.rollout_arms,
            "search_seed": args.search_seed,
        },
        "summary": summarize(rows),
        "rows": rows,
        "replay_checks": replay_checks,
        "elapsed_seconds": elapsed,
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "source": args.source,
                "actions": args.actions,
                "metadata": args.metadata,
                "genomes": args.genomes,
                "merged_receipt": args.merged_receipt,
            }.items()
        },
    }
    if args.backbone:
        payload["inputs"]["backbone"] = {
            "path": str(args.backbone),
            "sha256": sha256(args.backbone),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "summary": payload["summary"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

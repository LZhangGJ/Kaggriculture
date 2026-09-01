#!/usr/bin/env python3
"""Search general adaptive-planner genomes against a passive farm in C++."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def sample_genomes(
    count: int,
    seed: int,
    base: np.ndarray | None = None,
    *,
    global_fraction: float = 0.20,
    mutation_rate: float = 0.45,
    local_scale: float = 0.14,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    names = list(adaptive_genome_names())
    default = np.asarray(adaptive_default_genome(), dtype=np.float64)
    anchor = default if base is None else np.asarray(base, dtype=np.float64)
    if anchor.shape != default.shape:
        raise ValueError((anchor.shape, default.shape))
    genomes = np.tile(anchor, (count, 1))
    ranges: dict[str, tuple[float, float, bool]] = {
        "cash_reserve": (40, 800, False),
        "action_cost": (0, 32, False),
        "move_cost": (0, 20, False),
        "risk_multiplier": (0.70, 1.60, False),
        "market_impact_weight": (0.30, 1.80, False),
        "opponent_supply_weight": (0.0, 1.30, False),
        "demand_drift_weight": (0.0, 1.60, False),
        "sell_drop_limit": (0.04, 0.65, False),
        "price_replan_fraction": (0.05, 0.80, False),
        "task_stickiness": (0, 240, False),
        "deadline_weight": (5, 220, False),
        "fertilizer_value_fraction": (0.55, 1.45, False),
        "max_hands": (7, 18, True),
        "max_quadrants": (2, 4, True),
        "max_total_animals": (6, 24, True),
        "max_cows": (2, 18, True),
        "max_sheep": (0, 14, True),
        "max_geese": (0, 8, True),
        "max_wheat": (8, 65, True),
        "max_carrot": (0, 30, True),
        "max_tomato": (0, 35, True),
        "max_strawberry": (6, 65, True),
        "max_melon": (3, 22, True),
        "min_wheat_buffer": (0, 10, True),
        "liquidation_day": (23, 29, True),
        "stop_new_animals_day": (6, 18, True),
        "stop_new_crops_day": (17, 27, True),
        "replan_interval_steps": (8, 96, True),
        "routine_priority_scale": (1.0, 14.0, False),
        "task_value_scale": (0.0, 1.5, False),
        "plant_priority": (480.0, 860.0, False),
        "drop_value_threshold": (200.0, 1600.0, False),
        "preempt_quantity": (1.0, 10.0, True),
        # W1 capabilities that passed isolated engineering ablations.  These
        # remain semantic parameters: no replay day, quantity, coordinate or
        # opponent identity is sampled here.
        "land_capacity_trigger_fraction": (0.68, 0.98, False),
        "capital_lockup_weight": (0.0, 0.30, False),
        "future_shop_expectation_weight": (0.0, 0.50, False),
        "portfolio_supply_impact_weight": (0.0, 6.0, False),
        "live_commitment_projection_weight": (0.0, 1.0, False),
    }
    global_count = int(round(count * np.clip(global_fraction, 0.0, 1.0)))
    global_mask = np.zeros(count, dtype=bool)
    if global_count > 0:
        global_mask[-global_count:] = True
    for column, name in enumerate(names):
        # New experimental capabilities default to their safe C++ values and
        # are searched only by a task-specific sampler after passing an
        # independent ablation.  This keeps the generic legacy sampler usable
        # as AdaptiveGenome grows without silently enabling unvalidated modes.
        if name not in ranges:
            continue
        lo, hi, integer = ranges[name]
        span = hi - lo
        local_values = np.clip(
            rng.normal(anchor[column], max(1e-9, local_scale * span), count),
            lo,
            hi,
        )
        mutate = rng.random(count) < mutation_rate
        values = np.where(mutate, local_values, anchor[column])
        if global_count > 0:
            values[global_mask] = rng.uniform(lo, hi, global_count)
        if integer:
            values = np.rint(values)
        genomes[:, column] = values
    genomes[0] = anchor
    # Deterministic, interpretable wide-production anchors.  They remain caps
    # and thresholds; the planner still decides what to buy from live state.
    anchors = [
        {"cash_reserve": 250, "action_cost": 2, "move_cost": 5,
         "risk_multiplier": 1.6, "market_impact_weight": 1,
         "opponent_supply_weight": 0.65, "demand_drift_weight": 0.6,
         "sell_drop_limit": 0.3, "price_replan_fraction": 0.2,
         "task_stickiness": 80, "deadline_weight": 160,
         "fertilizer_value_fraction": 0.9,
         "max_hands": 14, "max_quadrants": 3,
         "max_total_animals": 14, "max_cows": 10,
         "max_sheep": 4, "max_geese": 0, "max_wheat": 45,
         "max_carrot": 12, "max_tomato": 20, "max_strawberry": 45,
         "max_melon": 8, "min_wheat_buffer": 2, "liquidation_day": 26,
         "stop_new_animals_day": 14, "stop_new_crops_day": 24,
         "replan_interval_steps": 24},
        {"max_hands": 16, "max_total_animals": 12, "max_cows": 10,
         "max_sheep": 2, "max_geese": 0, "max_wheat": 50,
         "max_carrot": 8, "max_tomato": 24, "max_strawberry": 55,
         "max_melon": 8, "min_wheat_buffer": 1, "liquidation_day": 25},
        {"max_hands": 16, "max_total_animals": 18, "max_cows": 12,
         "max_sheep": 6, "max_geese": 0, "max_wheat": 45,
         "max_carrot": 10, "max_tomato": 18, "max_strawberry": 45,
         "max_melon": 7, "min_wheat_buffer": 3, "liquidation_day": 25},
    ]
    for row, changes in enumerate(anchors if base is None else (), 1):
        if row >= count:
            break
        genomes[row] = default
        for name, value in changes.items():
            genomes[row, names.index(name)] = value
    return genomes


def load_genome(path: Path, index: int, names: list[str]) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("genomes", [])
    if not (0 <= index < len(rows)):
        raise ValueError(f"base genome index {index} outside 0..{len(rows) - 1}")
    values = dict(zip(names, adaptive_default_genome(), strict=True))
    values.update({
        str(name): float(value)
        for name, value in payload.get("base_values", {}).items()
    })
    values.update({
        str(name): float(value)
        for name, value in rows[index].get("values", {}).items()
    })
    return np.asarray([values[name] for name in names], dtype=np.float64)


def evaluate(executor, genomes: np.ndarray, seed_start: int, seed_count: int):
    tasks = np.asarray([
        [genome, -1, seed, seat]
        for genome in range(len(genomes))
        for seed in range(seed_start, seed_start + seed_count)
        for seat in (0, 1)
    ], dtype=np.int64)
    started = time.perf_counter()
    rewards, diagnostics = executor.play_batch(genomes, tasks)
    elapsed = time.perf_counter() - started
    rewards = np.asarray(rewards, dtype=np.float64)
    diagnostics = np.asarray(diagnostics, dtype=np.int32)
    rows = []
    for genome in range(len(genomes)):
        indices = np.flatnonzero(tasks[:, 0] == genome)
        own = np.asarray([rewards[index, tasks[index, 3]] for index in indices])
        diag = diagnostics[indices]
        hard = diag[:, 2] + diag[:, 3]
        overflow = diag[:, 4]
        mean_hard = float(hard.mean())
        mean_overflow = float(overflow.mean())
        # Catastrophic failures must not be hidden by a lucky high-reward tail.
        objective = float(own.mean() - 12_000 * mean_hard - 800 * mean_overflow)
        rows.append({
            "genome_index": genome,
            "objective": objective,
            "mean_reward": float(own.mean()),
            "p10_reward": float(np.quantile(own, 0.10)),
            "min_reward": float(own.min()),
            "max_reward": float(own.max()),
            "hard_losses": int(hard.sum()),
            "end_overflow": int(overflow.sum()),
            "mean_replans": float(diag[:, 0].mean()),
        })
    rows.sort(key=lambda row: (-row["objective"], -row["p10_reward"]))
    return rows, len(tasks), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--candidate-count", type=int, default=2048)
    parser.add_argument("--base-genomes", type=Path)
    parser.add_argument("--base-genome-index", type=int, default=0)
    parser.add_argument(
        "--base-override",
        action="append",
        default=[],
        help="Set NAME=VALUE on the search anchor; repeatable.",
    )
    parser.add_argument("--global-fraction", type=float, default=0.20)
    parser.add_argument("--mutation-rate", type=float, default=0.45)
    parser.add_argument("--local-scale", type=float, default=0.14)
    parser.add_argument("--search-seed", type=int, default=19_440_001)
    parser.add_argument("--screen-seed-start", type=int, default=1_944_001)
    parser.add_argument("--screen-seed-count", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=64)
    parser.add_argument("--validation-seed-start", type=int, default=2_044_001)
    parser.add_argument("--validation-seed-count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    setup_seconds = time.perf_counter() - setup_started
    names = list(adaptive_genome_names())
    base = None
    if args.base_genomes is not None:
        base = load_genome(args.base_genomes, args.base_genome_index, names)
    elif args.base_override:
        base = np.asarray(adaptive_default_genome(), dtype=np.float64)
    if base is not None:
        for raw_override in args.base_override:
            name, raw_value = raw_override.split("=", 1)
            if name not in names:
                raise ValueError(f"unknown genome field: {name}")
            base[names.index(name)] = float(raw_value)
    genomes = sample_genomes(
        args.candidate_count,
        args.search_seed,
        base,
        global_fraction=args.global_fraction,
        mutation_rate=args.mutation_rate,
        local_scale=args.local_scale,
    )
    screen, screen_games, screen_seconds = evaluate(
        bundle.adaptive_executor, genomes, args.screen_seed_start, args.screen_seed_count
    )
    finalist_indices = [int(row["genome_index"]) for row in screen[: args.finalists]]
    finalist_genomes = genomes[finalist_indices]
    validation, validation_games, validation_seconds = evaluate(
        bundle.adaptive_executor,
        finalist_genomes,
        args.validation_seed_start,
        args.validation_seed_count,
    )
    for row in validation:
        local_index = int(row["genome_index"])
        original_index = finalist_indices[local_index]
        row["screen_genome_index"] = original_index
        row["values"] = {
            name: float(value) for name, value in zip(names, genomes[original_index])
        }
    payload = {
        "schema": "kaggriculture.native-adaptive-passive-search.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "policy_semantics": "general public-state planner; passive route discovery; no opponent identity",
        "candidate_count": args.candidate_count,
        "genome_names": names,
        "search_seed": args.search_seed,
        "search_distribution": {
            "base_genomes": None if args.base_genomes is None else str(args.base_genomes),
            "base_genome_index": args.base_genome_index,
            "base_overrides": list(args.base_override),
            "global_fraction": args.global_fraction,
            "mutation_rate": args.mutation_rate,
            "local_scale": args.local_scale,
        },
        "screen": {
            "seed_start": args.screen_seed_start,
            "seed_count": args.screen_seed_count,
            "games": screen_games,
            "seconds": screen_seconds,
            "games_per_second": screen_games / screen_seconds,
            "top": screen[:100],
        },
        "validation": {
            "seed_start": args.validation_seed_start,
            "seed_count": args.validation_seed_count,
            "games": validation_games,
            "seconds": validation_seconds,
            "games_per_second": validation_games / validation_seconds,
            "ranking": validation,
        },
        "setup_seconds": setup_seconds,
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "backbone": None if args.backbone is None else {
                "path": str(args.backbone), "sha256": sha256(args.backbone)
            },
            "base_genomes": None if args.base_genomes is None else {
                "path": str(args.base_genomes), "sha256": sha256(args.base_genomes)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "screen_games": screen_games,
        "screen_games_per_second": screen_games / screen_seconds,
        "validation_games": validation_games,
        "validation_games_per_second": validation_games / validation_seconds,
        "top_validation": validation[:10],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

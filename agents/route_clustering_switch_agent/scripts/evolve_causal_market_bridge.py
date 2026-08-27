#!/usr/bin/env python3
"""Genetically evolve market-only causal bridges without changing worker paths."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_causal_market_bridges import (
    _csv,
    _play_groups,
    apply_causal_market_genome,
    market_option_space,
    splice_suffix,
)
from search_native_public_trace_counters import _ints, _opponent, _trace


Genome = tuple[int, ...]


def _digest(genome: Genome) -> str:
    return hashlib.sha256(bytes(genome)).hexdigest()


def _fitness(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["completion_rate"]),
        float(row["minimum_opponent_raw_win_rate"]),
        float(row["combined_raw_win_rate"]),
        float(row["minimum_opponent_mean_margin"]),
        float(row["combined_mean_margin"]),
        -float(row.get("mean_macro_market_failures", 0.0)),
    )


def _mutate(
    genome: Genome, option_counts: Sequence[int], rng: random.Random
) -> Genome:
    result = list(genome)
    changes = rng.choices((1, 2, 3), weights=(.72, .22, .06), k=1)[0]
    for step_index in rng.sample(range(len(result)), changes):
        choices = [
            value for value in range(option_counts[step_index])
            if value != result[step_index]
        ]
        if choices:
            result[step_index] = rng.choice(choices)
    return tuple(result)


def _crossover(left: Genome, right: Genome, rng: random.Random) -> Genome:
    if len(left) != len(right):
        raise ValueError("market parent lengths differ")
    if len(left) <= 1:
        return left if rng.random() < .5 else right
    pivot_count = min(len(left) - 1, rng.choice((1, 1, 2, 3)))
    pivots = sorted(rng.sample(range(1, len(left)), pivot_count))
    boundaries = (0, *pivots, len(left))
    values = []
    use_left = bool(rng.randrange(2))
    for start, stop in zip(boundaries, boundaries[1:]):
        source = left if use_left else right
        values.extend(source[start:stop])
        use_left = not use_left
    return tuple(values)


def _descriptor(
    genome: Genome,
    options: Sequence[Sequence[list[Any]]],
    base_genome: Genome,
) -> tuple[int, int, int, int]:
    selected = [options[index][value] for index, value in enumerate(genome)]
    nonempty = sum(bool(value) for value in selected)
    order_count = sum(len(value) for value in selected)
    quantities = sum(
        int(order[2]) if len(order) >= 3 else 1
        for market in selected for order in market
    )
    changes = sum(left != right for left, right in zip(genome, base_genome))
    return nonempty // 4, order_count // 4, quantities // 10, changes // 3


def _route_genome(
    route: Sequence[Mapping[str, Any]],
    steps: Sequence[int],
    options: Sequence[Sequence[list[Any]]],
) -> Genome:
    result = []
    for step, values in zip(steps, options):
        key = json.dumps(
            route[step].get("market", ()) or (), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False,
        )
        lookup = {
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False): index
            for index, value in enumerate(values)
        }
        if key not in lookup:
            raise ValueError(f"seed route market is outside option space at step {step}")
        result.append(lookup[key])
    return tuple(result)


def _next_population(
    rows: Sequence[Mapping[str, Any]],
    population: int,
    elites: int,
    map_elites: int,
    option_counts: Sequence[int],
    rng: random.Random,
) -> list[Genome]:
    ordered = sorted(rows, key=_fitness, reverse=True)
    selected: list[Genome] = []
    seen: set[Genome] = set()

    def add(row: Mapping[str, Any]) -> None:
        genome = tuple(map(int, row["genome"]))
        if genome not in seen:
            seen.add(genome)
            selected.append(genome)

    for row in ordered[:elites]:
        add(row)
    cells = {}
    for row in ordered:
        cells.setdefault(tuple(row["descriptor"]), row)
    for row in sorted(cells.values(), key=_fitness, reverse=True)[:map_elites]:
        add(row)
    parents = selected[:]
    attempts = 0
    while len(selected) < population:
        attempts += 1
        child = _crossover(rng.choice(parents), rng.choice(parents), rng)
        child = _mutate(child, option_counts, rng)
        if child not in seen:
            seen.add(child)
            selected.append(child)
        if attempts > population * 200:
            raise RuntimeError("unable to fill a unique market population")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-family", required=True)
    parser.add_argument("--prefix-family", required=True)
    parser.add_argument("--donor-team", default="nt_68_route_pool")
    parser.add_argument("--seed-elites", type=Path, required=True)
    parser.add_argument("--bridge-start", type=int, required=True)
    parser.add_argument("--branch-step", type=int, required=True)
    parser.add_argument("--default-suffix", required=True)
    parser.add_argument("--branch-suffix", required=True)
    parser.add_argument("--default-opponent", type=_opponent, action="append", required=True)
    parser.add_argument("--branch-opponent", type=_opponent, action="append", required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--holdout-seeds", type=_ints, required=True)
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--map-elites", type=int, default=32)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--holdout-top", type=int, default=48)
    parser.add_argument("--random-seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elite-actions", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.random_seed)
    started = time.perf_counter()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    entries = list(metadata["opponent_routes"])
    route_ids = {str(row["family"]): str(row["route_id"]) for row in entries}
    required = {
        args.base_family, args.prefix_family,
        args.default_suffix, args.branch_suffix,
    }
    missing = sorted(required - set(route_ids))
    if missing:
        raise KeyError(f"unknown family: {missing[0]}")
    tapes = load_action_tapes(args.actions)
    base = tapes[route_ids[args.base_family]]
    prefix = tapes[route_ids[args.prefix_family]]
    donor_families = [
        str(row["family"]) for row in entries
        if str(row.get("team", "")) == args.donor_team
    ]
    donor_tapes = [tapes[route_ids[family]] for family in donor_families]
    steps, options = market_option_space(
        donor_tapes, args.bridge_start, args.branch_step
    )
    option_counts = tuple(len(value) for value in options)
    base_genome = _route_genome(base, steps, options)
    seed_tapes = json.loads(zlib.decompress(args.seed_elites.read_bytes()))
    seed_genomes = [
        _route_genome(route, steps, options)
        for family, route in seed_tapes.items() if family.endswith("_DEFAULT")
    ]
    seed_genomes.extend(_route_genome(route, steps, options) for route in donor_tapes)
    population = list(dict.fromkeys([base_genome, *seed_genomes]))
    while len(population) < args.population:
        candidate = _mutate(rng.choice(population), option_counts, rng)
        if candidate not in population:
            population.append(candidate)
    population = population[:args.population]
    default_suffix = tapes[route_ids[args.default_suffix]]
    branch_suffix = tapes[route_ids[args.branch_suffix]]
    default_opponents = {
        family: _trace(spec) for family, spec in args.default_opponent
    }
    branch_opponents = {
        family: _trace(spec) for family, spec in args.branch_opponent
    }
    archive: dict[str, dict[str, Any]] = {}
    history = []
    games = 0
    native_seconds = 0.0
    for generation in range(args.generations):
        fresh = [genome for genome in population if _digest(genome) not in archive]
        if fresh:
            bridges = [
                apply_causal_market_genome(
                    prefix, base, steps, options, genome, args.bridge_start
                )
                for genome in fresh
            ]
            routes = [(
                splice_suffix(bridge, default_suffix, args.branch_step),
                splice_suffix(bridge, branch_suffix, args.branch_step),
            ) for bridge in bridges]
            rows, elapsed, count = _play_groups(
                args.source, args.actions, args.metadata, routes,
                default_opponents, branch_opponents, args.seeds,
                f"MG{generation:02d}", capture_audit=False,
            )
            games += count
            native_seconds += elapsed
            for genome, row in zip(fresh, rows):
                row["genome"] = list(genome)
                row["genome_sha256"] = _digest(genome)
                row["descriptor"] = list(_descriptor(genome, options, base_genome))
                archive[row["genome_sha256"]] = row
        current = [archive[_digest(genome)] for genome in population]
        ordered = sorted(current, key=_fitness, reverse=True)
        history.append({
            "generation": generation,
            "population": len(population),
            "fresh": len(fresh),
            "archive_size": len(archive),
            "map_cells": len({tuple(row["descriptor"]) for row in current}),
            "best": ordered[0],
        })
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if generation + 1 < args.generations:
            population = _next_population(
                current, args.population, args.elites, args.map_elites,
                option_counts, rng,
            )
    archive_rows = sorted(archive.values(), key=_fitness, reverse=True)
    holdout_genomes = [
        tuple(map(int, row["genome"]))
        for row in archive_rows[:args.holdout_top]
    ]
    holdout_bridges = [
        apply_causal_market_genome(
            prefix, base, steps, options, genome, args.bridge_start
        )
        for genome in holdout_genomes
    ]
    holdout_routes = [(
        splice_suffix(bridge, default_suffix, args.branch_step),
        splice_suffix(bridge, branch_suffix, args.branch_step),
    ) for bridge in holdout_bridges]
    holdout_rows, elapsed, count = _play_groups(
        args.source, args.actions, args.metadata, holdout_routes,
        default_opponents, branch_opponents, args.holdout_seeds,
        "MGH", capture_audit=True,
    )
    games += count
    native_seconds += elapsed
    for genome, row in zip(holdout_genomes, holdout_rows):
        row["genome"] = list(genome)
        row["genome_sha256"] = _digest(genome)
        row["descriptor"] = list(_descriptor(genome, options, base_genome))
    holdout_rows.sort(key=_fitness, reverse=True)
    elite_tapes = {}
    for rank, row in enumerate(holdout_rows, 1):
        genome = tuple(map(int, row["genome"]))
        bridge = apply_causal_market_genome(
            prefix, base, steps, options, genome, args.bridge_start
        )
        default_id = f"MGA{rank:03d}_DEFAULT"
        branch_id = f"MGA{rank:03d}_YARN2"
        elite_tapes[default_id] = splice_suffix(
            bridge, default_suffix, args.branch_step
        )
        elite_tapes[branch_id] = splice_suffix(
            bridge, branch_suffix, args.branch_step
        )
        row["rank"] = rank
        row["default_route_id"] = default_id
        row["branch_route_id"] = branch_id
        row["market_genes"] = {
            str(step): copy.deepcopy(options[index][selected])
            for index, (step, selected) in enumerate(zip(steps, genome))
        }
    packed = zlib.compress(
        json.dumps(elite_tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    args.elite_actions.parent.mkdir(parents=True, exist_ok=True)
    args.elite_actions.write_bytes(packed)
    payload = {
        "schema": "causal-market-bridge-ga-v1",
        "engine": "compiled-cpp-fast-kaggriculture",
        "jax_used": False,
        "base_family": args.base_family,
        "prefix_family": args.prefix_family,
        "donor_team": args.donor_team,
        "donor_count": len(donor_tapes),
        "bridge_start": args.bridge_start,
        "branch_step": args.branch_step,
        "variable_steps": list(steps),
        "option_counts": list(option_counts),
        "default_suffix": args.default_suffix,
        "branch_suffix": args.branch_suffix,
        "training_seeds": list(args.seeds),
        "holdout_seeds": list(args.holdout_seeds),
        "population": args.population,
        "generations": args.generations,
        "archive_size": len(archive),
        "games": games,
        "native_seconds": native_seconds,
        "native_games_per_second": games / native_seconds,
        "wall_seconds": time.perf_counter() - started,
        "history": history,
        "holdout_ranking": holdout_rows,
        "elite_actions": str(args.elite_actions.resolve()),
        "elite_actions_sha256": hashlib.sha256(packed).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "games": games,
        "native_games_per_second": games / native_seconds,
        "wall_seconds": payload["wall_seconds"],
        "top": holdout_rows[:10],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

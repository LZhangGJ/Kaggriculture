#!/usr/bin/env python3
"""Evolve new daily market-intent profiles from a mature replay route."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_tape_mutations import _apply_genes, _evaluate, _ints


Genome = tuple[tuple[int, int, int], ...]
SHIFT_VALUES = (-10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10)
QUANTITY_DELTAS = (-3, -2, -1, 0, 1, 2, 3)


def _active_days(tape: Sequence[Mapping[str, Any]], operation: str, item: str) -> tuple[int, ...]:
    return tuple(sorted({
        step // 24
        for step, action in enumerate(tape)
        for order in action.get("market", []) or []
        if order
        and str(order[0]) == operation
        and (str(order[1]) if len(order) >= 2 else "") == item
    }))


def _canonical(values: Mapping[int, tuple[int, int]], days: Sequence[int]) -> Genome:
    return tuple((day, *values.get(day, (0, 0))) for day in days)


def _gene(genome: Genome, operation: str, item: str) -> list[dict[str, Any]]:
    shifts = {str(day): shift for day, shift, _ in genome if shift}
    quantities = {str(day): delta for day, _, delta in genome if delta}
    if not shifts and not quantities:
        return []
    return [{
        "operator": "market_profile",
        "operation": operation,
        "item": item,
        "daily_shift": shifts,
        "daily_quantity_delta": quantities,
    }]


def _mutate(genome: Genome, rng: random.Random, strength: int = 1) -> Genome:
    values = {day: (shift, quantity) for day, shift, quantity in genome}
    days = tuple(values)
    for _ in range(max(1, strength)):
        day = rng.choice(days)
        shift, quantity = values[day]
        if rng.random() < .72:
            choices = [value for value in SHIFT_VALUES if value != shift]
            shift = rng.choice(choices)
        else:
            choices = [value for value in QUANTITY_DELTAS if value != quantity]
            quantity = rng.choice(choices)
        values[day] = (shift, quantity)
    return _canonical(values, days)


def _crossover(left: Genome, right: Genome, rng: random.Random) -> Genome:
    return tuple(
        a if rng.random() < .5 else b
        for a, b in zip(left, right)
    )


def _fitness(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("one_sided_95pct_score_lower", row["mean_score"])),
        float(row["mean_score"]),
        float(row["one_sided_95pct_margin_lower"]),
        float(row["mean_margin"]),
    )


def _genome_from_row(
    row: Mapping[str, Any], days: Sequence[int], operation: str, item: str
) -> Genome | None:
    if row.get("genome"):
        raw = {
            int(day): (int(shift), int(quantity))
            for day, shift, quantity in row["genome"]
        }
        return _canonical(raw, days)
    for gene in row.get("genes", []) or []:
        if (
            str(gene.get("operator")) == "market_profile"
            and str(gene.get("operation")) == operation
            and str(gene.get("item")) == item
        ):
            shifts = {int(key): int(value) for key, value in gene.get("daily_shift", {}).items()}
            quantities = {
                int(key): int(value)
                for key, value in gene.get("daily_quantity_delta", {}).items()
            }
            return _canonical({
                day: (shifts.get(day, 0), quantities.get(day, 0)) for day in days
            }, days)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent", default="G006")
    parser.add_argument("--opening", default="G001")
    parser.add_argument("--operation", default="BUY_PRODUCT")
    parser.add_argument("--item", default="WHEAT")
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=20)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument("--initial-from", type=Path)
    parser.add_argument("--initial-top", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    rng = random.Random(args.random_seed)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    parent_entry = next(
        value for value in metadata["opponent_routes"]
        if str(value["family"]) == args.parent
    )
    parent_tape = load_action_tapes(args.actions)[str(parent_entry["route_id"])]
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    days = _active_days(parent_tape, args.operation, args.item)
    base = _canonical({}, days)
    global_early = _canonical({day: (-6, 0) for day in days if 6 <= day < 12}, days)
    population = {base, global_early}
    if args.initial_from:
        initial_payload = json.loads(args.initial_from.read_text(encoding="utf-8"))
        for row in list(initial_payload["ranking"])[:args.initial_top]:
            genome = _genome_from_row(row, days, args.operation, args.item)
            if genome is not None:
                population.add(genome)
    while len(population) < args.population:
        origins = tuple(population)
        origin = rng.choice(origins) if rng.random() < .8 else base
        population.add(_mutate(origin, rng, rng.randint(1, 4)))

    cache: dict[Genome, dict[str, Any]] = {}
    history = []
    for generation in range(args.generations):
        fresh = [genome for genome in population if genome not in cache]
        if fresh:
            families = [f"E{generation:02d}_{index:04d}" for index in range(len(fresh))]
            routes = {
                family: _apply_genes(
                    parent_tape, _gene(genome, args.operation, args.item)
                )
                for family, genome in zip(families, fresh)
            }
            bundle = NativeTeammateBundle(
                args.source, args.actions, args.metadata, additional_routes=routes
            )
            rows = _evaluate(bundle, families, args.seeds, policy, args.opening)
            for genome, row in zip(fresh, rows):
                row["genes"] = _gene(genome, args.operation, args.item)
                row["genome"] = [list(value) for value in genome]
                row["generation_first_seen"] = generation
                cache[genome] = row
        ranked = sorted(population, key=lambda genome: _fitness(cache[genome]), reverse=True)
        elites = ranked[:args.elites]
        best = cache[elites[0]]
        history.append({
            "generation": generation,
            "new_candidates": len(fresh),
            "unique_candidates": len(cache),
            "best": best,
        })
        print(json.dumps({
            "generation": generation,
            "new_candidates": len(fresh),
            "unique_candidates": len(cache),
            "elapsed_seconds": time.perf_counter() - started,
            "best": best,
        }), flush=True)
        next_population = set(elites)
        # Retain a few intentionally simpler genomes; they are harder to
        # overfit and keep the mutation path connected to the replay parent.
        next_population.update((base, global_early))
        while len(next_population) < args.population:
            if rng.random() < .7:
                child = _crossover(rng.choice(elites), rng.choice(elites), rng)
            else:
                child = rng.choice(elites)
            child = _mutate(child, rng, 1 if rng.random() < .75 else rng.randint(2, 4))
            next_population.add(child)
        population = next_population

    ranking = sorted(cache.values(), key=_fitness, reverse=True)
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    payload = {
        "schema": "native-intent-profile-evolution-v1",
        "parent": args.parent,
        "opening": args.opening,
        "operation": args.operation,
        "item": args.item,
        "active_days": list(days),
        "seeds": list(args.seeds),
        "population": args.population,
        "elites": args.elites,
        "generations": args.generations,
        "candidate_count": len(ranking),
        "games": len(ranking) * len(args.seeds) * 2,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "ranking": ranking,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "candidates": len(ranking),
        "games": payload["games"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "top10": ranking[:10],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

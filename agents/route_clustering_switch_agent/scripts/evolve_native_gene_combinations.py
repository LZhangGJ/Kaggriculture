#!/usr/bin/env python3
"""Evolve combinations of atomic replay-tape mutations against a policy."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_tape_mutations import _apply_genes, _evaluate, _ints


Genome = tuple[int, ...]


def _fitness(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["one_sided_95pct_score_lower"]),
        float(row["mean_score"]),
        float(row["one_sided_95pct_margin_lower"]),
        float(row["mean_margin"]),
    )


def _canonical(values: set[int], maximum: int, rng: random.Random) -> Genome:
    selected = sorted(values)
    if len(selected) > maximum:
        selected = sorted(rng.sample(selected, maximum))
    return tuple(selected)


def _mutate(
    genome: Genome,
    atom_count: int,
    maximum: int,
    rng: random.Random,
    strength: int = 1,
) -> Genome:
    values = set(genome)
    for _ in range(max(1, strength)):
        draw = rng.random()
        if draw < .48 and len(values) < maximum:
            values.add(rng.randrange(atom_count))
        elif draw < .72 and values:
            values.remove(rng.choice(tuple(values)))
        else:
            if values:
                values.remove(rng.choice(tuple(values)))
            values.add(rng.randrange(atom_count))
    return _canonical(values, maximum, rng)


def _crossover(left: Genome, right: Genome, maximum: int, rng: random.Random) -> Genome:
    values = {
        value for value in set(left) | set(right)
        if rng.random() < .5 or value in (set(left) & set(right))
    }
    if not values:
        source = left or right
        if source:
            values.add(rng.choice(source))
    return _canonical(values, maximum, rng)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--atoms-from", type=Path, required=True)
    parser.add_argument("--top-atoms", type=int, default=80)
    parser.add_argument("--parent", default="G001")
    parser.add_argument("--opening", default="G001")
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--elites", type=int, default=20)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--max-genes", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=20260826)
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
    atom_payload = json.loads(args.atoms_from.read_text(encoding="utf-8"))
    atoms: list[dict[str, Any]] = []
    atom_keys: set[str] = set()
    for row in atom_payload["ranking"]:
        genes = list(row.get("genes") or [])
        if len(genes) != 1:
            continue
        key = json.dumps(genes[0], sort_keys=True, separators=(",", ":"))
        if key in atom_keys:
            continue
        atom_keys.add(key)
        atoms.append(genes[0])
        if len(atoms) >= args.top_atoms:
            break
    if len(atoms) < 2:
        raise ValueError("at least two unique atomic genes are required")

    population: set[Genome] = {(), *((index,) for index in range(min(len(atoms), args.population - 1)))}
    while len(population) < args.population:
        size = rng.randint(2, min(args.max_genes, 4))
        population.add(tuple(sorted(rng.sample(range(len(atoms)), size))))

    cache: dict[Genome, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    for generation in range(args.generations):
        fresh = [genome for genome in population if genome not in cache]
        if fresh:
            names = [f"C{generation:02d}_{index:04d}" for index in range(len(fresh))]
            routes = {
                name: _apply_genes(parent_tape, [atoms[index] for index in genome])
                for name, genome in zip(names, fresh)
            }
            bundle = NativeTeammateBundle(
                args.source, args.actions, args.metadata, additional_routes=routes
            )
            rows = _evaluate(bundle, names, args.seeds, policy, args.opening)
            for genome, row in zip(fresh, rows):
                row["genome"] = list(genome)
                row["genes"] = [atoms[index] for index in genome]
                row["generation_first_seen"] = generation
                cache[genome] = row
        ranked = sorted(population, key=lambda value: _fitness(cache[value]), reverse=True)
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
            "best_score": best["mean_score"],
            "best_score_lower": best["one_sided_95pct_score_lower"],
            "best_margin": best["mean_margin"],
            "best_genome": best["genome"],
        }), flush=True)
        next_population = set(elites)
        next_population.add(())
        while len(next_population) < args.population:
            if rng.random() < .7:
                child = _crossover(rng.choice(elites), rng.choice(elites), args.max_genes, rng)
            else:
                child = rng.choice(elites)
            child = _mutate(
                child, len(atoms), args.max_genes, rng,
                1 if rng.random() < .8 else rng.randint(2, 3),
            )
            next_population.add(child)
        population = next_population

    ranking = sorted(cache.values(), key=_fitness, reverse=True)
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    payload = {
        "schema": "native-atomic-gene-combination-evolution-v1",
        "parent": args.parent,
        "opening": args.opening,
        "seeds": list(args.seeds),
        "atoms": atoms,
        "population": args.population,
        "elites": args.elites,
        "generations": args.generations,
        "max_genes": args.max_genes,
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
        "top10": [{
            key: row[key] for key in (
                "rank", "mean_score", "one_sided_95pct_score_lower",
                "mean_margin", "one_sided_95pct_margin_lower", "genome", "genes",
            )
        } for row in ranking[:10]],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

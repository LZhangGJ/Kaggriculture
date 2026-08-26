#!/usr/bin/env python3
"""Evolve executable replay-tape variants by marginal switch-library coverage."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_tape_mutations import _apply_genes, _candidate_genes, _ints


Genome = tuple[str, tuple[int, ...]]


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _fitness(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(row["incremental_oracle_score"]),
        float(row["incremental_oracle_margin"]),
        float(row["candidate_score"]),
        float(row["candidate_margin"]),
        -float(len(row["genome"])),
    )


def _canonical(values: set[int], maximum: int, rng: random.Random) -> tuple[int, ...]:
    result = sorted(values)
    if len(result) > maximum:
        result = sorted(rng.sample(result, maximum))
    return tuple(result)


def _mutate(
    genome: Genome,
    atoms: Mapping[str, list[dict[str, Any]]],
    parents: tuple[str, ...],
    maximum: int,
    rng: random.Random,
) -> Genome:
    parent, raw = genome
    if rng.random() < .04:
        parent = rng.choice([value for value in parents if value != parent])
        raw = ()
    values = set(raw)
    draw = rng.random()
    if draw < .52 and len(values) < maximum:
        values.add(rng.randrange(len(atoms[parent])))
    elif draw < .74 and values:
        values.remove(rng.choice(tuple(values)))
    else:
        if values:
            values.remove(rng.choice(tuple(values)))
        values.add(rng.randrange(len(atoms[parent])))
    return parent, _canonical(values, maximum, rng)


def _crossover(
    left: Genome, right: Genome, maximum: int, rng: random.Random
) -> Genome:
    if left[0] != right[0]:
        return rng.choice((left, right))
    values = {
        value for value in set(left[1]) | set(right[1])
        if rng.random() < .5 or value in (set(left[1]) & set(right[1]))
    }
    return left[0], _canonical(values, maximum, rng)


def _representative_opponents(
    families: tuple[str, ...], required: tuple[str, ...], limit: int
) -> tuple[str, ...]:
    selected = list(dict.fromkeys(value for value in required if value in families))
    if limit <= 0 or limit >= len(families):
        return families
    for index in np.linspace(0, len(families) - 1, limit, dtype=int):
        family = families[int(index)]
        if family not in selected:
            selected.append(family)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected.extend(value for value in families if value not in selected)
    return tuple(selected[:limit])


def _seed_genomes(
    archives: list[Path],
    atoms: Mapping[str, list[dict[str, Any]]],
) -> set[Genome]:
    atom_index = {
        parent: {
            json.dumps(gene, sort_keys=True, separators=(",", ":")): index
            for index, gene in enumerate(values)
        }
        for parent, values in atoms.items()
    }
    result: set[Genome] = set()
    for path in archives:
        payload = json.loads(path.read_text(encoding="utf-8"))
        parent = str(payload.get("parent", ""))
        if parent not in atoms:
            continue
        for row in payload.get("ranking", [])[:32]:
            indices = []
            for gene in row.get("genes", []) or []:
                key = json.dumps(gene, sort_keys=True, separators=(",", ":"))
                if key in atom_index[parent]:
                    indices.append(atom_index[parent][key])
            if indices:
                result.add((parent, tuple(sorted(set(indices)))))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parents", type=_csv)
    parser.add_argument("--openings", type=_csv, default=("G001",))
    parser.add_argument("--base-targets", type=_csv)
    parser.add_argument("--opponents", type=_csv)
    parser.add_argument("--opponent-limit", type=int, default=32)
    parser.add_argument("--checkpoints", type=_ints, default=(144, 168, 216))
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--elites-per-parent", type=int, default=2)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--max-genes", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument("--seed-archive", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    rng = random.Random(args.random_seed)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    entries = list(metadata["opponent_routes"])
    base_families = tuple(str(value["family"]) for value in entries)
    route_id = {str(value["family"]): str(value["route_id"]) for value in entries}
    tapes = load_action_tapes(args.actions)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    policy_targets = tuple(str(value) for value in policy.get("targets", ()))
    parents = args.parents or policy_targets
    base_targets = args.base_targets or policy_targets
    if args.population < len(parents):
        parser.error(
            f"--population must be at least the parent count ({len(parents)})"
        )
    for family in (*parents, *args.openings, *base_targets):
        if family not in route_id:
            raise KeyError(f"unknown base family: {family}")
    opponents = args.opponents or _representative_opponents(
        base_families, tuple(dict.fromkeys((*parents, *base_targets))), args.opponent_limit
    )
    for family in opponents:
        if family not in route_id:
            raise KeyError(f"unknown opponent family: {family}")

    atom_map: dict[str, list[dict[str, Any]]] = {}
    parent_tapes: dict[str, list[dict[str, Any]]] = {}
    for parent in parents:
        tape = tapes[route_id[parent]]
        parent_tapes[parent] = tape
        atom_map[parent] = [
            list(genes)[0] for genes in _candidate_genes(tape) if len(list(genes)) == 1
        ]
        if not atom_map[parent]:
            raise ValueError(f"no mutations available for parent {parent}")

    population: set[Genome] = {(parent, ()) for parent in parents}
    population.update(_seed_genomes(args.seed_archive, atom_map))
    while len(population) < args.population:
        parent = rng.choice(parents)
        size = rng.randint(1, min(3, args.max_genes))
        population.add((parent, tuple(sorted(rng.sample(range(len(atom_map[parent])), size)))))
    if len(population) > args.population:
        mandatory = {(parent, ()) for parent in parents}
        extras = sorted(population - mandatory)
        population = mandatory | set(rng.sample(extras, args.population - len(mandatory)))

    cache: dict[Genome, dict[str, Any]] = {}
    score_vectors: dict[Genome, np.ndarray] = {}
    margin_vectors: dict[Genome, np.ndarray] = {}
    history: list[dict[str, Any]] = []
    base_oracle_score = None
    base_oracle_margin = None
    for generation in range(args.generations):
        fresh = [genome for genome in population if genome not in cache]
        if fresh:
            names = [f"EV{generation:02d}_{index:04d}" for index in range(len(fresh))]
            routes = {
                name: _apply_genes(
                    parent_tapes[genome[0]],
                    [atom_map[genome[0]][index] for index in genome[1]],
                )
                for name, genome in zip(names, fresh)
            }
            bundle = NativeTeammateBundle(
                args.source, args.actions, args.metadata, additional_routes=routes
            )
            target_names = (*base_targets, *names)
            result = bundle.executor.switch_search(
                [bundle.index(value) for value in args.openings],
                [bundle.index(value) for value in target_names],
                args.checkpoints,
                args.seeds,
                [bundle.index(value) for value in opponents],
            )
            scores = np.asarray(result["outcome"], dtype=np.float32) * .5
            margins = np.asarray(result["margin"], dtype=np.float32)
            base_count = len(base_targets)
            base_scores = np.max(scores[:, :, :base_count], axis=2)
            base_margin = np.max(
                np.where(scores[:, :, :base_count] == base_scores[:, :, None],
                         margins[:, :, :base_count], -np.inf),
                axis=2,
            )
            if base_oracle_score is None:
                base_oracle_score = float(np.mean(base_scores))
                base_oracle_margin = float(np.mean(base_margin))
            for index, genome in enumerate(fresh):
                candidate_scores = scores[:, :, base_count + index]
                candidate_margins = margins[:, :, base_count + index]
                next_scores = np.maximum(base_scores, candidate_scores)
                next_margins = np.where(
                    candidate_scores > base_scores,
                    candidate_margins,
                    np.where(
                        candidate_scores < base_scores,
                        base_margin,
                        np.maximum(base_margin, candidate_margins),
                    ),
                )
                checkpoint_rows = []
                for checkpoint_index, checkpoint in enumerate(args.checkpoints):
                    checkpoint_rows.append({
                        "checkpoint": int(checkpoint),
                        "incremental_oracle_score": float(np.mean(
                            next_scores[:, checkpoint_index] - base_scores[:, checkpoint_index]
                        )),
                        "incremental_oracle_margin": float(np.mean(
                            next_margins[:, checkpoint_index] - base_margin[:, checkpoint_index]
                        )),
                        "candidate_score": float(np.mean(candidate_scores[:, checkpoint_index])),
                        "candidate_margin": float(np.mean(candidate_margins[:, checkpoint_index])),
                    })
                best_checkpoint = max(
                    checkpoint_rows,
                    key=lambda row: (
                        row["incremental_oracle_score"],
                        row["incremental_oracle_margin"],
                        row["candidate_score"],
                        row["candidate_margin"],
                    ),
                )
                row = {
                    "parent": genome[0],
                    "genome": list(genome[1]),
                    "genes": [atom_map[genome[0]][value] for value in genome[1]],
                    "generation_first_seen": generation,
                    **best_checkpoint,
                    "checkpoint_metrics": checkpoint_rows,
                }
                cache[genome] = row
                score_vectors[genome] = candidate_scores.reshape(-1).astype(np.float32)
                margin_vectors[genome] = candidate_margins.reshape(-1).astype(np.float32)

        ranked = sorted(population, key=lambda value: _fitness(cache[value]), reverse=True)
        elites = list(ranked[:args.elites])
        by_parent: dict[str, list[Genome]] = defaultdict(list)
        for genome in ranked:
            by_parent[genome[0]].append(genome)
        for parent in parents:
            for genome in by_parent[parent][:args.elites_per_parent]:
                if genome not in elites:
                    elites.append(genome)
        best = cache[ranked[0]]
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
            "best_parent": best["parent"],
            "best_checkpoint": best["checkpoint"],
            "incremental_oracle_score": best["incremental_oracle_score"],
            "candidate_score": best["candidate_score"],
        }), flush=True)
        next_population = set(elites)
        next_population.update((parent, ()) for parent in parents)
        while len(next_population) < args.population:
            if rng.random() < .65:
                child = _crossover(rng.choice(elites), rng.choice(elites), args.max_genes, rng)
            else:
                child = rng.choice(elites)
            child = _mutate(child, atom_map, parents, args.max_genes, rng)
            next_population.add(child)
        population = set(sorted(next_population, key=lambda value: (value[0], value[1]))[:args.population])

    ordered_genomes = sorted(cache, key=lambda value: _fitness(cache[value]), reverse=True)
    ranking = []
    for rank, genome in enumerate(ordered_genomes, 1):
        row = dict(cache[genome])
        row["rank"] = rank
        ranking.append(row)
    matrix = np.stack([score_vectors[value] for value in ordered_genomes])
    margin_matrix = np.stack([margin_vectors[value] for value in ordered_genomes])
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.matrix_output,
        scores=matrix,
        margins=margin_matrix,
        parents=np.asarray([value[0] for value in ordered_genomes]),
        genomes=np.asarray([json.dumps(value[1]) for value in ordered_genomes]),
        openings=np.asarray(args.openings),
        checkpoints=np.asarray(args.checkpoints, dtype=np.int16),
        opponents=np.asarray(opponents),
        seeds=np.asarray(args.seeds, dtype=np.int64),
    )
    payload = {
        "schema": "native-evolved-switch-route-library-v1",
        "source_actions": str(args.actions.resolve()),
        "source_metadata": str(args.metadata.resolve()),
        "parents": list(parents),
        "base_targets": list(base_targets),
        "openings": list(args.openings),
        "opponents": list(opponents),
        "checkpoints": list(args.checkpoints),
        "seeds": list(args.seeds),
        "population": args.population,
        "elites": args.elites,
        "generations": args.generations,
        "max_genes": args.max_genes,
        "candidate_count": len(ranking),
        "base_oracle_score": base_oracle_score,
        "base_oracle_margin": base_oracle_margin,
        "matrix_output": str(args.matrix_output.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "matrix_output": str(args.matrix_output.resolve()),
        "candidates": len(ranking),
        "base_oracle_score": base_oracle_score,
        "top10": ranking[:10],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

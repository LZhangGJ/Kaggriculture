#!/usr/bin/env python3
"""Evolve exact-prefix suffix routes against a dynamic tree-policy opponent."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import zlib
from pathlib import Path

import numpy as np

from evolve_native_route_library import (
    _apply_evolution_genes,
    _behavior_descriptor,
    _canonical,
    _crossover,
    _donor_atoms,
    _mutate,
)
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.search_route_policy import NumpySearchTree
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_tape_mutations import _candidate_genes


Genome = tuple[str, tuple[int, ...]]


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(part) for part in value.split(",") if part.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-actions", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--route-actions", type=Path, required=True)
    parser.add_argument("--route-metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--opening", default="NR295")
    parser.add_argument("--opponent-opening", default="G001")
    parser.add_argument("--donors", type=_csv, required=True)
    parser.add_argument("--checkpoint", type=int, default=96)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--max-genes", type=int, default=8)
    parser.add_argument("--seed-folds", type=int, default=4)
    parser.add_argument(
        "--selector-template", type=Path,
        help="Existing target matrix used to reward public-state complementarity.",
    )
    parser.add_argument("--random-seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--output-matrix", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    rng = random.Random(args.random_seed)
    metadata = json.loads(args.route_metadata.read_text(encoding="utf-8"))
    route_ids = {str(row["family"]): str(row["route_id"]) for row in metadata["opponent_routes"]}
    source_tapes = load_action_tapes(args.route_actions)
    parent_tapes = {
        family: source_tapes[route_ids[family]]
        for family in (args.opening, *args.donors)
    }
    atoms = [
        list(genes)[0]
        for genes in _candidate_genes(parent_tapes[args.opening])
        if len(list(genes)) == 1
    ]
    atoms.extend(_donor_atoms(args.opening, (args.opening, *args.donors), parent_tapes))
    atoms = [gene for gene in atoms if int(gene.get("start", 0)) >= args.checkpoint]
    atom_map = {args.opening: atoms}
    if not atoms:
        raise ValueError("no suffix mutation atoms")

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    nodes = [
        (int(node["selected"]["checkpoint"]), NumpySearchTree(node["selected"]["tree"]))
        for node in policy["nodes"]
        if node["selected"].get("enabled", True)
        and str(node["selected"]["opening"]) == args.opponent_opening
    ]
    policy_families = {
        args.opponent_opening,
        *(family for _, tree in nodes for family in tree.classes),
    }
    selector_base = None
    selector_state_ids = None
    selector_base_rate = 0.0
    if args.selector_template:
        with np.load(args.selector_template, allow_pickle=False) as saved:
            if not np.array_equal(saved["seeds"], np.asarray(args.seeds)):
                raise ValueError("selector template seeds do not match evolution seeds")
            selector_base = (
                saved["outcome"][0, 0, :, 0].reshape(len(saved["targets"]), -1).T
                == 2
            ).astype(np.float32)
            states = saved["states"][0, 0, 0].reshape(-1, saved["states"].shape[-1])
            _, selector_state_ids = np.unique(states, axis=0, return_inverse=True)

        selector_seed_ids = np.repeat(np.arange(len(args.seeds)), 2)
        selector_folds = np.array_split(
            np.arange(len(args.seeds)), min(args.seed_folds, len(args.seeds))
        )

        def selector_metrics(candidate: np.ndarray | None = None) -> dict[str, float]:
            scores = selector_base if candidate is None else np.column_stack(
                (selector_base, candidate.astype(np.float32))
            )
            predictions = np.zeros(len(selector_state_ids), dtype=np.int32)
            for state in np.unique(selector_state_ids):
                rows = selector_state_ids == state
                predictions[rows] = int(np.argmax(np.mean(scores[rows], axis=0)))
            training = float(np.mean(
                scores[np.arange(len(scores)), predictions] == 1.0
            ))
            cross_validation = []
            for valid_seeds in selector_folds:
                valid = np.isin(selector_seed_ids, valid_seeds)
                train = ~valid
                fold_predictions = np.zeros(np.sum(valid), dtype=np.int32)
                valid_states = selector_state_ids[valid]
                for state in np.unique(valid_states):
                    state_train = train & (selector_state_ids == state)
                    if np.any(state_train):
                        action = int(np.argmax(np.mean(scores[state_train], axis=0)))
                    else:
                        action = 0
                    fold_predictions[valid_states == state] = action
                valid_scores = scores[valid]
                cross_validation.append(float(np.mean(
                    valid_scores[np.arange(len(valid_scores)), fold_predictions] == 1.0
                )))
            return {
                "selector_raw_win_rate": training,
                "selector_cross_validation_raw_win_rate": float(np.mean(cross_validation)),
                "selector_minimum_fold_raw_win_rate": float(min(cross_validation)),
            }

        selector_base_metrics = selector_metrics()
        selector_base_rate = selector_base_metrics["selector_raw_win_rate"]
    else:
        selector_metrics = lambda candidate=None: {
            "selector_raw_win_rate": 0.0,
            "selector_cross_validation_raw_win_rate": 0.0,
            "selector_minimum_fold_raw_win_rate": 0.0,
        }
        selector_base_metrics = selector_metrics()

    def materialize(genome: Genome) -> list[dict]:
        tape = _apply_evolution_genes(
            parent_tapes[genome[0]], [atoms[index] for index in genome[1]], parent_tapes
        )
        if tape[:args.checkpoint] != parent_tapes[args.opening][:args.checkpoint]:
            raise AssertionError("mutation violated the exact-prefix invariant")
        return tape

    def fitness(row: dict) -> tuple[float, ...]:
        return (
            row["selector_minimum_fold_raw_win_rate"],
            row["selector_cross_validation_raw_win_rate"],
            row["selector_raw_win_rate"],
            row["incremental_selector_raw_win_rate"],
            row["raw_win_rate"],
            row["minimum_seed_fold_raw_win_rate"],
            row["minimum_seat_raw_win_rate"],
            row["mean_margin"],
            -len(row["genome"]),
        )

    population: set[Genome] = {(args.opening, ())}
    while len(population) < args.population:
        size = rng.randint(1, min(4, args.max_genes))
        values = set(rng.sample(range(len(atoms)), size))
        population.add((
            args.opening, _canonical(values, args.max_genes, rng, atoms)
        ))
    cache: dict[Genome, dict] = {}
    outcome_cache: dict[Genome, np.ndarray] = {}
    history = []
    for generation in range(args.generations):
        fresh = [genome for genome in population if genome not in cache]
        names = [f"DT{generation:02d}_{index:04d}" for index in range(len(fresh))]
        routes = {name: materialize(genome) for name, genome in zip(names, fresh)}
        if routes:
            bundle = NativeTeammateBundle(
                args.source, args.base_actions, args.base_metadata,
                additional_routes=routes,
                included_families=sorted(policy_families),
            )
            opponent_index = bundle.index(args.opponent_opening)
            samples = [
                (name, seed, seat)
                for name in names for seed in args.seeds for seat in (0, 1)
            ]
            opponent_steps = np.full(len(samples), -1, dtype=np.int64)
            opponent_targets = np.full(len(samples), opponent_index, dtype=np.int64)
            for checkpoint, tree in sorted(nodes):
                tasks = np.empty((len(samples), 6), dtype=np.int64)
                for row, (name, seed, seat) in enumerate(samples):
                    routes_pair = [opponent_index, opponent_index]
                    routes_pair[seat] = bundle.index(name)
                    tasks[row] = [
                        *routes_pair, seed, checkpoint, 1 - seat, opponent_index
                    ]
                features = np.asarray(bundle.executor.features_batch(tasks))
                for row, vector in enumerate(features):
                    if opponent_steps[row] >= 0:
                        continue
                    prediction = tree.predict(vector)
                    if prediction != args.opponent_opening:
                        opponent_steps[row] = checkpoint
                        opponent_targets[row] = bundle.index(prediction)
            games = np.empty((len(samples), 7), dtype=np.int64)
            for row, (name, seed, seat) in enumerate(samples):
                routes_pair = [opponent_index, opponent_index]
                routes_pair[seat] = bundle.index(name)
                switches = [opponent_steps[row], opponent_targets[row]] * 2
                switches[2 * seat:2 * seat + 2] = [-1, -1]
                games[row] = [*routes_pair, seed, *switches]
            rewards = np.asarray(bundle.executor.play_batch(games), dtype=np.float64)
            block = len(args.seeds) * 2
            for position, (name, genome) in enumerate(zip(names, fresh)):
                start, stop = position * block, (position + 1) * block
                candidate_seats = np.tile((0, 1), len(args.seeds))
                own = rewards[start:stop][np.arange(block), candidate_seats]
                other = rewards[start:stop][np.arange(block), 1 - candidate_seats]
                margins = own - other
                wins = margins > 0
                outcome_cache[genome] = wins.astype(np.uint8)
                seat_rates = wins.reshape(len(args.seeds), 2).mean(axis=0)
                fold_rates = [
                    float(np.mean(wins.reshape(len(args.seeds), 2)[indices]))
                    for indices in np.array_split(
                        np.arange(len(args.seeds)), min(args.seed_folds, len(args.seeds))
                    ) if len(indices)
                ]
                complement = selector_metrics(wins)
                cache[genome] = {
                    "parent": genome[0],
                    "genome": list(genome[1]),
                    "genes": [atoms[index] for index in genome[1]],
                    "descriptor": list(_behavior_descriptor(routes[name], parent_tapes[genome[0]])),
                    "generation_first_seen": generation,
                    "raw_win_rate": float(np.mean(wins)),
                    **complement,
                    "incremental_selector_raw_win_rate": (
                        complement["selector_raw_win_rate"] - selector_base_rate
                    ),
                    "minimum_seat_raw_win_rate": float(np.min(seat_rates)),
                    "seat_raw_win_rates": seat_rates.tolist(),
                    "minimum_seed_fold_raw_win_rate": float(min(fold_rates)),
                    "seed_fold_raw_win_rates": fold_rates,
                    "mean_margin": float(np.mean(margins)),
                    "minimum_margin": float(np.min(margins)),
                }
        ranked = sorted(population, key=lambda genome: fitness(cache[genome]), reverse=True)
        elites = ranked[:args.elites]
        best = cache[ranked[0]]
        history.append({"generation": generation, "unique": len(cache), "best": best})
        print(json.dumps({
            "generation": generation,
            "fresh": len(fresh),
            "unique": len(cache),
            "elapsed_seconds": time.perf_counter() - started,
            "best": {key: best[key] for key in (
                "selector_minimum_fold_raw_win_rate",
                "selector_cross_validation_raw_win_rate",
                "selector_raw_win_rate", "incremental_selector_raw_win_rate",
                "raw_win_rate", "minimum_seat_raw_win_rate",
                "minimum_seed_fold_raw_win_rate", "mean_margin", "genome",
            )},
        }), flush=True)
        next_population = set(elites)
        while len(next_population) < args.population:
            child = _crossover(
                rng.choice(elites), rng.choice(elites), args.max_genes, atom_map, rng
            )
            child = _mutate(
                child, atom_map, (args.opening,), args.max_genes, rng
            )
            next_population.add(child)
        population = next_population

    ranked = sorted(cache, key=lambda genome: fitness(cache[genome]), reverse=True)
    ranking = []
    deployed = {}
    for rank, genome in enumerate(ranked, 1):
        row = dict(cache[genome])
        row["rank"] = rank
        row["family"] = f"DT{rank:04d}"
        ranking.append(row)
        if rank <= 64:
            deployed[row["family"]] = materialize(genome)
    packed = zlib.compress(
        json.dumps(deployed, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_actions.parent.mkdir(parents=True, exist_ok=True)
    args.output_actions.write_bytes(packed)
    if args.output_matrix:
        args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_matrix,
            outcome=np.stack([outcome_cache[genome] for genome in ranked]).reshape(
                len(ranked), len(args.seeds), 2
            ),
            families=np.asarray([f"DT{rank:04d}" for rank in range(1, len(ranked) + 1)]),
            genomes=np.asarray([json.dumps(genome[1]) for genome in ranked]),
            seeds=np.asarray(args.seeds, dtype=np.int64),
            checkpoint=np.asarray(args.checkpoint, dtype=np.int16),
        )
    payload = {
        "schema": "dynamic-tree-counter-suffix-evolution-v1",
        "opening": args.opening,
        "checkpoint": args.checkpoint,
        "opponent_opening": args.opponent_opening,
        "donors": list(args.donors),
        "seeds": list(args.seeds),
        "population": args.population,
        "generations": args.generations,
        "atom_count": len(atoms),
        "prefix_invariant": f"route[:{args.checkpoint}] == {args.opening}[:{args.checkpoint}]",
        "actions": str(args.output_actions.resolve()),
        "actions_sha256": hashlib.sha256(packed).hexdigest(),
        "selector_template": str(args.selector_template.resolve()) if args.selector_template else None,
        "selector_base_raw_win_rate": selector_base_rate,
        "selector_base_metrics": selector_base_metrics,
        "matrix": str(args.output_matrix.resolve()) if args.output_matrix else None,
        "history": history,
        "ranking": ranking,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "actions": str(args.output_actions.resolve()),
        "unique": len(ranking),
        "best": ranking[0],
        "elapsed_seconds": time.perf_counter() - started,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

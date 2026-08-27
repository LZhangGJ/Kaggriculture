#!/usr/bin/env python3
"""Evolve causal multi-action route bridges with the compiled C++ engine."""

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

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_public_trace_counters import _ints, _opponent, _trace


Genome = tuple[tuple[int, ...], tuple[int, ...], int, int]


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _splice(
    prefix: Sequence[Mapping[str, Any]],
    donors: Sequence[Sequence[Mapping[str, Any]]],
    block_genes: Sequence[int],
    suffix: Sequence[Mapping[str, Any]],
    bridge_start: int,
    branch_step: int,
    block_size: int,
) -> list[dict[str, Any]]:
    if (branch_step - bridge_start) % block_size:
        raise ValueError("causal interval must be divisible by block size")
    if len(block_genes) != (branch_step - bridge_start) // block_size:
        raise ValueError("bridge genome length does not match causal interval")
    bridge = []
    for block, donor_index in enumerate(block_genes):
        start = bridge_start + block * block_size
        bridge.extend(donors[donor_index][start:start + block_size])
    return copy.deepcopy([
        *prefix[:bridge_start], *bridge, *suffix[branch_step:719]
    ])


def _mutate(
    genome: Genome,
    donor_count: int,
    default_suffix_count: int,
    branch_suffix_count: int,
    rng: random.Random,
) -> Genome:
    default_actions, branch_actions, default_suffix, branch_suffix = genome
    draw = rng.random()
    if draw < .68:
        mutate_default = rng.random() < .5
        values = list(default_actions if mutate_default else branch_actions)
        lengths = (1, 1, 1, 2, 2, 3)
        length = min(rng.choice(lengths), len(values))
        start = rng.randrange(len(values) - length + 1)
        donor = rng.randrange(donor_count)
        values[start:start + length] = [donor] * length
        if mutate_default:
            default_actions = tuple(values)
        else:
            branch_actions = tuple(values)
    elif draw < .84:
        default_suffix = rng.randrange(default_suffix_count)
    else:
        branch_suffix = rng.randrange(branch_suffix_count)
    return default_actions, branch_actions, default_suffix, branch_suffix


def _crossover(left: Genome, right: Genome, rng: random.Random) -> Genome:
    left_default_actions, left_branch_actions, left_default, left_branch = left
    right_default_actions, right_branch_actions, right_default, right_branch = right

    def cross_actions(left_actions: tuple[int, ...], right_actions: tuple[int, ...]) -> tuple[int, ...]:
        if len(left_actions) != len(right_actions):
            raise ValueError("parent bridge lengths differ")
        if len(left_actions) <= 1:
            return left_actions if rng.random() < .5 else right_actions
        pivot = rng.randrange(1, len(left_actions))
        return (*left_actions[:pivot], *right_actions[pivot:])

    default_actions = cross_actions(left_default_actions, right_default_actions)
    branch_actions = cross_actions(left_branch_actions, right_branch_actions)
    return (
        default_actions,
        branch_actions,
        left_default if rng.random() < .5 else right_default,
        left_branch if rng.random() < .5 else right_branch,
    )


def _descriptor(
    default_route: Sequence[Mapping[str, Any]],
    branch_route: Sequence[Mapping[str, Any]],
    bridge_start: int,
    branch_step: int,
    default_genes: Sequence[int],
    branch_genes: Sequence[int],
) -> tuple[int, int, int, int, int, int]:
    kinds = (
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
        "GOOSE", "COW", "SHEEP", "COOP", "PASTURE",
    )

    def summary(route: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
        counts = {kind: 0 for kind in kinds}
        market = 0
        for action in route[bridge_start:branch_step]:
            orders = [action.get("farmer"), *(action.get("hands", ()) or ())]
            for raw in orders:
                order = list(raw or ())
                if not order:
                    continue
                operation = str(order[0])
                item = str(order[1]) if len(order) > 1 else ""
                kind = (
                    item if operation in {"PLANT", "PLACE"}
                    else operation.removeprefix("BUILD_")
                )
                if kind in counts:
                    counts[kind] += 1
            market += len(action.get("market", ()) or ())
        dominant = max(range(len(kinds)), key=lambda index: counts[kinds[index]])
        return dominant, min(12, sum(counts.values()) // 2), min(12, market // 2)

    default = summary(default_route)
    branch = summary(branch_route)
    divergence = sum(left != right for left, right in zip(default_genes, branch_genes))
    return (*default[:2], *branch[:2], min(12, abs(default[2] - branch[2])), divergence)


def _digest(genome: Genome) -> str:
    return hashlib.sha256(
        json.dumps(genome, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _fitness(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["completion_rate"]),
        float(row["minimum_opponent_raw_win_rate"]),
        float(row["minimum_opponent_seat_win_rate"]),
        float(row["minimum_opponent_both_seats_win_rate"]),
        float(row["combined_raw_win_rate"]),
        float(row["minimum_opponent_mean_margin"]),
        float(row["combined_mean_margin"]),
    )


def _evaluate(
    source: Path,
    actions_path: Path,
    metadata_path: Path,
    prefix: Sequence[Mapping[str, Any]],
    donors: Sequence[Sequence[Mapping[str, Any]]],
    suffix_tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    genomes: Sequence[Genome],
    default_suffixes: Sequence[str],
    branch_suffixes: Sequence[str],
    default_opponents: Sequence[str],
    branch_opponents: Sequence[str],
    external_opponent_tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    seeds: Sequence[int],
    bridge_start: int,
    branch_step: int,
    block_size: int,
    label: str,
) -> tuple[list[dict[str, Any]], float, int]:
    generated: dict[str, list[dict[str, Any]]] = {}
    names = []
    for index, genome in enumerate(genomes):
        default_family = default_suffixes[genome[2]]
        branch_family = branch_suffixes[genome[3]]
        default_name = f"{label}D{index:04d}"
        branch_name = f"{label}B{index:04d}"
        generated[default_name] = _splice(
            prefix, donors, genome[0], suffix_tapes[default_family],
            bridge_start, branch_step, block_size,
        )
        generated[branch_name] = _splice(
            prefix, donors, genome[1], suffix_tapes[branch_family],
            bridge_start, branch_step, block_size,
        )
        names.append((default_name, branch_name))
    bundle = NativeTeammateBundle(
        source, actions_path, metadata_path,
        additional_routes={**external_opponent_tapes, **generated},
    )
    opponent_groups = (tuple(default_opponents), tuple(branch_opponents))
    task_rows = []
    tasks = np.empty(
        (len(genomes) * sum(map(len, opponent_groups)) * len(seeds) * 2, 7),
        dtype=np.int64,
    )
    row = 0
    for genome_index, route_names in enumerate(names):
        for group_index, opponents in enumerate(opponent_groups):
            route_index = bundle.index(route_names[group_index])
            for opponent_family in opponents:
                opponent_index = bundle.index(opponent_family)
                for seed in seeds:
                    for seat in (0, 1):
                        left, right = (
                            (route_index, opponent_index)
                            if seat == 0 else (opponent_index, route_index)
                        )
                        tasks[row] = (left, right, seed, -1, -1, -1, -1)
                        task_rows.append((genome_index, opponent_family, seat))
                        row += 1
    started = time.perf_counter()
    rewards = np.asarray(bundle.executor.play_batch(tasks), dtype=np.float64)
    elapsed = time.perf_counter() - started
    margins: list[dict[str, list[float]]] = [
        {family: [] for family in (*default_opponents, *branch_opponents)}
        for _ in genomes
    ]
    finite: list[list[bool]] = [[] for _ in genomes]
    for index, (genome_index, opponent_family, seat) in enumerate(task_rows):
        pair = rewards[index]
        finite[genome_index].append(bool(np.all(np.isfinite(pair))))
        margins[genome_index][opponent_family].append(
            float(pair[seat] - pair[1 - seat])
        )
    rows = []
    for genome_index, genome in enumerate(genomes):
        used = {
            family: values for family, values in margins[genome_index].items()
            if values
        }
        rates = {
            family: float(np.mean(np.asarray(values) > 0))
            for family, values in used.items()
        }
        paired = {
            family: np.asarray(values, dtype=np.float64).reshape(-1, 2) > 0
            for family, values in used.items()
        }
        seat_rates = {
            family: np.mean(values, axis=0).tolist()
            for family, values in paired.items()
        }
        both_seats_rates = {
            family: float(np.mean(np.all(values, axis=1)))
            for family, values in paired.items()
        }
        mean_margins = {
            family: float(np.mean(values)) for family, values in used.items()
        }
        all_margins = np.asarray(
            [value for values in used.values() for value in values], dtype=np.float64
        )
        default_route = generated[names[genome_index][0]]
        branch_route = generated[names[genome_index][1]]
        rows.append({
            "genome_sha256": _digest(genome),
            "genome": [list(genome[0]), list(genome[1]), genome[2], genome[3]],
            "descriptor": list(_descriptor(
                default_route, branch_route, bridge_start, branch_step,
                genome[0], genome[1],
            )),
            "default_suffix": default_suffixes[genome[2]],
            "branch_suffix": branch_suffixes[genome[3]],
            "completion_rate": float(np.mean(finite[genome_index])),
            "minimum_opponent_raw_win_rate": min(rates.values()),
            "minimum_opponent_seat_win_rate": min(
                min(values) for values in seat_rates.values()
            ),
            "minimum_opponent_both_seats_win_rate": min(both_seats_rates.values()),
            "combined_raw_win_rate": float(np.mean(all_margins > 0)),
            "minimum_opponent_mean_margin": min(mean_margins.values()),
            "combined_mean_margin": float(np.mean(all_margins)),
            "opponent_raw_win_rates": rates,
            "opponent_seat_win_rates": seat_rates,
            "opponent_both_seats_win_rates": both_seats_rates,
            "opponent_mean_margins": mean_margins,
        })
    return rows, elapsed, len(tasks)


def _next_population(
    rows: Sequence[Mapping[str, Any]],
    population_size: int,
    elite_count: int,
    map_elite_count: int,
    donor_count: int,
    default_suffix_count: int,
    branch_suffix_count: int,
    rng: random.Random,
) -> list[Genome]:
    ordered = sorted(rows, key=_fitness, reverse=True)
    selected: list[Genome] = []
    seen: set[Genome] = set()

    def add(row: Mapping[str, Any]) -> None:
        raw = row["genome"]
        genome = (
            tuple(map(int, raw[0])), tuple(map(int, raw[1])),
            int(raw[2]), int(raw[3]),
        )
        if genome not in seen:
            seen.add(genome)
            selected.append(genome)

    for row in ordered[:elite_count]:
        add(row)
    cells: dict[tuple[int, ...], Mapping[str, Any]] = {}
    for row in ordered:
        cell = tuple(map(int, row["descriptor"]))
        if cell not in cells:
            cells[cell] = row
    for row in sorted(cells.values(), key=_fitness, reverse=True)[:map_elite_count]:
        add(row)
    parent_pool = selected[:] or [
        (tuple(map(int, ordered[0]["genome"][0])),
         tuple(map(int, ordered[0]["genome"][1])),
         int(ordered[0]["genome"][2]), int(ordered[0]["genome"][3]))
    ]
    attempts = 0
    while len(selected) < population_size:
        attempts += 1
        left = rng.choice(parent_pool)
        right = rng.choice(parent_pool)
        child = _crossover(left, right, rng)
        for _ in range(1 + (rng.random() < .35)):
            child = _mutate(
                child, donor_count, default_suffix_count,
                branch_suffix_count, rng,
            )
        if child not in seen:
            seen.add(child)
            selected.append(child)
        if attempts > population_size * 100:
            raise RuntimeError("could not create a unique genetic population")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prefix-family", required=True)
    parser.add_argument("--bridge-start", type=int, required=True)
    parser.add_argument("--branch-step", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--bridge-team", default="nt_68_route_pool")
    parser.add_argument("--default-suffixes", type=_csv, required=True)
    parser.add_argument("--branch-suffixes", type=_csv, required=True)
    parser.add_argument("--default-opponent", type=_opponent, action="append", default=[])
    parser.add_argument("--branch-opponent", type=_opponent, action="append", default=[])
    parser.add_argument(
        "--default-opponent-family", type=_csv, action="append", default=[],
        help="Use one or more comma-separated route families already in metadata.",
    )
    parser.add_argument(
        "--branch-opponent-family", type=_csv, action="append", default=[],
        help="Use one or more comma-separated route families already in metadata.",
    )
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--holdout-seeds", type=_ints, required=True)
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--map-elites", type=int, default=32)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--holdout-top", type=int, default=48)
    parser.add_argument("--random-seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elite-actions", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.bridge_start < args.branch_step <= 719:
        parser.error("expected 0 <= bridge-start < branch-step <= 719")
    if args.block_size <= 0 or (args.branch_step - args.bridge_start) % args.block_size:
        parser.error("causal interval must be divisible by a positive block-size")
    if args.population < args.elites + args.map_elites:
        parser.error("population must be at least elites + map-elites")
    rng = random.Random(args.random_seed)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    entries = list(metadata["opponent_routes"])
    route_ids = {str(row["family"]): str(row["route_id"]) for row in entries}
    tapes = load_action_tapes(args.actions)
    required = {
        args.prefix_family, *args.default_suffixes, *args.branch_suffixes,
    }
    missing = sorted(required - set(route_ids))
    if missing:
        raise KeyError(f"unknown route family: {missing[0]}")
    raw_donors = list(dict.fromkeys([
        str(row["family"]) for row in entries
        if str(row.get("team", "")) == args.bridge_team
    ] + [args.prefix_family, *args.default_suffixes, *args.branch_suffixes]))
    donor_families = []
    donor_tapes = []
    segment_index: dict[str, int] = {}
    donor_index: dict[str, int] = {}
    for family in raw_donors:
        tape = tapes[route_ids[family]]
        digest = hashlib.sha256(json.dumps(
            tape[args.bridge_start:args.branch_step], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        if digest not in segment_index:
            segment_index[digest] = len(donor_tapes)
            donor_families.append(family)
            donor_tapes.append(tape)
        donor_index[family] = segment_index[digest]
    if not donor_tapes:
        raise ValueError("no bridge donors")
    prefix = tapes[route_ids[args.prefix_family]]
    suffix_tapes = {
        family: tapes[route_ids[family]]
        for family in (*args.default_suffixes, *args.branch_suffixes)
    }
    external_default_opponents = {
        family: _trace(spec) for family, spec in args.default_opponent
    }
    external_branch_opponents = {
        family: _trace(spec) for family, spec in args.branch_opponent
    }
    internal_default_opponents = tuple(
        family
        for group in args.default_opponent_family
        for family in group
    )
    internal_branch_opponents = tuple(
        family
        for group in args.branch_opponent_family
        for family in group
    )
    internal = (*internal_default_opponents, *internal_branch_opponents)
    missing_opponents = sorted(set(internal) - set(route_ids))
    if missing_opponents:
        raise KeyError(f"unknown opponent route family: {missing_opponents[0]}")
    default_opponents = (
        *external_default_opponents, *internal_default_opponents,
    )
    branch_opponents = (
        *external_branch_opponents, *internal_branch_opponents,
    )
    if not default_opponents and not branch_opponents:
        raise ValueError("at least one opponent is required")
    if len(set(default_opponents)) != len(default_opponents):
        raise ValueError("duplicate default opponent family")
    if len(set(branch_opponents)) != len(branch_opponents):
        raise ValueError("duplicate branch opponent family")
    overlap = set(default_opponents) & set(branch_opponents)
    if overlap:
        raise ValueError(f"opponent appears in both groups: {sorted(overlap)[0]}")
    external_opponent_tapes = {
        **external_default_opponents, **external_branch_opponents,
    }

    length = (args.branch_step - args.bridge_start) // args.block_size
    initial = []
    for default_index, default_family in enumerate(args.default_suffixes):
        for branch_index, branch_family in enumerate(args.branch_suffixes):
            initial.append((
                (donor_index[default_family],) * length,
                (donor_index[branch_family],) * length,
                default_index,
                branch_index,
            ))
    initial.extend(
        ((index,) * length, (index,) * length, 0, 0)
        for index in range(len(donor_tapes))
    )
    initial = list(dict.fromkeys(initial))
    while len(initial) < args.population:
        parent = rng.choice(initial)
        candidate = _mutate(
            parent, len(donor_tapes), len(args.default_suffixes),
            len(args.branch_suffixes), rng,
        )
        if candidate not in initial:
            initial.append(candidate)
    population = initial[:args.population]
    archive: dict[str, dict[str, Any]] = {}
    history = []
    total_games = 0
    native_seconds = 0.0
    started = time.perf_counter()
    for generation in range(args.generations):
        rows, elapsed, games = _evaluate(
            args.source, args.actions, args.metadata, prefix, donor_tapes,
            suffix_tapes, population, args.default_suffixes, args.branch_suffixes,
            default_opponents, branch_opponents, external_opponent_tapes,
            args.seeds,
            args.bridge_start, args.branch_step, args.block_size,
            f"G{generation:02d}",
        )
        native_seconds += elapsed
        total_games += games
        for row in rows:
            previous = archive.get(str(row["genome_sha256"]))
            if previous is None or _fitness(row) > _fitness(previous):
                archive[str(row["genome_sha256"])] = dict(row)
        ordered = sorted(rows, key=_fitness, reverse=True)
        history.append({
            "generation": generation,
            "population": len(rows),
            "archive_size": len(archive),
            "games": games,
            "native_seconds": elapsed,
            "best": ordered[0],
            "map_cells": len({tuple(row["descriptor"]) for row in rows}),
        })
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if generation + 1 < args.generations:
            population = _next_population(
                rows, args.population, args.elites, args.map_elites,
                len(donor_tapes), len(args.default_suffixes),
                len(args.branch_suffixes), rng,
            )

    archive_rows = sorted(archive.values(), key=_fitness, reverse=True)
    holdout_genomes = []
    for row in archive_rows[:args.holdout_top]:
        raw = row["genome"]
        holdout_genomes.append(
            (tuple(map(int, raw[0])), tuple(map(int, raw[1])),
             int(raw[2]), int(raw[3]))
        )
    holdout_rows, elapsed, games = _evaluate(
        args.source, args.actions, args.metadata, prefix, donor_tapes,
        suffix_tapes, holdout_genomes, args.default_suffixes, args.branch_suffixes,
        default_opponents, branch_opponents, external_opponent_tapes,
        args.holdout_seeds,
        args.bridge_start, args.branch_step, args.block_size, "H",
    )
    native_seconds += elapsed
    total_games += games
    holdout_rows.sort(key=_fitness, reverse=True)
    elite_tapes: dict[str, list[dict[str, Any]]] = {}
    elite_rows = []
    for index, row in enumerate(holdout_rows):
        raw = row["genome"]
        genome = (
            tuple(map(int, raw[0])), tuple(map(int, raw[1])),
            int(raw[2]), int(raw[3]),
        )
        default_id = f"CGA{index:03d}_DEFAULT"
        branch_id = f"CGA{index:03d}_YARN2"
        elite_tapes[default_id] = _splice(
            prefix, donor_tapes, genome[0],
            suffix_tapes[args.default_suffixes[genome[2]]],
            args.bridge_start, args.branch_step, args.block_size,
        )
        elite_tapes[branch_id] = _splice(
            prefix, donor_tapes, genome[1],
            suffix_tapes[args.branch_suffixes[genome[3]]],
            args.bridge_start, args.branch_step, args.block_size,
        )
        enriched = dict(row)
        enriched["rank"] = index + 1
        enriched["default_route_id"] = default_id
        enriched["branch_route_id"] = branch_id
        enriched["default_bridge_donor_families"] = [
            donor_families[value] for value in genome[0]
        ]
        enriched["branch_bridge_donor_families"] = [
            donor_families[value] for value in genome[1]
        ]
        elite_rows.append(enriched)
    packed = zlib.compress(
        json.dumps(elite_tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    args.elite_actions.parent.mkdir(parents=True, exist_ok=True)
    args.elite_actions.write_bytes(packed)
    payload = {
        "schema": "causal-route-bridge-ga-v2",
        "engine": "compiled-cpp-fast-kaggriculture",
        "jax_used": False,
        "source_actions": str(args.actions.resolve()),
        "source_metadata": str(args.metadata.resolve()),
        "prefix_family": args.prefix_family,
        "bridge_start": args.bridge_start,
        "branch_step": args.branch_step,
        "block_size": args.block_size,
        "raw_donor_count": len(raw_donors),
        "unique_donor_count": len(donor_tapes),
        "donor_families": donor_families,
        "default_suffixes": list(args.default_suffixes),
        "branch_suffixes": list(args.branch_suffixes),
        "default_opponents": list(default_opponents),
        "branch_opponents": list(branch_opponents),
        "internal_default_opponents": list(internal_default_opponents),
        "internal_branch_opponents": list(internal_branch_opponents),
        "training_seeds": list(args.seeds),
        "holdout_seeds": list(args.holdout_seeds),
        "population": args.population,
        "generations": args.generations,
        "archive_size": len(archive),
        "games": total_games,
        "native_seconds": native_seconds,
        "native_games_per_second": total_games / native_seconds,
        "wall_seconds": time.perf_counter() - started,
        "elite_actions": str(args.elite_actions.resolve()),
        "elite_actions_sha256": hashlib.sha256(packed).hexdigest(),
        "history": history,
        "holdout_ranking": elite_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "elite_actions": str(args.elite_actions.resolve()),
        "games": total_games,
        "native_games_per_second": total_games / native_seconds,
        "wall_seconds": payload["wall_seconds"],
        "top": elite_rows[:10],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

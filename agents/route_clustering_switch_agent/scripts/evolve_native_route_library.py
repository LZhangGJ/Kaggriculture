#!/usr/bin/env python3
"""Evolve executable replay-tape variants by marginal switch-library coverage."""

from __future__ import annotations

import argparse
import copy
import json
import random
import runpy
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_tape_mutations import _apply_genes, _candidate_genes, _ints


Genome = tuple[str, tuple[int, ...]]
TRANSFERABLE_MARKET_OPERATIONS = {
    "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL", "SELL",
}
DONOR_MARKET_WIDTHS = (24, 144)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _external_opponent(value: str) -> tuple[str, str]:
    family, separator, trace_spec = value.partition("=")
    if not separator or not family or not trace_spec:
        raise argparse.ArgumentTypeError(
            "external opponent must be FAMILY=PATH or FAMILY=PATH::VARIABLE::KEY"
        )
    return family, trace_spec


def _external_trace(trace_spec: str) -> list[dict[str, Any]]:
    parts = trace_spec.rsplit("::", 2)
    path = Path(parts[0]).resolve()
    namespace = runpy.run_path(str(path))
    if len(parts) == 1:
        trace = namespace.get("_ACTIONS", namespace.get("_TRACE"))
        description = "_ACTIONS/_TRACE"
    elif len(parts) == 3:
        variable, key = parts[1:]
        routes = namespace.get(variable)
        if not isinstance(routes, dict):
            raise ValueError(f"{variable} is not a route dictionary in {path}")
        trace = routes.get(key)
        description = f"{variable}[{key!r}]"
    else:
        raise ValueError(
            "trace spec must be PATH or PATH::VARIABLE::KEY: " + trace_spec
        )
    if not isinstance(trace, list) or len(trace) < 719:
        raise ValueError(f"no 719-step {description} in {path}")
    return [dict(value or {}) for value in trace[:719]]


def _fitness(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row.get(
            "robust_raw_win_rate",
            row.get("minimum_opponent_raw_win_rate", 0.0),
        )),
        float(row.get("minimum_opponent_raw_win_rate", 0.0)),
        float(row.get("combined_raw_win_rate", 0.0)),
        float(row.get("minimum_opponent_mean_margin", -float("inf"))),
        float(row["incremental_oracle_score"]),
        float(row["incremental_oracle_margin"]),
        float(row["candidate_score"]),
        float(row["candidate_margin"]),
        -float(len(row["genome"])),
    )


def _select_elites(
    ranked: Sequence[Genome],
    cache: Mapping[Genome, Mapping[str, Any]],
    parents: Sequence[str],
    opponents: Sequence[str],
    *,
    global_count: int,
    per_parent: int,
    per_opponent: int,
    population_size: int,
    map_elites: int = 0,
) -> list[Genome]:
    """Retain robust candidates plus parent and opponent MAP-Elites niches."""
    selected: list[Genome] = []
    seen: set[Genome] = set()

    def add(genome: Genome) -> None:
        if genome not in seen and len(selected) < population_size:
            seen.add(genome)
            selected.append(genome)

    # Unmodified parents remain recovery points if a mutation lineage regresses.
    for parent in parents:
        add((str(parent), ()))
    for genome in ranked[:global_count]:
        add(genome)
    cells: dict[tuple[int, ...], Genome] = {}
    for genome in ranked:
        descriptor = tuple(int(value) for value in cache[genome].get("descriptor", ()))
        if descriptor and descriptor not in cells:
            cells[descriptor] = genome
    for genome in sorted(
        cells.values(), key=lambda value: _fitness(cache[value]), reverse=True
    )[:map_elites]:
        add(genome)
    for opponent in opponents:
        niche = sorted(
            ranked,
            key=lambda genome: (
                min(cache[genome].get(
                    "opponent_seat_raw_win_rates", {}
                ).get(opponent, [
                    cache[genome]["opponent_raw_win_rates"][opponent]
                ])),
                min(cache[genome].get(
                    "opponent_seed_fold_raw_win_rates", {}
                ).get(opponent, [
                    cache[genome]["opponent_raw_win_rates"][opponent]
                ])),
                float(cache[genome]["opponent_raw_win_rates"][opponent]),
                float(cache[genome]["opponent_mean_margins"][opponent]),
                _fitness(cache[genome]),
            ),
            reverse=True,
        )
        for genome in niche[:per_opponent]:
            add(genome)
    for parent in parents:
        niche = [genome for genome in ranked if genome[0] == parent]
        for genome in niche[:per_parent]:
            add(genome)
    return selected


def _canonical(
    values: set[int],
    maximum: int,
    rng: random.Random,
    atoms: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[int, ...]:
    result = sorted(values)
    if atoms is not None:
        allele_groups: dict[tuple[str, str, str, int, int], list[int]] = defaultdict(list)
        for index in result:
            gene = atoms[index]
            operator = str(gene.get("operator", ""))
            if operator in {"scale_quantity", "shift_market", "reduce_hires"}:
                allele_groups[(
                    operator,
                    str(gene.get("operation", "")),
                    str(gene.get("item", "")),
                    int(gene["start"]),
                    int(gene["stop"]),
                )].append(index)
        rejected = {
            index
            for group in allele_groups.values()
            for index in (rng.sample(group, len(group) - 1) if len(group) > 1 else ())
        }
        donor_by_phase: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        for index in result:
            gene = atoms[index]
            operator = str(gene.get("operator", ""))
            if operator in {"replace_market_phase", "replace_worker_phase"}:
                donor_by_phase[(
                    operator, int(gene["start"]), int(gene["stop"])
                )].append(index)
        rejected.update(
            index
            for group in donor_by_phase.values()
            for index in (rng.sample(group, len(group) - 1) if len(group) > 1 else ())
        )
        result = [index for index in result if index not in rejected]

        # Prefer day-sized donor segments when a genome also contains a broad
        # donor segment covering the same turns. Overlapping replacements are
        # order-dependent and therefore do not form stable, recombinable genes.
        replacements = sorted(
            (
                index for index in result
                if str(atoms[index].get("operator", "")) == "replace_market_phase"
            ),
            key=lambda index: (
                int(atoms[index]["stop"]) - int(atoms[index]["start"]), index
            ),
        )
        kept_replacements: list[int] = []
        for index in replacements:
            gene = atoms[index]
            start, stop = int(gene["start"]), int(gene["stop"])
            if any(
                start < int(atoms[other]["stop"])
                and int(atoms[other]["start"]) < stop
                for other in kept_replacements
            ):
                rejected.add(index)
            else:
                kept_replacements.append(index)
        result = [index for index in result if index not in rejected]

        # Market replacements run after ordinary market mutations. Drop a
        # local mutation when donor segments cover every turn it could affect;
        # otherwise the genome contains a phenotypically inert gene.
        replacement_ranges = [
            (int(atoms[index]["start"]), int(atoms[index]["stop"]))
            for index in kept_replacements if index not in rejected
        ]
        for index in result:
            gene = atoms[index]
            if str(gene.get("operator", "")) not in {"scale_quantity", "shift_market"}:
                continue
            start, stop = int(gene["start"]), int(gene["stop"])
            if all(
                any(left <= step < right for left, right in replacement_ranges)
                for step in range(start, stop)
            ):
                rejected.add(index)
        result = [index for index in result if index not in rejected]
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
    if len(parents) > 1 and rng.random() < .04:
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
    return parent, _canonical(values, maximum, rng, atoms[parent])


def _crossover(
    left: Genome,
    right: Genome,
    maximum: int,
    atoms: Mapping[str, list[dict[str, Any]]],
    rng: random.Random,
) -> Genome:
    if left[0] != right[0]:
        return rng.choice((left, right))
    values = {
        value for value in set(left[1]) | set(right[1])
        if rng.random() < .5 or value in (set(left[1]) & set(right[1]))
    }
    return left[0], _canonical(values, maximum, rng, atoms[left[0]])


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


def _replace_market_phase(
    tape: list[dict[str, Any]],
    donor: list[dict[str, Any]],
    start: int,
    stop: int,
) -> list[dict[str, Any]]:
    """Recombine economics without splicing incompatible worker positions."""

    result = copy.deepcopy(tape)
    for step in range(max(0, start), min(stop, len(result), len(donor))):
        structural = [
            list(order)
            for order in result[step].get("market", ()) or ()
            if not order or str(order[0]) not in TRANSFERABLE_MARKET_OPERATIONS
        ]
        transferred = [
            list(order)
            for order in donor[step].get("market", ()) or ()
            if order and str(order[0]) in TRANSFERABLE_MARKET_OPERATIONS
        ]
        result[step]["market"] = (structural + transferred)[:10]
    return result


def _replace_worker_phase(
    tape: list[dict[str, Any]],
    donor: list[dict[str, Any]],
    start: int,
    stop: int,
) -> list[dict[str, Any]]:
    """Transfer a synchronized farmer/hand day while keeping base economics."""
    result = copy.deepcopy(tape)
    for step in range(max(0, start), min(stop, len(result), len(donor))):
        result[step]["farmer"] = copy.deepcopy(donor[step].get("farmer", ["PASS"]))
        result[step]["hands"] = copy.deepcopy(donor[step].get("hands", []))
    return result


def _apply_evolution_genes(
    tape: list[dict[str, Any]],
    genes: list[dict[str, Any]],
    parent_tapes: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    result = copy.deepcopy(tape)
    for gene in genes:
        if str(gene["operator"]) == "replace_market_phase":
            result = _replace_market_phase(
                result,
                parent_tapes[str(gene["donor"])],
                int(gene["start"]),
                int(gene["stop"]),
            )
        elif str(gene["operator"]) == "replace_worker_phase":
            result = _replace_worker_phase(
                result,
                parent_tapes[str(gene["donor"])],
                int(gene["start"]),
                int(gene["stop"]),
            )
        else:
            result = _apply_genes(result, [gene])
    return result


def _donor_atoms(
    parent: str,
    parents: tuple[str, ...],
    parent_tapes: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    atoms = []
    tape = parent_tapes[parent]
    for donor in parents:
        if donor == parent:
            continue
        donor_tape = parent_tapes[donor]
        for width in DONOR_MARKET_WIDTHS:
            for start in range(0, 719, width):
                stop = min(719, start + width)
                base_profile = [
                    order
                    for step in tape[start:stop]
                    for order in step.get("market", ()) or ()
                    if order and str(order[0]) in TRANSFERABLE_MARKET_OPERATIONS
                ]
                donor_profile = [
                    order
                    for step in donor_tape[start:stop]
                    for order in step.get("market", ()) or ()
                    if order and str(order[0]) in TRANSFERABLE_MARKET_OPERATIONS
                ]
                if base_profile != donor_profile:
                    atoms.append({
                        "operator": "replace_market_phase",
                        "donor": donor,
                        "start": start,
                        "stop": stop,
                    })
        for start in range(0, 719, 24):
            stop = min(719, start + 24)
            base_segment = tape[start:stop]
            donor_segment = donor_tape[start:stop]
            # Copy all workers together and only between schedules with the
            # same active-hand count. This evolves assignments without
            # splicing individual paths onto non-existent workers.
            compatible = all(
                len(base_action.get("hands", ()) or ())
                == len(donor_action.get("hands", ()) or ())
                for base_action, donor_action in zip(base_segment, donor_segment)
            )
            base_workers = [
                (action.get("farmer", ["PASS"]), action.get("hands", []))
                for action in base_segment
            ]
            donor_workers = [
                (action.get("farmer", ["PASS"]), action.get("hands", []))
                for action in donor_segment
            ]
            if compatible and base_workers != donor_workers:
                atoms.append({
                    "operator": "replace_worker_phase",
                    "donor": donor,
                    "start": start,
                    "stop": stop,
                })
    return atoms


def _robust_checkpoint_metrics(
    scores: np.ndarray,
    margins: np.ndarray,
    opponents: Sequence[str],
    seed_folds: int,
) -> dict[str, Any]:
    """Measure a checkpoint without hiding opponent, seat, or seed-cohort failures."""

    wins = scores == 1.0
    opponent_raw = np.mean(wins, axis=(0, 2, 3))
    opponent_margin = np.mean(margins, axis=(0, 2, 3))
    opponent_seat = np.mean(wins, axis=(0, 2))
    folds = np.array_split(
        np.arange(scores.shape[2]), min(max(1, seed_folds), scores.shape[2])
    )
    fold_rates = [
        np.mean(wins[:, :, indices, :], axis=(0, 2, 3))
        for indices in folds if len(indices)
    ]
    minimum_seat = float(np.min(opponent_seat))
    minimum_fold = float(min(np.min(values) for values in fold_rates))
    return {
        "robust_raw_win_rate": min(minimum_seat, minimum_fold),
        "minimum_opponent_raw_win_rate": float(np.min(opponent_raw)),
        "combined_raw_win_rate": float(np.mean(wins)),
        "minimum_opponent_seat_raw_win_rate": minimum_seat,
        "minimum_opponent_seed_fold_raw_win_rate": minimum_fold,
        "opponent_raw_win_rates": {
            family: float(opponent_raw[index])
            for index, family in enumerate(opponents)
        },
        "opponent_mean_margins": {
            family: float(opponent_margin[index])
            for index, family in enumerate(opponents)
        },
        "opponent_seat_raw_win_rates": {
            family: [float(value) for value in opponent_seat[index]]
            for index, family in enumerate(opponents)
        },
        "opponent_seed_fold_raw_win_rates": {
            family: [float(values[index]) for values in fold_rates]
            for index, family in enumerate(opponents)
        },
        "minimum_opponent_mean_margin": float(np.min(opponent_margin)),
    }


def _behavior_descriptor(
    tape: Sequence[Mapping[str, Any]],
    parent: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int, int, int]:
    """Describe the changed executable behavior, independent of reward."""

    changed_market_days = 0
    changed_worker_days = 0
    for start in range(0, min(len(tape), len(parent)), 24):
        stop = min(len(tape), len(parent), start + 24)
        if any(
            (left.get("market", ()) or ()) != (right.get("market", ()) or ())
            for left, right in zip(tape[start:stop], parent[start:stop])
        ):
            changed_market_days += 1
        if any(
            (
                left.get("farmer", ["PASS"]), left.get("hands", ()) or ()
            ) != (
                right.get("farmer", ["PASS"]), right.get("hands", ()) or ()
            )
            for left, right in zip(tape[start:stop], parent[start:stop])
        ):
            changed_worker_days += 1

    def balance(values: Sequence[Mapping[str, Any]], start: int, stop: int) -> int:
        result = 0
        for action in values[start:stop]:
            for raw in action.get("market", ()) or ():
                order = list(raw or ())
                if not order:
                    continue
                quantity = int(order[2]) if len(order) >= 3 else 1
                operation = str(order[0])
                if operation.startswith("BUY_") or operation == "HIRE":
                    result += quantity
                elif operation == "SELL":
                    result -= quantity
        return result

    flow_bins = []
    for start, stop in ((0, 288), (288, 576), (576, 719)):
        delta = balance(tape, start, stop) - balance(parent, start, stop)
        flow_bins.append(max(-4, min(4, round(delta / 20))) + 4)
    return (
        min(7, changed_market_days // 2),
        *flow_bins,
        min(5, changed_worker_days // 2),
    )


def _seed_genomes(
    archives: list[Path],
    atoms: Mapping[str, list[dict[str, Any]]],
    per_opponent: int = 4,
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
        ranking = list(payload.get("ranking", []))
        rows = list(ranking[:32])
        for opponent in payload.get("opponents", ()):
            niche = sorted(
                ranking,
                key=lambda row: (
                    float(row.get("opponent_raw_win_rates", {}).get(opponent, -1.0)),
                    float(row.get("opponent_mean_margins", {}).get(
                        opponent, -float("inf")
                    )),
                ),
                reverse=True,
            )
            for row in niche[:per_opponent]:
                if row not in rows:
                    rows.append(row)
        for row in rows:
            parent = str(row.get("parent", payload.get("parent", "")))
            if parent not in atoms:
                continue
            indices = []
            for gene in row.get("genes", []) or []:
                key = json.dumps(gene, sort_keys=True, separators=(",", ":"))
                if key in atom_index[parent]:
                    indices.append(atom_index[parent][key])
            if indices:
                result.add((
                    parent,
                    _canonical(
                        set(indices), len(indices),
                        random.Random(json.dumps(row.get("genes", []), sort_keys=True)),
                        atoms[parent],
                    ),
                ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parents", type=_csv)
    parser.add_argument(
        "--donor-family", type=_csv, action="append", default=[],
        help="Add route families as mutation donors without breeding them as parents.",
    )
    parser.add_argument("--openings", type=_csv, default=("G001",))
    parser.add_argument("--base-targets", type=_csv)
    parser.add_argument("--opponents", type=_csv)
    parser.add_argument(
        "--external-opponent", type=_external_opponent, action="append", default=[],
        help="Add FAMILY=PATH or FAMILY=PATH::VARIABLE::KEY to the fitness pool.",
    )
    parser.add_argument("--opponent-limit", type=int, default=32)
    parser.add_argument("--checkpoints", type=_ints, default=(144, 168, 216))
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--elites-per-parent", type=int, default=2)
    parser.add_argument("--elites-per-opponent", type=int, default=2)
    parser.add_argument("--map-elites", type=int, default=16)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--max-genes", type=int, default=5)
    parser.add_argument("--donor-seeds-per-parent", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument("--seed-archive", type=Path, action="append", default=[])
    parser.add_argument("--seed-archive-per-opponent", type=int, default=4)
    parser.add_argument(
        "--seed-folds", type=int, default=4,
        help="Rank candidates by their worst opponent/seat/seed-cohort win rate.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    if args.donor_seeds_per_parent < 0:
        parser.error("--donor-seeds-per-parent must be non-negative")
    if args.seed_folds <= 0:
        parser.error("--seed-folds must be positive")
    rng = random.Random(args.random_seed)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    entries = list(metadata["opponent_routes"])
    base_families = tuple(str(value["family"]) for value in entries)
    route_id = {str(value["family"]): str(value["route_id"]) for value in entries}
    tapes = load_action_tapes(args.actions)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    external_opponents = {
        family: _external_trace(trace_spec)
        for family, trace_spec in args.external_opponent
    }
    duplicate_external = sorted(set(base_families) & set(external_opponents))
    if duplicate_external:
        raise ValueError(
            f"external opponent family already exists: {duplicate_external[0]}"
        )
    policy_targets = tuple(str(value) for value in policy.get("targets", ()))
    parents = args.parents or policy_targets
    donor_families = tuple(dict.fromkeys(
        family for group in args.donor_family for family in group
        if family not in parents
    ))
    base_targets = args.base_targets or policy_targets
    if args.population < len(parents):
        parser.error(
            f"--population must be at least the parent count ({len(parents)})"
        )
    for family in parents:
        if family not in route_id and family not in external_opponents:
            raise KeyError(f"unknown parent family: {family}")
    for family in donor_families:
        if family not in route_id and family not in external_opponents:
            raise KeyError(f"unknown donor family: {family}")
    for family in (*args.openings, *base_targets):
        if family not in route_id:
            raise KeyError(f"unknown base family: {family}")
    opponents = args.opponents or (
        tuple(external_opponents)
        if external_opponents else _representative_opponents(
            base_families,
            tuple(dict.fromkeys((*parents, *base_targets))),
            args.opponent_limit,
        )
    )
    for family in opponents:
        if family not in route_id and family not in external_opponents:
            raise KeyError(f"unknown opponent family: {family}")

    atom_map: dict[str, list[dict[str, Any]]] = {}
    parent_tapes: dict[str, list[dict[str, Any]]] = {}
    for parent in (*parents, *donor_families):
        tape = (
            external_opponents[parent]
            if parent in external_opponents else tapes[route_id[parent]]
        )
        parent_tapes[parent] = tape
    for parent in parents:
        tape = parent_tapes[parent]
        atom_map[parent] = [
            list(genes)[0] for genes in _candidate_genes(tape) if len(list(genes)) == 1
        ]
        atom_map[parent].extend(_donor_atoms(
            parent, (*parents, *donor_families), parent_tapes
        ))
        if not atom_map[parent]:
            raise ValueError(f"no mutations available for parent {parent}")

    population: set[Genome] = {(parent, ()) for parent in parents}
    population.update(_seed_genomes(
        args.seed_archive, atom_map, args.seed_archive_per_opponent
    ))
    for parent in parents:
        for operator in ("replace_market_phase", "replace_worker_phase"):
            donor_indices = [
                index
                for index, gene in enumerate(atom_map[parent])
                if str(gene.get("operator", "")) == operator
            ]
            for index in rng.sample(
                donor_indices, min(args.donor_seeds_per_parent, len(donor_indices))
            ):
                population.add((parent, (index,)))
    while len(population) < args.population:
        parent = rng.choice(parents)
        size = rng.randint(1, min(3, args.max_genes))
        sampled = set(rng.sample(range(len(atom_map[parent])), size))
        population.add((
            parent,
            _canonical(sampled, args.max_genes, rng, atom_map[parent]),
        ))
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
            routes = {}
            descriptors = {}
            for name, genome in zip(names, fresh):
                routes[name] = _apply_evolution_genes(
                    parent_tapes[genome[0]],
                    [atom_map[genome[0]][index] for index in genome[1]],
                    parent_tapes,
                )
                descriptors[name] = _behavior_descriptor(
                    routes[name], parent_tapes[genome[0]]
                )
            bundle = NativeTeammateBundle(
                args.source,
                args.actions,
                args.metadata,
                additional_routes={**external_opponents, **routes},
                included_families=tuple(dict.fromkeys(
                    (*args.openings, *base_targets, *opponents)
                )),
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
                    checkpoint_scores = candidate_scores[:, checkpoint_index]
                    checkpoint_margins = candidate_margins[:, checkpoint_index]
                    checkpoint_rows.append({
                        "checkpoint": int(checkpoint),
                        **_robust_checkpoint_metrics(
                            checkpoint_scores, checkpoint_margins,
                            opponents, args.seed_folds,
                        ),
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
                        row["robust_raw_win_rate"],
                        row["minimum_opponent_raw_win_rate"],
                        row["combined_raw_win_rate"],
                        row["minimum_opponent_mean_margin"],
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
                    "descriptor": list(descriptors[names[index]]),
                    "generation_first_seen": generation,
                    **best_checkpoint,
                    "checkpoint_metrics": checkpoint_rows,
                }
                cache[genome] = row
                score_vectors[genome] = candidate_scores.reshape(-1).astype(np.float32)
                margin_vectors[genome] = candidate_margins.reshape(-1).astype(np.float32)

        ranked = sorted(population, key=lambda value: _fitness(cache[value]), reverse=True)
        elites = _select_elites(
            ranked, cache, parents, opponents,
            global_count=args.elites,
            per_parent=args.elites_per_parent,
            per_opponent=args.elites_per_opponent,
            population_size=args.population,
            map_elites=args.map_elites,
        )
        best = cache[ranked[0]]
        history.append({
            "generation": generation,
            "new_candidates": len(fresh),
            "unique_candidates": len(cache),
            "retained_elites": len(elites),
            "map_cells": len({
                tuple(cache[genome].get("descriptor", ())) for genome in population
            }),
            "best": best,
        })
        print(json.dumps({
            "generation": generation,
            "new_candidates": len(fresh),
            "unique_candidates": len(cache),
            "elapsed_seconds": time.perf_counter() - started,
            "best_parent": best["parent"],
            "best_checkpoint": best["checkpoint"],
            "robust_raw_win_rate": best["robust_raw_win_rate"],
            "minimum_opponent_seat_raw_win_rate": (
                best["minimum_opponent_seat_raw_win_rate"]
            ),
            "minimum_opponent_seed_fold_raw_win_rate": (
                best["minimum_opponent_seed_fold_raw_win_rate"]
            ),
            "minimum_opponent_raw_win_rate": best["minimum_opponent_raw_win_rate"],
            "incremental_oracle_score": best["incremental_oracle_score"],
            "candidate_score": best["candidate_score"],
        }), flush=True)
        next_population = set(elites)
        while len(next_population) < args.population:
            if rng.random() < .65:
                child = _crossover(
                    rng.choice(elites), rng.choice(elites), args.max_genes, atom_map, rng
                )
            else:
                child = rng.choice(elites)
            child = _mutate(child, atom_map, parents, args.max_genes, rng)
            next_population.add(child)
        population = next_population

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
        "donor_families": list(donor_families),
        "base_targets": list(base_targets),
        "openings": list(args.openings),
        "opponents": list(opponents),
        "external_opponents": [
            {"family": family, "trace_spec": trace_spec}
            for family, trace_spec in args.external_opponent
        ],
        "checkpoints": list(args.checkpoints),
        "seeds": list(args.seeds),
        "population": args.population,
        "elites": args.elites,
        "elites_per_parent": args.elites_per_parent,
        "elites_per_opponent": args.elites_per_opponent,
        "map_elites": args.map_elites,
        "generations": args.generations,
        "max_genes": args.max_genes,
        "donor_seeds_per_parent": args.donor_seeds_per_parent,
        "seed_archive_per_opponent": args.seed_archive_per_opponent,
        "seed_folds": args.seed_folds,
        "candidate_count": len(ranking),
        "generation_method": {
            "within_parent_atoms": [
                "scale_quantity", "shift_market", "reduce_hires"
            ],
            "cross_parent_atoms": ["replace_market_phase"],
            "cross_parent_phase_widths": list(DONOR_MARKET_WIDTHS),
            "cross_parent_worker_atoms": ["replace_worker_phase"],
            "cross_parent_worker_phase_width": 24,
            "worker_constraint": (
                "farmer and every hand are transferred as one synchronized day; "
                "donor and parent must have identical active-hand counts per step"
            ),
        },
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

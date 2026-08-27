#!/usr/bin/env python3
"""Exhaustively recombine state-compatible market genes in a causal bridge."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_public_trace_counters import _ints, _opponent, _trace


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def market_option_space(
    tapes: Sequence[Sequence[Mapping[str, Any]]], start: int, stop: int
) -> tuple[tuple[int, ...], tuple[tuple[list[Any], ...], ...]]:
    steps = []
    options = []
    for step in range(start, stop):
        unique: dict[str, list[Any]] = {}
        for tape in tapes:
            market = copy.deepcopy(list(tape[step].get("market", ()) or ()))
            unique.setdefault(_canonical(market), market)
        if len(unique) > 1:
            steps.append(step)
            options.append(tuple(unique.values()))
    return tuple(steps), tuple(options)


def apply_market_genome(
    base: Sequence[Mapping[str, Any]],
    steps: Sequence[int],
    options: Sequence[Sequence[list[Any]]],
    genome: Sequence[int],
) -> list[dict[str, Any]]:
    if not (len(steps) == len(options) == len(genome)):
        raise ValueError("market genome shape mismatch")
    result = copy.deepcopy(list(base))
    for step, values, selected in zip(steps, options, genome):
        result[step]["market"] = copy.deepcopy(values[int(selected)])
    return result


def apply_causal_market_genome(
    prefix: Sequence[Mapping[str, Any]],
    base: Sequence[Mapping[str, Any]],
    steps: Sequence[int],
    options: Sequence[Sequence[list[Any]]],
    genome: Sequence[int],
    bridge_start: int,
) -> list[dict[str, Any]]:
    """Apply market genes while enforcing the actually executed causal prefix."""

    result = apply_market_genome(base, steps, options, genome)
    result[:bridge_start] = copy.deepcopy(list(prefix[:bridge_start]))
    return result


def splice_suffix(
    bridge: Sequence[Mapping[str, Any]],
    suffix: Sequence[Mapping[str, Any]],
    branch_step: int,
) -> list[dict[str, Any]]:
    return copy.deepcopy([*bridge[:branch_step], *suffix[branch_step:719]])


def _rank_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["completion_rate"]),
        float(row["minimum_opponent_raw_win_rate"]),
        float(row["combined_raw_win_rate"]),
        float(row["minimum_opponent_mean_margin"]),
        float(row["combined_mean_margin"]),
    )


def _play_groups(
    source: Path,
    actions: Path,
    metadata: Path,
    routes: Sequence[tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]],
    default_opponents: Mapping[str, Sequence[Mapping[str, Any]]],
    branch_opponents: Mapping[str, Sequence[Mapping[str, Any]]],
    seeds: Sequence[int],
    label: str,
    capture_audit: bool = True,
) -> tuple[list[dict[str, Any]], float, int]:
    generated = {}
    names = []
    for index, (default_route, branch_route) in enumerate(routes):
        default_name, branch_name = f"{label}D{index:04d}", f"{label}B{index:04d}"
        generated[default_name] = list(default_route)
        generated[branch_name] = list(branch_route)
        names.append((default_name, branch_name))
    opponent_tapes = {**default_opponents, **branch_opponents}
    bundle = NativeTeammateBundle(
        source, actions, metadata,
        additional_routes={**opponent_tapes, **generated},
    )
    groups = (tuple(default_opponents), tuple(branch_opponents))
    count = len(routes) * sum(map(len, groups)) * len(seeds) * 2
    tasks = np.empty((count, 7), dtype=np.int64)
    task_map = []
    row = 0
    for route_index, route_names in enumerate(names):
        for group_index, opponents in enumerate(groups):
            own = bundle.index(route_names[group_index])
            for opponent in opponents:
                other = bundle.index(opponent)
                for seed in seeds:
                    for seat in (0, 1):
                        left, right = (own, other) if seat == 0 else (other, own)
                        tasks[row] = (left, right, seed, -1, -1, -1, -1)
                        task_map.append((route_index, opponent, seat))
                        row += 1
    started = time.perf_counter()
    if capture_audit:
        rewards, audit = bundle.executor.play_audit_batch(tasks)
    else:
        rewards = bundle.executor.play_batch(tasks)
        audit = np.zeros((len(tasks), 2, 3), dtype=np.int32)
    elapsed = time.perf_counter() - started
    rewards = np.asarray(rewards, dtype=np.float64)
    audit = np.asarray(audit, dtype=np.int32)
    margins = [dict() for _ in routes]
    finite = [[] for _ in routes]
    unit_failures = [[] for _ in routes]
    market_failures = [[] for _ in routes]
    for index, (route_index, opponent, seat) in enumerate(task_map):
        pair = rewards[index]
        margins[route_index].setdefault(opponent, []).append(
            float(pair[seat] - pair[1 - seat])
        )
        finite[route_index].append(bool(np.all(np.isfinite(pair))))
        unit_failures[route_index].append(int(audit[index, seat, 0]))
        market_failures[route_index].append(int(audit[index, seat, 1]))
    rows = []
    for index in range(len(routes)):
        rates = {
            opponent: float(np.mean(np.asarray(values) > 0))
            for opponent, values in margins[index].items()
        }
        means = {
            opponent: float(np.mean(values))
            for opponent, values in margins[index].items()
        }
        all_values = np.asarray(
            [value for values in margins[index].values() for value in values],
            dtype=np.float64,
        )
        rows.append({
            "completion_rate": float(np.mean(finite[index])),
            "minimum_opponent_raw_win_rate": min(rates.values()),
            "combined_raw_win_rate": float(np.mean(all_values > 0)),
            "minimum_opponent_mean_margin": min(means.values()),
            "combined_mean_margin": float(np.mean(all_values)),
            "opponent_raw_win_rates": rates,
            "opponent_mean_margins": means,
            "mean_macro_unit_failures": float(np.mean(unit_failures[index])),
            "mean_macro_market_failures": float(np.mean(market_failures[index])),
        })
    return rows, elapsed, len(tasks)


def _best_suffixes(
    source: Path,
    actions: Path,
    metadata: Path,
    bridges: Sequence[Sequence[Mapping[str, Any]]],
    suffix_tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    default_suffixes: Sequence[str],
    branch_suffixes: Sequence[str],
    default_opponents: Mapping[str, Sequence[Mapping[str, Any]]],
    branch_opponents: Mapping[str, Sequence[Mapping[str, Any]]],
    seeds: Sequence[int],
    branch_step: int,
) -> tuple[list[dict[str, Any]], float, int]:
    generated = {}
    default_names = np.empty((len(bridges), len(default_suffixes)), dtype=object)
    branch_names = np.empty((len(bridges), len(branch_suffixes)), dtype=object)
    for bridge_index, bridge in enumerate(bridges):
        for suffix_index, family in enumerate(default_suffixes):
            name = f"SD{bridge_index:04d}_{suffix_index:02d}"
            default_names[bridge_index, suffix_index] = name
            generated[name] = splice_suffix(bridge, suffix_tapes[family], branch_step)
        for suffix_index, family in enumerate(branch_suffixes):
            name = f"SB{bridge_index:04d}_{suffix_index:02d}"
            branch_names[bridge_index, suffix_index] = name
            generated[name] = splice_suffix(bridge, suffix_tapes[family], branch_step)
    opponent_tapes = {**default_opponents, **branch_opponents}
    bundle = NativeTeammateBundle(
        source, actions, metadata,
        additional_routes={**opponent_tapes, **generated},
    )
    output = {}
    elapsed = 0.0
    games = 0
    for label, names, opponents in (
        ("default", default_names, tuple(default_opponents)),
        ("branch", branch_names, tuple(branch_opponents)),
    ):
        shape = (len(bridges), names.shape[1], len(opponents), len(seeds), 2)
        margins = np.empty(shape, dtype=np.float64)
        tasks = np.empty((int(np.prod(shape)), 7), dtype=np.int64)
        task_map = []
        row = 0
        for bridge_index in range(len(bridges)):
            for suffix_index in range(names.shape[1]):
                own = bundle.index(str(names[bridge_index, suffix_index]))
                for opponent_index, opponent in enumerate(opponents):
                    other = bundle.index(opponent)
                    for seed_index, seed in enumerate(seeds):
                        for seat in (0, 1):
                            left, right = (own, other) if seat == 0 else (other, own)
                            tasks[row] = (left, right, seed, -1, -1, -1, -1)
                            task_map.append((bridge_index, suffix_index, opponent_index, seed_index, seat))
                            row += 1
        started = time.perf_counter()
        rewards = np.asarray(bundle.executor.play_batch(tasks), dtype=np.float64)
        elapsed += time.perf_counter() - started
        games += len(tasks)
        for index, target in enumerate(task_map):
            seat = target[-1]
            margins[target] = rewards[index, seat] - rewards[index, 1 - seat]
        output[label] = margins
    rows = []
    for bridge_index in range(len(bridges)):
        chosen = {}
        all_rates = []
        all_means = []
        all_margins = []
        for label, suffixes, opponents in (
            ("default", default_suffixes, tuple(default_opponents)),
            ("branch", branch_suffixes, tuple(branch_opponents)),
        ):
            block = output[label][bridge_index]
            candidate_rows = []
            for suffix_index, family in enumerate(suffixes):
                values = block[suffix_index]
                rates = np.mean(values > 0, axis=(1, 2))
                means = np.mean(values, axis=(1, 2))
                candidate_rows.append((
                    float(np.min(rates)), float(np.mean(values > 0)),
                    float(np.min(means)), float(np.mean(values)), suffix_index,
                    family, rates, means, values,
                ))
            best = max(candidate_rows, key=lambda value: value[:4])
            chosen[label] = {
                "suffix_index": best[4], "suffix": best[5],
                "minimum_opponent_raw_win_rate": best[0],
                "combined_raw_win_rate": best[1],
                "minimum_opponent_mean_margin": best[2],
                "combined_mean_margin": best[3],
                "opponent_raw_win_rates": {
                    opponent: float(value) for opponent, value in zip(opponents, best[6])
                },
            }
            all_rates.extend(map(float, best[6]))
            all_means.extend(map(float, best[7]))
            all_margins.extend(np.ravel(best[8]).tolist())
        rows.append({
            "completion_rate": 1.0,
            "minimum_opponent_raw_win_rate": min(all_rates),
            "combined_raw_win_rate": float(np.mean(np.asarray(all_margins) > 0)),
            "minimum_opponent_mean_margin": min(all_means),
            "combined_mean_margin": float(np.mean(all_margins)),
            "default": chosen["default"],
            "branch": chosen["branch"],
        })
    return rows, elapsed, games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-family", required=True)
    parser.add_argument(
        "--prefix-family",
        help="Route actually executed before bridge-start; defaults to base-family.",
    )
    parser.add_argument("--market-donors", type=_csv, required=True)
    parser.add_argument("--bridge-start", type=int, required=True)
    parser.add_argument("--branch-step", type=int, required=True)
    parser.add_argument("--default-suffixes", type=_csv, required=True)
    parser.add_argument("--branch-suffixes", type=_csv, required=True)
    parser.add_argument("--default-opponent", type=_opponent, action="append", required=True)
    parser.add_argument("--branch-opponent", type=_opponent, action="append", required=True)
    parser.add_argument("--screen-seeds", type=_ints, required=True)
    parser.add_argument("--selection-seeds", type=_ints, required=True)
    parser.add_argument("--holdout-seeds", type=_ints, required=True)
    parser.add_argument("--selection-top", type=int, default=96)
    parser.add_argument("--holdout-top", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elite-actions", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    route_ids = {
        str(row["family"]): str(row["route_id"])
        for row in metadata["opponent_routes"]
    }
    required = {
        args.base_family, *(tuple([args.prefix_family]) if args.prefix_family else ()),
        *args.market_donors,
        *args.default_suffixes, *args.branch_suffixes,
    }
    missing = sorted(required - set(route_ids))
    if missing:
        raise KeyError(f"unknown family: {missing[0]}")
    tapes = load_action_tapes(args.actions)
    base = tapes[route_ids[args.base_family]]
    prefix_family = args.prefix_family or args.base_family
    prefix = tapes[route_ids[prefix_family]]
    donor_tapes = [tapes[route_ids[family]] for family in args.market_donors]
    steps, options = market_option_space(
        donor_tapes, args.bridge_start, args.branch_step
    )
    genome_count = int(np.prod([len(value) for value in options], dtype=np.int64))
    if genome_count > 200000:
        raise ValueError(f"market Cartesian product is too large: {genome_count}")
    genomes = list(itertools.product(*(range(len(value)) for value in options)))
    bridges = [
        apply_causal_market_genome(
            prefix, base, steps, options, genome, args.bridge_start
        )
        for genome in genomes
    ]
    suffix_tapes = {
        family: tapes[route_ids[family]]
        for family in (*args.default_suffixes, *args.branch_suffixes)
    }
    default_opponents = {
        family: _trace(spec) for family, spec in args.default_opponent
    }
    branch_opponents = {
        family: _trace(spec) for family, spec in args.branch_opponent
    }
    fixed_routes = [(
        splice_suffix(bridge, suffix_tapes[args.default_suffixes[0]], args.branch_step),
        splice_suffix(bridge, suffix_tapes[args.branch_suffixes[0]], args.branch_step),
    ) for bridge in bridges]
    screen_rows, screen_seconds, screen_games = _play_groups(
        args.source, args.actions, args.metadata, fixed_routes,
        default_opponents, branch_opponents, args.screen_seeds, "M0",
    )
    for index, row in enumerate(screen_rows):
        row["genome_index"] = index
        row["genome"] = list(genomes[index])
    balanced = sorted(screen_rows, key=_rank_key, reverse=True)
    default_rank = sorted(screen_rows, key=lambda row: (
        min(row["opponent_raw_win_rates"][family] for family in default_opponents),
        row["combined_raw_win_rate"], row["combined_mean_margin"],
    ), reverse=True)
    branch_rank = sorted(screen_rows, key=lambda row: (
        min(row["opponent_raw_win_rates"][family] for family in branch_opponents),
        row["combined_raw_win_rate"], row["combined_mean_margin"],
    ), reverse=True)
    selected_indices = []
    for rows, limit in (
        (balanced, max(1, args.selection_top * 2 // 3)),
        (default_rank, max(1, args.selection_top // 6)),
        (branch_rank, max(1, args.selection_top // 6)),
        (balanced, args.selection_top),
    ):
        for row in rows[:limit]:
            index = int(row["genome_index"])
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= args.selection_top:
                break
        if len(selected_indices) >= args.selection_top:
            break
    selected_bridges = [bridges[index] for index in selected_indices]
    selection_rows, selection_seconds, selection_games = _best_suffixes(
        args.source, args.actions, args.metadata, selected_bridges, suffix_tapes,
        args.default_suffixes, args.branch_suffixes,
        default_opponents, branch_opponents, args.selection_seeds,
        args.branch_step,
    )
    for local_index, row in enumerate(selection_rows):
        genome_index = selected_indices[local_index]
        row["genome_index"] = genome_index
        row["genome"] = list(genomes[genome_index])
    selection_rows.sort(key=_rank_key, reverse=True)
    holdout_candidates = selection_rows[:args.holdout_top]
    holdout_routes = []
    for row in holdout_candidates:
        bridge = bridges[int(row["genome_index"])]
        holdout_routes.append((
            splice_suffix(bridge, suffix_tapes[row["default"]["suffix"]], args.branch_step),
            splice_suffix(bridge, suffix_tapes[row["branch"]["suffix"]], args.branch_step),
        ))
    holdout_rows, holdout_seconds, holdout_games = _play_groups(
        args.source, args.actions, args.metadata, holdout_routes,
        default_opponents, branch_opponents, args.holdout_seeds, "MH",
    )
    for index, row in enumerate(holdout_rows):
        source_row = holdout_candidates[index]
        row["genome_index"] = source_row["genome_index"]
        row["genome"] = source_row["genome"]
        row["default_suffix"] = source_row["default"]["suffix"]
        row["branch_suffix"] = source_row["branch"]["suffix"]
    holdout_rows.sort(key=_rank_key, reverse=True)
    elite_tapes = {}
    for rank, row in enumerate(holdout_rows, 1):
        bridge = bridges[int(row["genome_index"])]
        default_id = f"CMG{rank:03d}_DEFAULT"
        branch_id = f"CMG{rank:03d}_YARN2"
        elite_tapes[default_id] = splice_suffix(
            bridge, suffix_tapes[row["default_suffix"]], args.branch_step
        )
        elite_tapes[branch_id] = splice_suffix(
            bridge, suffix_tapes[row["branch_suffix"]], args.branch_step
        )
        row["rank"] = rank
        row["default_route_id"] = default_id
        row["branch_route_id"] = branch_id
        row["market_genes"] = {
            str(step): copy.deepcopy(options[index][selected])
            for step, index, selected in zip(steps, range(len(options)), row["genome"])
        }
    packed = zlib.compress(
        json.dumps(elite_tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    args.elite_actions.parent.mkdir(parents=True, exist_ok=True)
    args.elite_actions.write_bytes(packed)
    games = screen_games + selection_games + holdout_games
    native_seconds = screen_seconds + selection_seconds + holdout_seconds
    payload = {
        "schema": "causal-market-bridge-search-v1",
        "engine": "compiled-cpp-fast-kaggriculture",
        "jax_used": False,
        "base_family": args.base_family,
        "prefix_family": prefix_family,
        "market_donors": list(args.market_donors),
        "bridge_start": args.bridge_start,
        "branch_step": args.branch_step,
        "variable_steps": list(steps),
        "option_counts": [len(value) for value in options],
        "candidate_count": len(genomes),
        "screen_seeds": list(args.screen_seeds),
        "selection_seeds": list(args.selection_seeds),
        "holdout_seeds": list(args.holdout_seeds),
        "screen_top": balanced[:32],
        "selection_ranking": selection_rows,
        "holdout_ranking": holdout_rows,
        "games": games,
        "native_seconds": native_seconds,
        "native_games_per_second": games / native_seconds,
        "wall_seconds": time.perf_counter() - started,
        "elite_actions": str(args.elite_actions.resolve()),
        "elite_actions_sha256": hashlib.sha256(packed).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_count": len(genomes),
        "games": games,
        "native_games_per_second": games / native_seconds,
        "wall_seconds": payload["wall_seconds"],
        "top": holdout_rows[:10],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

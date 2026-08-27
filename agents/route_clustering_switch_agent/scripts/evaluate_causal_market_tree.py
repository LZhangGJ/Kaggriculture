#!/usr/bin/env python3
"""Evaluate a causal route tree against real seed-driven shop trajectories."""

from __future__ import annotations

import argparse
import json
import runpy
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.route_switch_features import route_switch_feature_names
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_public_trace_counters import _ints


def select_public_route(
    first: str | None,
    second: str | None,
    third: str | None,
    assets: dict[str, int],
    config: Any,
) -> tuple[str, int]:
    if first == "YARN_STORE":
        return "yarn_first", int(config.yarn_first_start)
    if first != "YARN_STORE" and second == "YARN_STORE":
        return "yarn_second", int(config.yarn_second_start)
    if (
        bool(config.bakery_capital_enabled)
        and first == "BAKERY"
        and second in set(config.bakery_capital_second_shops)
        and assets["COW"] >= int(config.bakery_capital_minimum_cows)
        and assets["SHEEP"] >= int(config.bakery_capital_minimum_sheep)
        and assets["MELON"] >= int(config.bakery_capital_minimum_melons)
        and assets["GOOSE"] <= int(config.bakery_capital_maximum_geese)
    ):
        return "bakery_capital", int(config.bakery_capital_start)
    prefix = (first, second)
    allowed = {tuple(map(str, value[:2])) for value in config.yarn_third_prefixes}
    third_allowed = not allowed or prefix in allowed
    vetoed = (
        bool(config.yarn_third_pet_brunch_veto)
        and prefix == ("PET_CAFE", "BRUNCH_SPOT")
    )
    if (
        bool(config.yarn_third_enabled)
        and third == "YARN_STORE"
        and third_allowed
        and not vetoed
    ):
        return "yarn_third", int(config.yarn_third_start)
    return "default", -1


def _metrics(
    margins: np.ndarray, selected: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    margins = np.asarray(margins, dtype=np.float64)
    if margins.size % 2:
        raise ValueError("the full margin panel must contain both seats")
    mask = (
        np.ones(margins.size, dtype=bool)
        if selected is None else np.asarray(selected, dtype=bool)
    )
    if mask.shape != margins.shape:
        raise ValueError("selection mask and margin panel differ")
    values = margins[mask]
    paired = margins.reshape(-1, 2)
    paired_mask = mask.reshape(-1, 2).all(axis=1)
    complete_pairs = paired[paired_mask]
    return {
        "samples": int(values.size),
        "paired_seed_samples": int(complete_pairs.shape[0]),
        "raw_win_rate": float(np.mean(values > 0)),
        "both_seats_win_rate": (
            float(np.mean(np.all(complete_pairs > 0, axis=1)))
            if len(complete_pairs) else None
        ),
        "mean_margin": float(np.mean(values)),
        "minimum_margin": float(np.min(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--public-agent", type=Path, required=True)
    parser.add_argument("--default-route", required=True)
    parser.add_argument("--yarn-first-route", required=True)
    parser.add_argument("--yarn-second-route", required=True)
    parser.add_argument("--yarn-third-route", required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    namespace = runpy.run_path(str(args.public_agent.resolve()))
    public_routes = namespace["_V44_ROUTES"]
    config = namespace["_V44_CONFIG"]
    opponent_names = {name: f"PUBLIC_{name.upper()}" for name in public_routes}
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata,
        additional_routes={opponent_names[name]: tape for name, tape in public_routes.items()},
    )
    candidate_names = {
        "default": args.default_route,
        "yarn_first": args.yarn_first_route,
        "yarn_second": args.yarn_second_route,
        "yarn_third": args.yarn_third_route,
        "bakery_capital": args.default_route,
    }

    tapes = load_action_tapes(args.actions)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    route_ids = {
        str(value["family"]): str(value["route_id"])
        for value in metadata["opponent_routes"]
    }
    default_tape = tapes[route_ids[args.default_route]]
    prefix_checks = {
        "yarn_first_to_120": default_tape[:120]
        == tapes[route_ids[args.yarn_first_route]][:120],
        "yarn_second_to_144": default_tape[:144]
        == tapes[route_ids[args.yarn_second_route]][:144],
        "yarn_third_to_216": default_tape[:216]
        == tapes[route_ids[args.yarn_third_route]][:216],
        "opponent_yarn_first_to_88": public_routes["default"][:88]
        == public_routes["yarn_first"][:88],
        "opponent_yarn_second_to_153": public_routes["default"][:153]
        == public_routes["yarn_second"][:153],
        "opponent_bakery_to_160": public_routes["default"][:160]
        == public_routes["bakery_capital"][:160],
        "opponent_yarn_third_to_216": public_routes["default"][:216]
        == public_routes["yarn_third"][:216],
    }
    if not all(prefix_checks.values()):
        raise ValueError(f"non-causal route prefix: {prefix_checks}")

    names = route_switch_feature_names()
    name_index = {name: index for index, name in enumerate(names)}
    shop_indices = {
        name.removeprefix("shop_").upper(): index
        for index, name in enumerate(names) if name.startswith("shop_")
    }
    probe_candidate = bundle.index(args.default_route)
    probe_opponent = bundle.index(opponent_names["default"])
    samples = [(seed, seat) for seed in args.seeds for seat in (0, 1)]
    features = {}
    feature_started = time.perf_counter()
    for checkpoint in (88, 153, 160, 216):
        tasks = []
        for seed, seat in samples:
            if seat == 0:
                tasks.append((
                    probe_candidate, probe_opponent, seed,
                    checkpoint, seat, probe_candidate,
                ))
            else:
                tasks.append((
                    probe_opponent, probe_candidate, seed,
                    checkpoint, seat, probe_candidate,
                ))
        features[checkpoint] = np.asarray(
            bundle.executor.features_batch(np.asarray(tasks, dtype=np.int64)),
            dtype=np.float32,
        )
    feature_seconds = time.perf_counter() - feature_started

    selections = []
    for index, (seed, seat) in enumerate(samples):
        shop_sets = []
        for checkpoint in (88, 153, 216):
            vector = features[checkpoint][index]
            shop_sets.append({
                shop for shop, feature in shop_indices.items()
                if vector[feature] > 0.5
            })
        first_values = shop_sets[0]
        second_values = shop_sets[1] - shop_sets[0]
        third_values = shop_sets[2] - shop_sets[1]
        first = next(iter(first_values), None) if len(first_values) == 1 else None
        second = next(iter(second_values), None) if len(second_values) == 1 else None
        third = next(iter(third_values), None) if len(third_values) == 1 else None
        vector160 = features[160][index]
        assets = {
            item: int(round(float(vector160[name_index[f"self_{item.lower()}_count"]])))
            for item in ("COW", "SHEEP", "MELON", "GOOSE")
        }
        public_route, switch_step = select_public_route(
            first, second, third, assets, config
        )
        selections.append({
            "seed": int(seed), "seat": seat,
            "shops": [first, second, third], "assets160": assets,
            "public_route": public_route, "public_switch_step": switch_step,
            "candidate_route": candidate_names[public_route],
        })

    def task(row: dict[str, Any], candidate_family: str) -> tuple[int, ...]:
        candidate = bundle.index(candidate_family)
        opponent = bundle.index(opponent_names["default"])
        switch_step = int(row["public_switch_step"])
        switch_target = bundle.index(opponent_names[row["public_route"]])
        if row["seat"] == 0:
            return (
                candidate, opponent, row["seed"], -1, -1,
                switch_step, switch_target,
            )
        return (
            opponent, candidate, row["seed"], switch_step, switch_target,
            -1, -1,
        )

    baseline_tasks = np.asarray([
        task(row, args.default_route) for row in selections
    ], dtype=np.int64)
    tree_tasks = np.asarray([
        task(row, row["candidate_route"]) for row in selections
    ], dtype=np.int64)
    game_started = time.perf_counter()
    rewards = np.asarray(
        bundle.executor.play_batch(np.concatenate((baseline_tasks, tree_tasks))),
        dtype=np.float64,
    )
    game_seconds = time.perf_counter() - game_started
    baseline_rewards, tree_rewards = np.split(rewards, 2)
    seats = np.asarray([row["seat"] for row in selections], dtype=np.int64)
    indices = np.arange(len(seats))
    baseline_margins = (
        baseline_rewards[indices, seats] - baseline_rewards[indices, 1 - seats]
    )
    tree_margins = tree_rewards[indices, seats] - tree_rewards[indices, 1 - seats]
    groups = {}
    for route in sorted(set(row["public_route"] for row in selections)):
        selected = np.asarray([
            row["public_route"] == route for row in selections
        ])
        groups[route] = {
            "samples": int(np.count_nonzero(selected)),
            "baseline": _metrics(baseline_margins, selected),
            "causal_tree": _metrics(tree_margins, selected),
        }
    payload = {
        "schema": "causal-real-market-tree-evaluation-v1",
        "engine": "compiled-cpp-fast-kaggriculture",
        "jax_used": False,
        "market_source": "seed-driven simulator shops",
        "opponent_model": (
            "public V44 child tapes with exact shop/bakery route-selection triggers; "
            "stateful clone-preemption and market-maker overlays excluded"
        ),
        "seeds": list(args.seeds),
        "games": int(len(rewards)),
        "feature_seconds": feature_seconds,
        "game_seconds": game_seconds,
        "games_per_second": len(rewards) / game_seconds,
        "causal_prefix_checks": prefix_checks,
        "route_counts": dict(Counter(
            row["public_route"] for row in selections
        )),
        "candidate_counts": dict(Counter(
            row["candidate_route"] for row in selections
        )),
        "baseline": _metrics(baseline_margins),
        "causal_tree": _metrics(tree_margins),
        "groups": groups,
        "selections": selections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "games": payload["games"],
        "games_per_second": payload["games_per_second"],
        "prefix_checks": prefix_checks,
        "route_counts": payload["route_counts"],
        "baseline": payload["baseline"],
        "causal_tree": payload["causal_tree"],
        "groups": groups,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Screen a causal market route tree against every fixed C++ route in a pool."""

from __future__ import annotations

import argparse
import json
import runpy
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_causal_market_tree import select_public_route
from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.route_switch_features import route_switch_feature_names
from search_native_public_trace_counters import _ints


def _metrics(margins: np.ndarray) -> dict[str, float | int]:
    paired = margins.reshape(-1, 2)
    return {
        "samples": int(margins.size),
        "raw_win_rate": float(np.mean(margins > 0)),
        "both_seats_win_rate": float(np.mean(np.all(paired > 0, axis=1))),
        "mean_margin": float(np.mean(margins)),
        "minimum_margin": float(np.min(margins)),
        "completion_rate": 1.0,
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
    parser.add_argument("--opponent-team", default="nt_68_route_pool")
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    namespace = runpy.run_path(str(args.public_agent.resolve()))
    config = namespace["_V44_CONFIG"]
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    entries = [
        row for row in metadata["opponent_routes"]
        if str(row.get("team")) == args.opponent_team
    ]
    if not entries:
        raise ValueError(f"no routes for opponent team {args.opponent_team}")
    opponents = [str(row["family"]) for row in entries]
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)

    names = route_switch_feature_names()
    shop_indices = {
        name.removeprefix("shop_").upper(): index
        for index, name in enumerate(names) if name.startswith("shop_")
    }
    probe = bundle.index(args.default_route)
    samples = [(seed, seat) for seed in args.seeds for seat in (0, 1)]
    features = {}
    feature_started = time.perf_counter()
    for checkpoint in (88, 153, 216):
        tasks = np.asarray([
            (
                probe, probe, seed, checkpoint, seat, probe
            ) if seat == 0 else (
                probe, probe, seed, checkpoint, seat, probe
            )
            for seed, seat in samples
        ], dtype=np.int64)
        features[checkpoint] = np.asarray(
            bundle.executor.features_batch(tasks), dtype=np.float32
        )
    feature_seconds = time.perf_counter() - feature_started

    candidate_names = {
        "default": args.default_route,
        "bakery_capital": args.default_route,
        "yarn_first": args.yarn_first_route,
        "yarn_second": args.yarn_second_route,
        "yarn_third": args.yarn_third_route,
    }
    selected_candidates = []
    shop_paths = []
    empty_assets = {"COW": 0, "SHEEP": 0, "MELON": 0, "GOOSE": 0}
    for index, _ in enumerate(samples):
        active = []
        for checkpoint in (88, 153, 216):
            vector = features[checkpoint][index]
            active.append({
                shop for shop, feature in shop_indices.items()
                if vector[feature] > 0.5
            })
        first_values = active[0]
        second_values = active[1] - active[0]
        third_values = active[2] - active[1]
        first = next(iter(first_values), None) if len(first_values) == 1 else None
        second = next(iter(second_values), None) if len(second_values) == 1 else None
        third = next(iter(third_values), None) if len(third_values) == 1 else None
        route, _ = select_public_route(first, second, third, empty_assets, config)
        selected_candidates.append(candidate_names[route])
        shop_paths.append((first, second, third))

    task_rows = []
    task_meta = []
    for opponent in opponents:
        opponent_index = bundle.index(opponent)
        for sample_index, (seed, seat) in enumerate(samples):
            candidate_index = bundle.index(selected_candidates[sample_index])
            pair = (
                (candidate_index, opponent_index)
                if seat == 0 else (opponent_index, candidate_index)
            )
            task_rows.append((*pair, seed, -1, -1, -1, -1))
            task_meta.append((opponent, seat))
    game_started = time.perf_counter()
    rewards = np.asarray(
        bundle.executor.play_batch(np.asarray(task_rows, dtype=np.int64)),
        dtype=np.float64,
    )
    game_seconds = time.perf_counter() - game_started
    seats = np.asarray([seat for _, seat in task_meta], dtype=np.int64)
    row_indices = np.arange(len(seats))
    margins = rewards[row_indices, seats] - rewards[row_indices, 1 - seats]
    margins = margins.reshape(len(opponents), len(args.seeds), 2)

    entry_by_family = {str(row["family"]): row for row in entries}
    route_rows = []
    member_routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, opponent in enumerate(opponents):
        metrics = _metrics(margins[index].reshape(-1))
        entry = entry_by_family[opponent]
        sources = list(entry.get("provenance", {}).get("sources", ()))
        route_rows.append({
            "family": opponent, "alias": str(entry.get("alias", "")),
            "metrics": metrics, "sources": sources,
        })
        for source_row in sources:
            if "runtime_routing_table" not in source_row.get("selection_reasons", ()):
                continue
            for member in source_row.get("member_names", ()):
                member_routes[str(member)].append({
                    "family": opponent, "metrics": metrics,
                })
    route_rows.sort(key=lambda row: (
        row["metrics"]["raw_win_rate"], row["metrics"]["mean_margin"], row["family"]
    ))
    member_rows = []
    for member, rows in member_routes.items():
        member_rows.append({
            "member": member,
            "runtime_route_count": len(rows),
            "minimum_route_raw_win_rate": min(
                row["metrics"]["raw_win_rate"] for row in rows
            ),
            "minimum_route_completion_rate": min(
                row["metrics"]["completion_rate"] for row in rows
            ),
            "routes": sorted(rows, key=lambda row: row["family"]),
        })
    member_rows.sort(key=lambda row: (
        row["minimum_route_raw_win_rate"], row["member"]
    ))
    payload = {
        "schema": "causal-tree-fixed-route-pool-screen-v1",
        "engine": "compiled-cpp-fast-kaggriculture",
        "jax_used": False,
        "opponent_team": args.opponent_team,
        "seed_count": len(args.seeds),
        "route_count": len(opponents),
        "games": len(task_rows),
        "feature_seconds": feature_seconds,
        "game_seconds": game_seconds,
        "games_per_second": len(task_rows) / game_seconds,
        "candidate_route_counts": dict(Counter(selected_candidates)),
        "minimum_fixed_route_raw_win_rate": route_rows[0]["metrics"]["raw_win_rate"],
        "fixed_routes_at_or_above_90pct": sum(
            row["metrics"]["raw_win_rate"] >= 0.9 for row in route_rows
        ),
        "runtime_members_with_exact_routes": len(member_rows),
        "runtime_members_at_or_above_90pct_strict_envelope": sum(
            row["minimum_route_raw_win_rate"] >= 0.9 for row in member_rows
        ),
        "members": member_rows,
        "routes": route_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: payload[key] for key in (
            "route_count", "games", "games_per_second", "candidate_route_counts",
            "minimum_fixed_route_raw_win_rate", "fixed_routes_at_or_above_90pct",
            "runtime_members_with_exact_routes",
            "runtime_members_at_or_above_90pct_strict_envelope",
        )
    } | {
        "worst_routes": [
            {"family": row["family"], "alias": row["alias"], **row["metrics"]}
            for row in route_rows[:10]
        ],
        "worst_members": [
            {key: row[key] for key in (
                "member", "runtime_route_count", "minimum_route_raw_win_rate",
                "minimum_route_completion_rate",
            )}
            for row in member_rows[:10]
        ],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

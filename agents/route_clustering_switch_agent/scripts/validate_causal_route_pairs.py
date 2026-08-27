#!/usr/bin/env python3
"""Validate paired causal leaves on unseen seeds with the compiled C++ engine."""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from search_causal_market_bridges import _play_groups
from search_native_public_trace_counters import _ints, _opponent, _trace


def pair_routes(
    routes: Mapping[str, Sequence[Mapping[str, Any]]],
    default_suffix: str = "_DEFAULT",
    branch_suffix: str = "_YARN2",
) -> list[tuple[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]]:
    result = []
    for family, route in routes.items():
        if not family.endswith(default_suffix):
            continue
        prefix = family[: -len(default_suffix)]
        branch_family = prefix + branch_suffix
        if branch_family not in routes:
            raise KeyError(f"missing paired branch route: {branch_family}")
        result.append((prefix, route, routes[branch_family]))
    if not result:
        raise ValueError("candidate action archive contains no route pairs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--candidate-actions", type=Path, required=True)
    parser.add_argument("--default-opponent", type=_opponent, action="append", required=True)
    parser.add_argument("--branch-opponent", type=_opponent, action="append", required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    routes = json.loads(zlib.decompress(args.candidate_actions.read_bytes()))
    pairs = pair_routes(routes)
    default_opponents = {
        family: _trace(spec) for family, spec in args.default_opponent
    }
    branch_opponents = {
        family: _trace(spec) for family, spec in args.branch_opponent
    }
    rows, native_seconds, games = _play_groups(
        args.source, args.actions, args.metadata,
        [(default, branch) for _, default, branch in pairs],
        default_opponents, branch_opponents, args.seeds,
        "V", capture_audit=True,
    )
    for (family, _, _), row in zip(pairs, rows):
        row["family"] = family
    rows.sort(key=lambda row: (
        row["completion_rate"], row["minimum_opponent_raw_win_rate"],
        row["combined_raw_win_rate"], row["minimum_opponent_mean_margin"],
        row["combined_mean_margin"], -row["mean_macro_market_failures"],
    ), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    payload = {
        "schema": "causal-route-pair-validation-v1",
        "engine": "compiled-cpp-fast-kaggriculture",
        "jax_used": False,
        "candidate_actions": str(args.candidate_actions.resolve()),
        "seeds": list(args.seeds),
        "games": games,
        "native_seconds": native_seconds,
        "native_games_per_second": games / native_seconds,
        "wall_seconds": time.perf_counter() - started,
        "ranking": rows,
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
        "top": rows[:20],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

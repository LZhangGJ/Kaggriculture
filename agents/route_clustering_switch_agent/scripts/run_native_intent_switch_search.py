#!/usr/bin/env python3
"""Thin serializer for the compact all-C++ counterfactual switch search."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from search_native_public_trace_counters import _opponent, _trace


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
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--openings", type=_csv, required=True)
    parser.add_argument("--targets", type=_csv)
    parser.add_argument("--checkpoints", type=_ints, required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument(
        "--route-archive", type=Path, action="append", default=[],
        help="Add a zlib-compressed JSON mapping of route family to action tape.",
    )
    parser.add_argument(
        "--external-opponent", type=_opponent, action="append", default=[],
        help="Add FAMILY=PATH or FAMILY=PATH::VARIABLE::KEY.",
    )
    parser.add_argument("--opponents", type=_csv)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    for parent in {args.output.parent, args.summary.parent}:
        with tempfile.TemporaryFile(dir=parent):
            pass

    setup_started = time.perf_counter()
    additional_routes = {}
    for path in args.route_archive:
        archive = json.loads(zlib.decompress(path.read_bytes()))
        for family, tape in archive.items():
            if family in additional_routes:
                raise ValueError(f"duplicate additional route: {family}")
            additional_routes[str(family)] = list(tape)
    for family, trace_spec in args.external_opponent:
        if family in additional_routes:
            raise ValueError(f"duplicate additional route: {family}")
        additional_routes[family] = _trace(trace_spec)
    included_families = (
        tuple(dict.fromkeys((*args.openings, *args.targets, *args.opponents)))
        if args.targets and args.opponents else None
    )
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata,
        additional_routes=additional_routes,
        included_families=included_families,
    )
    openings = args.openings
    external_opponents = {family for family, _ in args.external_opponent}
    targets = args.targets or tuple(
        family for family in bundle.families if family not in external_opponents
    )
    opening_indices = [bundle.index(value) for value in openings]
    target_indices = [bundle.index(value) for value in targets]
    opponents = args.opponents or bundle.families
    opponent_indices = [bundle.index(value) for value in opponents]
    setup_seconds = time.perf_counter() - setup_started

    started = time.perf_counter()
    result = bundle.executor.switch_search(
        opening_indices, target_indices, args.checkpoints, args.seeds,
        opponent_indices,
    )
    simulation_seconds = time.perf_counter() - started
    outcome = np.asarray(result["outcome"], dtype=np.uint8)
    margin = np.asarray(result["margin"], dtype=np.float32)
    states = np.asarray(result["states"], dtype=np.float32)
    games = int(outcome.size)

    np.savez(
        args.output,
        outcome=outcome,
        margin=margin,
        states=states,
        openings=np.asarray(openings),
        targets=np.asarray(targets),
        opponents=np.asarray(opponents),
        checkpoints=np.asarray(args.checkpoints, dtype=np.int16),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        engine=np.asarray("C++ NativeTeammateExecutor.switch_search"),
    )
    payload = {
        "schema_version": 1,
        "engine": "C++ NativeTeammateExecutor.switch_search",
        "layout": "opening, checkpoint, target, opponent, seed, seat",
        "openings": list(openings),
        "target_count": len(targets),
        "opponent_count": len(opponents),
        "checkpoints": list(args.checkpoints),
        "seed_count": len(args.seeds),
        "pairing": "identical seeds and both seats",
        "games": games,
        "state_vectors": int(np.prod(states.shape[:-1])),
        "feature_count": int(states.shape[-1]),
        "setup_seconds": setup_seconds,
        "simulation_seconds": simulation_seconds,
        "games_per_second": games / simulation_seconds,
        "output": str(args.output),
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

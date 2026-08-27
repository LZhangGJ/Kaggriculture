#!/usr/bin/env python3
"""Search causal bridge segments with the compiled C++ match engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_public_trace_counters import _ints, _opponent, _trace


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def splice_three(
    prefix: Sequence[Mapping[str, Any]],
    bridge: Sequence[Mapping[str, Any]],
    suffix: Sequence[Mapping[str, Any]],
    bridge_start: int,
    branch_step: int,
) -> list[dict[str, Any]]:
    if any(len(tape) < 719 for tape in (prefix, bridge, suffix)):
        raise ValueError("all route tapes must contain 719 actions")
    if not 0 <= bridge_start <= branch_step <= 719:
        raise ValueError("expected 0 <= bridge_start <= branch_step <= 719")
    return [
        dict(value)
        for value in (
            *prefix[:bridge_start],
            *bridge[bridge_start:branch_step],
            *suffix[branch_step:719],
        )
    ]


def _segment_digest(
    tape: Sequence[Mapping[str, Any]], start: int, stop: int
) -> str:
    raw = json.dumps(
        tape[start:stop], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _best_suffix(
    margins: np.ndarray, suffixes: Sequence[str]
) -> tuple[int, dict[str, Any]]:
    rows = []
    for index, suffix in enumerate(suffixes):
        block = margins[index]
        rates = np.mean(block > 0, axis=(1, 2))
        rows.append({
            "suffix": suffix,
            "minimum_opponent_raw_win_rate": float(np.min(rates)),
            "combined_raw_win_rate": float(np.mean(block > 0)),
            "minimum_opponent_mean_margin": float(np.min(np.mean(block, axis=(1, 2)))),
            "combined_mean_margin": float(np.mean(block)),
            "opponent_raw_win_rates": [float(value) for value in rates],
        })
    best = max(range(len(rows)), key=lambda index: (
        rows[index]["minimum_opponent_raw_win_rate"],
        rows[index]["combined_raw_win_rate"],
        rows[index]["minimum_opponent_mean_margin"],
        rows[index]["combined_mean_margin"],
    ))
    return best, rows[best]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prefix-family", required=True)
    parser.add_argument("--bridge-start", type=int, required=True)
    parser.add_argument("--branch-step", type=int, required=True)
    parser.add_argument("--bridges", type=_csv)
    parser.add_argument("--bridge-team", default="nt_68_route_pool")
    parser.add_argument("--default-suffixes", type=_csv, required=True)
    parser.add_argument("--branch-suffixes", type=_csv, required=True)
    parser.add_argument(
        "--default-opponent", type=_opponent, action="append", required=True
    )
    parser.add_argument(
        "--branch-opponent", type=_opponent, action="append", required=True
    )
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.bridge_start <= args.branch_step <= 719:
        parser.error("expected 0 <= bridge-start <= branch-step <= 719")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    entries = list(metadata["opponent_routes"])
    route_by_family = {
        str(value["family"]): str(value["route_id"]) for value in entries
    }
    tapes = load_action_tapes(args.actions)
    required = {
        args.prefix_family, *args.default_suffixes, *args.branch_suffixes
    }
    missing = sorted(required - set(route_by_family))
    if missing:
        raise KeyError(f"unknown route family: {missing[0]}")
    raw_bridges = args.bridges or tuple(
        str(value["family"])
        for value in entries
        if str(value.get("team", "")) == args.bridge_team
    )
    unique_bridges = []
    seen_segments = set()
    for family in raw_bridges:
        if family not in route_by_family:
            raise KeyError(f"unknown bridge family: {family}")
        digest = _segment_digest(
            tapes[route_by_family[family]], args.bridge_start, args.branch_step
        )
        if digest not in seen_segments:
            seen_segments.add(digest)
            unique_bridges.append((family, digest))

    prefix = tapes[route_by_family[args.prefix_family]]
    generated: dict[str, list[dict[str, Any]]] = {}
    default_names = np.empty(
        (len(unique_bridges), len(args.default_suffixes)), dtype=object
    )
    branch_names = np.empty(
        (len(unique_bridges), len(args.branch_suffixes)), dtype=object
    )
    for bridge_index, (bridge_family, _digest) in enumerate(unique_bridges):
        bridge = tapes[route_by_family[bridge_family]]
        for suffix_index, suffix_family in enumerate(args.default_suffixes):
            name = f"CBD_{bridge_index:04d}_{suffix_index:03d}"
            default_names[bridge_index, suffix_index] = name
            generated[name] = splice_three(
                prefix, bridge, tapes[route_by_family[suffix_family]],
                args.bridge_start, args.branch_step,
            )
        for suffix_index, suffix_family in enumerate(args.branch_suffixes):
            name = f"CBB_{bridge_index:04d}_{suffix_index:03d}"
            branch_names[bridge_index, suffix_index] = name
            generated[name] = splice_three(
                prefix, bridge, tapes[route_by_family[suffix_family]],
                args.bridge_start, args.branch_step,
            )

    default_opponents = {
        family: _trace(path) for family, path in args.default_opponent
    }
    branch_opponents = {
        family: _trace(path) for family, path in args.branch_opponent
    }
    overlap = set(default_opponents) & set(branch_opponents)
    if overlap:
        raise ValueError(f"opponent appears in both groups: {sorted(overlap)[0]}")
    bundle = NativeTeammateBundle(
        args.source,
        args.actions,
        args.metadata,
        additional_routes={**default_opponents, **branch_opponents, **generated},
    )

    groups = (
        ("default", default_names, tuple(default_opponents)),
        ("branch", branch_names, tuple(branch_opponents)),
    )
    output_margins: dict[str, np.ndarray] = {}
    elapsed = 0.0
    games = 0
    for label, names, opponent_families in groups:
        shape = (
            len(unique_bridges), names.shape[1], len(opponent_families),
            len(args.seeds), 2,
        )
        margins = np.empty(shape, dtype=np.float64)
        tasks = np.empty((int(np.prod(shape[:-1])) * 2, 7), dtype=np.int64)
        row_map = []
        row = 0
        for bridge_index in range(names.shape[0]):
            for suffix_index in range(names.shape[1]):
                route = bundle.index(str(names[bridge_index, suffix_index]))
                for opponent_index, opponent_family in enumerate(opponent_families):
                    opponent = bundle.index(opponent_family)
                    for seed_index, seed in enumerate(args.seeds):
                        for seat in (0, 1):
                            left, right = (
                                (route, opponent) if seat == 0 else (opponent, route)
                            )
                            tasks[row] = (left, right, seed, -1, -1, -1, -1)
                            row_map.append((bridge_index, suffix_index, opponent_index, seed_index, seat))
                            row += 1
        started = time.perf_counter()
        rewards = np.asarray(bundle.executor.play_batch(tasks), dtype=np.float64)
        elapsed += time.perf_counter() - started
        games += len(tasks)
        for index, target in enumerate(row_map):
            seat = target[-1]
            margins[target] = rewards[index, seat] - rewards[index, 1 - seat]
        output_margins[label] = margins

    ranking = []
    for bridge_index, (bridge_family, digest) in enumerate(unique_bridges):
        default_index, default_row = _best_suffix(
            output_margins["default"][bridge_index], args.default_suffixes
        )
        branch_index, branch_row = _best_suffix(
            output_margins["branch"][bridge_index], args.branch_suffixes
        )
        ranking.append({
            "bridge_family": bridge_family,
            "bridge_segment_sha256": digest,
            "bridge_start": args.bridge_start,
            "branch_step": args.branch_step,
            "minimum_branch_raw_win_rate": min(
                default_row["minimum_opponent_raw_win_rate"],
                branch_row["minimum_opponent_raw_win_rate"],
            ),
            "combined_raw_win_rate": float(np.mean([
                default_row["combined_raw_win_rate"],
                branch_row["combined_raw_win_rate"],
            ])),
            "default": {"suffix_index": default_index, **default_row},
            "branch": {"suffix_index": branch_index, **branch_row},
        })
    ranking.sort(key=lambda value: (
        -value["minimum_branch_raw_win_rate"],
        -value["combined_raw_win_rate"],
        -value["default"]["minimum_opponent_mean_margin"],
        -value["branch"]["minimum_opponent_mean_margin"],
        value["bridge_family"],
    ))

    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.matrix_output,
        bridge_families=np.asarray([value[0] for value in unique_bridges]),
        bridge_digests=np.asarray([value[1] for value in unique_bridges]),
        default_suffixes=np.asarray(args.default_suffixes),
        branch_suffixes=np.asarray(args.branch_suffixes),
        default_opponents=np.asarray(tuple(default_opponents)),
        branch_opponents=np.asarray(tuple(branch_opponents)),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        default_margins=output_margins["default"],
        branch_margins=output_margins["branch"],
    )
    payload = {
        "schema": "causal-route-bridge-search-v1",
        "prefix_family": args.prefix_family,
        "bridge_start": args.bridge_start,
        "branch_step": args.branch_step,
        "raw_bridge_count": len(raw_bridges),
        "unique_bridge_count": len(unique_bridges),
        "default_suffixes": list(args.default_suffixes),
        "branch_suffixes": list(args.branch_suffixes),
        "default_opponents": [
            {"family": family, "trace_spec": path}
            for family, path in args.default_opponent
        ],
        "branch_opponents": [
            {"family": family, "trace_spec": path}
            for family, path in args.branch_opponent
        ],
        "seeds": list(args.seeds),
        "games": games,
        "elapsed_seconds": elapsed,
        "games_per_second": games / elapsed,
        "matrix_output": str(args.matrix_output.resolve()),
        "ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "games": games,
        "elapsed_seconds": elapsed,
        "games_per_second": games / elapsed,
        "raw_bridges": len(raw_bridges),
        "unique_bridges": len(unique_bridges),
        "top": ranking[:20],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract candidate-visible checkpoint features against an existing policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.route_switch_features import route_switch_feature_names
from search_native_counter_schedules import _policy_response_grid


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
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-opening", default="G001")
    parser.add_argument("--candidate-opening", default="G001")
    parser.add_argument("--checkpoints", default="144,168,216")
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoints = tuple(int(value) for value in args.checkpoints.split(","))
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    nodes = sorted(
        (
            value["selected"] for value in policy["nodes"]
            if value["selected"].get("enabled", True)
            and str(value["selected"]["opening"]) == args.policy_opening
        ),
        key=lambda value: int(value["checkpoint"]),
    )
    candidate_route = bundle.index(args.candidate_opening)
    policy_route = bundle.index(args.policy_opening)
    samples = [(int(seed), seat) for seed in args.seeds for seat in (0, 1)]
    output: dict[str, np.ndarray] = {
        "seeds": np.asarray(args.seeds, dtype=np.int64),
        "feature_names": np.asarray(route_switch_feature_names()),
    }

    for checkpoint in checkpoints:
        policy_step, policy_target = _policy_response_grid(
            bundle, nodes, args.policy_opening, args.candidate_opening,
            (args.candidate_opening,), checkpoint, args.seeds,
        )
        tasks = np.empty((len(samples), 10), dtype=np.int64)
        for index, (seed, candidate_seat) in enumerate(samples):
            response_step = int(policy_step[0, index])
            response_target = int(policy_target[0, index])
            if candidate_seat == 0:
                tasks[index] = (
                    candidate_route, policy_route, seed, checkpoint, 0, candidate_route,
                    -1, -1, response_step, response_target,
                )
            else:
                tasks[index] = (
                    policy_route, candidate_route, seed, checkpoint, 1, candidate_route,
                    response_step, response_target, -1, -1,
                )
        output[f"features_{checkpoint}"] = np.asarray(
            bundle.executor.features_with_switch_batch(tasks), dtype=np.float32
        ).reshape(len(args.seeds), 2, -1)
        output[f"policy_steps_{checkpoint}"] = policy_step.reshape(
            len(args.seeds), 2
        )
        output[f"policy_targets_{checkpoint}"] = policy_target.reshape(
            len(args.seeds), 2
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "seeds": len(args.seeds),
        "samples": len(samples),
        "checkpoints": list(checkpoints),
        "features": int(output[f"features_{checkpoints[0]}"].shape[-1]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

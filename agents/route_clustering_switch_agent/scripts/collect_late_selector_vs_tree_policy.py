#!/usr/bin/env python3
"""Collect a late selector matrix after the opponent may already have switched."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from meta_agent.src.search_route_policy import NumpySearchTree
from meta_agent.src.teammate_expanded_routes import load_action_tapes


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(part) for part in value.split(",") if part.strip()
    )


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-actions", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--opponent-policy", type=Path, required=True)
    parser.add_argument("--opponent-opening", default="G001")
    parser.add_argument("--candidate-parent-actions", type=Path, required=True)
    parser.add_argument("--candidate-parent-metadata", type=Path, required=True)
    parser.add_argument("--shared-actions", type=Path, required=True)
    parser.add_argument("--opening", default="NR295")
    parser.add_argument("--targets", type=_csv, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parent_metadata = json.loads(args.candidate_parent_metadata.read_text(encoding="utf-8"))
    parent_row = next(row for row in parent_metadata["opponent_routes"] if row["family"] == args.opening)
    parent_actions = load_action_tapes(args.candidate_parent_actions)
    shared_actions = load_action_tapes(args.shared_actions)
    additional = {args.opening: parent_actions[str(parent_row["route_id"])]}
    additional.update({family: shared_actions[family] for family in args.targets if family != args.opening})
    policy = json.loads(args.opponent_policy.read_text(encoding="utf-8"))
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
    bundle = NativeTeammateBundle(
        args.source, args.base_actions, args.base_metadata,
        additional_routes=additional, included_families=sorted(policy_families),
    )
    candidate_index = bundle.index(args.opening)
    opponent_index = bundle.index(args.opponent_opening)
    target_indices = np.asarray([bundle.index(family) for family in args.targets])
    samples = [(seed, seat) for seed in args.seeds for seat in (0, 1)]

    def routes(seed: int, seat: int) -> list[int]:
        values = [opponent_index, opponent_index]
        values[seat] = candidate_index
        return values

    opponent_steps = np.full(len(samples), -1, dtype=np.int64)
    opponent_targets = np.full(len(samples), opponent_index, dtype=np.int64)
    for node_checkpoint, tree in sorted(nodes):
        if node_checkpoint > args.checkpoint:
            continue
        tasks = np.empty((len(samples), 6), dtype=np.int64)
        for row, (seed, seat) in enumerate(samples):
            tasks[row] = [
                *routes(seed, seat), seed, node_checkpoint, 1 - seat, opponent_index
            ]
        features = np.asarray(bundle.executor.features_batch(tasks))
        for row, vector in enumerate(features):
            if opponent_steps[row] >= 0:
                continue
            prediction = tree.predict(vector)
            if prediction != args.opponent_opening:
                opponent_steps[row] = node_checkpoint
                opponent_targets[row] = bundle.index(prediction)

    state_tasks = np.empty((len(samples), 10), dtype=np.int64)
    for row, (seed, seat) in enumerate(samples):
        switches = [-1, -1, -1, -1]
        opponent_seat = 1 - seat
        switches[2 * opponent_seat:2 * opponent_seat + 2] = [
            opponent_steps[row], opponent_targets[row]
        ]
        state_tasks[row] = [
            *routes(seed, seat), seed, args.checkpoint, seat, candidate_index, *switches
        ]
    states = np.asarray(bundle.executor.features_with_switch_batch(state_tasks))
    outcome = np.empty((len(args.targets), len(samples)), dtype=np.uint8)
    margin = np.empty((len(args.targets), len(samples)), dtype=np.float32)

    for target_position, target_index in enumerate(target_indices):
        candidate_step = -1 if target_index == candidate_index else args.checkpoint
        final_opponent_steps = opponent_steps.copy()
        final_opponent_targets = opponent_targets.copy()
        for node_checkpoint, tree in sorted(nodes):
            if node_checkpoint <= args.checkpoint:
                continue
            tasks = np.empty((len(samples), 10), dtype=np.int64)
            for row, (seed, seat) in enumerate(samples):
                switches = [-1, -1, -1, -1]
                switches[2 * seat:2 * seat + 2] = [candidate_step, target_index]
                tasks[row] = [
                    *routes(seed, seat), seed, node_checkpoint, 1 - seat,
                    opponent_index, *switches,
                ]
            features = np.asarray(bundle.executor.features_with_switch_batch(tasks))
            for row, vector in enumerate(features):
                if final_opponent_steps[row] >= 0:
                    continue
                prediction = tree.predict(vector)
                if prediction != args.opponent_opening:
                    final_opponent_steps[row] = node_checkpoint
                    final_opponent_targets[row] = bundle.index(prediction)
        games = np.empty((len(samples), 7), dtype=np.int64)
        for row, (seed, seat) in enumerate(samples):
            switches = [
                final_opponent_steps[row], final_opponent_targets[row],
                final_opponent_steps[row], final_opponent_targets[row],
            ]
            switches[2 * seat:2 * seat + 2] = [candidate_step, target_index]
            games[row] = [*routes(seed, seat), seed, *switches]
        rewards = np.asarray(bundle.executor.play_batch(games))
        own = np.asarray([rewards[row, seat] for row, (_, seat) in enumerate(samples)])
        other = np.asarray([rewards[row, 1 - seat] for row, (_, seat) in enumerate(samples)])
        differences = own - other
        outcome[target_position] = np.where(differences > 0, 2, np.where(differences < 0, 0, 1))
        margin[target_position] = differences

    shape = (1, 1, len(args.targets), 1, len(args.seeds), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, outcome=outcome.reshape(shape), margin=margin.reshape(shape),
        states=states.reshape(1, 1, 1, len(args.seeds), 2, states.shape[-1]),
        openings=np.asarray([args.opening]), targets=np.asarray(args.targets),
        opponents=np.asarray(["CURRENT_STRONGEST"]),
        checkpoints=np.asarray([args.checkpoint], dtype=np.int16),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        engine=np.asarray("C++ late selector after dynamic opponent switch"),
    )
    raw = outcome == 2
    print(json.dumps({
        "output": str(args.output.resolve()), "checkpoint": args.checkpoint,
        "games": int(outcome.size),
        "route_oracle_raw_win_rate": float(np.mean(np.max(raw, axis=0))),
        "best_static_raw_win_rate": float(np.max(np.mean(raw, axis=1))),
    }, indent=2))


if __name__ == "__main__":
    main()

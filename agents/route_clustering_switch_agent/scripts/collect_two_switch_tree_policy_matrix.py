#!/usr/bin/env python3
"""Collect outcomes for a real 48->96 two-switch public route policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle
from collect_tree_policy_selector_matrix import _policy_nodes
from meta_agent.src.teammate_expanded_routes import load_action_tapes


KEEP = "__KEEP__"


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
    parser.add_argument("--extra-actions", type=Path, action="append", default=[])
    parser.add_argument("--opening", default="NR295")
    parser.add_argument("--early-targets", type=_csv, required=True)
    parser.add_argument("--late-targets", type=_csv, required=True)
    parser.add_argument("--early-checkpoint", type=int, default=48)
    parser.add_argument("--late-checkpoint", type=int, default=96)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.early_checkpoint >= args.late_checkpoint:
        raise ValueError("early checkpoint must precede late checkpoint")

    metadata = json.loads(args.candidate_parent_metadata.read_text(encoding="utf-8"))
    parent_row = next(
        row for row in metadata["opponent_routes"]
        if str(row["family"]) == args.opening
    )
    parent_actions = load_action_tapes(args.candidate_parent_actions)
    shared = load_action_tapes(args.shared_actions)
    for path in args.extra_actions:
        shared.update(load_action_tapes(path))
    tapes = {args.opening: parent_actions[str(parent_row["route_id"])]}
    for family in {*args.early_targets, *args.late_targets} - {args.opening}:
        tapes[family] = shared[family]
    opening_tape = tapes[args.opening]
    for family in args.early_targets:
        if tapes[family][:args.early_checkpoint] != opening_tape[:args.early_checkpoint]:
            raise ValueError(f"early route {family} violates the public prefix")
    for family in args.late_targets:
        if tapes[family][:args.late_checkpoint] != opening_tape[:args.late_checkpoint]:
            raise ValueError(f"late route {family} violates the public prefix")

    policy = json.loads(args.opponent_policy.read_text(encoding="utf-8"))
    nodes = _policy_nodes(policy, args.opponent_opening)
    opponent_families = {args.opponent_opening}
    if "targets" in policy:
        opponent_families.update(str(value) for value in policy["targets"])
    else:
        opponent_families.update(
            family for _, predictor in nodes for family in predictor.classes
        )
    base_families = {
        str(row["family"]) for row in json.loads(
            args.base_metadata.read_text(encoding="utf-8")
        )["opponent_routes"]
    }
    additional = {
        family: tape for family, tape in tapes.items()
        if family not in base_families
    }
    bundle = NativeTeammateBundle(
        args.source, args.base_actions, args.base_metadata,
        additional_routes=additional, included_families=sorted(opponent_families),
    )
    opening_index = bundle.index(args.opening)
    opponent_index = bundle.index(args.opponent_opening)
    early_indices = np.asarray([bundle.index(value) for value in args.early_targets])
    late_families = (KEEP, *args.late_targets)
    late_indices = np.asarray([
        -1 if value == KEEP else bundle.index(value) for value in late_families
    ])
    samples = [(seed, seat) for seed in args.seeds for seat in (0, 1)]

    def routes(candidate: int, seat: int) -> list[int]:
        values = [opponent_index, opponent_index]
        values[seat] = candidate
        return values

    early_tasks = np.empty((len(samples), 6), dtype=np.int64)
    for row, (seed, seat) in enumerate(samples):
        early_tasks[row] = [
            *routes(opening_index, seat), seed, args.early_checkpoint,
            seat, opening_index,
        ]
    early_states = np.asarray(bundle.executor.features_batch(early_tasks), dtype=np.float32)
    late_states = np.empty(
        (len(early_indices), len(samples), early_states.shape[-1]), dtype=np.float32
    )
    outcome = np.empty(
        (len(early_indices), len(late_indices), len(samples)), dtype=np.uint8
    )
    margin = np.empty_like(outcome, dtype=np.float32)

    for early_position, early_index in enumerate(early_indices):
        state_tasks = np.empty((len(samples), 6), dtype=np.int64)
        for row, (seed, seat) in enumerate(samples):
            state_tasks[row] = [
                *routes(int(early_index), seat), seed, args.late_checkpoint,
                seat, int(early_index),
            ]
        late_states[early_position] = bundle.executor.features_batch(state_tasks)

        for late_position, late_index in enumerate(late_indices):
            candidate_step = -1 if late_index < 0 else args.late_checkpoint
            candidate_target = int(early_index if late_index < 0 else late_index)
            opponent_steps = np.full(len(samples), -1, dtype=np.int64)
            opponent_targets = np.full(len(samples), opponent_index, dtype=np.int64)
            for node_checkpoint, tree in sorted(nodes):
                tasks = np.empty((len(samples), 10), dtype=np.int64)
                for row, (seed, seat) in enumerate(samples):
                    switches = [-1, -1, -1, -1]
                    switches[2 * seat:2 * seat + 2] = [
                        candidate_step, candidate_target,
                    ]
                    tasks[row] = [
                        *routes(int(early_index), seat), seed, node_checkpoint,
                        1 - seat, opponent_index, *switches,
                    ]
                features = np.asarray(bundle.executor.features_with_switch_batch(tasks))
                for row, vector in enumerate(features):
                    if opponent_steps[row] >= 0:
                        continue
                    prediction = tree.predict(vector)
                    if prediction != args.opponent_opening:
                        opponent_steps[row] = node_checkpoint
                        opponent_targets[row] = bundle.index(prediction)

            games = np.empty((len(samples), 7), dtype=np.int64)
            for row, (seed, seat) in enumerate(samples):
                switches = [opponent_steps[row], opponent_targets[row]] * 2
                switches[2 * seat:2 * seat + 2] = [
                    candidate_step, candidate_target,
                ]
                games[row] = [*routes(int(early_index), seat), seed, *switches]
            rewards = np.asarray(bundle.executor.play_batch(games), dtype=np.float64)
            own = np.asarray([
                rewards[row, seat] for row, (_, seat) in enumerate(samples)
            ])
            other = np.asarray([
                rewards[row, 1 - seat] for row, (_, seat) in enumerate(samples)
            ])
            difference = own - other
            outcome[early_position, late_position] = np.where(
                difference > 0, 2, np.where(difference < 0, 0, 1)
            )
            margin[early_position, late_position] = difference

    shape = (len(args.seeds), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        early_states=early_states.reshape(*shape, early_states.shape[-1]),
        late_states=late_states.reshape(
            len(early_indices), *shape, late_states.shape[-1]
        ),
        outcome=outcome.reshape(len(early_indices), len(late_indices), *shape),
        margin=margin.reshape(len(early_indices), len(late_indices), *shape),
        opening=np.asarray(args.opening),
        early_targets=np.asarray(args.early_targets),
        late_targets=np.asarray(late_families),
        checkpoints=np.asarray(
            [args.early_checkpoint, args.late_checkpoint], dtype=np.int16
        ),
        opponents=np.asarray(["CURRENT_STRONGEST"]),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        feature_schema=np.asarray("semantic_route_switch_v1"),
        engine=np.asarray("C++ true two-switch route policy vs dynamic public-state policy"),
    )
    raw = outcome == 2
    print(json.dumps({
        "output": str(args.output.resolve()),
        "games": int(outcome.size),
        "early_targets": len(early_indices),
        "late_actions": len(late_indices),
        "hidden_pair_oracle_raw_win_rate": float(np.max(raw, axis=(0, 1)).mean()),
        "best_fixed_pair_raw_win_rate": float(raw.mean(axis=2).max()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

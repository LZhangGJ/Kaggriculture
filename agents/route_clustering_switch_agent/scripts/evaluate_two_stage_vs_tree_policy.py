#!/usr/bin/env python3
"""Evaluate the deployable two-stage selector against the prior tree agent."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _lookup(states, actions, targets) -> dict[bytes, str]:
    return {
        state.tobytes(): str(targets[int(action)])
        for state, action in zip(states.astype(np.float32), actions)
    }


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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.model, allow_pickle=False) as saved:
        opening = str(saved["opening"])
        early_checkpoint = int(saved["early_checkpoint"])
        late_checkpoint = int(saved["late_checkpoint"])
        early_lookup = _lookup(
            saved["early_states"], saved["early_actions"], saved["early_targets"].astype(str)
        )
        late_lookup = _lookup(
            saved["late_states"], saved["late_actions"], saved["late_targets"].astype(str)
        )
        used = {opening, *early_lookup.values(), *late_lookup.values()}
    parent_metadata = json.loads(args.candidate_parent_metadata.read_text(encoding="utf-8"))
    parent_row = next(row for row in parent_metadata["opponent_routes"] if row["family"] == opening)
    parent_actions = load_action_tapes(args.candidate_parent_actions)
    shared_actions = load_action_tapes(args.shared_actions)
    for path in args.extra_actions:
        shared_actions.update(load_action_tapes(path))
    additional = {opening: parent_actions[str(parent_row["route_id"])]}
    additional.update({family: shared_actions[family] for family in used if family != opening})
    opponent_policy = json.loads(args.opponent_policy.read_text(encoding="utf-8"))
    opponent_nodes = [
        (int(node["selected"]["checkpoint"]), NumpySearchTree(node["selected"]["tree"]))
        for node in opponent_policy["nodes"]
        if node["selected"].get("enabled", True)
        and str(node["selected"]["opening"]) == args.opponent_opening
    ]
    opponent_families = {
        args.opponent_opening,
        *(family for _, tree in opponent_nodes for family in tree.classes),
    }
    bundle = NativeTeammateBundle(
        args.source, args.base_actions, args.base_metadata,
        additional_routes=additional,
        included_families=sorted(opponent_families),
    )
    candidate_index = bundle.index(opening)
    opponent_index = bundle.index(args.opponent_opening)
    samples = [(seed, seat) for seed in args.seeds for seat in (0, 1)]

    def feature_tasks(checkpoint: int, player_kind: str, switches=None) -> np.ndarray:
        width = 10 if switches is not None else 6
        tasks = np.empty((len(samples), width), dtype=np.int64)
        for row, (seed, candidate_seat) in enumerate(samples):
            routes = [opponent_index, opponent_index]
            routes[candidate_seat] = candidate_index
            player = candidate_seat if player_kind == "candidate" else 1 - candidate_seat
            feature_route = candidate_index if player_kind == "candidate" else opponent_index
            values = [*routes, seed, checkpoint, player, feature_route]
            if switches is not None:
                candidate_steps, candidate_targets = switches
                switch_values = [-1, -1, -1, -1]
                switch_values[2 * candidate_seat] = int(candidate_steps[row])
                switch_values[2 * candidate_seat + 1] = int(candidate_targets[row])
                values.extend(switch_values)
            tasks[row] = values
        return tasks

    early_features = np.asarray(bundle.executor.features_batch(
        feature_tasks(early_checkpoint, "candidate")
    ))
    early_families = np.asarray([
        early_lookup.get(row.tobytes(), opening) for row in early_features
    ], dtype=object)
    deferred = early_families == opening
    late_features = np.asarray(bundle.executor.features_batch(
        feature_tasks(late_checkpoint, "candidate")
    ))
    late_families = np.asarray([
        late_lookup.get(row.tobytes(), opening) for row in late_features
    ], dtype=object)
    final_families = early_families.copy()
    final_families[deferred] = late_families[deferred]
    candidate_steps = np.where(
        ~deferred, early_checkpoint,
        np.where(final_families != opening, late_checkpoint, -1),
    ).astype(np.int64)
    candidate_targets = np.asarray([
        bundle.index(family) if family != opening else candidate_index
        for family in final_families
    ], dtype=np.int64)

    opponent_steps = np.full(len(samples), -1, dtype=np.int64)
    opponent_targets = np.full(len(samples), opponent_index, dtype=np.int64)
    for checkpoint, tree in sorted(opponent_nodes):
        features = np.asarray(bundle.executor.features_with_switch_batch(
            feature_tasks(checkpoint, "opponent", (candidate_steps, candidate_targets))
        ))
        for row, vector in enumerate(features):
            if opponent_steps[row] >= 0:
                continue
            prediction = tree.predict(vector)
            if prediction != args.opponent_opening:
                opponent_steps[row] = checkpoint
                opponent_targets[row] = bundle.index(prediction)

    tasks = np.empty((len(samples), 7), dtype=np.int64)
    for row, (seed, candidate_seat) in enumerate(samples):
        routes = [opponent_index, opponent_index]
        routes[candidate_seat] = candidate_index
        switch_values = [opponent_steps[row], opponent_targets[row]] * 2
        switch_values[2 * candidate_seat:2 * candidate_seat + 2] = [
            candidate_steps[row], candidate_targets[row]
        ]
        tasks[row] = [*routes, seed, *switch_values]
    rewards, audit = bundle.executor.play_audit_batch(tasks)
    rewards = np.asarray(rewards, dtype=np.float64)
    own = np.asarray([rewards[row, seat] for row, (_, seat) in enumerate(samples)])
    other = np.asarray([rewards[row, 1 - seat] for row, (_, seat) in enumerate(samples)])
    margins = own - other
    scores = (margins > 0).astype(float) + .5 * (margins == 0)
    paired_scores = scores.reshape(len(args.seeds), 2)
    completed = np.isfinite(rewards).all(axis=1)
    trace_lengths = []
    for row in np.linspace(0, len(tasks) - 1, min(8, len(tasks)), dtype=int):
        trace_lengths.append(len(bundle.executor.play(*tasks[row], True)["trace"]))
    payload = {
        "schema": "two-stage-exact-agent-vs-prior-tree-agent-v1",
        "engine": "C++ NativeTeammateExecutor; both selectors use public state",
        "candidate_opening": opening,
        "opponent_opening": args.opponent_opening,
        "seeds": list(args.seeds),
        "games": len(samples),
        "raw_win_rate": float(np.mean(scores)),
        "seat_win_rates": [float(np.mean(scores[0::2])), float(np.mean(scores[1::2]))],
        "both_seats_win_rate": float(np.mean(np.all(paired_scores == 1.0, axis=1))),
        "mean_margin": float(np.mean(margins)),
        "minimum_margin": float(np.min(margins)),
        "completion_rate": float(np.mean(completed)),
        "completed_games": int(np.sum(completed)),
        "trace_length_checks": trace_lengths,
        "macro_unit_failure_rate": float(np.mean(np.asarray(audit)[:, :, 0] > 0)),
        "macro_market_failure_rate": float(np.mean(np.asarray(audit)[:, :, 1] > 0)),
        "candidate_target_counts": dict(Counter(final_families.tolist())),
        "opponent_target_counts": dict(Counter(
            bundle.families[int(index)] for index in opponent_targets
        )),
        "early_seen_state_rate": float(np.mean([
            row.tobytes() in early_lookup for row in early_features
        ])),
        "late_seen_state_rate_on_deferred": float(np.mean([
            row.tobytes() in late_lookup for row in late_features[deferred]
        ])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

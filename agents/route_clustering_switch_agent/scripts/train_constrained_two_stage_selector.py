#!/usr/bin/env python3
"""Tune exact-state actions for explicit pool and priority-opponent floors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_compact_route_q_selector import load_node


def _state_ids(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.unique(features, axis=0, return_inverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-search", type=Path, required=True)
    parser.add_argument("--late-search", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--priority-opponent", required=True)
    parser.add_argument("--pool-floor", type=float, default=.82)
    parser.add_argument("--overall-floor", type=float, default=.92)
    parser.add_argument("--priority-target", type=float, default=.88)
    parser.add_argument("--max-changes", type=int, default=100)
    parser.add_argument("--initialize-priority-optimum", action="store_true")
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    early = load_node(args.early_search, 48)
    late = load_node(args.late_search, 96)
    early_states, early_ids = _state_ids(early["features"])
    late_states, late_ids = _state_ids(late["features"])
    if not np.array_equal(early["opponents"], late["opponents"]):
        raise ValueError("opponents are not aligned")
    priority_positions = np.flatnonzero(early["opponents"] == args.priority_opponent)
    if len(priority_positions) != 1:
        raise ValueError("priority opponent is absent or duplicated")
    priority_position = int(priority_positions[0])
    opponent_ids = np.broadcast_to(
        np.arange(len(early["opponents"]))[:, None, None], early["sample_shape"]
    ).reshape(-1)
    priority_rows = np.flatnonzero(opponent_ids == priority_position)
    early_opening = int(np.flatnonzero(early["targets"] == early["opening"])[0])

    with np.load(args.initial_model, allow_pickle=False) as saved:
        early_lookup = {
            state.tobytes(): str(saved["early_targets"][int(action)])
            for state, action in zip(saved["early_states"], saved["early_actions"])
        }
        late_lookup = {
            state.tobytes(): str(saved["late_targets"][int(action)])
            for state, action in zip(saved["late_states"], saved["late_actions"])
        }
    early_target_index = {family: i for i, family in enumerate(early["targets"])}
    late_target_index = {family: i for i, family in enumerate(late["targets"])}
    early_actions = np.asarray([
        early_target_index[early_lookup.get(state.tobytes(), early["opening"])]
        for state in early_states
    ], dtype=np.int32)
    late_actions = np.asarray([
        late_target_index[late_lookup.get(state.tobytes(), late["opening"])]
        for state in late_states
    ], dtype=np.int32)
    early_candidate_states = np.unique(early_ids[priority_rows])
    late_candidate_states = np.unique(late_ids[priority_rows])

    if args.initialize_priority_optimum:
        for state in late_candidate_states:
            rows = priority_rows[late_ids[priority_rows] == state]
            late_actions[state] = int(np.argmax(np.mean(late["scores"][rows], axis=0)))
        downstream = late["scores"][
            np.arange(len(late_ids)), late_actions[late_ids]
        ]
        effective_early = early["scores"].copy()
        effective_early[:, early_opening] = downstream
        for state in early_candidate_states:
            rows = priority_rows[early_ids[priority_rows] == state]
            early_actions[state] = int(np.argmax(np.mean(effective_early[rows], axis=0)))

    def evaluate() -> tuple[tuple[float, ...], dict]:
        early_choice = early_actions[early_ids]
        deferred = early_choice == early_opening
        chosen = np.where(
            deferred,
            late["scores"][np.arange(len(deferred)), late_actions[late_ids]],
            early["scores"][np.arange(len(deferred)), early_choice],
        )
        wins = chosen == 1.0
        counts = np.bincount(opponent_ids, weights=wins, minlength=len(early["opponents"]))
        rates = counts / (early["sample_shape"][1] * 2)
        pool_rates = np.delete(rates, priority_position)
        priority_rate = float(rates[priority_position])
        pool_min = float(np.min(pool_rates))
        overall = float(np.mean(pool_rates))
        penalty = (
            max(0.0, args.priority_target - priority_rate)
            + max(0.0, args.pool_floor - pool_min)
            + max(0.0, args.overall_floor - overall)
        )
        metrics = {
            "priority_raw_win_rate": priority_rate,
            "pool_minimum_raw_win_rate": pool_min,
            "pool_overall_raw_win_rate": overall,
            "constraint_penalty": penalty,
        }
        return (-penalty, priority_rate, pool_min, overall), metrics

    objective, initial = evaluate()
    changes = []
    for _ in range(args.max_changes):
        best = objective
        best_change = None
        for stage, states, actions, target_count in (
            ("early", early_candidate_states, early_actions, len(early["targets"])),
            ("late", late_candidate_states, late_actions, len(late["targets"])),
        ):
            for state in states:
                old = int(actions[state])
                for action in range(target_count):
                    if action == old:
                        continue
                    actions[state] = action
                    candidate, metrics = evaluate()
                    actions[state] = old
                    if candidate > best:
                        best = candidate
                        best_change = (stage, int(state), action, metrics)
        if best_change is None:
            break
        stage, state, action, metrics = best_change
        actions = early_actions if stage == "early" else late_actions
        old = int(actions[state])
        actions[state] = action
        objective = best
        targets = early["targets"] if stage == "early" else late["targets"]
        changes.append({
            "stage": stage,
            "state": state,
            "from": str(targets[old]),
            "to": str(targets[action]),
            **metrics,
        })
    _, final = evaluate()

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_model,
        early_states=early_states.astype(np.float32), early_actions=early_actions,
        early_targets=early["targets"], early_checkpoint=np.asarray(48),
        late_states=late_states.astype(np.float32), late_actions=late_actions,
        late_targets=late["targets"], late_checkpoint=np.asarray(96),
        opening=np.asarray(early["opening"]),
    )
    report = {
        "schema": "constrained-two-stage-exact-selector-v1",
        "priority_opponent": args.priority_opponent,
        "constraints": {
            "pool_floor": args.pool_floor,
            "overall_floor": args.overall_floor,
            "priority_target": args.priority_target,
        },
        "initialize_priority_optimum": bool(args.initialize_priority_optimum),
        "initial": initial,
        "final": final,
        "change_count": len(changes),
        "changes": changes,
        "used_targets": sorted({
            *early["targets"][early_actions].tolist(),
            *late["targets"][late_actions].tolist(),
        }),
        "model_bytes": args.output_model.stat().st_size,
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

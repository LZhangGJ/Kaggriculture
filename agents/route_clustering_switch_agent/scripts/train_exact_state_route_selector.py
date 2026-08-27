#!/usr/bin/env python3
"""Train a safe exact-public-state route selector with opening fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from train_compact_route_q_selector import load_node, policy_metrics


def fit_state_actions(
    state_ids: np.ndarray,
    scores: np.ndarray,
    margins: np.ndarray,
    indices: np.ndarray,
    state_count: int,
    fallback: int,
    margin_weight: float = 0.0,
    opponent_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Choose a state action, maximin over indistinguishable opponents."""

    utility = scores + margin_weight * np.tanh(margins / 10000.0)
    actions = np.full(state_count, fallback, dtype=np.int32)
    for state in np.unique(state_ids[indices]):
        rows = indices[state_ids[indices] == state]
        if opponent_ids is None:
            actions[state] = int(np.argmax(np.mean(utility[rows], axis=0)))
            continue
        per_opponent = np.stack([
            np.mean(utility[rows[opponent_ids[rows] == opponent]], axis=0)
            for opponent in np.unique(opponent_ids[rows])
        ])
        worst = np.min(per_opponent, axis=0)
        finalists = np.flatnonzero(worst == np.max(worst))
        actions[state] = int(finalists[np.argmax(
            np.mean(per_opponent[:, finalists], axis=0)
        )])
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--margin-weights", default="0,0.02")
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    node = load_node(args.search, args.checkpoint)
    states, state_ids = np.unique(
        node["features"], axis=0, return_inverse=True
    )
    opening = int(np.flatnonzero(node["targets"] == node["opening"])[0])
    opponent_ids = np.broadcast_to(
        np.arange(len(node["opponents"]))[:, None, None], node["sample_shape"]
    ).reshape(-1)
    splits = list(GroupKFold(args.folds).split(
        node["features"], groups=node["groups"]
    ))
    trials = []
    weights = tuple(float(value) for value in args.margin_weights.split(",") if value)
    for margin_weight in weights:
        predictions = np.full(len(state_ids), opening, dtype=np.int32)
        for train, valid in splits:
            actions = fit_state_actions(
                state_ids, node["scores"], node["margins"], train,
                len(states), opening, margin_weight, opponent_ids,
            )
            predictions[valid] = actions[state_ids[valid]]
        trials.append({
            "margin_weight": margin_weight,
            "cross_validation": policy_metrics(
                node["scores"], node["margins"], predictions, node["sample_shape"]
            ),
        })
    trials.sort(key=lambda row: (
        -row["cross_validation"]["minimum_opponent_raw_win_rate"],
        -row["cross_validation"]["paired_seed_raw_win_lower_95pct"],
        -row["cross_validation"]["raw_win_rate"],
        row["margin_weight"],
    ))
    selected = trials[0]
    all_indices = np.arange(len(state_ids))
    actions = fit_state_actions(
        state_ids, node["scores"], node["margins"], all_indices,
        len(states), opening, float(selected["margin_weight"]), opponent_ids,
    )
    predictions = actions[state_ids]
    training = policy_metrics(
        node["scores"], node["margins"], predictions, node["sample_shape"]
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_model,
        states=states.astype(np.float32),
        actions=actions,
        targets=node["targets"],
        opening=np.asarray(node["opening"]),
        checkpoint=np.asarray(args.checkpoint, dtype=np.int16),
    )
    report: dict[str, Any] = {
        "schema": "exact-public-state-route-selector-v1",
        "source": str(args.search.resolve()),
        "checkpoint": args.checkpoint,
        "fallback": node["opening"],
        "selection_objective": "state_conditional_minimum_opponent_utility",
        "state_count": len(states),
        "used_targets": sorted(set(node["targets"][actions].tolist())),
        "selected": selected,
        "training": training,
        "trials": trials,
        "model_bytes": args.output_model.stat().st_size,
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

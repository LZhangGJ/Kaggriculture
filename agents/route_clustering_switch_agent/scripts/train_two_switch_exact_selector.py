#!/usr/bin/env python3
"""Train a true 48->96 exact-state policy from a two-switch matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from train_compact_route_q_selector import policy_metrics
from train_exact_state_route_selector import fit_state_actions


def _states(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.unique(features, axis=0, return_inverse=True)


def fit_policy(
    early_features: np.ndarray,
    late_features: np.ndarray,
    scores: np.ndarray,
    margins: np.ndarray,
    train: np.ndarray,
    opening_action: int,
    margin_weight: float,
) -> dict[str, Any]:
    """Fit late decisions per early route, then the early decision."""

    early_states, early_ids = _states(early_features)
    late_tables = []
    effective_scores = np.empty((len(early_features), len(late_features)), dtype=np.float32)
    effective_margins = np.empty_like(effective_scores)
    rows = np.arange(len(early_features))
    for early in range(len(late_features)):
        states, state_ids = _states(late_features[early])
        actions = fit_state_actions(
            state_ids, scores[early].T, margins[early].T, train,
            len(states), 0, margin_weight,
        )
        predictions = actions[state_ids]
        effective_scores[:, early] = scores[early, predictions, rows]
        effective_margins[:, early] = margins[early, predictions, rows]
        late_tables.append((states, actions, state_ids))
    early_actions = fit_state_actions(
        early_ids, effective_scores, effective_margins, train,
        len(early_states), opening_action, margin_weight,
    )
    return {
        "early_states": early_states,
        "early_ids": early_ids,
        "early_actions": early_actions,
        "late_tables": late_tables,
        "effective_scores": effective_scores,
        "effective_margins": effective_margins,
    }


def selected_values(policy: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(policy["early_ids"]))
    early = policy["early_actions"][policy["early_ids"]]
    return (
        policy["effective_scores"][rows, early],
        policy["effective_margins"][rows, early],
    )


def _metrics(scores: np.ndarray, margins: np.ndarray, seeds: int) -> dict[str, Any]:
    return policy_metrics(
        scores[:, None], margins[:, None], np.zeros(len(scores), dtype=np.int32),
        (1, seeds, 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--margin-weights", default="0,0.02")
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.matrix, allow_pickle=False) as saved:
        early_features = saved["early_states"].reshape(-1, saved["early_states"].shape[-1])
        late_features = saved["late_states"].reshape(
            len(saved["early_targets"]), -1, saved["late_states"].shape[-1]
        )
        outcome = saved["outcome"].reshape(
            len(saved["early_targets"]), len(saved["late_targets"]), -1
        )
        margins = saved["margin"].reshape(outcome.shape).astype(np.float32)
        scores = outcome.astype(np.float32) * 0.5
        opening = str(saved["opening"])
        early_targets = saved["early_targets"].astype(str)
        late_targets = saved["late_targets"].astype(str)
        checkpoints = saved["checkpoints"].astype(np.int16)
        seeds = saved["seeds"].astype(np.int64)
    if late_targets[0] != "__KEEP__":
        raise ValueError("late action zero must be __KEEP__")
    opening_action = int(np.flatnonzero(early_targets == opening)[0])
    groups = np.repeat(seeds, 2)
    splits = list(GroupKFold(args.folds).split(early_features, groups=groups))
    trials = []
    for margin_weight in (
        float(value) for value in args.margin_weights.split(",") if value
    ):
        cv_scores = np.empty(len(early_features), dtype=np.float32)
        cv_margins = np.empty_like(cv_scores)
        for train, valid in splits:
            policy = fit_policy(
                early_features, late_features, scores, margins, train,
                opening_action, margin_weight,
            )
            selected_scores, selected_margins = selected_values(policy)
            cv_scores[valid] = selected_scores[valid]
            cv_margins[valid] = selected_margins[valid]
        trials.append({
            "margin_weight": margin_weight,
            "cross_validation": _metrics(cv_scores, cv_margins, len(seeds)),
        })
    trials.sort(key=lambda row: (
        -row["cross_validation"]["paired_seed_raw_win_lower_95pct"],
        -row["cross_validation"]["raw_win_rate"], row["margin_weight"],
    ))
    selected = trials[0]
    all_rows = np.arange(len(early_features))
    policy = fit_policy(
        early_features, late_features, scores, margins, all_rows,
        opening_action, float(selected["margin_weight"]),
    )
    training_scores, training_margins = selected_values(policy)
    training = _metrics(training_scores, training_margins, len(seeds))

    late_offsets = [0]
    late_states = []
    late_actions = []
    for states, actions, _ in policy["late_tables"]:
        late_states.append(states.astype(np.float32))
        late_actions.append(actions.astype(np.int32))
        late_offsets.append(late_offsets[-1] + len(states))
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_model,
        opening=np.asarray(opening), checkpoints=checkpoints,
        early_states=policy["early_states"].astype(np.float32),
        early_actions=policy["early_actions"].astype(np.int32),
        early_targets=early_targets,
        late_states=np.concatenate(late_states),
        late_actions=np.concatenate(late_actions),
        late_offsets=np.asarray(late_offsets, dtype=np.int32),
        late_targets=late_targets,
    )
    raw = scores == 1.0
    report = {
        "schema": "true-two-switch-exact-public-state-selector-v1",
        "source": str(args.matrix.resolve()),
        "checkpoints": checkpoints.astype(int).tolist(),
        "opening": opening,
        "fallbacks": {"early": opening, "late": "__KEEP__"},
        "early_targets": early_targets.tolist(),
        "late_targets": late_targets.tolist(),
        "state_counts": {
            "early": len(policy["early_states"]),
            "late_by_early": [len(value[0]) for value in policy["late_tables"]],
        },
        "hidden_pair_oracle_raw_win_rate": float(np.max(raw, axis=(0, 1)).mean()),
        "best_fixed_pair_raw_win_rate": float(raw.mean(axis=2).max()),
        "selected": selected,
        "training": training,
        "used_early_targets": sorted(set(
            early_targets[policy["early_actions"]].tolist()
        )),
        "used_late_actions": sorted(set(
            late_targets[np.concatenate(late_actions)].tolist()
        )),
        "model_bytes": args.output_model.stat().st_size,
        "trials": trials,
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

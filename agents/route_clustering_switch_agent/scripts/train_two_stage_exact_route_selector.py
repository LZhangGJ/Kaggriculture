#!/usr/bin/env python3
"""Train a two-checkpoint exact-state selector with a safe defer action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from train_compact_route_q_selector import load_node, policy_metrics, selected_values
from train_exact_state_route_selector import fit_state_actions


def _state_ids(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.unique(features, axis=0, return_inverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-search", type=Path, required=True)
    parser.add_argument("--late-search", type=Path, required=True)
    parser.add_argument("--early-checkpoint", type=int, required=True)
    parser.add_argument("--late-checkpoint", type=int, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--margin-weight", type=float, default=0.02)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    early = load_node(args.early_search, args.early_checkpoint)
    late = load_node(args.late_search, args.late_checkpoint)
    for key in ("opening", "sample_shape"):
        if early[key] != late[key]:
            raise ValueError(f"unaligned selector matrices: {key}")
    for key in ("opponents", "seeds", "groups"):
        if not np.array_equal(early[key], late[key]):
            raise ValueError(f"unaligned selector matrices: {key}")
    early_states, early_ids = _state_ids(early["features"])
    late_states, late_ids = _state_ids(late["features"])
    early_opening = int(np.flatnonzero(early["targets"] == early["opening"])[0])
    late_opening = int(np.flatnonzero(late["targets"] == late["opening"])[0])
    opponent_ids = np.broadcast_to(
        np.arange(len(early["opponents"]))[:, None, None], early["sample_shape"]
    ).reshape(-1)
    splits = list(GroupKFold(args.folds).split(
        early["features"], groups=early["groups"]
    ))
    predictions = np.full(len(early_ids), early_opening, dtype=np.int32)
    effective_scores = early["scores"].copy()
    effective_margins = early["margins"].copy()
    for train, valid in splits:
        late_actions = fit_state_actions(
            late_ids, late["scores"], late["margins"], train, len(late_states),
            late_opening, args.margin_weight, opponent_ids,
        )
        late_predictions = late_actions[late_ids]
        downstream_scores = selected_values(late["scores"], late_predictions)
        downstream_margins = selected_values(late["margins"], late_predictions)
        fold_scores = early["scores"].copy()
        fold_margins = early["margins"].copy()
        fold_scores[:, early_opening] = downstream_scores
        fold_margins[:, early_opening] = downstream_margins
        early_actions = fit_state_actions(
            early_ids, fold_scores, fold_margins, train, len(early_states),
            early_opening, args.margin_weight, opponent_ids,
        )
        predictions[valid] = early_actions[early_ids[valid]]
        effective_scores[valid, early_opening] = downstream_scores[valid]
        effective_margins[valid, early_opening] = downstream_margins[valid]
    cross_validation = policy_metrics(
        effective_scores, effective_margins, predictions, early["sample_shape"]
    )

    all_rows = np.arange(len(early_ids))
    late_actions = fit_state_actions(
        late_ids, late["scores"], late["margins"], all_rows, len(late_states),
        late_opening, args.margin_weight, opponent_ids,
    )
    late_predictions = late_actions[late_ids]
    full_scores = early["scores"].copy()
    full_margins = early["margins"].copy()
    full_scores[:, early_opening] = selected_values(late["scores"], late_predictions)
    full_margins[:, early_opening] = selected_values(late["margins"], late_predictions)
    early_actions = fit_state_actions(
        early_ids, full_scores, full_margins, all_rows, len(early_states),
        early_opening, args.margin_weight, opponent_ids,
    )
    training = policy_metrics(
        full_scores, full_margins, early_actions[early_ids], early["sample_shape"]
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_model,
        early_states=early_states.astype(np.float32), early_actions=early_actions,
        early_targets=early["targets"], early_checkpoint=np.asarray(args.early_checkpoint),
        late_states=late_states.astype(np.float32), late_actions=late_actions,
        late_targets=late["targets"], late_checkpoint=np.asarray(args.late_checkpoint),
        opening=np.asarray(early["opening"]),
    )
    report = {
        "schema": "two-stage-exact-public-state-route-selector-v1",
        "early_search": str(args.early_search.resolve()),
        "late_search": str(args.late_search.resolve()),
        "opening": early["opening"],
        "defer_action": early["opening"],
        "checkpoints": [args.early_checkpoint, args.late_checkpoint],
        "state_counts": [len(early_states), len(late_states)],
        "margin_weight": args.margin_weight,
        "cross_validation": cross_validation,
        "training": training,
        "early_used_targets": sorted(set(early["targets"][early_actions].tolist())),
        "late_used_targets": sorted(set(late["targets"][late_actions].tolist())),
        "model_bytes": args.output_model.stat().st_size,
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a frozen Candidate8 two-stage selector on a wholly new pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from train_candidate8_day6_safe_uplift import (
    correlation,
    make_features,
    paired_mean_margin_target,
    paired_targets,
)
from train_candidate8_two_stage_uplift import auc, passes, select, summarize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def state_groups(
    state_id: np.ndarray,
    values: np.ndarray,
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for state in np.unique(state_id):
        index = int(np.flatnonzero(state_id == state)[0])
        result.setdefault(str(values[index]), set()).add(int(state))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--z", type=float, default=1.0)
    args = parser.parse_args()

    model = joblib.load(args.model)
    training = json.loads(args.training_receipt.read_text(encoding="utf-8"))
    thresholds = model["thresholds"]
    with np.load(args.dataset, allow_pickle=False) as data:
        required = {
            "state_change_count_full", "action_change_count_full",
            "future_own_cash", "future_opponent_cash", "features",
            "feature_names", "family", "seat", "state_id", "opponent",
            "prefix_seed", "decision_day", "signature",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"blind dataset lacks fields: {sorted(missing)}")
        state_id = np.asarray(data["state_id"])
        opponents = np.asarray(data["opponent"])
        prefix_seeds = np.asarray(data["prefix_seed"])
        decision_day = np.asarray(data["decision_day"])
        seat = np.asarray(data["seat"])
        family = np.asarray(data["family"])
        signatures = np.asarray(data["signature"])
        raw_features = np.asarray(data["features"])
        feature_names = [str(value) for value in data["feature_names"]]
        own = np.asarray(data["future_own_cash"], dtype=np.float64)
        rival = np.asarray(data["future_opponent_cash"], dtype=np.float64)
        state_effect = np.asarray(data["state_change_count_full"]) > 0
        action_effect = np.asarray(data["action_change_count_full"]) > 0

    x, model_feature_names = make_features(
        raw_features, family, seat, feature_names
    )
    if model_feature_names != list(model["feature_names"]):
        raise ValueError("blind feature contract differs from frozen model")

    score, margin_lcb, _ = paired_targets(state_id, own, rival, args.z)
    margin_mean = paired_mean_margin_target(state_id, own, rival)
    score_effect = np.abs(score) > 1e-12
    score_improve = score > 1e-12
    score_regress = score < -1e-12

    p_state = model["state_effect_model"].predict_proba(x)[:, 1]
    p_score_effect = model["score_effect_model"].predict_proba(x)[:, 1]
    p_score_improve = model["score_improve_model"].predict_proba(x)[:, 1]
    p_score_regress = model["score_regress_model"].predict_proba(x)[:, 1]
    predicted_score = model["score_model"].predict(x)
    predicted_margin_mean = model["margin_mean_model"].predict(x)
    predicted_margin_lcb = model["margin_lcb_model"].predict(x)

    all_states = {int(value) for value in np.unique(state_id)}
    non_keep = np.ones(len(state_id), dtype=bool)
    for state in all_states:
        non_keep[np.flatnonzero(state_id == state)[0]] = False

    def evaluate(states: set[int]) -> tuple[dict, list[dict]]:
        rows = select(
            states, state_id, p_state, p_score_effect,
            p_score_improve, p_score_regress, predicted_score,
            predicted_margin_mean, predicted_margin_lcb,
            state_effect, action_effect, score, margin_mean, margin_lcb,
            thresholds["state_effect_threshold"],
            thresholds["score_effect_threshold"],
            thresholds["score_improve_threshold"],
            thresholds["score_regress_ceiling"],
            thresholds["score_edge_threshold"],
            thresholds["margin_lcb_threshold"],
        )
        return summarize(rows), rows

    overall, selected_rows = evaluate(all_states)
    grouped = {}
    for label, values in (
        ("opponent", opponents),
        ("seat", seat),
        ("decision_day", decision_day),
    ):
        grouped[label] = {
            key: evaluate(states)[0]
            for key, states in state_groups(state_id, values).items()
        }

    selected_enriched = []
    for row in selected_rows:
        state = row["state_id"]
        indices = np.flatnonzero(state_id == state)
        index = int(indices[row["selected_local_rank"]])
        item = dict(row)
        item.update({
            "opponent": str(opponents[index]),
            "prefix_seed": int(prefix_seeds[index]),
            "seat": int(seat[index]),
            "decision_day": int(decision_day[index]),
            "family": str(family[index]),
            "signature": int(signatures[index]),
        })
        selected_enriched.append(item)

    indices = np.flatnonzero(non_keep)
    diagnostics = {
        "rows": int(len(indices)),
        "state_effect_auc": auc(state_effect[indices], p_state[indices]),
        "score_effect_auc": auc(score_effect[indices], p_score_effect[indices]),
        "score_improve_auc": auc(score_improve[indices], p_score_improve[indices]),
        "score_regress_auc": auc(score_regress[indices], p_score_regress[indices]),
        "score_correlation": correlation(score[indices], predicted_score[indices]),
        "margin_mean_correlation": correlation(
            margin_mean[indices], predicted_margin_mean[indices]
        ),
        "margin_lcb_correlation": correlation(
            margin_lcb[indices], predicted_margin_lcb[indices]
        ),
    }
    blind_safety_passed = passes(overall)
    training_calibration_passed = bool(
        training["selected_thresholds"].get("passed", False)
    )
    deployment_gate = training_calibration_passed and blind_safety_passed
    payload = {
        "schema": "kaggriculture.candidate8-two-stage-blind.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Frozen model and thresholds; all blind futures are used only for "
            "this final audit. No opponent name, prefix seed, future seed or "
            "private opponent state is present in model features."
        ),
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "training_receipt": {
            "path": str(args.training_receipt),
            "sha256": sha256(args.training_receipt),
        },
        "coverage": {
            "states": len(all_states),
            "rows": len(state_id),
            "future_count": int(own.shape[1]),
            "opponents": sorted(str(value) for value in np.unique(opponents)),
            "prefix_seeds": sorted(int(value) for value in np.unique(prefix_seeds)),
            "seats": sorted(int(value) for value in np.unique(seat)),
            "decision_days": sorted(int(value) for value in np.unique(decision_day)),
        },
        "label_support": {
            "state_effect_rate_non_keep": float(np.mean(state_effect[non_keep])),
            "action_effect_rate_non_keep": float(np.mean(action_effect[non_keep])),
            "score_effect_rate_non_keep": float(np.mean(score_effect[non_keep])),
            "score_improve_rate_non_keep": float(np.mean(score_improve[non_keep])),
            "score_regress_rate_non_keep": float(np.mean(score_regress[non_keep])),
        },
        "frozen_thresholds": thresholds,
        "prediction_diagnostics": diagnostics,
        "overall": overall,
        "grouped": grouped,
        "selected_rows": selected_enriched,
        "gates": {
            "training_calibration_passed": training_calibration_passed,
            "blind_safety_passed": blind_safety_passed,
            "deployment_gate_passed": deployment_gate,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "coverage": payload["coverage"],
        "label_support": payload["label_support"],
        "prediction_diagnostics": diagnostics,
        "overall": overall,
        "gates": payload["gates"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

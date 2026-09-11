#!/usr/bin/env python3
"""Strict blind evaluation for a frozen O1.3 stage-pairwise selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from train_candidate8_day6_safe_uplift import (
    make_features,
    paired_mean_margin_target,
    paired_targets,
)
from train_candidate8_o13_stage_pairwise import (
    STAGES,
    auc,
    passes,
    predict_candidate_vs_keep,
    select,
    summarize,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


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
    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        opponents = np.asarray(data["opponent"])
        prefix_seeds = np.asarray(data["prefix_seed"])
        day = np.asarray(data["decision_day"])
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
    x = np.asarray(x, dtype=np.float32)
    if model_feature_names != list(model["feature_names"]):
        raise ValueError("blind feature contract differs from frozen model")
    score, lcb, _ = paired_targets(state_id, own, rival, args.z)
    mean = paired_mean_margin_target(state_id, own, rival)
    p_state = model["state_effect_model"].predict_proba(x)[:, 1]

    non_keep = np.ones(len(state_id), dtype=bool)
    all_states = {int(value) for value in np.unique(state_id)}
    for state in all_states:
        non_keep[np.flatnonzero(state_id == state)[0]] = False

    stage_rows = {}
    stage_metrics = {}
    diagnostics = {}
    all_rows = []
    for stage, days in STAGES.items():
        indices = np.flatnonzero(np.isin(day, days))
        states = {
            int(value) for value in np.unique(state_id[indices])
        }
        stage_model = model["stage_models"][stage]
        p_pair = np.zeros(len(state_id), dtype=np.float64)
        p_positive = np.zeros(len(state_id), dtype=np.float64)
        p_negative = np.zeros(len(state_id), dtype=np.float64)
        p_score_up = np.zeros(len(state_id), dtype=np.float64)
        p_score_down = np.zeros(len(state_id), dtype=np.float64)
        p_pair[indices] = predict_candidate_vs_keep(
            stage_model["pair_model"], x, state_id, indices
        )
        p_positive[indices] = stage_model["positive_lcb_model"].predict_proba(
            x[indices]
        )[:, 1]
        p_negative[indices] = stage_model["negative_lcb_model"].predict_proba(
            x[indices]
        )[:, 1]
        p_score_up[indices] = stage_model["score_improve_model"].predict_proba(
            x[indices]
        )[:, 1]
        p_score_down[indices] = stage_model["score_regress_model"].predict_proba(
            x[indices]
        )[:, 1]
        rows = select(
            states, state_id, p_state, p_pair, p_positive, p_negative,
            p_score_up, p_score_down, state_effect, action_effect,
            score, mean, lcb,
            model["thresholds_by_stage"][stage]["thresholds"],
        )
        for row in rows:
            row["stage"] = stage
        stage_rows[stage] = rows
        stage_metrics[stage] = summarize(rows)
        all_rows.extend(rows)
        candidate_indices = indices[non_keep[indices]]
        pair_better = (
            (score[candidate_indices] > 1e-12)
            | (
                (np.abs(score[candidate_indices]) <= 1e-12)
                & (lcb[candidate_indices] > 0.0)
            )
        )
        diagnostics[stage] = {
            "rows": int(len(candidate_indices)),
            "state_effect_auc": auc(
                state_effect[candidate_indices], p_state[candidate_indices]
            ),
            "positive_lcb_auc": auc(
                lcb[candidate_indices] > 0.0, p_positive[candidate_indices]
            ),
            "negative_lcb_auc": auc(
                lcb[candidate_indices] < 0.0, p_negative[candidate_indices]
            ),
            "score_improve_auc": auc(
                score[candidate_indices] > 1e-12,
                p_score_up[candidate_indices],
            ),
            "score_regress_auc": auc(
                score[candidate_indices] < -1e-12,
                p_score_down[candidate_indices],
            ),
            "pair_better_auc": auc(pair_better, p_pair[candidate_indices]),
        }

    overall = summarize(all_rows)

    def grouped(values: np.ndarray) -> dict:
        result = {}
        for value in np.unique(values):
            states = {
                int(state) for state in all_states
                if str(values[int(np.flatnonzero(state_id == state)[0])])
                == str(value)
            }
            rows = [row for row in all_rows if row["state_id"] in states]
            result[str(value)] = summarize(rows)
        return result

    enriched = []
    for row in all_rows:
        indices = np.flatnonzero(state_id == row["state_id"])
        index = int(indices[row["selected_local_rank"]])
        item = dict(row)
        item.update({
            "opponent": str(opponents[index]),
            "prefix_seed": int(prefix_seeds[index]),
            "seat": int(seat[index]),
            "decision_day": int(day[index]),
            "family": int(family[index]),
            "signature": int(signatures[index]),
        })
        enriched.append(item)

    stage_blind_pass = {
        stage: passes(metrics) for stage, metrics in stage_metrics.items()
    }
    blind_pass = all(stage_blind_pass.values()) and passes(overall)
    training_pass = bool(training["gates"]["deployment_gate_passed"])
    payload = {
        "schema": "kaggriculture.candidate8-o13-stage-pairwise-blind.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Frozen model and per-stage thresholds. The blind set was not "
            "used for model, threshold or feature selection. Opponent names "
            "and seeds are grouping keys only."
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
            "decision_days": sorted(int(value) for value in np.unique(day)),
        },
        "prediction_diagnostics": diagnostics,
        "metrics": {
            "overall": overall,
            "by_stage": stage_metrics,
            "by_opponent": grouped(opponents),
            "by_seat": grouped(seat),
        },
        "selected_rows": enriched,
        "gates": {
            "training_and_internal_gate_passed": training_pass,
            "blind_stage_passed": stage_blind_pass,
            "blind_overall_passed": passes(overall),
            "blind_all_passed": blind_pass,
            "deployment_gate_passed": training_pass and blind_pass,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "coverage": payload["coverage"],
        "prediction_diagnostics": diagnostics,
        "metrics": payload["metrics"],
        "gates": payload["gates"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

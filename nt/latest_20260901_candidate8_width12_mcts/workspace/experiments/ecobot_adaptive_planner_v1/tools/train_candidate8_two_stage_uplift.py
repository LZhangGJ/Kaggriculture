#!/usr/bin/env python3
"""Train a behavior-effect gate plus Candidate8-vs-KEEP uplift selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from train_candidate8_day6_safe_uplift import (
    correlation,
    make_features,
    paired_mean_margin_target,
    paired_targets,
    parse_strings,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def auc(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(np.unique(target)) < 2:
        return 0.5
    return float(roc_auc_score(target, prediction))


def select(
    states: set[int],
    state_id: np.ndarray,
    predicted_state_effect: np.ndarray,
    predicted_score_effect: np.ndarray,
    predicted_score_improve: np.ndarray,
    predicted_score_regress: np.ndarray,
    predicted_score: np.ndarray,
    predicted_margin_mean: np.ndarray,
    predicted_margin_lcb: np.ndarray,
    actual_state_effect: np.ndarray,
    actual_action_effect: np.ndarray,
    score_target: np.ndarray,
    margin_mean_target: np.ndarray,
    margin_lcb_target: np.ndarray,
    state_threshold: float,
    score_effect_threshold: float,
    score_improve_threshold: float,
    score_regress_ceiling: float,
    score_edge_threshold: float,
    margin_threshold: float,
) -> list[dict]:
    rows = []
    for state in sorted(states):
        indices = np.flatnonzero(state_id == state)
        keep = int(indices[0])
        eligible = [
            int(index) for index in indices[1:]
            if predicted_state_effect[index] >= state_threshold
            and predicted_score_effect[index] >= score_effect_threshold
            and predicted_score_improve[index] >= score_improve_threshold
            and predicted_score_regress[index] <= score_regress_ceiling
            and (
                predicted_score_improve[index]
                - predicted_score_regress[index]
            ) >= score_edge_threshold
            and predicted_margin_lcb[index] >= margin_threshold
        ]
        chosen = keep if not eligible else max(
            eligible,
            key=lambda index: (
                float(
                    predicted_score_improve[index]
                    - predicted_score_regress[index]
                ),
                float(predicted_score[index]),
                float(predicted_score_effect[index]),
                float(predicted_margin_mean[index]),
                -int(index),
            ),
        )
        candidate_indices = indices[1:]
        safe_opportunity = np.any(
            actual_state_effect[candidate_indices]
            & (score_target[candidate_indices] >= 0.0)
            & (margin_lcb_target[candidate_indices] > 0.0)
        )
        win_opportunity = np.any(
            actual_state_effect[candidate_indices]
            & (score_target[candidate_indices] > 0.0)
        )
        rows.append({
            "state_id": int(state),
            "activated": chosen != keep,
            "selected_local_rank": int(np.flatnonzero(indices == chosen)[0]),
            "state_effect": bool(actual_state_effect[chosen]),
            "action_effect": bool(actual_action_effect[chosen]),
            "score_delta": float(score_target[chosen]),
            "margin_mean_delta": float(margin_mean_target[chosen]),
            "margin_lcb_delta": float(margin_lcb_target[chosen]),
            "safe_opportunity": bool(safe_opportunity),
            "win_opportunity": bool(win_opportunity),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    active = [row for row in rows if row["activated"]]

    def mean(rows_: list[dict], key: str) -> float:
        return float(np.mean([row[key] for row in rows_])) if rows_ else 0.0

    opportunity = mean(rows, "safe_opportunity")
    win_opportunity = mean(rows, "win_opportunity")
    # A score-changing candidate is rare (about 7% of non-KEEP arms in the
    # frozen pool). Requiring five percent of *all* states to activate forced
    # the score-effect gate open and defeated the point of stage one. Demand a
    # small but non-zero sample instead, scaled by the actually available win
    # opportunities. Margin-only opportunities are the fallback when a split
    # has no possible win-rate change.
    target_opportunity = win_opportunity if win_opportunity > 0.0 else opportunity
    required_activation = min(0.02, 0.25 * target_opportunity)
    return {
        "states": len(rows),
        "safe_opportunity_rate": opportunity,
        "win_opportunity_rate": win_opportunity,
        "required_activation_rate": required_activation,
        "activation_rate": len(active) / len(rows) if rows else 0.0,
        "active_state_effect_rate": mean(active, "state_effect"),
        "active_action_effect_rate": mean(active, "action_effect"),
        "mean_score_delta_all": mean(rows, "score_delta"),
        "mean_margin_delta_all": mean(rows, "margin_mean_delta"),
        "mean_margin_lcb_delta_all": mean(rows, "margin_lcb_delta"),
        "mean_score_delta_active": mean(active, "score_delta"),
        "mean_margin_delta_active": mean(active, "margin_mean_delta"),
        "mean_margin_lcb_delta_active": mean(active, "margin_lcb_delta"),
        "active_score_regression_rate": float(np.mean([
            row["score_delta"] < 0.0 for row in active
        ])) if active else 0.0,
        "active_margin_lcb_negative_rate": float(np.mean([
            row["margin_lcb_delta"] < 0.0 for row in active
        ])) if active else 0.0,
    }


def passes(metrics: dict) -> bool:
    if metrics["activation_rate"] < metrics["required_activation_rate"]:
        return False
    if metrics["activation_rate"] == 0.0:
        return (
            metrics["safe_opportunity_rate"] == 0.0
            and metrics["win_opportunity_rate"] == 0.0
        )
    return (
        metrics["active_state_effect_rate"] >= 0.90
        and metrics["active_score_regression_rate"] <= 0.10
        and metrics["active_margin_lcb_negative_rate"] <= 0.20
        and metrics["mean_score_delta_active"] >= 0.0
        and metrics["mean_margin_delta_active"] > 0.0
        and metrics["mean_margin_lcb_delta_active"] > 0.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--holdout-opponents", type=parse_strings,
        default=parse_strings("G001,G003,G049,G245"),
    )
    parser.add_argument("--holdout-seed-count", type=int, default=2)
    parser.add_argument("--calibration-seed-count", type=int, default=2)
    parser.add_argument("--score-nonzero-weight", type=float, default=16.0)
    parser.add_argument("--z", type=float, default=1.0)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        required = {
            "state_change_count_full", "action_change_count_full",
            "future_own_cash", "future_opponent_cash",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"dataset lacks effect labels: {sorted(missing)}")
        state_id = np.asarray(data["state_id"])
        opponents = np.asarray(data["opponent"])
        seeds = np.asarray(data["prefix_seed"])
        seat = np.asarray(data["seat"])
        family = np.asarray(data["family"])
        raw_features = np.asarray(data["features"])
        feature_names = [str(value) for value in data["feature_names"]]
        own = np.asarray(data["future_own_cash"], dtype=np.float64)
        rival = np.asarray(data["future_opponent_cash"], dtype=np.float64)
        state_effect = np.asarray(data["state_change_count_full"]) > 0
        action_effect = np.asarray(data["action_change_count_full"]) > 0

    if own.shape[1] < 8 or own.shape[1] % 2:
        raise ValueError("requires an even future count of at least eight")
    x, model_feature_names = make_features(
        raw_features, family, seat, feature_names
    )
    bank_a = np.arange(0, own.shape[1], 2, dtype=np.int64)
    bank_b = np.arange(1, own.shape[1], 2, dtype=np.int64)
    score_a, margin_lcb_a, _ = paired_targets(
        state_id, own[:, bank_a], rival[:, bank_a], args.z
    )
    score_b, margin_lcb_b, _ = paired_targets(
        state_id, own[:, bank_b], rival[:, bank_b], args.z
    )
    margin_mean_a = paired_mean_margin_target(
        state_id, own[:, bank_a], rival[:, bank_a]
    )
    margin_mean_b = paired_mean_margin_target(
        state_id, own[:, bank_b], rival[:, bank_b]
    )
    score_effect_a = np.abs(score_a) > 1e-12
    score_effect_b = np.abs(score_b) > 1e-12
    score_improve_a = score_a > 1e-12
    score_improve_b = score_b > 1e-12
    score_regress_a = score_a < -1e-12
    score_regress_b = score_b < -1e-12

    unique_seeds = sorted(int(value) for value in np.unique(seeds))
    test_seeds = set(unique_seeds[-args.holdout_seed_count:])
    calibration_start = -(
        args.holdout_seed_count + args.calibration_seed_count
    )
    calibration_seeds = set(
        unique_seeds[calibration_start:-args.holdout_seed_count]
    )
    train_seeds = set(unique_seeds[:calibration_start])
    all_opponents = {str(value) for value in np.unique(opponents)}
    test_opponents = all_opponents & args.holdout_opponents
    train_opponents = all_opponents - test_opponents
    state_meta = {}
    for state in np.unique(state_id):
        index = int(np.flatnonzero(state_id == state)[0])
        state_meta[int(state)] = (str(opponents[index]), int(seeds[index]))

    def states_for(opponent_set: set[str], seed_set: set[int]) -> set[int]:
        return {
            state for state, (opponent, seed) in state_meta.items()
            if opponent in opponent_set and seed in seed_set
        }

    partitions = {
        "fit": states_for(train_opponents, train_seeds),
        "calibration": states_for(train_opponents, calibration_seeds),
        "unseen_seeds": states_for(train_opponents, test_seeds),
        "unseen_opponents": states_for(test_opponents, train_seeds),
        "joint_unseen": states_for(test_opponents, test_seeds),
    }
    non_keep = np.ones(len(state_id), dtype=bool)
    for state in np.unique(state_id):
        non_keep[np.flatnonzero(state_id == state)[0]] = False
    fit = np.isin(state_id, list(partitions["fit"])) & non_keep
    fit_indices = np.flatnonzero(fit)

    common = dict(
        n_estimators=300, learning_rate=0.035, num_leaves=15,
        max_depth=5, min_child_samples=40, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=2.0,
        random_state=20260831, n_jobs=16, verbosity=-1,
    )
    state_model = lgb.LGBMClassifier(**common, class_weight="balanced")
    score_effect_model = lgb.LGBMClassifier(**common, class_weight="balanced")
    score_improve_model = lgb.LGBMClassifier(**common, class_weight="balanced")
    score_regress_model = lgb.LGBMClassifier(**common, class_weight="balanced")
    score_model = lgb.LGBMRegressor(**common)
    margin_mean_model = lgb.LGBMRegressor(**common)
    margin_lcb_model = lgb.LGBMRegressor(**common)
    state_model.fit(x[fit_indices], state_effect[fit_indices])
    score_effect_model.fit(x[fit_indices], score_effect_a[fit_indices])
    score_improve_model.fit(x[fit_indices], score_improve_a[fit_indices])
    score_regress_model.fit(x[fit_indices], score_regress_a[fit_indices])
    score_weight = np.ones(len(fit_indices), dtype=np.float64)
    score_weight[score_effect_a[fit_indices]] = args.score_nonzero_weight
    score_model.fit(
        x[fit_indices], score_a[fit_indices], sample_weight=score_weight
    )
    margin_mean_model.fit(x[fit_indices], margin_mean_a[fit_indices])
    margin_lcb_model.fit(x[fit_indices], margin_lcb_a[fit_indices])
    p_state = state_model.predict_proba(x)[:, 1]
    p_score_effect = score_effect_model.predict_proba(x)[:, 1]
    p_score_improve = score_improve_model.predict_proba(x)[:, 1]
    p_score_regress = score_regress_model.predict_proba(x)[:, 1]
    predicted_score = score_model.predict(x)
    predicted_margin_mean = margin_mean_model.predict(x)
    predicted_margin_lcb = margin_lcb_model.predict(x)

    diagnostics = {}
    for name, states in partitions.items():
        indices = np.flatnonzero(np.isin(state_id, list(states)) & non_keep)
        test_bank = name not in {"fit", "calibration"}
        score_target = score_b if test_bank else score_a
        effect_target = score_effect_b if test_bank else score_effect_a
        improve_target = score_improve_b if test_bank else score_improve_a
        regress_target = score_regress_b if test_bank else score_regress_a
        mean_target = margin_mean_b if test_bank else margin_mean_a
        lcb_target = margin_lcb_b if test_bank else margin_lcb_a
        diagnostics[name] = {
            "rows": int(len(indices)),
            "state_effect_auc": auc(state_effect[indices], p_state[indices]),
            "score_effect_auc": auc(effect_target[indices], p_score_effect[indices]),
            "score_improve_auc": auc(
                improve_target[indices], p_score_improve[indices]
            ),
            "score_regress_auc": auc(
                regress_target[indices], p_score_regress[indices]
            ),
            "score_correlation": correlation(
                score_target[indices], predicted_score[indices]
            ),
            "margin_mean_correlation": correlation(
                mean_target[indices], predicted_margin_mean[indices]
            ),
            "margin_lcb_correlation": correlation(
                lcb_target[indices], predicted_margin_lcb[indices]
            ),
        }

    calibration_indices = np.flatnonzero(np.isin(
        state_id, list(partitions["calibration"])
    ) & non_keep)
    positive_margin = np.maximum(0.0, predicted_margin_lcb[calibration_indices])
    margin_grid = np.unique(np.concatenate([
        [0.0], np.quantile(positive_margin, [0.50, 0.75, 0.90])
    ]))
    candidates = []
    for state_threshold in (0.30, 0.50, 0.70, 0.85, 0.95):
        for effect_threshold in (0.0, 0.20, 0.40, 0.60, 0.80):
            for improve_threshold in (0.0, 0.10, 0.20, 0.35, 0.50):
                for regress_ceiling in (0.20, 0.35, 0.50):
                    for score_edge_threshold in (-0.10, 0.0, 0.10, 0.20):
                        for margin_threshold in margin_grid:
                            rows = select(
                                partitions["calibration"], state_id,
                                p_state, p_score_effect,
                                p_score_improve, p_score_regress,
                                predicted_score, predicted_margin_mean,
                                predicted_margin_lcb, state_effect,
                                action_effect, score_a, margin_mean_a,
                                margin_lcb_a, state_threshold,
                                effect_threshold, improve_threshold,
                                regress_ceiling, score_edge_threshold,
                                float(margin_threshold),
                            )
                            metrics = summarize(rows)
                            candidates.append({
                                "state_effect_threshold": state_threshold,
                                "score_effect_threshold": effect_threshold,
                                "score_improve_threshold": improve_threshold,
                                "score_regress_ceiling": regress_ceiling,
                                "score_edge_threshold": score_edge_threshold,
                                "margin_lcb_threshold": float(margin_threshold),
                                "metrics": metrics,
                                "passed": passes(metrics),
                            })
    eligible = [candidate for candidate in candidates if candidate["passed"]]
    active_candidates = [
        candidate for candidate in candidates
        if candidate["metrics"]["activation_rate"]
        >= candidate["metrics"]["required_activation_rate"]
        and candidate["metrics"]["activation_rate"] > 0.0
        and candidate["metrics"]["active_state_effect_rate"] >= 0.90
    ]
    threshold_search_diagnostics = {
        "candidate_count": len(candidates),
        "passed_count": len(eligible),
        "active_structurally_valid_count": len(active_candidates),
        "best_score_candidate": max(
            active_candidates,
            key=lambda candidate: (
                candidate["metrics"]["mean_score_delta_all"],
                candidate["metrics"]["mean_margin_lcb_delta_all"],
            ),
            default=None,
        ),
        "best_margin_lcb_candidate": max(
            active_candidates,
            key=lambda candidate: (
                candidate["metrics"]["mean_margin_lcb_delta_all"],
                candidate["metrics"]["mean_score_delta_all"],
            ),
            default=None,
        ),
    }
    if eligible:
        chosen = max(eligible, key=lambda candidate: (
            candidate["metrics"]["mean_score_delta_all"],
            candidate["metrics"]["mean_margin_lcb_delta_all"],
            candidate["metrics"]["mean_margin_delta_all"],
            candidate["metrics"]["active_state_effect_rate"],
        ))
        chosen["research_only"] = False
    elif active_candidates:
        # Keep the best calibration arm frozen for a diagnostic blind test,
        # while preserving the failed safety status. Blind performance is not
        # allowed to retroactively rescue a calibration failure.
        chosen = max(active_candidates, key=lambda candidate: (
            candidate["metrics"]["mean_score_delta_all"],
            candidate["metrics"]["mean_margin_lcb_delta_all"],
            candidate["metrics"]["mean_margin_delta_all"],
        ))
        chosen["passed"] = False
        chosen["research_only"] = True
    else:
        chosen = {
            "state_effect_threshold": float("inf"),
            "score_effect_threshold": float("inf"),
            "score_improve_threshold": float("inf"),
            "score_regress_ceiling": float("-inf"),
            "score_edge_threshold": float("inf"),
            "margin_lcb_threshold": float("inf"),
            "metrics": summarize(select(
                partitions["calibration"], state_id,
                p_state, p_score_effect,
                p_score_improve, p_score_regress, predicted_score,
                predicted_margin_mean, predicted_margin_lcb,
                state_effect, action_effect, score_a,
                margin_mean_a, margin_lcb_a,
                float("inf"), float("inf"), float("inf"),
                float("-inf"), float("inf"), float("inf"),
            )),
            "passed": False,
            "research_only": True,
        }

    metrics = {}
    selected_rows = {}
    for name, states in partitions.items():
        test_bank = name not in {"fit", "calibration"}
        rows = select(
            states, state_id, p_state, p_score_effect,
            p_score_improve, p_score_regress, predicted_score,
            predicted_margin_mean, predicted_margin_lcb,
            state_effect, action_effect,
            score_b if test_bank else score_a,
            margin_mean_b if test_bank else margin_mean_a,
            margin_lcb_b if test_bank else margin_lcb_a,
            chosen["state_effect_threshold"],
            chosen["score_effect_threshold"],
            chosen["score_improve_threshold"],
            chosen["score_regress_ceiling"],
            chosen["score_edge_threshold"],
            chosen["margin_lcb_threshold"],
        )
        metrics[name] = summarize(rows)
        selected_rows[name] = rows

    deployment_gate = (
        chosen["passed"]
        and all(passes(metrics[name]) for name in (
            "unseen_seeds", "unseen_opponents", "joint_unseen"
        ))
        and sum(metrics[name]["mean_margin_lcb_delta_all"] for name in (
            "unseen_seeds", "unseen_opponents", "joint_unseen"
        )) > 0.0
    )
    model_payload = {
        "state_effect_model": state_model,
        "score_effect_model": score_effect_model,
        "score_improve_model": score_improve_model,
        "score_regress_model": score_regress_model,
        "score_model": score_model,
        "margin_mean_model": margin_mean_model,
        "margin_lcb_model": margin_lcb_model,
        "feature_names": model_feature_names,
        "thresholds": chosen,
        "z": args.z,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, args.model_output)
    payload = {
        "schema": "kaggriculture.candidate8-two-stage-uplift.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Stage one predicts whether a candidate changes the real executed "
            "state chain and whether its paired win result is non-zero. Stage "
            "two predicts candidate-vs-KEEP win-rate, mean-margin and margin-LCB "
            "uplift. Opponent identity and seeds are split keys only."
        ),
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "model": {"path": str(args.model_output), "sha256": sha256(args.model_output)},
        "split": {
            "train_opponents": sorted(train_opponents),
            "test_opponents": sorted(test_opponents),
            "train_seeds": sorted(train_seeds),
            "calibration_seeds": sorted(calibration_seeds),
            "test_seeds": sorted(test_seeds),
            "bank_a_columns": bank_a.tolist(),
            "bank_b_columns": bank_b.tolist(),
        },
        "label_support": {
            "state_effect_rate_non_keep": float(np.mean(state_effect[non_keep])),
            "action_effect_rate_non_keep": float(np.mean(action_effect[non_keep])),
            "score_effect_rate_bank_a_non_keep": float(np.mean(score_effect_a[non_keep])),
            "score_effect_rate_bank_b_non_keep": float(np.mean(score_effect_b[non_keep])),
        },
        "selected_thresholds": chosen,
        "threshold_search_diagnostics": threshold_search_diagnostics,
        "prediction_diagnostics": diagnostics,
        "metrics": metrics,
        "selected_rows": selected_rows,
        "deployment_gate_passed": deployment_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "label_support": payload["label_support"],
        "selected_thresholds": chosen,
        "threshold_search_diagnostics": threshold_search_diagnostics,
        "prediction_diagnostics": diagnostics,
        "metrics": metrics,
        "deployment_gate_passed": deployment_gate,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

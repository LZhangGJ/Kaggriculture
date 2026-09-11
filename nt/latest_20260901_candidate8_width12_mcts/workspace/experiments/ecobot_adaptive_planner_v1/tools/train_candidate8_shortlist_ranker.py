#!/usr/bin/env python3
"""Train and group-validate a candidate pre-ranker from C++ continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def feature_matrix(
    raw_features: np.ndarray, family: np.ndarray, feature_names: list[str]
) -> np.ndarray:
    family_one_hot = np.eye(9, dtype=np.float64)[family.astype(np.int64)]
    values = raw_features.astype(np.float64)
    # C++ exports monetary/workload fields in deterministic x100 fixed point.
    # Restore human-auditable units by schema name; context columns make the
    # old "last three columns" shortcut invalid.
    for index, name in enumerate(feature_names):
        if "x100" in name:
            values[:, index] /= 100.0
    return np.concatenate([values, family_one_hot], axis=1)


def select_diverse_top64(
    prediction: np.ndarray, family: np.ndarray, signature: np.ndarray
) -> np.ndarray:
    order = np.lexsort((signature, -prediction))
    selected: list[int] = []
    selected_set: set[int] = set()
    # KEEP plus one representative of every live family are safety/diversity
    # reservations.  The remaining budget is model-ranked globally.
    for family_id in range(9):
        members = [index for index in order if family[index] == family_id]
        if members:
            selected.append(members[0])
            selected_set.add(members[0])
    for index in order:
        if len(selected) >= min(64, len(order)):
            break
        if int(index) in selected_set:
            continue
        selected.append(int(index))
        selected_set.add(int(index))
    return np.asarray(selected, dtype=np.int64)


def evaluate_split(
    prediction: np.ndarray,
    seed: np.ndarray,
    state_id: np.ndarray,
    signature: np.ndarray,
    family: np.ndarray,
    gain: np.ndarray,
    in_old_shortlist: np.ndarray,
    test_seeds: set[int],
    safe_threshold: float,
) -> dict:
    rows = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        if int(seed[indices[0]]) not in test_seeds:
            continue
        local_prediction = prediction[indices]
        local_gain = gain[indices]
        local_family = family[indices]
        local_signature = signature[indices]
        selected = select_diverse_top64(
            local_prediction, local_family, local_signature
        )
        oracle = int(np.argmax(local_gain))
        model_best = int(selected[np.argmax(local_prediction[selected])])
        keep_members = np.flatnonzero(local_family == 0)
        if not len(keep_members):
            raise RuntimeError(f"state {state} has no KEEP candidate")
        keep = int(keep_members[0])
        predicted_uplift = float(
            local_prediction[model_best] - local_prediction[keep]
        )
        safe_best = model_best if predicted_uplift > safe_threshold else keep
        old = np.flatnonzero(in_old_shortlist[indices])
        old_best = int(old[np.argmax(local_gain[old])])
        rows.append(
            {
                "state_id": int(state),
                "seed": int(seed[indices[0]]),
                "arms": len(indices),
                "oracle_family": int(local_family[oracle]),
                "oracle_gain": float(local_gain[oracle]),
                "oracle_in_model64": bool(oracle in set(selected.tolist())),
                "model64_oracle_regret": float(
                    local_gain[oracle] - np.max(local_gain[selected])
                ),
                "model_top1_gain": float(local_gain[model_best]),
                "predicted_top1_uplift": predicted_uplift,
                "safe_policy_activated": safe_best != keep,
                "safe_policy_gain": float(local_gain[safe_best]),
                "old64_oracle_recall": bool(in_old_shortlist[indices[oracle]]),
                "old64_oracle_regret": float(
                    local_gain[oracle] - local_gain[old_best]
                ),
            }
        )
    return {
        "states": len(rows),
        "oracle_recall_at_model64": float(
            np.mean([row["oracle_in_model64"] for row in rows])
        ),
        "oracle_recall_at_old64": float(
            np.mean([row["old64_oracle_recall"] for row in rows])
        ),
        "mean_model64_oracle_regret": float(
            np.mean([row["model64_oracle_regret"] for row in rows])
        ),
        "mean_old64_oracle_regret": float(
            np.mean([row["old64_oracle_regret"] for row in rows])
        ),
        "mean_model_top1_gain": float(
            np.mean([row["model_top1_gain"] for row in rows])
        ),
        "positive_model_top1_rate": float(
            np.mean([row["model_top1_gain"] > 0 for row in rows])
        ),
        "near_oracle_recall_within_500": float(
            np.mean([row["model64_oracle_regret"] <= 500.0 for row in rows])
        ),
        "near_oracle_recall_within_1000": float(
            np.mean([row["model64_oracle_regret"] <= 1000.0 for row in rows])
        ),
        "safe_threshold": float(safe_threshold),
        "safe_policy_activation_rate": float(
            np.mean([row["safe_policy_activated"] for row in rows])
        ),
        "mean_safe_policy_gain": float(
            np.mean([row["safe_policy_gain"] for row in rows])
        ),
        "safe_policy_positive_rate": float(
            np.mean([row["safe_policy_gain"] > 0 for row in rows])
        ),
        "safe_policy_negative_rate": float(
            np.mean([row["safe_policy_gain"] < 0 for row in rows])
        ),
        "rows": rows,
    }


def calibrate_safe_threshold(
    prediction: np.ndarray,
    seed: np.ndarray,
    state_id: np.ndarray,
    family: np.ndarray,
    gain: np.ndarray,
    train_seeds: set[int],
) -> tuple[float, dict]:
    choices = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        if int(seed[indices[0]]) not in train_seeds:
            continue
        keep = int(np.flatnonzero(family[indices] == 0)[0])
        top = int(np.argmax(prediction[indices]))
        choices.append(
            (
                float(prediction[indices[top]] - prediction[indices[keep]]),
                float(gain[indices[top]]),
            )
        )
    predicted = np.asarray([row[0] for row in choices], dtype=np.float64)
    realized = np.asarray([row[1] for row in choices], dtype=np.float64)
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, np.inf]),
                np.quantile(predicted, np.linspace(0.0, 0.95, 40)),
            ]
        )
    )
    candidates = []
    for threshold in thresholds:
        active = predicted > threshold
        selected_gain = np.where(active, realized, 0.0)
        candidates.append(
            {
                "threshold": float(threshold),
                "activation_rate": float(active.mean()),
                "mean_gain": float(selected_gain.mean()),
                "negative_rate": float(np.mean(selected_gain < 0)),
                "positive_rate": float(np.mean(selected_gain > 0)),
            }
        )
    eligible = [
        row
        for row in candidates
        if row["negative_rate"] <= 0.05 and row["activation_rate"] >= 0.05
    ]
    best = max(
        eligible or candidates,
        key=lambda row: (row["mean_gain"], -row["negative_rate"]),
    )
    return float(best["threshold"]), best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-seed-fraction", type=float, default=0.25)
    parser.add_argument("--calibration-seed-fraction", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=20260829)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        seed_key = "seed" if "seed" in data.files else "prefix_seed"
        seed = np.asarray(data[seed_key])
        signature = np.asarray(data["signature"])
        family = np.asarray(data["family"])
        raw_features = np.asarray(data["features"])
        gain = np.asarray(data["gain_vs_keep"])
        in_old_shortlist = np.asarray(data["in_shortlist"])
        feature_names = [str(value) for value in data["feature_names"]]

    unique_seeds = np.unique(seed)
    rng = np.random.default_rng(args.random_state)
    shuffled = rng.permutation(unique_seeds)
    test_count = max(1, int(round(len(shuffled) * args.test_seed_fraction)))
    test_seeds = {int(value) for value in shuffled[:test_count]}
    remaining = shuffled[test_count:]
    calibration_count = max(
        1, int(round(len(remaining) * args.calibration_seed_fraction))
    )
    calibration_seeds = {
        int(value) for value in remaining[:calibration_count]
    }
    fit_seeds = {
        int(value) for value in remaining[calibration_count:]
    }
    fit_mask = np.isin(seed, list(fit_seeds))
    calibration_mask = np.isin(seed, list(calibration_seeds))
    test_mask = np.isin(seed, list(test_seeds))
    x = feature_matrix(raw_features, family, feature_names)

    model = ExtraTreesRegressor(
        n_estimators=512,
        max_depth=16,
        min_samples_leaf=5,
        max_features=0.9,
        n_jobs=16,
        random_state=args.random_state,
    )
    model.fit(x[fit_mask], gain[fit_mask])
    prediction = model.predict(x)
    safe_threshold, safe_calibration = calibrate_safe_threshold(
        prediction, seed, state_id, family, gain, calibration_seeds
    )
    evaluation = evaluate_split(
        prediction,
        seed,
        state_id,
        signature,
        family,
        gain,
        in_old_shortlist,
        test_seeds,
        safe_threshold,
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": [
                *feature_names,
                *(f"family_{family_id}" for family_id in range(9)),
            ],
            "training_seeds": [
                int(value) for value in sorted(fit_seeds)
            ],
            "calibration_seeds": sorted(calibration_seeds),
            "test_seeds": sorted(test_seeds),
            "safe_threshold": safe_threshold,
        },
        args.model_output,
        compress=3,
    )
    payload = {
        "schema": "kaggriculture.candidate8_shortlist_ranker.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "ExtraTreesRegressor",
        "parameters": model.get_params(),
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "rows": len(seed),
            "states": int(len(np.unique(state_id))),
            "unique_seeds": int(len(unique_seeds)),
            "fit_rows": int(fit_mask.sum()),
            "calibration_rows": int(calibration_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "fit_seed_count": len(fit_seeds),
            "calibration_seed_count": len(calibration_seeds),
            "test_seed_count": int(len(test_seeds)),
        },
        "evaluation": evaluation,
        "safe_threshold_calibration": safe_calibration,
        "model_artifact": {
            "path": str(args.model_output),
            "sha256": sha256(args.model_output),
        },
        "gate": {
            "oracle_recall_at_64_target": 0.90,
            "oracle_recall_at_64_pass": evaluation[
                "oracle_recall_at_model64"
            ]
            >= 0.90,
            "mean_regret_improves": evaluation[
                "mean_model64_oracle_regret"
            ]
            < evaluation["mean_old64_oracle_regret"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**evaluation, "rows": None}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fit one prefix-safe step-120 margin router from multiple exact JAX panels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from fit_step120_public_margin_router import (
    make_model,
    portable_predict,
    route_frequency,
    score_key,
    select_with_threshold,
    sha256,
    summarize,
)


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def balanced_weights(opponent_index: np.ndarray) -> np.ndarray:
    weights = np.zeros(opponent_index.size, dtype=np.float32)
    for opponent in np.unique(opponent_index):
        mask = opponent_index == opponent
        weights[mask] = 1.0 / float(np.sum(mask))
    weights *= float(weights.size) / float(np.sum(weights))
    return weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--prefix-audit", type=Path, required=True)
    parser.add_argument("--prefix-anchor-route", default="route04_rank02_10c_4s_75l")
    parser.add_argument("--feature-step", type=int, default=120)
    parser.add_argument("--routes", default="")
    parser.add_argument(
        "--target-mode",
        choices=("clipped_margin", "downside_2x", "downside_3x"),
        default="clipped_margin",
        help=(
            "Training utility. Downside modes still use terminal margin, but "
            "penalize negative terminal margins two or three times so route "
            "selection better matches a high-win-rate acceptance gate."
        ),
    )
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = resolve(args.panel_manifest)
    prefix_path = resolve(args.prefix_audit)
    model_path = resolve(args.model_output)
    receipt_path = resolve(args.receipt_output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    panels = manifest["panels"]
    if len(panels) < 2:
        raise ValueError("multibank fit requires at least two panels")

    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    group = next(
        (
            row
            for row in prefix["groups"][str(args.feature_step)]
            if args.prefix_anchor_route in row["routes"]
        ),
        None,
    )
    if group is None:
        raise ValueError(
            f"prefix anchor is absent from the step-{args.feature_step} group"
        )
    route_names = [str(value) for value in group["routes"]]
    requested_routes = [value.strip() for value in args.routes.split(",") if value.strip()]
    if requested_routes:
        missing = sorted(set(requested_routes) - set(route_names))
        if missing:
            raise ValueError(
                f"routes outside the step-{args.feature_step} prefix group: {missing}"
            )
        if len(requested_routes) != len(set(requested_routes)):
            raise ValueError("duplicate route in --routes")
        route_names = requested_routes

    x_chunks: list[np.ndarray] = []
    margin_chunks: list[np.ndarray] = []
    opponent_chunks: list[np.ndarray] = []
    group_chunks: list[np.ndarray] = []
    train_chunks: list[np.ndarray] = []
    opponent_names: list[str] = []
    feature_names: list[str] | None = None
    source_rows: list[dict[str, Any]] = []
    source_candidate_ids: list[int] | None = None
    source_action_hashes: list[str] | None = None

    for panel_index, panel in enumerate(panels):
        features_path = resolve(panel["features"])
        outcomes_path = resolve(panel["outcomes"])
        bank_path = resolve(panel["bank_receipt"])
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        candidate_names = [str(value) for value in bank["candidate_names"]]
        panel_opponent_names = [str(row["name"]) for row in bank["opponents"]]
        route_ids = [candidate_names.index(name) for name in route_names]
        panel_action_hashes = [
            str(bank["skeletons"][route_id]["action_sha256"])
            for route_id in route_ids
        ]
        if source_candidate_ids is None:
            source_candidate_ids = route_ids
            source_action_hashes = panel_action_hashes
        elif panel_action_hashes != source_action_hashes:
            raise ValueError(
                f"panel {panel_index} route action streams differ from the first panel"
            )
        selected_names = [str(value) for value in panel["opponents"]]
        missing_opponents = sorted(set(selected_names) - set(panel_opponent_names))
        if missing_opponents:
            raise ValueError(f"panel {panel_index} missing opponents: {missing_opponents}")

        with np.load(features_path, allow_pickle=False) as data:
            panel_features = np.asarray(
                data[f"features{args.feature_step}"], dtype=np.float32
            )
            panel_feature_names = [
                str(value) for value in data[f"feature_names{args.feature_step}"]
            ]
            feature_opponent_ids = np.asarray(
                data["opponent_ids"], dtype=np.int32
            )
            seeds = np.asarray(data["seeds"], dtype=np.int64)
            feature_step = int(data[f"feature_step{args.feature_step}"])
        with np.load(outcomes_path, allow_pickle=False) as data:
            panel_margins = np.asarray(data["margins"], dtype=np.float32)[:, :, 0]
            outcome_opponent_ids = np.asarray(data["opponent_ids"], dtype=np.int32)
            outcome_seeds = np.asarray(data["seeds"], dtype=np.int64)
        if feature_step != args.feature_step:
            raise ValueError(f"panel {panel_index} feature step is {feature_step}")
        if feature_names is None:
            feature_names = panel_feature_names
        elif feature_names != panel_feature_names:
            raise ValueError("feature names differ across panels")
        if panel_margins.shape[0] != len(candidate_names):
            raise ValueError(f"panel {panel_index} candidate shape mismatch")
        if not np.array_equal(seeds, outcome_seeds):
            raise ValueError(f"panel {panel_index} feature/outcome seeds differ")
        train_seed_count = int(panel["train_seed_count"])
        if not 0 <= train_seed_count <= seeds.size:
            raise ValueError(f"panel {panel_index} has invalid train seed count")

        for selected_name in selected_names:
            compiled_opponent_id = panel_opponent_names.index(selected_name)
            outcome_matches = np.flatnonzero(
                outcome_opponent_ids == compiled_opponent_id
            )
            feature_matches = np.flatnonzero(
                feature_opponent_ids == compiled_opponent_id
            )
            if outcome_matches.size != 1 or feature_matches.size != 1:
                raise ValueError(
                    f"panel {panel_index} is missing opponent {selected_name}"
                )
            outcome_opponent_id = int(outcome_matches[0])
            feature_opponent_id = int(feature_matches[0])
            if selected_name in opponent_names:
                opponent_id = opponent_names.index(selected_name)
            else:
                opponent_id = len(opponent_names)
                opponent_names.append(selected_name)
            # Flatten in seat-major order so features and margins stay aligned.
            values = panel_features[:, feature_opponent_id].reshape(
                -1, panel_features.shape[-1]
            )
            margins = panel_margins[route_ids, :, outcome_opponent_id, :].reshape(
                len(route_names), -1
            )
            seat_count = panel_features.shape[0]
            seed_groups = np.tile(seeds, seat_count) + panel_index * 1_000_000
            train_mask = np.tile(
                np.arange(seeds.size) < train_seed_count, seat_count
            )
            x_chunks.append(values)
            margin_chunks.append(margins)
            opponent_chunks.append(
                np.full(values.shape[0], opponent_id, dtype=np.int16)
            )
            group_chunks.append(seed_groups)
            train_chunks.append(train_mask)

        source_rows.append(
            {
                "name": str(panel.get("name", f"panel_{panel_index}")),
                "features": str(features_path),
                "features_sha256": sha256(features_path),
                "outcomes": str(outcomes_path),
                "outcomes_sha256": sha256(outcomes_path),
                "bank_receipt": str(bank_path),
                "bank_receipt_sha256": sha256(bank_path),
                "selected_opponents": selected_names,
                "seed_start": int(seeds[0]),
                "seed_end": int(seeds[-1]),
                "seed_count": int(seeds.size),
                "train_seed_count": train_seed_count,
            }
        )

    assert feature_names is not None
    assert source_candidate_ids is not None
    x = np.concatenate(x_chunks, axis=0)
    margins = np.concatenate(margin_chunks, axis=1)
    opponent_index = np.concatenate(opponent_chunks)
    event_group = np.concatenate(group_chunks)
    train_mask = np.concatenate(train_chunks)
    blind_mask = ~train_mask
    if not np.any(train_mask) or not np.any(blind_mask):
        raise ValueError("panels must provide both training and blind games")
    x_train = x[train_mask]
    train_groups = event_group[train_mask]
    train_margins = margins[:, train_mask]
    train_opponents = opponent_index[train_mask]
    train_weights = balanced_weights(train_opponents)

    def target_values(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -20000.0, 20000.0)
        if args.target_mode == "clipped_margin":
            return clipped
        multiplier = 2.0 if args.target_mode == "downside_2x" else 3.0
        return np.where(clipped < 0.0, clipped * multiplier, clipped)

    target_name = {
        "clipped_margin": "clipped_terminal_margin_per_route",
        "downside_2x": "downside_2x_clipped_terminal_margin_per_route",
        "downside_3x": "downside_3x_clipped_terminal_margin_per_route",
    }[args.target_mode]

    fixed_rows = []
    for route_index, route in enumerate(route_names):
        row = summarize(train_margins[route_index], train_opponents, opponent_names)
        row.update({"route_index": route_index, "route": route})
        fixed_rows.append(row)
    fixed_rows.sort(key=score_key, reverse=True)
    baseline_index = int(fixed_rows[0]["route_index"])

    specs = [
        {"name": "tiny_depth2", "n_estimators": 32, "learning_rate": 0.05, "num_leaves": 4, "max_depth": 2, "min_child_samples": 64, "reg_lambda": 40.0, "colsample_bytree": 0.80},
        {"name": "small_depth3", "n_estimators": 64, "learning_rate": 0.04, "num_leaves": 7, "max_depth": 3, "min_child_samples": 64, "reg_lambda": 30.0, "colsample_bytree": 0.85},
        {"name": "medium_depth4", "n_estimators": 96, "learning_rate": 0.03, "num_leaves": 12, "max_depth": 4, "min_child_samples": 80, "reg_lambda": 40.0, "colsample_bytree": 0.85},
    ]
    thresholds = (0.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0)
    splitter = GroupKFold(n_splits=4)
    cv_rows = []
    best = None
    for spec_index, model_spec in enumerate(specs):
        oof = np.zeros_like(train_margins, dtype=np.float32)
        for fold, (fit_index, validation_index) in enumerate(
            splitter.split(x_train, groups=train_groups)
        ):
            for route_index in range(len(route_names)):
                model = make_model(
                    model_spec, 32000 + spec_index * 1000 + fold * 100 + route_index
                )
                target = target_values(train_margins[route_index, fit_index])
                model.fit(
                    x_train[fit_index],
                    target,
                    sample_weight=train_weights[fit_index],
                )
                oof[route_index, validation_index] = model.predict(
                    x_train[validation_index]
                )
        for threshold in thresholds:
            selected_margin, selected = select_with_threshold(
                oof, train_margins, baseline_index, threshold
            )
            summary = summarize(selected_margin, train_opponents, opponent_names)
            row = {
                "model": model_spec["name"],
                "threshold": threshold,
                "baseline_route": route_names[baseline_index],
                "route_count_used": len(np.unique(selected)),
                **{key: value for key, value in summary.items() if key != "per_opponent"},
            }
            cv_rows.append(row)
            key = score_key(summary)
            if best is None or key > best[0]:
                best = (key, model_spec, threshold, oof.copy())
    assert best is not None
    _, selected_spec, selected_threshold, selected_oof = best

    model_dumps = []
    blind_predictions = np.zeros(
        (len(route_names), int(np.sum(blind_mask))), dtype=np.float32
    )
    gain = np.zeros(len(feature_names), dtype=np.float64)
    portable_max_error = 0.0
    probe = x[blind_mask][: min(128, int(np.sum(blind_mask)))]
    for route_index, route in enumerate(route_names):
        model = make_model(selected_spec, 36000 + route_index)
        target = target_values(train_margins[route_index])
        model.fit(x_train, target, sample_weight=train_weights)
        blind_predictions[route_index] = model.predict(x[blind_mask]).astype(np.float32)
        dump = model.booster_.dump_model()
        model_dumps.append(
            {
                "route_index": route_index,
                "route": route,
                "source_candidate_id": source_candidate_ids[route_index],
                "model": dump,
            }
        )
        gain += model.booster_.feature_importance(importance_type="gain")
        portable_max_error = max(
            portable_max_error,
            float(
                np.max(
                    np.abs(portable_predict(dump, probe) - model.predict(probe))
                )
            ),
        )

    def partition(mask: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
        partition_margins = margins[:, mask]
        routed_margin, selected = select_with_threshold(
            predictions, partition_margins, baseline_index, selected_threshold
        )
        return {
            "baseline": summarize(
                partition_margins[baseline_index], opponent_index[mask], opponent_names
            ),
            "router": summarize(routed_margin, opponent_index[mask], opponent_names),
            "hindsight_oracle": summarize(
                np.max(partition_margins, axis=0), opponent_index[mask], opponent_names
            ),
            "route_frequency": route_frequency(selected, route_names),
        }

    train_result = partition(train_mask, selected_oof)
    blind_result = partition(blind_mask, blind_predictions)
    importance = sorted(
        (
            {"feature": name, "gain": float(value)}
            for name, value in zip(feature_names, gain, strict=True)
        ),
        key=lambda row: -row["gain"],
    )
    model_payload = {
        "schema": "kaggriculture-public-margin-router-model-v2",
        "decision_step": args.feature_step,
        "feature_names": feature_names,
        "route_names": route_names,
        "baseline_route_index": baseline_index,
        "baseline_route": route_names[baseline_index],
        "switch_threshold": selected_threshold,
        "training_target": target_name,
        "model_spec": selected_spec,
        "models": model_dumps,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        json.dumps(model_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": "kaggriculture-public-margin-router-multibank-fit-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "JAX_SCREEN_ONLY",
        "decision_step": args.feature_step,
        "common_prefix_actions": 120,
        "prefix_anchor_route": args.prefix_anchor_route,
        "route_selection": "explicit_subset" if requested_routes else "all_routes_in_prefix_group",
        "route_count": len(route_names),
        "route_names": route_names,
        "selected_opponents": opponent_names,
        "opponent_balanced_training_weights": True,
        "forbidden_runtime_inputs": [
            "opponent_identity", "seed", "seat", "future_events",
            "terminal_outcome", "private_opponent_state",
        ],
        "target": target_name,
        "selected_model": selected_spec,
        "selected_threshold": selected_threshold,
        "baseline_route": route_names[baseline_index],
        "cv_candidates": cv_rows,
        "train_oof": train_result,
        "blind": blind_result,
        "top_feature_gain": importance[:20],
        "portable_tree_max_abs_error": portable_max_error,
        "model_output": str(model_path),
        "model_sha256": sha256(model_path),
        "sources": {
            "panel_manifest": str(manifest_path),
            "panel_manifest_sha256": sha256(manifest_path),
            "panels": source_rows,
            "prefix_audit": str(prefix_path),
            "prefix_audit_sha256": sha256(prefix_path),
        },
        "truth_boundary": (
            "Blind JAX screening against six stepwise-exact opponents. Route choice "
            "is prefix-safe and public-state-only; official Python 1.32.7 independent "
            "holdout remains mandatory."
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "route_count": result["route_count"],
                "selected_model": selected_spec["name"],
                "selected_threshold": selected_threshold,
                "baseline_route": result["baseline_route"],
                "train_oof": train_result,
                "blind": blind_result,
                "portable_tree_max_abs_error": portable_max_error,
                "model": str(model_path),
                "receipt": str(receipt_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

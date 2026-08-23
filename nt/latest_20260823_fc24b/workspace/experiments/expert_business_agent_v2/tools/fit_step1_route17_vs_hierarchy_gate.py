#!/usr/bin/env python3
"""Fit a legal step-1 gate between Route17 and a later hierarchy.

Both options execute the same action at step 0.  The gate sees only the
candidate's public BASE48 state immediately before step 1.  The downstream
option's training outcomes come from OOF hierarchical choices, while the
independent panel uses the hierarchy fitted only on the two earlier panels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from fit_step120_public_margin_router import make_model, score_key, sha256, summarize
from fit_step120_public_margin_router_multibank import balanced_weights


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def select(
    route_prediction: np.ndarray,
    downstream_prediction: np.ndarray,
    route_margin: np.ndarray,
    downstream_margin: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    choose_route = route_prediction >= downstream_prediction + threshold
    margin = np.where(choose_route, route_margin, downstream_margin)
    return margin, choose_route


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = resolve(args.manifest)
    receipt_path = resolve(args.receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downstream_path = resolve(manifest["downstream_predictions"])
    with np.load(downstream_path, allow_pickle=False) as data:
        downstream_margin = np.asarray(data["selected_margin"], dtype=np.float32)
        downstream_train_mask = np.asarray(data["train_mask"], dtype=bool)
        downstream_opponents = np.asarray(data["opponent_index"], dtype=np.int16)
        downstream_panels = np.asarray(data["panel_index"], dtype=np.int16)
        downstream_groups = np.asarray(data["event_groups"], dtype=np.int64)

    x_chunks: list[np.ndarray] = []
    route_margin_chunks: list[np.ndarray] = []
    train_chunks: list[np.ndarray] = []
    opponent_chunks: list[np.ndarray] = []
    panel_chunks: list[np.ndarray] = []
    group_chunks: list[np.ndarray] = []
    opponent_names: list[str] = []
    feature_names: list[str] | None = None
    source_rows: list[dict[str, Any]] = []

    for panel_index, panel in enumerate(manifest["panels"]):
        features_path = resolve(panel["features"])
        outcomes_path = resolve(panel["outcomes"])
        bank_path = resolve(panel["bank_receipt"])
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        candidate_names = [str(value) for value in bank["candidate_names"]]
        bank_opponents = [str(row["name"]) for row in bank["opponents"]]
        route_id = candidate_names.index(str(manifest["route"]))
        with np.load(features_path, allow_pickle=False) as data:
            features = np.asarray(data["features1"], dtype=np.float32)
            names = [str(value) for value in data["feature_names1"]]
            feature_opponent_ids = np.asarray(data["opponent_ids"], dtype=np.int32)
            seeds = np.asarray(data["seeds"], dtype=np.int64)
        with np.load(outcomes_path, allow_pickle=False) as data:
            margins = np.asarray(data["margins"], dtype=np.float32)[:, :, 0]
            outcome_opponent_ids = np.asarray(data["opponent_ids"], dtype=np.int32)
            outcome_seeds = np.asarray(data["seeds"], dtype=np.int64)
        if not np.array_equal(seeds, outcome_seeds):
            raise ValueError(f"panel {panel_index} feature/outcome seeds differ")
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("step-1 feature schema differs across panels")
        train_seed_count = int(panel["train_seed_count"])
        if not 0 <= train_seed_count <= seeds.size:
            raise ValueError(f"invalid train count in panel {panel_index}")
        for selected_name_value in panel["opponents"]:
            selected_name = str(selected_name_value)
            compiled_id = bank_opponents.index(selected_name)
            feature_match = np.flatnonzero(feature_opponent_ids == compiled_id)
            outcome_match = np.flatnonzero(outcome_opponent_ids == compiled_id)
            if feature_match.size != 1 or outcome_match.size != 1:
                raise ValueError(f"panel {panel_index} is missing {selected_name}")
            values = features[:, int(feature_match[0])].reshape(-1, features.shape[-1])
            route_values = margins[
                route_id, :, int(outcome_match[0]), :
            ].reshape(-1)
            if selected_name in opponent_names:
                opponent_id = opponent_names.index(selected_name)
            else:
                opponent_id = len(opponent_names)
                opponent_names.append(selected_name)
            repeats = values.shape[0] // seeds.size
            x_chunks.append(values)
            route_margin_chunks.append(route_values)
            train_chunks.append(
                np.tile(np.arange(seeds.size) < train_seed_count, repeats)
            )
            opponent_chunks.append(
                np.full(values.shape[0], opponent_id, dtype=np.int16)
            )
            panel_chunks.append(
                np.full(values.shape[0], panel_index, dtype=np.int16)
            )
            group_chunks.append(
                np.tile(seeds, repeats) + panel_index * 1_000_000
            )
        source_rows.append(
            {
                "name": str(panel["name"]),
                "features": str(features_path),
                "features_sha256": sha256(features_path),
                "outcomes": str(outcomes_path),
                "outcomes_sha256": sha256(outcomes_path),
                "bank_receipt": str(bank_path),
                "bank_receipt_sha256": sha256(bank_path),
                "seed_start": int(seeds[0]),
                "seed_count": int(seeds.size),
                "train_seed_count": train_seed_count,
            }
        )

    assert feature_names is not None
    x = np.concatenate(x_chunks)
    route_margin = np.concatenate(route_margin_chunks)
    train_mask = np.concatenate(train_chunks)
    opponent_index = np.concatenate(opponent_chunks)
    panel_index_values = np.concatenate(panel_chunks)
    groups = np.concatenate(group_chunks)
    if downstream_margin.size != x.shape[0]:
        raise ValueError("downstream predictions and gate panels have different sizes")
    if not np.array_equal(downstream_train_mask, train_mask):
        raise ValueError("downstream and gate train masks differ")
    if not np.array_equal(downstream_opponents, opponent_index):
        raise ValueError("downstream and gate opponent order differs")
    if not np.array_equal(downstream_panels, panel_index_values):
        raise ValueError("downstream and gate panel order differs")
    if not np.array_equal(downstream_groups, groups):
        raise ValueError("downstream and gate event groups differ")
    blind_mask = ~train_mask
    train_indices = np.flatnonzero(train_mask)
    train_groups = groups[train_mask]
    weights = balanced_weights(opponent_index[train_mask])

    specs = [
        {"name": "stump", "n_estimators": 24, "learning_rate": 0.04, "num_leaves": 2, "max_depth": 1, "min_child_samples": 48, "reg_lambda": 60.0, "colsample_bytree": 0.75},
        {"name": "tiny_depth2", "n_estimators": 32, "learning_rate": 0.05, "num_leaves": 4, "max_depth": 2, "min_child_samples": 48, "reg_lambda": 50.0, "colsample_bytree": 0.80},
        {"name": "small_depth3", "n_estimators": 64, "learning_rate": 0.04, "num_leaves": 7, "max_depth": 3, "min_child_samples": 64, "reg_lambda": 50.0, "colsample_bytree": 0.85},
    ]
    thresholds = [-1000.0, -500.0, 0.0, 500.0, 1000.0, 2000.0]
    splitter = GroupKFold(n_splits=4)
    cv_rows = []
    best: tuple[Any, ...] | None = None
    for spec_index, spec in enumerate(specs):
        route_oof = np.zeros(train_indices.size, dtype=np.float32)
        downstream_oof = np.zeros(train_indices.size, dtype=np.float32)
        for fold, (fit_local, valid_local) in enumerate(
            splitter.split(x[train_mask], groups=train_groups)
        ):
            route_model = make_model(spec, 141000 + spec_index * 100 + fold)
            downstream_model = make_model(spec, 142000 + spec_index * 100 + fold)
            route_model.fit(
                x[train_indices[fit_local]],
                np.clip(route_margin[train_indices[fit_local]], -20000.0, 20000.0),
                sample_weight=weights[fit_local],
            )
            downstream_model.fit(
                x[train_indices[fit_local]],
                np.clip(downstream_margin[train_indices[fit_local]], -20000.0, 20000.0),
                sample_weight=weights[fit_local],
            )
            route_oof[valid_local] = route_model.predict(
                x[train_indices[valid_local]]
            )
            downstream_oof[valid_local] = downstream_model.predict(
                x[train_indices[valid_local]]
            )
        for threshold in thresholds:
            selected_margin, selected = select(
                route_oof,
                downstream_oof,
                route_margin[train_mask],
                downstream_margin[train_mask],
                threshold,
            )
            summary = summarize(
                selected_margin, opponent_index[train_mask], opponent_names
            )
            row = {
                "model": spec["name"],
                "threshold": threshold,
                "route17_fraction": float(np.mean(selected)),
                **{key: value for key, value in summary.items() if key != "per_opponent"},
            }
            cv_rows.append(row)
            key = score_key(summary)
            if best is None or key > best[0]:
                best = (key, spec, threshold, route_oof, downstream_oof)
    assert best is not None
    _, selected_spec, selected_threshold, route_oof, downstream_oof = best

    route_model = make_model(selected_spec, 151001)
    downstream_model = make_model(selected_spec, 151002)
    route_model.fit(
        x[train_mask],
        np.clip(route_margin[train_mask], -20000.0, 20000.0),
        sample_weight=weights,
    )
    downstream_model.fit(
        x[train_mask],
        np.clip(downstream_margin[train_mask], -20000.0, 20000.0),
        sample_weight=weights,
    )
    route_blind = route_model.predict(x[blind_mask])
    downstream_blind = downstream_model.predict(x[blind_mask])

    def partition(
        mask: np.ndarray,
        route_prediction: np.ndarray,
        downstream_prediction: np.ndarray,
    ) -> dict[str, Any]:
        selected_margin, selected = select(
            route_prediction,
            downstream_prediction,
            route_margin[mask],
            downstream_margin[mask],
            float(selected_threshold),
        )
        option_oracle = np.maximum(route_margin[mask], downstream_margin[mask])
        result = {
            "route17": summarize(route_margin[mask], opponent_index[mask], opponent_names),
            "downstream_hierarchy": summarize(
                downstream_margin[mask], opponent_index[mask], opponent_names
            ),
            "step1_gate": summarize(
                selected_margin, opponent_index[mask], opponent_names
            ),
            "two_option_oracle": summarize(
                option_oracle, opponent_index[mask], opponent_names
            ),
            "route17_fraction": float(np.mean(selected)),
        }
        partition_panels = panel_index_values[mask]
        partition_opponents = opponent_index[mask]
        result["per_panel"] = {}
        for panel_id, panel in enumerate(manifest["panels"]):
            local = partition_panels == panel_id
            if not np.any(local):
                continue
            result["per_panel"][str(panel["name"])] = {
                "games": int(np.sum(local)),
                "route17": summarize(
                    route_margin[mask][local], partition_opponents[local], opponent_names
                ),
                "downstream_hierarchy": summarize(
                    downstream_margin[mask][local], partition_opponents[local], opponent_names
                ),
                "step1_gate": summarize(
                    selected_margin[local], partition_opponents[local], opponent_names
                ),
                "two_option_oracle": summarize(
                    option_oracle[local], partition_opponents[local], opponent_names
                ),
            }
        return result

    train_result = partition(train_mask, route_oof, downstream_oof)
    blind_result = partition(blind_mask, route_blind, downstream_blind)
    result = {
        "schema": "kaggriculture-step1-route17-vs-hierarchy-gate-fit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "JAX_SCREEN_ONLY",
        "decision_step": 1,
        "route_option": str(manifest["route"]),
        "downstream_option": "step120_step168_step216_hierarchy",
        "selected_model": selected_spec,
        "selected_threshold": selected_threshold,
        "cv_candidates": cv_rows,
        "train_oof": train_result,
        "blind": blind_result,
        "forbidden_runtime_inputs": [
            "opponent_identity",
            "seed",
            "future_events",
            "terminal_outcome",
            "private_opponent_state",
        ],
        "sources": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "downstream_predictions": str(downstream_path),
            "downstream_predictions_sha256": sha256(downstream_path),
            "panels": source_rows,
        },
        "truth_boundary": (
            "Offline legal step-1 JAX screen. The blind panel was not used for "
            "model or threshold selection. No deployable Agent was generated."
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

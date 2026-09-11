#!/usr/bin/env python3
"""Fit a prefix-safe 120-step family / 168-step route terminal-margin router."""

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
    route_frequency,
    score_key,
    sha256,
    summarize,
)
from fit_step120_public_margin_router_multibank import balanced_weights


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def choose(
    root_predictions: np.ndarray,
    child_predictions: list[np.ndarray],
    family_route_indices: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    family_choice = np.argmax(root_predictions, axis=0)
    route_choice = np.zeros(family_choice.size, dtype=np.int16)
    for family_index, route_indices in enumerate(family_route_indices):
        mask = family_choice == family_index
        if not np.any(mask):
            continue
        local = np.argmax(child_predictions[family_index][:, mask], axis=0)
        route_choice[mask] = np.asarray(route_indices, dtype=np.int16)[local]
    return family_choice, route_choice


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = resolve(args.panel_manifest)
    family_path = resolve(args.families)
    receipt_path = resolve(args.receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    family_spec = json.loads(family_path.read_text(encoding="utf-8"))
    families = list(family_spec["families"])
    family_names = [str(row["name"]) for row in families]
    route_names = [str(route) for row in families for route in row["routes"]]
    if len(route_names) != len(set(route_names)):
        raise ValueError("a route occurs in more than one family")
    route_to_index = {route: index for index, route in enumerate(route_names)}
    family_route_indices = [
        [route_to_index[str(route)] for route in row["routes"]] for row in families
    ]

    root_chunks: list[np.ndarray] = []
    child_chunks: list[list[np.ndarray]] = [[] for _ in families]
    margin_chunks: list[np.ndarray] = []
    opponent_chunks: list[np.ndarray] = []
    group_chunks: list[np.ndarray] = []
    train_chunks: list[np.ndarray] = []
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
        route_ids = [candidate_names.index(route) for route in route_names]
        anchor_ids = [candidate_names.index(str(row["anchor"])) for row in families]
        with np.load(features_path, allow_pickle=False) as data:
            feature_candidate_ids = np.asarray(data["candidate_ids"], dtype=np.int32)
            feature_opponent_ids = np.asarray(data["opponent_ids"], dtype=np.int32)
            seeds = np.asarray(data["seeds"], dtype=np.int64)
            root_features = np.asarray(
                data["features120_by_candidate"], dtype=np.float32
            )
            child_features168 = np.asarray(
                data["features168_by_candidate"], dtype=np.float32
            )
            child_features216 = np.asarray(
                data["features216_by_candidate"], dtype=np.float32
            )
            panel_feature_names = [str(value) for value in data["feature_names120"]]
        with np.load(outcomes_path, allow_pickle=False) as data:
            panel_margins = np.asarray(data["margins"], dtype=np.float32)[:, :, 0]
            outcome_opponent_ids = np.asarray(data["opponent_ids"], dtype=np.int32)
            outcome_seeds = np.asarray(data["seeds"], dtype=np.int64)
        if not np.array_equal(seeds, outcome_seeds):
            raise ValueError(f"panel {panel_index} feature/outcome seeds differ")
        if feature_names is None:
            feature_names = panel_feature_names
        elif feature_names != panel_feature_names:
            raise ValueError("feature schema differs across panels")
        anchor_positions = []
        for anchor_id in anchor_ids:
            matches = np.flatnonzero(feature_candidate_ids == anchor_id)
            if matches.size != 1:
                raise ValueError(f"panel {panel_index} is missing anchor {anchor_id}")
            anchor_positions.append(int(matches[0]))
        train_seed_count = int(panel["train_seed_count"])
        if not 0 < train_seed_count < seeds.size:
            raise ValueError(f"panel {panel_index} leaves no blind suffix")

        for selected_name in panel["opponents"]:
            compiled_id = bank_opponents.index(str(selected_name))
            feature_match = np.flatnonzero(feature_opponent_ids == compiled_id)
            outcome_match = np.flatnonzero(outcome_opponent_ids == compiled_id)
            if feature_match.size != 1 or outcome_match.size != 1:
                raise ValueError(f"panel {panel_index} is missing {selected_name}")
            feature_opponent = int(feature_match[0])
            outcome_opponent = int(outcome_match[0])
            root_values = root_features[
                anchor_positions[0], :, feature_opponent
            ].reshape(-1, root_features.shape[-1])
            # Every family shares the exact root prefix through step 120.
            for anchor_position in anchor_positions[1:]:
                alternate = root_features[
                    anchor_position, :, feature_opponent
                ].reshape(-1, root_features.shape[-1])
                if not np.array_equal(root_values, alternate):
                    raise AssertionError("root family features differ before step 120")
            root_chunks.append(root_values)
            for family_index, anchor_position in enumerate(anchor_positions):
                child_step = int(families[family_index].get("child_step", 168))
                selected_child_features = (
                    child_features216 if child_step == 216 else child_features168
                )
                if child_step not in (168, 216):
                    raise ValueError(f"unsupported child step {child_step}")
                child_chunks[family_index].append(
                    selected_child_features[
                        anchor_position, :, feature_opponent
                    ].reshape(-1, selected_child_features.shape[-1])
                )
            margin_chunks.append(
                panel_margins[route_ids, :, outcome_opponent, :].reshape(
                    len(route_names), -1
                )
            )
            selected_name = str(selected_name)
            if selected_name in opponent_names:
                opponent_id = opponent_names.index(selected_name)
            else:
                opponent_id = len(opponent_names)
                opponent_names.append(selected_name)
            opponent_chunks.append(
                np.full(root_values.shape[0], opponent_id, dtype=np.int16)
            )
            group_chunks.append(
                np.tile(seeds, root_features.shape[1]) + panel_index * 1_000_000
            )
            train_chunks.append(
                np.tile(
                    np.arange(seeds.size) < train_seed_count,
                    root_features.shape[1],
                )
            )
        source_rows.append({
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
        })

    assert feature_names is not None
    root_x = np.concatenate(root_chunks)
    child_x = [np.concatenate(chunks) for chunks in child_chunks]
    margins = np.concatenate(margin_chunks, axis=1)
    opponent_index = np.concatenate(opponent_chunks)
    event_groups = np.concatenate(group_chunks)
    train_mask = np.concatenate(train_chunks)
    blind_mask = ~train_mask
    train_indices = np.flatnonzero(train_mask)
    weights = balanced_weights(opponent_index[train_mask])

    fixed = []
    for route_index, route in enumerate(route_names):
        summary = summarize(
            margins[route_index, train_mask], opponent_index[train_mask], opponent_names
        )
        fixed.append({"route": route, "route_index": route_index, **summary})
    fixed.sort(key=score_key, reverse=True)
    baseline_index = int(fixed[0]["route_index"])

    specs = [
        {"name": "tiny_depth2", "n_estimators": 32, "learning_rate": 0.05, "num_leaves": 4, "max_depth": 2, "min_child_samples": 32, "reg_lambda": 40.0, "colsample_bytree": 0.80},
        {"name": "small_depth3", "n_estimators": 64, "learning_rate": 0.04, "num_leaves": 7, "max_depth": 3, "min_child_samples": 48, "reg_lambda": 30.0, "colsample_bytree": 0.85},
        {"name": "medium_depth4", "n_estimators": 96, "learning_rate": 0.03, "num_leaves": 12, "max_depth": 4, "min_child_samples": 64, "reg_lambda": 40.0, "colsample_bytree": 0.85},
    ]
    splitter = GroupKFold(n_splits=4)
    cv_rows = []
    best: tuple[Any, ...] | None = None
    train_groups = event_groups[train_mask]
    train_margins = margins[:, train_mask]
    for spec_index, spec in enumerate(specs):
        root_oof = np.zeros((len(families), train_indices.size), dtype=np.float32)
        child_oof = [
            np.zeros((len(indices), train_indices.size), dtype=np.float32)
            for indices in family_route_indices
        ]
        for fold, (fit_local, validation_local) in enumerate(
            splitter.split(root_x[train_mask], groups=train_groups)
        ):
            for family_index, indices in enumerate(family_route_indices):
                root_target = np.clip(
                    np.max(train_margins[indices][:, fit_local], axis=0),
                    -20000.0,
                    20000.0,
                )
                root_model = make_model(
                    spec, 41000 + spec_index * 1000 + fold * 100 + family_index
                )
                root_model.fit(
                    root_x[train_indices[fit_local]],
                    root_target,
                    sample_weight=weights[fit_local],
                )
                root_oof[family_index, validation_local] = root_model.predict(
                    root_x[train_indices[validation_local]]
                )
                for local_route, route_index in enumerate(indices):
                    child_model = make_model(
                        spec,
                        51000
                        + spec_index * 2000
                        + fold * 200
                        + route_index,
                    )
                    child_model.fit(
                        child_x[family_index][train_indices[fit_local]],
                        np.clip(
                            train_margins[route_index, fit_local], -20000.0, 20000.0
                        ),
                        sample_weight=weights[fit_local],
                    )
                    child_oof[family_index][local_route, validation_local] = (
                        child_model.predict(
                            child_x[family_index][train_indices[validation_local]]
                        )
                    )
        family_choice, route_choice = choose(
            root_oof, child_oof, family_route_indices
        )
        selected_margin = train_margins[
            route_choice, np.arange(route_choice.size)
        ]
        summary = summarize(
            selected_margin, opponent_index[train_mask], opponent_names
        )
        row = {
            "model": spec["name"],
            "family_count_used": int(np.unique(family_choice).size),
            "route_count_used": int(np.unique(route_choice).size),
            **{key: value for key, value in summary.items() if key != "per_opponent"},
        }
        cv_rows.append(row)
        key = score_key(summary)
        if best is None or key > best[0]:
            best = (key, spec, root_oof, child_oof)
    assert best is not None
    _, selected_spec, root_oof, child_oof = best

    blind_indices = np.flatnonzero(blind_mask)
    root_blind = np.zeros((len(families), blind_indices.size), dtype=np.float32)
    child_blind = [
        np.zeros((len(indices), blind_indices.size), dtype=np.float32)
        for indices in family_route_indices
    ]
    for family_index, indices in enumerate(family_route_indices):
        root_model = make_model(selected_spec, 61000 + family_index)
        root_model.fit(
            root_x[train_mask],
            np.clip(np.max(train_margins[indices], axis=0), -20000.0, 20000.0),
            sample_weight=weights,
        )
        root_blind[family_index] = root_model.predict(root_x[blind_mask])
        for local_route, route_index in enumerate(indices):
            child_model = make_model(selected_spec, 71000 + route_index)
            child_model.fit(
                child_x[family_index][train_mask],
                np.clip(train_margins[route_index], -20000.0, 20000.0),
                sample_weight=weights,
            )
            child_blind[family_index][local_route] = child_model.predict(
                child_x[family_index][blind_mask]
            )

    def partition(
        mask: np.ndarray,
        root_predictions: np.ndarray,
        child_predictions: list[np.ndarray],
    ) -> dict[str, Any]:
        partition_margins = margins[:, mask]
        family_choice, route_choice = choose(
            root_predictions, child_predictions, family_route_indices
        )
        selected_margin = partition_margins[
            route_choice, np.arange(route_choice.size)
        ]
        return {
            "fixed_baseline": summarize(
                partition_margins[baseline_index], opponent_index[mask], opponent_names
            ),
            "hierarchical_router": summarize(
                selected_margin, opponent_index[mask], opponent_names
            ),
            "hindsight_oracle": summarize(
                np.max(partition_margins, axis=0), opponent_index[mask], opponent_names
            ),
            "family_frequency": [
                {
                    "family": family_names[index],
                    "games": int(np.sum(family_choice == index)),
                    "fraction": float(np.mean(family_choice == index)),
                }
                for index in range(len(families))
                if np.any(family_choice == index)
            ],
            "route_frequency": route_frequency(route_choice, route_names),
        }

    train_result = partition(train_mask, root_oof, child_oof)
    blind_result = partition(blind_mask, root_blind, child_blind)
    result = {
        "schema": "kaggriculture-step120-step168-hierarchical-router-fit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "JAX_SCREEN_ONLY",
        "root_step": int(family_spec["root_step"]),
        "child_steps": {
            str(row["name"]): int(row.get("child_step", 168)) for row in families
        },
        "selected_model": selected_spec,
        "route_count": len(route_names),
        "family_count": len(families),
        "families": families,
        "baseline_route": route_names[baseline_index],
        "cv_candidates": cv_rows,
        "train_oof": train_result,
        "blind": blind_result,
        "forbidden_runtime_inputs": [
            "opponent_identity", "seed", "future_events", "terminal_outcome",
            "private_opponent_state",
        ],
        "sources": {
            "panel_manifest": str(manifest_path),
            "panel_manifest_sha256": sha256(manifest_path),
            "family_spec": str(family_path),
            "family_spec_sha256": sha256(family_path),
            "panels": source_rows,
        },
        "truth_boundary": (
            "Offline JAX screening of a prefix-safe hierarchical policy. No "
            "deployable Agent has been generated; official Python 1.32.7 holdout "
            "is mandatory if blind screening improves."
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "selected_model": selected_spec["name"],
        "train_oof": train_result,
        "blind": blind_result,
        "receipt": str(receipt_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

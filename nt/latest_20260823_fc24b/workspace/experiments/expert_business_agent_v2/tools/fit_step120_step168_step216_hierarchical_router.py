#!/usr/bin/env python3
"""Fit a legal 120 -> 168 -> 216 public-state terminal-margin router.

The hierarchy is constrained by real shared action prefixes:

* step 120 selects a route family;
* step 168 selects a subgroup whose routes still share that prefix;
* step 216 selects the concrete route within the chosen subgroup.

No later feature is visible to an earlier decision.  This script is an offline
JAX screen only; it deliberately does not emit a deployable Agent.
"""

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


def _flatten_routes(families: list[dict[str, Any]]) -> list[str]:
    routes = [
        str(route)
        for family in families
        for subgroup in family["subgroups"]
        for route in subgroup["routes"]
    ]
    if len(routes) != len(set(routes)):
        raise ValueError("a route occurs in more than one subgroup")
    return routes


def choose(
    root_predictions: np.ndarray,
    subgroup_predictions: list[np.ndarray],
    route_predictions: list[list[np.ndarray]],
    subgroup_route_indices: list[list[list[int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the hierarchy without using an unchosen branch's later state."""

    family_choice = np.argmax(root_predictions, axis=0).astype(np.int16)
    subgroup_choice = np.zeros(family_choice.size, dtype=np.int16)
    route_choice = np.zeros(family_choice.size, dtype=np.int16)
    for family_index, subgroups in enumerate(subgroup_route_indices):
        family_mask = family_choice == family_index
        if not np.any(family_mask):
            continue
        local_subgroup = np.argmax(
            subgroup_predictions[family_index][:, family_mask], axis=0
        ).astype(np.int16)
        subgroup_choice[family_mask] = local_subgroup
        family_positions = np.flatnonzero(family_mask)
        for subgroup_index, route_indices in enumerate(subgroups):
            local_mask = local_subgroup == subgroup_index
            if not np.any(local_mask):
                continue
            positions = family_positions[local_mask]
            local_route = np.argmax(
                route_predictions[family_index][subgroup_index][:, positions],
                axis=0,
            )
            route_choice[positions] = np.asarray(route_indices, dtype=np.int16)[
                local_route
            ]
    return family_choice, subgroup_choice, route_choice


def _load_feature_files(
    paths: list[Path],
) -> tuple[
    dict[int, dict[int, np.ndarray]],
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Merge candidate anchors from multiple feature probes."""

    anchors: dict[int, dict[int, np.ndarray]] = {}
    reference_opponents: np.ndarray | None = None
    reference_seeds: np.ndarray | None = None
    reference_names: list[str] | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            candidate_ids = np.asarray(data["candidate_ids"], dtype=np.int32)
            opponent_ids = np.asarray(data["opponent_ids"], dtype=np.int32)
            seeds = np.asarray(data["seeds"], dtype=np.int64)
            names = [str(value) for value in data["feature_names120"]]
            step_arrays = {
                120: np.asarray(data["features120_by_candidate"], dtype=np.float32),
                168: np.asarray(data["features168_by_candidate"], dtype=np.float32),
                216: np.asarray(data["features216_by_candidate"], dtype=np.float32),
            }
        if reference_opponents is None:
            reference_opponents = opponent_ids
            reference_seeds = seeds
            reference_names = names
        else:
            if not np.array_equal(reference_opponents, opponent_ids):
                raise ValueError(f"feature opponent ids differ in {path}")
            if not np.array_equal(reference_seeds, seeds):
                raise ValueError(f"feature seeds differ in {path}")
            if reference_names != names:
                raise ValueError(f"feature schema differs in {path}")
        for position, candidate_id in enumerate(candidate_ids.tolist()):
            candidate_id = int(candidate_id)
            values = {
                step: array[position].copy() for step, array in step_arrays.items()
            }
            if candidate_id in anchors:
                for step in (120, 168, 216):
                    if not np.array_equal(anchors[candidate_id][step], values[step]):
                        raise ValueError(
                            f"duplicate candidate {candidate_id} differs at step {step}"
                        )
            else:
                anchors[candidate_id] = values
    if reference_opponents is None or reference_seeds is None or reference_names is None:
        raise ValueError("no feature files")
    return anchors, reference_opponents, reference_seeds, reference_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    args = parser.parse_args()

    manifest_path = resolve(args.panel_manifest)
    hierarchy_path = resolve(args.hierarchy)
    receipt_path = resolve(args.receipt)
    predictions_path = (
        resolve(args.predictions_output) if args.predictions_output is not None else None
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    families = list(hierarchy["families"])
    family_names = [str(row["name"]) for row in families]
    subgroup_names = [
        [str(subgroup["name"]) for subgroup in family["subgroups"]]
        for family in families
    ]
    route_names = _flatten_routes(families)
    route_to_index = {route: index for index, route in enumerate(route_names)}
    subgroup_route_indices = [
        [
            [route_to_index[str(route)] for route in subgroup["routes"]]
            for subgroup in family["subgroups"]
        ]
        for family in families
    ]
    family_route_indices = [
        [route_index for subgroup in subgroups for route_index in subgroup]
        for subgroups in subgroup_route_indices
    ]

    root_chunks: list[np.ndarray] = []
    family_chunks: list[list[np.ndarray]] = [[] for _ in families]
    subgroup_chunks: list[list[list[np.ndarray]]] = [
        [[] for _ in family["subgroups"]] for family in families
    ]
    margin_chunks: list[np.ndarray] = []
    opponent_chunks: list[np.ndarray] = []
    panel_chunks: list[np.ndarray] = []
    group_chunks: list[np.ndarray] = []
    train_chunks: list[np.ndarray] = []
    opponent_names: list[str] = []
    feature_names: list[str] | None = None
    source_rows: list[dict[str, Any]] = []

    for panel_index, panel in enumerate(manifest["panels"]):
        feature_paths = [resolve(path) for path in panel["feature_files"]]
        outcomes_path = resolve(panel["outcomes"])
        bank_path = resolve(panel["bank_receipt"])
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        candidate_names = [str(value) for value in bank["candidate_names"]]
        bank_opponents = [str(row["name"]) for row in bank["opponents"]]
        route_ids = [candidate_names.index(route) for route in route_names]
        root_anchor_ids = [candidate_names.index(str(row["anchor"])) for row in families]
        subgroup_anchor_ids = [
            [
                candidate_names.index(str(subgroup["anchor"]))
                if len(subgroup["routes"]) > 1
                else root_anchor_ids[family_index]
                for subgroup in family["subgroups"]
            ]
            for family_index, family in enumerate(families)
        ]
        anchors, feature_opponent_ids, seeds, panel_feature_names = (
            _load_feature_files(feature_paths)
        )
        required_anchor_ids = set(root_anchor_ids)
        required_anchor_ids.update(
            anchor for family_anchors in subgroup_anchor_ids for anchor in family_anchors
        )
        missing = sorted(required_anchor_ids.difference(anchors))
        if missing:
            raise ValueError(f"panel {panel_index} is missing anchors {missing}")
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
        train_seed_count = int(panel["train_seed_count"])
        if not 0 <= train_seed_count <= seeds.size:
            raise ValueError(f"panel {panel_index} has invalid train seed count")

        for selected_name_value in panel["opponents"]:
            selected_name = str(selected_name_value)
            compiled_id = bank_opponents.index(selected_name)
            feature_match = np.flatnonzero(feature_opponent_ids == compiled_id)
            outcome_match = np.flatnonzero(outcome_opponent_ids == compiled_id)
            if feature_match.size != 1 or outcome_match.size != 1:
                raise ValueError(f"panel {panel_index} is missing {selected_name}")
            feature_opponent = int(feature_match[0])
            outcome_opponent = int(outcome_match[0])

            root_values = anchors[root_anchor_ids[0]][120][
                :, feature_opponent
            ].reshape(-1, len(panel_feature_names))
            # All families genuinely share the same state through step 120.
            for anchor_id in root_anchor_ids[1:]:
                alternate = anchors[anchor_id][120][
                    :, feature_opponent
                ].reshape(-1, len(panel_feature_names))
                if not np.array_equal(root_values, alternate):
                    raise AssertionError("family roots differ before step 120")
            root_chunks.append(root_values)

            for family_index, root_anchor_id in enumerate(root_anchor_ids):
                family_values = anchors[root_anchor_id][168][
                    :, feature_opponent
                ].reshape(-1, len(panel_feature_names))
                family_chunks[family_index].append(family_values)
                for subgroup_index, subgroup_anchor_id in enumerate(
                    subgroup_anchor_ids[family_index]
                ):
                    subgroup_values = anchors[subgroup_anchor_id][216][
                        :, feature_opponent
                    ].reshape(-1, len(panel_feature_names))
                    subgroup_chunks[family_index][subgroup_index].append(
                        subgroup_values
                    )

            margin_chunks.append(
                panel_margins[route_ids, :, outcome_opponent, :].reshape(
                    len(route_names), -1
                )
            )
            if selected_name in opponent_names:
                opponent_id = opponent_names.index(selected_name)
            else:
                opponent_id = len(opponent_names)
                opponent_names.append(selected_name)
            opponent_chunks.append(
                np.full(root_values.shape[0], opponent_id, dtype=np.int16)
            )
            panel_chunks.append(
                np.full(root_values.shape[0], panel_index, dtype=np.int16)
            )
            group_chunks.append(
                np.tile(seeds, root_values.shape[0] // seeds.size)
                + panel_index * 1_000_000
            )
            train_chunks.append(
                np.tile(
                    np.arange(seeds.size) < train_seed_count,
                    root_values.shape[0] // seeds.size,
                )
            )
        source_rows.append(
            {
                "name": str(panel["name"]),
                "feature_files": [
                    {"path": str(path), "sha256": sha256(path)}
                    for path in feature_paths
                ],
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
    root_x = np.concatenate(root_chunks)
    family_x = [np.concatenate(chunks) for chunks in family_chunks]
    subgroup_x = [
        [np.concatenate(chunks) for chunks in family_chunks]
        for family_chunks in subgroup_chunks
    ]
    margins = np.concatenate(margin_chunks, axis=1)
    opponent_index = np.concatenate(opponent_chunks)
    panel_index_values = np.concatenate(panel_chunks)
    event_groups = np.concatenate(group_chunks)
    train_mask = np.concatenate(train_chunks)
    blind_mask = ~train_mask
    if not np.any(train_mask) or not np.any(blind_mask):
        raise ValueError("manifest must provide both training and blind games")
    train_indices = np.flatnonzero(train_mask)
    weights = balanced_weights(opponent_index[train_mask])
    train_groups = event_groups[train_mask]
    train_margins = margins[:, train_mask]

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
    cv_rows: list[dict[str, Any]] = []
    best: tuple[Any, ...] | None = None

    for spec_index, spec in enumerate(specs):
        n_train = train_indices.size
        root_oof = np.zeros((len(families), n_train), dtype=np.float32)
        subgroup_oof = [
            np.zeros((len(subgroups), n_train), dtype=np.float32)
            for subgroups in subgroup_route_indices
        ]
        route_oof = [
            [np.zeros((len(routes), n_train), dtype=np.float32) for routes in subgroups]
            for subgroups in subgroup_route_indices
        ]
        for fold, (fit_local, validation_local) in enumerate(
            splitter.split(root_x[train_mask], groups=train_groups)
        ):
            for family_index, family_routes in enumerate(family_route_indices):
                root_model = make_model(
                    spec, 81000 + spec_index * 4000 + fold * 400 + family_index
                )
                root_model.fit(
                    root_x[train_indices[fit_local]],
                    np.clip(
                        np.max(train_margins[family_routes][:, fit_local], axis=0),
                        -20000.0,
                        20000.0,
                    ),
                    sample_weight=weights[fit_local],
                )
                root_oof[family_index, validation_local] = root_model.predict(
                    root_x[train_indices[validation_local]]
                )

                subgroups = subgroup_route_indices[family_index]
                if len(subgroups) > 1:
                    for subgroup_index, subgroup_routes in enumerate(subgroups):
                        subgroup_model = make_model(
                            spec,
                            91000
                            + spec_index * 5000
                            + fold * 500
                            + family_index * 20
                            + subgroup_index,
                        )
                        subgroup_model.fit(
                            family_x[family_index][train_indices[fit_local]],
                            np.clip(
                                np.max(
                                    train_margins[subgroup_routes][:, fit_local], axis=0
                                ),
                                -20000.0,
                                20000.0,
                            ),
                            sample_weight=weights[fit_local],
                        )
                        subgroup_oof[family_index][
                            subgroup_index, validation_local
                        ] = subgroup_model.predict(
                            family_x[family_index][train_indices[validation_local]]
                        )

                for subgroup_index, subgroup_routes in enumerate(subgroups):
                    if len(subgroup_routes) == 1:
                        continue
                    for local_route, route_index in enumerate(subgroup_routes):
                        route_model = make_model(
                            spec,
                            101000
                            + spec_index * 7000
                            + fold * 700
                            + route_index,
                        )
                        route_model.fit(
                            subgroup_x[family_index][subgroup_index][
                                train_indices[fit_local]
                            ],
                            np.clip(
                                train_margins[route_index, fit_local],
                                -20000.0,
                                20000.0,
                            ),
                            sample_weight=weights[fit_local],
                        )
                        route_oof[family_index][subgroup_index][
                            local_route, validation_local
                        ] = route_model.predict(
                            subgroup_x[family_index][subgroup_index][
                                train_indices[validation_local]
                            ]
                        )

        family_choice, subgroup_choice, route_choice = choose(
            root_oof, subgroup_oof, route_oof, subgroup_route_indices
        )
        selected_margin = train_margins[route_choice, np.arange(n_train)]
        summary = summarize(selected_margin, opponent_index[train_mask], opponent_names)
        row = {
            "model": spec["name"],
            "family_count_used": int(np.unique(family_choice).size),
            "route_count_used": int(np.unique(route_choice).size),
            **{key: value for key, value in summary.items() if key != "per_opponent"},
        }
        cv_rows.append(row)
        key = score_key(summary)
        if best is None or key > best[0]:
            best = (key, spec, root_oof, subgroup_oof, route_oof)
    assert best is not None
    _, selected_spec, root_oof, subgroup_oof, route_oof = best

    blind_indices = np.flatnonzero(blind_mask)
    n_blind = blind_indices.size
    root_blind = np.zeros((len(families), n_blind), dtype=np.float32)
    subgroup_blind = [
        np.zeros((len(subgroups), n_blind), dtype=np.float32)
        for subgroups in subgroup_route_indices
    ]
    route_blind = [
        [np.zeros((len(routes), n_blind), dtype=np.float32) for routes in subgroups]
        for subgroups in subgroup_route_indices
    ]
    for family_index, family_routes in enumerate(family_route_indices):
        root_model = make_model(selected_spec, 111000 + family_index)
        root_model.fit(
            root_x[train_mask],
            np.clip(np.max(train_margins[family_routes], axis=0), -20000.0, 20000.0),
            sample_weight=weights,
        )
        root_blind[family_index] = root_model.predict(root_x[blind_mask])
        subgroups = subgroup_route_indices[family_index]
        if len(subgroups) > 1:
            for subgroup_index, subgroup_routes in enumerate(subgroups):
                subgroup_model = make_model(
                    selected_spec, 121000 + family_index * 20 + subgroup_index
                )
                subgroup_model.fit(
                    family_x[family_index][train_mask],
                    np.clip(
                        np.max(train_margins[subgroup_routes], axis=0),
                        -20000.0,
                        20000.0,
                    ),
                    sample_weight=weights,
                )
                subgroup_blind[family_index][subgroup_index] = subgroup_model.predict(
                    family_x[family_index][blind_mask]
                )
        for subgroup_index, subgroup_routes in enumerate(subgroups):
            if len(subgroup_routes) == 1:
                continue
            for local_route, route_index in enumerate(subgroup_routes):
                route_model = make_model(selected_spec, 131000 + route_index)
                route_model.fit(
                    subgroup_x[family_index][subgroup_index][train_mask],
                    np.clip(train_margins[route_index], -20000.0, 20000.0),
                    sample_weight=weights,
                )
                route_blind[family_index][subgroup_index][local_route] = (
                    route_model.predict(
                        subgroup_x[family_index][subgroup_index][blind_mask]
                    )
                )

    def partition(
        mask: np.ndarray,
        root_predictions: np.ndarray,
        subgroup_predictions: list[np.ndarray],
        concrete_predictions: list[list[np.ndarray]],
    ) -> dict[str, Any]:
        partition_margins = margins[:, mask]
        family_choice, subgroup_choice, route_choice = choose(
            root_predictions,
            subgroup_predictions,
            concrete_predictions,
            subgroup_route_indices,
        )
        selected_margin = partition_margins[route_choice, np.arange(route_choice.size)]
        partition_opponents = opponent_index[mask]
        partition_panels = panel_index_values[mask]
        subgroup_frequency = []
        for family_index, names in enumerate(subgroup_names):
            family_mask = family_choice == family_index
            for subgroup_index, name in enumerate(names):
                count = int(
                    np.sum(family_mask & (subgroup_choice == subgroup_index))
                )
                if count:
                    subgroup_frequency.append(
                        {
                            "family": family_names[family_index],
                            "subgroup": name,
                            "games": count,
                            "fraction": float(count / route_choice.size),
                        }
                    )
        per_panel = {}
        for panel_id, panel in enumerate(manifest["panels"]):
            local = partition_panels == panel_id
            if not np.any(local):
                continue
            per_panel[str(panel["name"])] = {
                "games": int(np.sum(local)),
                "fixed_baseline": summarize(
                    partition_margins[baseline_index, local],
                    partition_opponents[local],
                    opponent_names,
                ),
                "hierarchical_router": summarize(
                    selected_margin[local],
                    partition_opponents[local],
                    opponent_names,
                ),
                "hindsight_oracle": summarize(
                    np.max(partition_margins[:, local], axis=0),
                    partition_opponents[local],
                    opponent_names,
                ),
            }
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
            "subgroup_frequency": subgroup_frequency,
            "route_frequency": route_frequency(route_choice, route_names),
            "per_panel": per_panel,
        }

    train_result = partition(
        train_mask, root_oof, subgroup_oof, route_oof
    )
    blind_result = partition(
        blind_mask, root_blind, subgroup_blind, route_blind
    )
    train_family, train_subgroup, train_route = choose(
        root_oof, subgroup_oof, route_oof, subgroup_route_indices
    )
    blind_family, blind_subgroup, blind_route = choose(
        root_blind, subgroup_blind, route_blind, subgroup_route_indices
    )
    full_family = np.empty(margins.shape[1], dtype=np.int16)
    full_subgroup = np.empty(margins.shape[1], dtype=np.int16)
    full_route = np.empty(margins.shape[1], dtype=np.int16)
    full_family[train_mask], full_family[blind_mask] = train_family, blind_family
    full_subgroup[train_mask], full_subgroup[blind_mask] = (
        train_subgroup,
        blind_subgroup,
    )
    full_route[train_mask], full_route[blind_mask] = train_route, blind_route
    full_selected_margin = margins[full_route, np.arange(margins.shape[1])]
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            predictions_path,
            train_mask=train_mask,
            family_choice=full_family,
            subgroup_choice=full_subgroup,
            route_choice=full_route,
            selected_margin=full_selected_margin,
            opponent_index=opponent_index,
            panel_index=panel_index_values,
            event_groups=event_groups,
            route_names=np.asarray(route_names),
            family_names=np.asarray(family_names),
        )
    result = {
        "schema": "kaggriculture-step120-step168-step216-hierarchical-router-fit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "JAX_SCREEN_ONLY",
        "decision_steps": [
            int(hierarchy["root_step"]),
            int(hierarchy["subgroup_step"]),
            int(hierarchy["route_step"]),
        ],
        "selected_model": selected_spec,
        "route_count": len(route_names),
        "family_count": len(families),
        "subgroup_count": int(sum(len(row["subgroups"]) for row in families)),
        "hierarchy": families,
        "baseline_route": route_names[baseline_index],
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
            "panel_manifest": str(manifest_path),
            "panel_manifest_sha256": sha256(manifest_path),
            "hierarchy": str(hierarchy_path),
            "hierarchy_sha256": sha256(hierarchy_path),
            "panels": source_rows,
        },
        "truth_boundary": (
            "Offline JAX screening of a prefix-safe three-stage policy. No "
            "deployable Agent has been generated; independent JAX and official "
            "Python 1.32.7 holdouts remain mandatory."
        ),
    }
    if predictions_path is not None:
        result["predictions_output"] = str(predictions_path)
        result["predictions_output_sha256"] = sha256(predictions_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_model": selected_spec["name"],
                "train_oof": train_result,
                "blind": blind_result,
                "receipt": str(receipt_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

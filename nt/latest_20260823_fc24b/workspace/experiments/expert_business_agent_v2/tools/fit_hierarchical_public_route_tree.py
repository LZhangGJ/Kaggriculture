#!/usr/bin/env python3
"""Fit prefix-safe hierarchical public-state route trees from official games."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gold_adaptive_rule_v2" / "tools"))
from fit_public_route_tree import public_features, summarize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def parse_seeds(raw: str) -> set[int]:
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def utility(row: dict[str, Any], margin_scale: float) -> float:
    margin = float(np.clip(float(row["margin"]), -100000.0, 100000.0))
    return float(bool(row["win"])) + margin_scale * margin


def tree_payload(model: DecisionTreeRegressor, choices: list[str], feature_names: list[str]) -> dict[str, Any]:
    tree = model.tree_
    node_choice = [choices[int(np.argmax(tree.value[node].reshape(-1)))] for node in range(tree.node_count)]
    return {
        "choices": choices,
        "tree_text": export_text(model, feature_names=feature_names),
        "children_left": tree.children_left.astype(int).tolist(),
        "children_right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "node_choice": node_choice,
    }


def predict_choice(model: DecisionTreeRegressor, choices: list[str], values: list[float]) -> str:
    prediction = np.asarray(model.predict(np.asarray([values], dtype=np.float64))).reshape(-1)
    return choices[int(np.argmax(prediction))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--validation-seeds", default="")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--margin-scale", type=float, default=0.000001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    family_spec = json.loads(args.families.read_text(encoding="utf-8"))
    families = list(family_spec["families"])
    root_step = int(family_spec["root_step"])
    allowed_routes = {route for family in families for route in family["routes"]}
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in matrix["rows"]:
        route = str(row["candidate"])
        if route not in allowed_routes:
            continue
        key = (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        grouped[key][route] = row
    missing = {
        str(key): sorted(allowed_routes - set(rows))
        for key, rows in grouped.items()
        if set(rows) != allowed_routes
    }
    if missing:
        raise ValueError(f"matrix is missing routes for {len(missing)} cases")

    case_keys = sorted(grouped)
    validation_seeds = parse_seeds(args.validation_seeds)
    training_indices = [i for i, key in enumerate(case_keys) if key[1] not in validation_seeds]
    validation_indices = [i for i, key in enumerate(case_keys) if key[1] in validation_seeds]
    if not training_indices:
        raise ValueError("validation seeds consume every case")
    if validation_seeds and not validation_indices:
        raise ValueError("requested validation seeds do not exist")

    feature_names: list[str] | None = None
    feature_cache: dict[tuple[int, int], list[float]] = {}
    steps = {root_step} | {int(family["child_step"]) for family in families if "child_step" in family}
    for case_index, key in enumerate(case_keys):
        rows = grouped[key]
        for step in steps:
            relevant_routes = list(allowed_routes)
            for family in families:
                if int(family.get("child_step", -1)) == step:
                    relevant_routes = list(family["routes"])
                    break
            snapshots = [rows[route]["public_state_steps"][str(step)] for route in relevant_routes]
            signatures = {json.dumps(value, sort_keys=True, separators=(",", ":")) for value in snapshots}
            if len(signatures) != 1:
                raise AssertionError(f"public prefix mismatch at step {step} for {key}")
            names, values = public_features(snapshots[0])
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise AssertionError("feature schema changed")
            feature_cache[(case_index, step)] = values
    if feature_names is None:
        raise ValueError("no cases")

    family_names = [str(family["name"]) for family in families]
    root_y = np.asarray([
        [
            max(utility(grouped[key][route], args.margin_scale) for route in family["routes"])
            for family in families
        ]
        for key in case_keys
    ], dtype=np.float64)
    root_x = np.asarray([feature_cache[(i, root_step)] for i in range(len(case_keys))], dtype=np.float64)
    root_model = DecisionTreeRegressor(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=20260817,
    )
    root_model.fit(root_x[training_indices], root_y[training_indices])

    child_models: dict[str, DecisionTreeRegressor] = {}
    child_payloads: dict[str, Any] = {}
    for family in families:
        routes = [str(route) for route in family["routes"]]
        if len(routes) == 1:
            continue
        step = int(family["child_step"])
        x = np.asarray([feature_cache[(i, step)] for i in range(len(case_keys))], dtype=np.float64)
        y = np.asarray([
            [utility(grouped[key][route], args.margin_scale) for route in routes]
            for key in case_keys
        ], dtype=np.float64)
        model = DecisionTreeRegressor(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=20260817 + step,
        )
        model.fit(x[training_indices], y[training_indices])
        child_models[str(family["name"])] = model
        child_payloads[str(family["name"])] = {
            "decision_step": step,
            **tree_payload(model, routes, feature_names),
        }

    def select_route(case_index: int) -> str:
        root_values = feature_cache[(case_index, root_step)]
        family_name = predict_choice(root_model, family_names, root_values)
        family = next(value for value in families if value["name"] == family_name)
        routes = [str(route) for route in family["routes"]]
        if len(routes) == 1:
            return routes[0]
        step = int(family["child_step"])
        return predict_choice(child_models[family_name], routes, feature_cache[(case_index, step)])

    selected_routes = [select_route(i) for i in range(len(case_keys))]

    def metrics(indices: list[int]) -> dict[str, Any] | None:
        if not indices:
            return None
        selected_rows = [grouped[case_keys[i]][selected_routes[i]] for i in indices]
        oracle_rows = [
            max(grouped[case_keys[i]].values(), key=lambda row: (bool(row["win"]), float(row["margin"])))
            for i in indices
        ]
        return {
            "case_count": len(indices),
            "seeds": sorted({case_keys[i][1] for i in indices}),
            "selected": summarize(selected_rows),
            "oracle": summarize(oracle_rows),
            "route_counts": {
                route: sum(selected_routes[i] == route for i in indices)
                for route in sorted(allowed_routes)
            },
        }

    output = {
        "schema": "hierarchical-public-route-tree-v1",
        "source_matrix": str(args.matrix),
        "source_matrix_sha256": sha256(args.matrix),
        "family_spec": str(args.families),
        "family_spec_sha256": sha256(args.families),
        "runtime_forbidden_inputs": ["opponent identity", "seed", "future events", "private opponent state", "reward"],
        "case_count": len(case_keys),
        "training_case_count": len(training_indices),
        "validation_case_count": len(validation_indices),
        "validation_seeds": sorted(validation_seeds),
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "margin_scale": args.margin_scale,
        "feature_names": feature_names,
        "root": {
            "decision_step": root_step,
            **tree_payload(root_model, family_names, feature_names),
        },
        "families": families,
        "children": child_payloads,
        "training": metrics(training_indices),
        "validation": metrics(validation_indices),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "case_count": output["case_count"],
        "training": output["training"],
        "validation": output["validation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose V1 route regret and exact feature aliasing on the opened fresh panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ROUTES = ("10C-4S-75L", "8C-6S-75L", "6C-12S-100L", "6C-8S-75L")


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def metrics(contexts: list[dict[str, Any]], routes: list[str]) -> dict[str, Any]:
    selected = [context["outcomes"][route] for context, route in zip(contexts, routes)]
    return {
        "games": len(selected),
        "wins": sum(bool(row["win"]) for row in selected),
        "win_rate": sum(bool(row["win"]) for row in selected) / len(selected),
        "mean_margin": sum(float(row["margin"]) for row in selected) / len(selected),
        "route_counts": dict(Counter(routes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counterfactual", type=Path, required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel_path = resolve(args.counterfactual)
    arena_path = resolve(args.arena)
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    arena = json.loads(arena_path.read_text(encoding="utf-8"))
    rows_by_context: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in panel["rows"]:
        key = (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        rows_by_context[key][str(row["candidate"])] = row
    choice_by_context = {}
    for row in arena["rows"]:
        if row["candidate"] != "counterfactual_lgbm_v1":
            continue
        key = (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        choice_by_context[key] = str(row["candidate_route_probe"]["route"])
    contexts = []
    for key in sorted(rows_by_context):
        by_route = rows_by_context[key]
        reference = by_route[ROUTES[0]]
        outcomes = {
            route: {
                "utility": float(by_route[route]["utility"]),
                "win": bool(by_route[route]["win"]),
                "margin": float(by_route[route]["margin"]),
            }
            for route in ROUTES
        }
        best = max(ROUTES, key=lambda route: (outcomes[route]["utility"], outcomes[route]["margin"]))
        contexts.append({
            "key": key,
            "features": tuple(float(value) for value in reference["features"]),
            "fallback": str(reference["fallback_route"]),
            "selected": choice_by_context[key],
            "best": best,
            "outcomes": outcomes,
        })
    fallback_routes = [context["fallback"] for context in contexts]
    model_routes = [context["selected"] for context in contexts]
    oracle_routes = [context["best"] for context in contexts]
    feature_groups: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for index, context in enumerate(contexts):
        feature_groups[context["features"]].append(index)
    feature_oracle_routes = ["" for _ in contexts]
    conflicting_groups = []
    for indices in feature_groups.values():
        selected = max(
            ROUTES,
            key=lambda route: (
                sum(contexts[index]["outcomes"][route]["win"] for index in indices),
                sum(contexts[index]["outcomes"][route]["margin"] for index in indices),
            ),
        )
        for index in indices:
            feature_oracle_routes[index] = selected
        labels = Counter(contexts[index]["best"] for index in indices)
        if len(labels) > 1:
            conflicting_groups.append({
                "contexts": len(indices),
                "best_route_counts": dict(labels),
                "feature_oracle_route": selected,
                "keys": [list(contexts[index]["key"]) for index in indices],
            })
    confusion = Counter(f"{context['selected']}->{context['best']}" for context in contexts if context["selected"] != context["best"])
    result = {
        "schema": "kawashigi-counterfactual-fresh-feature-aliasing-v1",
        "truth_boundary": "post-result diagnosis only; forbidden for V1 retuning or promotion",
        "counterfactual_panel": str(panel_path),
        "counterfactual_panel_sha256": sha256(panel_path),
        "arena": str(arena_path),
        "arena_sha256": sha256(arena_path),
        "context_count": len(contexts),
        "unique_exact_feature_vectors": len(feature_groups),
        "duplicate_feature_contexts": sum(len(indices) for indices in feature_groups.values() if len(indices) > 1),
        "conflicting_feature_groups": len(conflicting_groups),
        "contexts_in_conflicting_groups": sum(row["contexts"] for row in conflicting_groups),
        "fallback": metrics(contexts, fallback_routes),
        "model_v1": metrics(contexts, model_routes),
        "context_oracle": metrics(contexts, oracle_routes),
        "exact_feature_oracle": metrics(contexts, feature_oracle_routes),
        "model_exact_best_route_accuracy": sum(left == right for left, right in zip(model_routes, oracle_routes)) / len(contexts),
        "fallback_exact_best_route_accuracy": sum(left == right for left, right in zip(fallback_routes, oracle_routes)) / len(contexts),
        "model_to_best_confusion": dict(confusion),
        "conflicting_groups": conflicting_groups,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("context_count", "unique_exact_feature_vectors", "duplicate_feature_contexts", "conflicting_feature_groups", "contexts_in_conflicting_groups", "fallback", "model_v1", "context_oracle", "exact_feature_oracle", "model_exact_best_route_accuracy", "fallback_exact_best_route_accuracy", "model_to_best_confusion")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

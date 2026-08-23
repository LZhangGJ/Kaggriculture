#!/usr/bin/env python3
"""Audit portable Agent predictions against the frozen LightGBM dump."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("kcr_agent_parity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def tree_value(node: dict[str, Any], values: list[float]) -> float:
    while "leaf_value" not in node:
        feature = int(node["split_feature"])
        node = node["left_child"] if values[feature] <= float(node["threshold"]) else node["right_child"]
    return float(node["leaf_value"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    agent_path = args.agent.resolve() if args.agent.is_absolute() else (ROOT / args.agent).resolve()
    model_path = args.model.resolve() if args.model.is_absolute() else (ROOT / args.model).resolve()
    panel_path = args.panel.resolve() if args.panel.is_absolute() else (ROOT / args.panel).resolve()
    module = load_module(agent_path)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    probes = []
    max_error = 0.0
    mismatch = 0
    grouped = {}
    for row in panel["rows"]:
        key = (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        grouped.setdefault(key, row)
    for key, source_row in sorted(grouped.items()):
        values = [float(value) for value in source_row["features"]]
        reference_scores = []
        for route in model["routes"]:
            reference_scores.append(sum(tree_value(tree["tree_structure"], values) for tree in model["models"][route]["tree_info"]))
        agent_scores = [float(value) for value in module._kcr_scores(values)]
        error = max(abs(left - right) for left, right in zip(reference_scores, agent_scores))
        max_error = max(max_error, error)
        fallback = str(source_row["fallback_route"])
        best = model["routes"][max(range(len(reference_scores)), key=lambda index: reference_scores[index])]
        fallback_index = model["routes"].index(fallback)
        best_index = model["routes"].index(best)
        expected = best if reference_scores[best_index] - reference_scores[fallback_index] >= module._KCR_THRESHOLD else fallback
        actual = module._kcr_choose_from_features(values, fallback)
        mismatch += int(expected != actual)
        probes.append({"context": list(key), "fallback": fallback, "expected": expected, "actual": actual, "max_score_error": error})
    passed = max_error <= 1e-8 and mismatch == 0 and hasattr(module, "kaggriculture_kawashigi_counterfactual_lgbm")
    result = {
        "schema": "kawashigi-counterfactual-agent-model-parity-v1",
        "status": "PASS" if passed else "FAIL",
        "contexts": len(probes),
        "max_abs_score_error": max_error,
        "route_mismatches": mismatch,
        "unique_last_entrypoint": hasattr(module, "kaggriculture_kawashigi_counterfactual_lgbm"),
        "agent": str(agent_path),
        "agent_sha256": sha256(agent_path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "panel": str(panel_path),
        "panel_sha256": sha256(panel_path),
        "probes": probes,
    }
    output = args.output.resolve() if args.output.is_absolute() else (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "contexts", "max_abs_score_error", "route_mismatches", "unique_last_entrypoint")}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

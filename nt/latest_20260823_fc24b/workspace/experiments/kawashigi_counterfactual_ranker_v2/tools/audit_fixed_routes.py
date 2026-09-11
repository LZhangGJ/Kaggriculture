#!/usr/bin/env python3
"""Audit the frozen KAWASHIGI route set before official counterfactual runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/gold_adaptive_rule_v2/agents/gold_imitations_current_20260816_0955_v2/rank01_team/main.py"
MANIFEST = ROOT / "experiments/kawashigi_counterfactual_ranker_v2/configs/fixed_routes_v1.json"
ROUTES = ("10C-4S-75L", "8C-6S-75L", "6C-12S-100L", "6C-8S-75L")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load_module(SOURCE, "counterfactual_route_source")
    streams = source._CGR_STREAMS
    if tuple(streams) != ROUTES:
        raise AssertionError(tuple(streams))
    prefix_equal_through_144 = all(
        streams[route][:145] == streams[ROUTES[0]][:145] for route in ROUTES
    )
    divergent_at_145 = len({json.dumps(streams[route][145], sort_keys=True) for route in ROUTES}) > 1
    first_global_divergence = next(
        step
        for step in range(719)
        if len({json.dumps(streams[route][step], sort_keys=True) for route in ROUTES}) > 1
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = []
    for index, item in enumerate(manifest["candidates"]):
        path = (ROOT / item["path"]).resolve()
        module = load_module(path, f"fixed_route_{index}")
        files.append({
            "name": item["name"],
            "path": str(path),
            "sha256": sha256(path),
            "fixed_route": str(module._CGR_FIXED_ROUTE),
            "stream_length": len(module._CGR_STREAMS[item["name"]]),
            "has_step_recovery": hasattr(module, "_cgr_fixed_with_step"),
            "has_unique_last_entrypoint": hasattr(module, "kaggriculture_fixed_current_gold_route"),
        })
    passed = bool(
        prefix_equal_through_144
        and divergent_at_145
        and first_global_divergence == 145
        and all(
            row["name"] == row["fixed_route"]
            and row["stream_length"] == 719
            and row["has_step_recovery"]
            and row["has_unique_last_entrypoint"]
            for row in files
        )
    )
    result = {
        "schema": "kawashigi-counterfactual-fixed-route-audit-v1",
        "status": "PASS" if passed else "FAIL",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "routes": list(ROUTES),
        "stream_lengths": {route: len(streams[route]) for route in ROUTES},
        "prefix_equal_steps_0_to_144": prefix_equal_through_144,
        "first_global_divergence_step": first_global_divergence,
        "divergent_actions_at_step145": divergent_at_145,
        "fixed_agents": files,
    }
    output = args.output.resolve() if args.output.is_absolute() else (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

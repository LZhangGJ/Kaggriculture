#!/usr/bin/env python3
"""Add a narrow public shop-state override to an existing EBA21 router."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


TAIL = r'''
# --- Public shop-state Route18 override (generated) ---
_Y18_BASE_CHOOSE_ROUTE = _s120_choose_route
_Y18_ROUTE = __ROUTE__
_Y18_REQUIRED_SHOP = "YARN_STORE"
_Y18_EXCLUDED_SHOP = "ICE_CREAM_SHOP"


def _s120_choose_route(obs):
    route, predictions = _Y18_BASE_CHOOSE_ROUTE(obs)
    town = _get(obs, "town", {}) or {}
    shops = {
        str(value)
        for value in list(_get(town, "unlocked_shops", []) or [])
    }
    if _Y18_REQUIRED_SHOP in shops and _Y18_EXCLUDED_SHOP not in shops:
        route = _Y18_ROUTE
    return route, predictions


def kaggriculture_step168_yarn_route18_override(obs, configuration=None):
    return agent(obs, configuration)


__version__ = __VERSION__
'''


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("yarn_override_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-agent", type=Path, required=True)
    parser.add_argument("--output-agent", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--route", default="route18_rank06_6c_12s_100l")
    parser.add_argument("--agent-version", default="eba26-step168-yarn-route18-override-v1")
    args = parser.parse_args()

    base = resolve(args.base_agent)
    output = resolve(args.output_agent)
    receipt = resolve(args.receipt)
    module = load_module(base)
    if int(getattr(module, "_S120_DECISION_STEP", -1)) != 168:
        raise ValueError("base agent is not a step-168 router")
    route_names = tuple(getattr(module, "_S120_ROUTE_NAMES", ()))
    if args.route not in route_names:
        raise ValueError(f"Route18 override target is absent: {args.route}")

    tail = TAIL.replace("__ROUTE__", repr(args.route)).replace(
        "__VERSION__", repr(args.agent_version)
    )
    source = base.read_text(encoding="utf-8").rstrip() + "\n" + tail.lstrip()
    compile(source, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8", newline="\n")

    result = {
        "schema": "kaggriculture-step168-yarn-route18-override-build-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT_OFFICIAL_VALIDATION_REQUIRED",
        "agent_version": args.agent_version,
        "decision_step": 168,
        "override_route": args.route,
        "condition": {
            "required_public_shop": "YARN_STORE",
            "excluded_public_shop": "ICE_CREAM_SHOP",
        },
        "runtime_inputs": ["public unlocked shops"],
        "forbidden_runtime_inputs": ["opponent identity", "seed", "future events", "terminal outcome", "private opponent state"],
        "base_agent": str(base),
        "base_agent_sha256": sha256(base),
        "output_agent": str(output),
        "output_agent_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "truth_boundary": "Public-state candidate; independent official Python 1.32.7 holdout is mandatory.",
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

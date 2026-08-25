from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GOLD_TOOLS = ROOT / "experiments" / "gold_adaptive_rule_v2" / "tools"
sys.path.insert(0, str(GOLD_TOOLS))

from build_current_gold_gpu_trace_bank import FIELDS, _encode_stream  # noqa: E402


SOURCE = (
    ROOT
    / "public_notebooks"
    / "recent_latest_20260825_scan_v1"
    / "kaitofukami__40-40-early-floor-39-46-top-10-v48-fast-routes"
    / "output"
    / "main.py"
)
BASE_BANK = (
    ROOT
    / "experiments"
    / "expert_business_agent_v2"
    / "artifacts"
    / "latest_public6_20260822_route_bank_v1.npz"
)
BASE_RECEIPT = (
    ROOT
    / "experiments"
    / "expert_business_agent_v2"
    / "receipts"
    / "latest_public6_20260822_route_bank_v1.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_action_sha256(stream: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        stream[:719], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("recent_kaito_v48_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    module = load_module(SOURCE)
    source_routes = dict(module._V48_ROUTES)
    expected = (
        "default",
        "yarn_fast",
        "farm_fast",
        "yarn_second",
        "yarn_third",
        "bakery_capital",
    )
    if tuple(source_routes) != expected:
        raise ValueError(f"unexpected V48 route keys: {tuple(source_routes)}")

    with np.load(BASE_BANK, allow_pickle=False) as data:
        arrays = {
            field: [row.copy() for row in np.asarray(data[field])] for field in FIELDS
        }
    base_receipt = json.loads(BASE_RECEIPT.read_text(encoding="utf-8"))
    routes = [dict(row) for row in base_receipt["routes"]]
    by_hash = {row["action_sha256"]: int(row["route_id"]) for row in routes}
    aliases = []

    for name in expected:
        stream = list(source_routes[name])
        if len(stream) != 719:
            raise ValueError(f"{name}: expected 719 actions, got {len(stream)}")
        action_sha = canonical_action_sha256(stream)
        is_new = action_sha not in by_hash
        if is_new:
            route_id = len(routes)
            by_hash[action_sha] = route_id
            encoded = _encode_stream(stream)
            for field in FIELDS:
                arrays[field].append(encoded[field])
            routes.append(
                {
                    "route_id": route_id,
                    "canonical_name": f"kaito_v48__{name}",
                    "action_sha256": action_sha,
                }
            )
        route_id = by_hash[action_sha]
        aliases.append(
            {
                "alias": f"kaito_v48__{name}",
                "route_name": name,
                "route_id": route_id,
                "action_sha256": action_sha,
                "new_unique_route": is_new,
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, **{field: np.stack(values, axis=0) for field, values in arrays.items()}
    )
    receipt = {
        "schema": "kaggriculture.public-recent-20260825.kaito-v48-route-bank.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "base_bank": str(BASE_BANK.resolve()),
        "base_bank_sha256": sha256(BASE_BANK),
        "base_routes": len(base_receipt["routes"]),
        "unique_routes": len(routes),
        "aliases": aliases,
        "routes": routes,
        "bank": str(output),
        "bank_sha256": sha256(output),
        "truth_boundary": (
            "This freezes only the six public raw 719-step V48 routes. "
            "Dynamic routing and overlays require separate official/JAX stepwise parity."
        ),
    }
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "base_routes": receipt["base_routes"],
                "unique_routes": receipt["unique_routes"],
                "aliases": aliases,
                "bank": str(output),
                "receipt": str(receipt_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

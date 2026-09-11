from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GOLD_TOOLS = ROOT / "experiments" / "gold_adaptive_rule_v2" / "tools"
sys.path.insert(0, str(GOLD_TOOLS))

from build_current_gold_gpu_trace_bank import FIELDS, _encode_stream  # noqa: E402


LATEST = ROOT / "references" / "public_latest6_20260822"
BASE_BANK = (
    ROOT
    / "experiments"
    / "expert_business_agent_v2"
    / "artifacts"
    / "latest_public8_route_bank_v1.npz"
)
BASE_RECEIPT = (
    ROOT
    / "experiments"
    / "expert_business_agent_v2"
    / "receipts"
    / "latest_public8_route_bank_v1.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_action_sha256(stream: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        stream[:719], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_module(path: Path, tag: str):
    spec = importlib.util.spec_from_file_location(tag, path)
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

    manifest_path = LATEST / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_by_slug = {
        row["slug"]: LATEST / row["slug"] / "main.py" for row in manifest["agents"]
    }
    modules = {
        slug: load_module(path, "latest6_20260822_" + slug)
        for slug, path in source_by_slug.items()
    }

    with np.load(BASE_BANK, allow_pickle=False) as data:
        arrays = {
            field: [row.copy() for row in np.asarray(data[field])] for field in FIELDS
        }
    base_receipt = json.loads(BASE_RECEIPT.read_text(encoding="utf-8"))
    routes = [dict(row) for row in base_receipt["routes"]]
    by_hash = {row["action_sha256"]: int(row["route_id"]) for row in routes}
    aliases: list[dict[str, Any]] = []

    def add(alias: str, slug: str, attribute: str, stream: list[dict]) -> int:
        stream = list(stream[:719])
        if len(stream) != 719:
            raise ValueError(f"{alias}: expected 719 actions, got {len(stream)}")
        action_sha = canonical_action_sha256(stream)
        if action_sha not in by_hash:
            route_id = len(routes)
            by_hash[action_sha] = route_id
            encoded = _encode_stream(stream)
            for field in FIELDS:
                arrays[field].append(encoded[field])
            routes.append(
                {
                    "route_id": route_id,
                    "canonical_name": alias,
                    "action_sha256": action_sha,
                }
            )
        route_id = by_hash[action_sha]
        aliases.append(
            {
                "alias": alias,
                "slug": slug,
                "attribute": attribute,
                "route_id": route_id,
                "action_sha256": action_sha,
                "source": str(source_by_slug[slug].resolve()),
                "source_sha256": sha256(source_by_slug[slug]),
            }
        )
        return route_id

    shared = (
        ("10c4s_3q", "_ACTIONS_10C4S_3Q"),
        ("8c6s_3q", "_ACTIONS_8C6S_3Q"),
        ("6c8s_3q", "_ACTIONS_6C8S_3Q"),
        ("6c12s_4q_first_yarn", "_ACTIONS_6C12S_4Q_FIRST_YARN"),
        ("6c12s_4q_second_yarn", "_ACTIONS_6C12S_4Q_SECOND_YARN"),
        ("legacy_10c4s_3q", "_LEGACY_ACTIONS_10C4S_3Q"),
        ("legacy_8c6s_3q", "_LEGACY_ACTIONS_8C6S_3Q"),
        ("legacy_6c8s_3q", "_LEGACY_ACTIONS_6C8S_3Q"),
        ("legacy_6c12s_4q_first_yarn", "_LEGACY_ACTIONS_6C12S_4Q_FIRST_YARN"),
        ("legacy_6c12s_4q_second_yarn", "_LEGACY_ACTIONS_6C12S_4Q_SECOND_YARN"),
    )

    moon = modules["prvsiyan_moon_v92_latest"]
    for route_name, attribute in shared:
        add(
            f"prvsiyan_moon_v92_latest__{route_name}",
            "prvsiyan_moon_v92_latest",
            attribute,
            getattr(moon, attribute),
        )

    boatlee = modules["boatlee_v21_latest"]
    for route_name, attribute in shared:
        add(
            f"boatlee_v21_latest__moon__{route_name}",
            "boatlee_v21_latest",
            f"_MOON.{attribute}",
            getattr(boatlee._MOON, attribute),
        )
    for child_name, child in (
        ("mutoy", boatlee._MUTOY),
        ("munib_base", boatlee._MUNIB_BASE),
        ("munib_front", boatlee._MUNIB_FR),
    ):
        add(
            f"boatlee_v21_latest__{child_name}",
            "boatlee_v21_latest",
            f"_{child_name.upper()}._ACTIONS",
            child._ACTIONS,
        )

    soil = modules["prvsiyan_soil_v26h_latest"]
    add(
        "prvsiyan_soil_v26h_latest__modal",
        "prvsiyan_soil_v26h_latest",
        "_MODAL_NS['_ROUTE']",
        soil._MODAL_NS["_ROUTE"],
    )
    add(
        "prvsiyan_soil_v26h_latest__rc5",
        "prvsiyan_soil_v26h_latest",
        "_RC5_NS['_ACTIONS']",
        soil._RC5_NS["_ACTIONS"],
    )
    kaito = modules["kaito_v39_history_gate_latest"]
    for alias, module, attribute in (
        ("base_v36", kaito._V39_BASE._V39_V36, "_V36_ROUTE"),
        ("base_v37", kaito._V39_BASE._V39_V37, "_V37_ROUTE"),
        ("wool_v37", kaito._V39_WOOL, "_V37_ROUTE"),
        ("alt_v37", kaito._V39_ALT, "_V37_ROUTE"),
    ):
        add(
            f"kaito_v39_history_gate_latest__{alias}",
            "kaito_v39_history_gate_latest",
            attribute,
            getattr(module, attribute),
        )

    for slug in ("steven_e284_hadouken_latest", "salem_harvestforge_x_latest"):
        module = modules[slug]
        add(f"{slug}__low", slug, "_E279_LOW_ACTIONS", module._E279_LOW_ACTIONS)
        add(
            f"{slug}__high", slug, "_E279_HIGH_ACTIONS", module._E279_HIGH_ACTIONS
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, **{field: np.stack(values, axis=0) for field, values in arrays.items()}
    )
    receipt = {
        "schema": "kaggriculture.latest_public6_20260822_route_bank.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "base_bank": str(BASE_BANK.resolve()),
        "base_bank_sha256": sha256(BASE_BANK),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "base_routes": len(base_receipt["routes"]),
        "unique_routes": len(routes),
        "aliases": aliases,
        "routes": routes,
        "bank": str(output),
        "bank_sha256": sha256(output),
        "truth_boundary": (
            "This bank freezes raw 719-step action tapes only. Dynamic Python "
            "routing and overlays require separate controller parity acceptance."
        ),
    }
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "base_routes": receipt["base_routes"],
                "unique_routes": receipt["unique_routes"],
                "aliases": len(aliases),
                "bank": str(output),
                "receipt": str(receipt_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

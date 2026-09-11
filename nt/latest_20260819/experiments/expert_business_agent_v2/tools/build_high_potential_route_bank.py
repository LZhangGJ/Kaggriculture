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


FROZEN_ROOT = ROOT / "references" / "public_high_potential_20260819"


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


def route_sources() -> list[dict[str, Any]]:
    boatlee = FROZEN_ROOT / "boatlee_v20_multi_route" / "main.py"
    rayk = FROZEN_ROOT / "rayk_k320_adaptive_rank1" / "main.py"
    tetsutani = FROZEN_ROOT / "tetsutani_adaptive_premium_queue" / "main.py"
    kaito = FROZEN_ROOT / "kaito_v27_midgame_reset" / "main.py"
    flex = FROZEN_ROOT / "flexonafft_v59_multi_route" / "main.py"
    modules = {
        "boatlee_v20": (boatlee, load_module(boatlee, "hp_boatlee_v20")),
        "rayk_k320": (rayk, load_module(rayk, "hp_rayk_k320")),
        "tetsutani_adaptive": (tetsutani, load_module(tetsutani, "hp_tetsutani")),
        "kaito_v27": (kaito, load_module(kaito, "hp_kaito_v27")),
        "flexonafft_v59": (flex, load_module(flex, "hp_flex_v59")),
    }
    rows: list[dict[str, Any]] = []
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
    for family in ("boatlee_v20", "rayk_k320", "tetsutani_adaptive"):
        path, module = modules[family]
        for route_name, attribute in shared:
            rows.append(
                {
                    "family": family,
                    "route_name": route_name,
                    "attribute": attribute,
                    "source": path,
                    "stream": getattr(module, attribute),
                    "source_runtime_layers": (
                        "route selection + weed repair + feed/capacity guards + "
                        "sell ordering + market counters + terminal liquidation"
                    ),
                }
            )

    kaito_path, kaito_module = modules["kaito_v27"]
    for route_name, attribute in (
        ("legacy", "_LEGACY_ACTIONS"),
        ("rebalance", "_REBALANCE_ACTIONS"),
    ):
        rows.append(
            {
                "family": "kaito_v27",
                "route_name": route_name,
                "attribute": attribute,
                "source": kaito_path,
                "stream": getattr(kaito_module, attribute),
                "source_runtime_layers": "weed repair + sell ordering",
            }
        )

    flex_path, flex_module = modules["flexonafft_v59"]
    rows.append(
        {
            "family": "flexonafft_v59",
            "route_name": "modal_route",
            "attribute": "_MODAL_NS._ROUTE",
            "source": flex_path,
            "stream": flex_module._MODAL_NS["_ROUTE"],
            "source_runtime_layers": "RC5 weed repair",
        }
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the deduplicated JAX route bank from frozen public agents."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    by_hash: dict[str, int] = {}
    arrays: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
    routes: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for row in route_sources():
        stream = list(row.pop("stream")[:719])
        if len(stream) != 719:
            raise ValueError(
                f"{row['family']}/{row['route_name']}: expected 719 actions, got {len(stream)}"
            )
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
                    "canonical_name": f"{row['family']}__{row['route_name']}",
                    "action_sha256": action_sha,
                    "jax_screen_controller": "generic_rc5_weed_sell_rank_terminal",
                    "source_agent_exact": False,
                    "source_agent_exact_reason": (
                        "the raw 719-step production tape is exact, but source-specific "
                        "adaptive runtime overlays are not all applied by this screening controller"
                    ),
                }
            )
        route_id = by_hash[action_sha]
        source_path = Path(row["source"]).resolve()
        aliases.append(
            {
                "route_id": route_id,
                "family": row["family"],
                "route_name": row["route_name"],
                "attribute": row["attribute"],
                "source": str(source_path),
                "source_sha256": sha256(source_path),
                "source_runtime_layers": row["source_runtime_layers"],
                "action_sha256": action_sha,
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{field: np.stack(values, axis=0) for field, values in arrays.items()},
    )
    receipt = {
        "schema": "kaggriculture.high_potential_route_bank.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_public_root": str(FROZEN_ROOT.resolve()),
        "input_aliases": len(aliases),
        "unique_routes": len(routes),
        "bank": str(output),
        "bank_sha256": sha256(output),
        "truth_boundary": (
            "Exact raw public production tapes, deduplicated by canonical action SHA256. "
            "GPU screening uses the shared weed/sell-rank/terminal controller and is not "
            "yet exact parity for every source agent's adaptive overlays."
        ),
        "routes": routes,
        "aliases": aliases,
    }
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "bank": str(output),
                "bank_sha256": receipt["bank_sha256"],
                "input_aliases": len(aliases),
                "unique_routes": len(routes),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

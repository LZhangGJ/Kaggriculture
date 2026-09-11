#!/usr/bin/env python3
"""Build a standalone stdlib Agent from a prefix-safe step-120 margin router."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any
import zlib


ROOT = Path(__file__).resolve().parents[3]


TAIL = r'''
# --- Prefix-safe public-state terminal-margin router (generated) ---
__PREEMPT_OVERRIDE__
__PREEMPT_PRODUCTS_OVERRIDE__
__PREEMPT_ENABLED_OVERRIDE__
_S120_STREAMS = json.loads(zlib.decompress(base64.b85decode(__STREAMS__)).decode("utf-8"))
_S120_MODEL = json.loads(zlib.decompress(base64.b85decode(__MODEL__)).decode("utf-8"))
_S120_DECISION_STEP = int(_S120_MODEL["decision_step"])
_S120_BASELINE_INDEX = int(_S120_MODEL["baseline_route_index"])
_S120_THRESHOLD = float(_S120_MODEL["switch_threshold"])
_S120_ROUTE_NAMES = tuple(_S120_MODEL["route_names"])
_S120_STATE = {
    0: {"last_step": -1, "route": None, "predictions": None},
    1: {"last_step": -1, "route": None, "predictions": None},
}


def _s120_tree_value(node, values):
    while "leaf_value" not in node:
        feature = int(node["split_feature"])
        threshold = float(node["threshold"])
        node = node["left_child"] if values[feature] <= threshold else node["right_child"]
    return float(node["leaf_value"])


def _s120_model_value(model, values):
    return sum(
        _s120_tree_value(tree["tree_structure"], values)
        for tree in model["tree_info"]
    )


def _s120_choose_route(obs):
    values = _cgr_features(obs)
    predictions = [
        _s120_model_value(row["model"], values)
        for row in _S120_MODEL["models"]
    ]
    best = max(range(len(predictions)), key=predictions.__getitem__)
    if predictions[best] - predictions[_S120_BASELINE_INDEX] <= _S120_THRESHOLD:
        best = _S120_BASELINE_INDEX
    return _S120_ROUTE_NAMES[best], predictions


def _s120_route(obs, step):
    seat = _seat(obs)
    state = _S120_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "route": None, "predictions": None}
        _S120_STATE[seat] = state
    state["last_step"] = step
    if state.get("route") is None and step >= _S120_DECISION_STEP:
        route, predictions = _s120_choose_route(obs)
        state["route"] = route
        state["predictions"] = predictions
    return state.get("route") or _S120_ROUTE_NAMES[_S120_BASELINE_INDEX]


def agent(obs, configuration=None):
    global _ACTIONS
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), 718)
        route = _s120_route(obs, step)
        actions = _S120_STREAMS[route]
        _ACTIONS = actions
        action = _weed_repair_action(obs, _copy_action(actions[step]), step)
        # Keep the exact market-aware execution chain used by every frozen
        # route source.  Omitting these two transforms makes the router's
        # "baseline route" materially different from the validated fixed
        # agent even when no route switch occurs.
        action = _repay_shift(obs, action, step)
        action = _rank_sell_slots(obs, action, configuration)
        action = _preempt_shift(obs, action, step)
        action = _terminal_liquidation(obs, action, step)
        return _align_hands(action, obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def kaggriculture_step120_public_margin_router(obs, configuration=None):
    return agent(obs, configuration)


__version__ = __VERSION__
'''


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def encode(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b85encode(zlib.compress(raw, 9)).decode("ascii")


def load_module(path: Path, label: str):
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_stream(module: Any) -> list[dict[str, Any]]:
    route = getattr(module, "_FCGR_ROUTE", getattr(module, "_CGR_FIXED_ROUTE", None))
    streams = getattr(module, "_CGR_STREAMS", None)
    stream = streams.get(route) if isinstance(streams, dict) and route in streams else None
    if not isinstance(stream, list):
        stream = getattr(module, "_ACTIONS", None)
    if not isinstance(stream, list) or len(stream) != 719:
        raise ValueError("candidate does not expose one 719-action fixed stream")
    return stream


def validate_models(payload: dict[str, Any]) -> None:
    route_names = payload.get("route_names", [])
    models = payload.get("models", [])
    if len(route_names) != len(models) or not route_names:
        raise ValueError("route/model count mismatch")
    decision_step = int(payload.get("decision_step", -1))
    if not 0 <= decision_step <= 718:
        raise ValueError("model decision step is outside the game")

    def visit(node: dict[str, Any]) -> None:
        if "leaf_value" in node:
            return
        if str(node.get("decision_type", "<=")) != "<=":
            raise ValueError(f"unsupported decision type: {node.get('decision_type')}")
        visit(node["left_child"])
        visit(node["right_child"])

    for row in models:
        for tree in row["model"]["tree_info"]:
            visit(tree["tree_structure"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-agent", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--output-agent", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--agent-version", default="eba18-step120-public-margin-router-v1")
    parser.add_argument(
        "--preempt-horizons",
        default="",
        help=(
            "Optional comma-separated positive lookahead horizons for the "
            "market-aware base executor. Empty preserves the base agent."
        ),
    )
    parser.add_argument(
        "--preempt-products",
        default="",
        help=(
            "Optional comma-separated product names eligible for exact-repay "
            "preemption. Empty preserves the base agent."
        ),
    )
    parser.add_argument(
        "--disable-preempt",
        action="store_true",
        help="Disable exact-repay market preemption while preserving all other execution logic.",
    )
    args = parser.parse_args()

    base_path = resolve(args.base_agent).resolve()
    model_path = resolve(args.model).resolve()
    fit_path = resolve(args.fit_receipt).resolve()
    bank_path = resolve(args.bank_receipt).resolve()
    output_path = resolve(args.output_agent).resolve()
    receipt_path = resolve(args.receipt).resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    validate_models(model)
    decision_step = int(model["decision_step"])
    accepted_targets = {
        "clipped_terminal_margin_per_route",
        "downside_2x_clipped_terminal_margin_per_route",
        "downside_3x_clipped_terminal_margin_per_route",
    }
    if fit.get("target") not in accepted_targets:
        raise ValueError("fit receipt target is not an accepted per-route terminal-margin utility")
    if float(fit.get("portable_tree_max_abs_error", 1.0)) > 1e-8:
        raise ValueError("portable tree evaluator did not match LightGBM")

    streams: dict[str, list[dict[str, Any]]] = {}
    source_rows = []
    for index, route_name in enumerate(model["route_names"]):
        source_candidate_id = int(model["models"][index]["source_candidate_id"])
        skeleton = bank["skeletons"][source_candidate_id]
        source = Path(skeleton["source"]).resolve()
        module = load_module(source, f"step120_source_{index}")
        stream = fixed_stream(module)
        streams[str(route_name)] = stream
        source_rows.append({
            "route": str(route_name),
            "source": str(source),
            "source_sha256": sha256(source),
            "action_sha256": str(skeleton["action_sha256"]),
        })

    baseline = streams[model["baseline_route"]]
    minimum_prefix = min(
        next((i for i, (left, right) in enumerate(zip(baseline, stream, strict=True)) if left != right), 719)
        for stream in streams.values()
    )
    if minimum_prefix < decision_step:
        raise ValueError(f"unsafe route set: minimum common prefix={minimum_prefix}")

    preempt_horizons: tuple[int, ...] | None = None
    if args.preempt_horizons.strip():
        parsed = tuple(
            int(value.strip())
            for value in args.preempt_horizons.split(",")
            if value.strip()
        )
        if not parsed or any(value <= 0 or value > 96 for value in parsed):
            raise ValueError("preempt horizons must be between 1 and 96")
        if len(parsed) != len(set(parsed)):
            raise ValueError("preempt horizons must be unique")
        preempt_horizons = parsed

    preempt_products: tuple[str, ...] | None = None
    if args.preempt_products.strip():
        allowed_products = {
            "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER",
        }
        parsed_products = tuple(
            value.strip().upper()
            for value in args.preempt_products.split(",")
            if value.strip()
        )
        if not parsed_products or any(value not in allowed_products for value in parsed_products):
            raise ValueError("preempt products contain an unknown product")
        if len(parsed_products) != len(set(parsed_products)):
            raise ValueError("preempt products must be unique")
        preempt_products = parsed_products

    tail = TAIL
    replacements = {
        "__PREEMPT_OVERRIDE__": (
            f"_PREEMPT_HORIZONS = {preempt_horizons!r}"
            if preempt_horizons is not None
            else "# Preserve base-agent preemption horizons."
        ),
        "__PREEMPT_PRODUCTS_OVERRIDE__": (
            f"_PREMIUM = {preempt_products!r}"
            if preempt_products is not None
            else "# Preserve base-agent preemption products."
        ),
        "__PREEMPT_ENABLED_OVERRIDE__": (
            "_PREEMPT_ENABLED = False"
            if args.disable_preempt
            else "# Preserve base-agent preemption enabled state."
        ),
        "__STREAMS__": repr(encode(streams)),
        "__MODEL__": repr(encode(model)),
        "__VERSION__": repr(args.agent_version),
    }
    for old, new in replacements.items():
        tail = tail.replace(old, new)
    source = base_path.read_text(encoding="utf-8").rstrip() + "\n" + tail.lstrip()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8", newline="\n")

    result = {
        "schema": "kaggriculture-public-margin-router-agent-build-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT_OFFICIAL_VALIDATION_REQUIRED",
        "agent_version": args.agent_version,
        "decision_step": decision_step,
        "minimum_common_prefix_actions": minimum_prefix,
        "route_count": len(streams),
        "routes": source_rows,
        "target": fit["target"],
        "switch_threshold": float(model["switch_threshold"]),
        "preempt_horizons_override": list(preempt_horizons) if preempt_horizons is not None else None,
        "preempt_products_override": list(preempt_products) if preempt_products is not None else None,
        "preempt_disabled": bool(args.disable_preempt),
        "baseline_route": model["baseline_route"],
        "tree_count": sum(len(row["model"]["tree_info"]) for row in model["models"]),
        "portable_tree_max_abs_error": fit["portable_tree_max_abs_error"],
        "runtime_inputs": "public observation only",
        "forbidden_inputs": fit["forbidden_runtime_inputs"],
        "online_dependencies": ["python-standard-library"],
        "base_agent": str(base_path),
        "base_agent_sha256": sha256(base_path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "fit_receipt": str(fit_path),
        "fit_receipt_sha256": sha256(fit_path),
        "bank_receipt": str(bank_path),
        "bank_receipt_sha256": sha256(bank_path),
        "output_agent": str(output_path),
        "output_agent_sha256": sha256(output_path),
        "output_bytes": output_path.stat().st_size,
        "truth_boundary": "JAX-screened candidate; official Python 1.32.7 holdout is mandatory.",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

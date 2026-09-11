#!/usr/bin/env python3
"""Embed the accepted JAX-screened margin router in a stdlib Kaggle agent."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib


TAIL = r'''
# --- Route37/Route38 public-state terminal-margin router (generated) ---
_R3738_MODEL = json.loads(zlib.decompress(base64.b85decode(__MODEL__)).decode("utf-8"))
_R3738_THRESHOLD = __THRESHOLD__
_R3738_DECISION_STEP = 192
_R3738_TREND_INDICES = (8, 11, 12, 14, 15, 23, 24, 27, 28)
_R3738_ROUTE37 = "8C-4S-75L"
_R3738_ROUTE38 = "6C-6S-75L"
_R3738_STATE = {
    0: {"last_step": -1, "anchor120": None, "anchor168": None, "route": None, "predicted_delta": None},
    1: {"last_step": -1, "anchor120": None, "anchor168": None, "route": None, "predicted_delta": None},
}


def _r3738_tree_value(node, values):
    while "leaf_value" not in node:
        feature = int(node["split_feature"])
        threshold = float(node["threshold"])
        node = node["left_child"] if values[feature] <= threshold else node["right_child"]
    return float(node["leaf_value"])


def _r3738_predict(values):
    return sum(
        _r3738_tree_value(tree["tree_structure"], values)
        for tree in _R3738_MODEL["tree_info"]
    )


def _r3738_route(obs, step):
    seat = _seat(obs)
    state = _R3738_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "anchor120": None, "anchor168": None, "route": None, "predicted_delta": None}
        _R3738_STATE[seat] = state
    state["last_step"] = step

    base = _cgr_features(obs)
    if step == 120:
        state["anchor120"] = list(base)
    if step == 168:
        state["anchor168"] = list(base)
    if state.get("route") is None and step >= _R3738_DECISION_STEP:
        anchor120 = state.get("anchor120") or base
        anchor168 = state.get("anchor168") or base
        values = list(base)
        values += [base[index] - anchor120[index] for index in _R3738_TREND_INDICES]
        values += [base[index] - anchor168[index] for index in _R3738_TREND_INDICES]
        predicted_delta = _r3738_predict(values)
        state["predicted_delta"] = predicted_delta
        state["route"] = (
            _R3738_ROUTE38
            if predicted_delta > _R3738_THRESHOLD
            else _R3738_ROUTE37
        )
    return state.get("route") or _R3738_ROUTE37


def agent(obs, configuration=None):
    global _ACTIONS
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), 718)
        route = _r3738_route(obs, step)
        actions = _CGR_STREAMS[route]
        _ACTIONS = actions
        action = _weed_repair_action(obs, _copy_action(actions[step]), step)
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


def kaggriculture_route37_route38_margin_router(obs, configuration=None):
    return agent(obs, configuration)


__version__ = __VERSION__
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def encode(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return base64.b85encode(zlib.compress(raw, 9)).decode("ascii")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("route37_route38_agent_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_numeric_trees(model: dict) -> None:
    def visit(node: dict) -> None:
        if "leaf_value" in node:
            return
        if str(node.get("decision_type", "<=")) != "<=":
            raise ValueError(f"unsupported decision type: {node.get('decision_type')}")
        visit(node["left_child"])
        visit(node["right_child"])

    for tree in model["tree_info"]:
        visit(tree["tree_structure"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-agent", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--output-agent", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--agent-version", default="route37-route38-margin-router-v1")
    args = parser.parse_args()

    base_path = args.base_agent.resolve()
    model_path = args.model.resolve()
    fit_path = args.fit_receipt.resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    validate_numeric_trees(model)
    if fit.get("target") != "terminal_margin_route38_minus_route37":
        raise ValueError("fit receipt is not a Route38-minus-Route37 margin model")
    if float(fit.get("portable_tree_max_abs_error", 1.0)) > 1e-8:
        raise ValueError("portable tree evaluator did not match LightGBM")

    module = load_module(base_path)
    streams = getattr(module, "_CGR_STREAMS", {})
    route37 = streams.get("8C-4S-75L")
    route38 = streams.get("6C-6S-75L")
    if not isinstance(route37, list) or not isinstance(route38, list):
        raise ValueError("base agent does not contain both Rank16 routes")
    if len(route37) != 719 or len(route38) != 719:
        raise ValueError("route stream length is not 719")
    common_prefix = 0
    for left, right in zip(route37, route38, strict=True):
        if left != right:
            break
        common_prefix += 1
    if common_prefix < 192:
        raise ValueError(f"unsafe decision point: common prefix={common_prefix}")

    threshold = float(fit["selected_threshold"])
    tail = TAIL
    for old, new in {
        "__MODEL__": repr(encode(model)),
        "__THRESHOLD__": repr(threshold),
        "__VERSION__": repr(args.agent_version),
    }.items():
        tail = tail.replace(old, new)
    source = base_path.read_text(encoding="utf-8").rstrip() + "\n" + tail.lstrip()
    args.output_agent.parent.mkdir(parents=True, exist_ok=True)
    args.output_agent.write_text(source, encoding="utf-8", newline="\n")
    result = {
        "schema": "route37-route38-margin-router-agent-build-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT_OFFICIAL_VALIDATION_REQUIRED",
        "agent_version": args.agent_version,
        "decision_step": 192,
        "common_prefix_actions": common_prefix,
        "target": fit["target"],
        "threshold": threshold,
        "tree_count": len(model["tree_info"]),
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
        "output_agent": str(args.output_agent.resolve()),
        "output_agent_sha256": sha256(args.output_agent),
        "output_bytes": args.output_agent.stat().st_size,
        "truth_boundary": "JAX-screened candidate; official Python 1.32.7 holdout is mandatory.",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

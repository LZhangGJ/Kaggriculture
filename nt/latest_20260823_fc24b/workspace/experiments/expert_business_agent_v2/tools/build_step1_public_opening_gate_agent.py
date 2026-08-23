#!/usr/bin/env python3
"""Build a standalone public-state step-1 gate around an existing router.

The baseline agent is kept byte-for-byte as the first part of the generated
file.  The alternate route may only be selected after action 0, and the
builder rejects it unless its frozen action 0 is identical to the baseline
stream.  Runtime selection uses only the opponent farm in the public
observation; opponent names and seeds never enter the generated agent.
"""

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
# --- Public-state opening gate (generated) ---
_S1_BASE_AGENT = agent
_S1_ALT_STREAM = json.loads(zlib.decompress(base64.b85decode(__ALT_STREAM__)).decode("utf-8"))
_S1_GATE = __GATE__
_S1_GATE_STATE = {
    0: {"last_step": -1, "route": None},
    1: {"last_step": -1, "route": None},
}


def _s1_pasture_count(farm):
    count = 0
    for row in list(_get(farm, "tiles", []) or []):
        for tile in list(row or []):
            if isinstance(tile, dict) and str(tile.get("kind") or "") == "PASTURE":
                count += 1
    return count


def _s1_choose_route(obs):
    seat = _seat(obs)
    opponent = _farm(obs, 1 - seat)
    money = float(_get(opponent, "money", 0) or 0)
    hands = len(_get(opponent, "hands", []) or [])
    pastures = _s1_pasture_count(opponent)
    alternate = (
        float(_S1_GATE["opponent_money_min"]) <= money
        <= float(_S1_GATE["opponent_money_max"])
        and hands == int(_S1_GATE["opponent_hands"])
        and pastures == int(_S1_GATE["opponent_pastures"])
    )
    return "alternate" if alternate else "baseline"


def _s1_route(obs, step):
    seat = _seat(obs)
    state = _S1_GATE_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "route": None}
        _S1_GATE_STATE[seat] = state
    state["last_step"] = step
    if state.get("route") is None and step >= 1:
        state["route"] = _s1_choose_route(obs)
    return state.get("route") or "baseline"


def agent(obs, configuration=None):
    global _ACTIONS
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), 718)
        if _s1_route(obs, step) == "baseline":
            return _S1_BASE_AGENT(obs, configuration)
        _ACTIONS = _S1_ALT_STREAM
        action = _weed_repair_action(obs, _copy_action(_S1_ALT_STREAM[step]), step)
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


def kaggriculture_step1_public_opening_gate(obs, configuration=None):
    return agent(obs, configuration)


__version__ = __VERSION__
'''


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


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
        raise ValueError("alternate agent does not expose one 719-action fixed stream")
    return stream


def baseline_stream(module: Any) -> list[dict[str, Any]]:
    streams = getattr(module, "_S120_STREAMS", None)
    names = getattr(module, "_S120_ROUTE_NAMES", None)
    index = getattr(module, "_S120_BASELINE_INDEX", None)
    if isinstance(streams, dict) and names is not None and index is not None:
        stream = streams[str(names[int(index)])]
        if isinstance(stream, list) and len(stream) == 719:
            return stream
    stream = getattr(module, "_ACTIONS", None)
    if not isinstance(stream, list) or len(stream) != 719:
        raise ValueError("baseline agent does not expose a 719-action baseline stream")
    return stream


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-agent", type=Path, required=True)
    parser.add_argument("--alternate-agent", type=Path, required=True)
    parser.add_argument("--output-agent", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--opponent-money-min", type=float, default=100.0)
    parser.add_argument("--opponent-money-max", type=float, default=400.0)
    parser.add_argument("--opponent-hands", type=int, default=5)
    parser.add_argument("--opponent-pastures", type=int, default=1)
    parser.add_argument("--agent-version", default="eba25-step1-public-opening-gate-v1")
    args = parser.parse_args()

    base_path = resolve(args.base_agent)
    alternate_path = resolve(args.alternate_agent)
    output_path = resolve(args.output_agent)
    receipt_path = resolve(args.receipt)
    base_module = load_module(base_path, "s1_gate_base")
    alternate_module = load_module(alternate_path, "s1_gate_alternate")
    base_actions = baseline_stream(base_module)
    alternate_actions = fixed_stream(alternate_module)
    if base_actions[0] != alternate_actions[0]:
        raise ValueError("unsafe gate: baseline and alternate action 0 differ")

    gate = {
        "opponent_money_min": float(args.opponent_money_min),
        "opponent_money_max": float(args.opponent_money_max),
        "opponent_hands": int(args.opponent_hands),
        "opponent_pastures": int(args.opponent_pastures),
    }
    tail = TAIL
    for old, new in {
        "__ALT_STREAM__": repr(encode(alternate_actions)),
        "__GATE__": repr(gate),
        "__VERSION__": repr(args.agent_version),
    }.items():
        tail = tail.replace(old, new)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        base_path.read_text(encoding="utf-8").rstrip() + "\n" + tail.lstrip(),
        encoding="utf-8",
        newline="\n",
    )

    result = {
        "schema": "kaggriculture-step1-public-opening-gate-build-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT_OFFICIAL_VALIDATION_REQUIRED",
        "agent_version": args.agent_version,
        "decision_step": 1,
        "common_prefix_actions": 1,
        "gate": gate,
        "runtime_inputs": ["public opponent money", "public opponent hand count", "public opponent pasture count"],
        "forbidden_runtime_inputs": ["opponent identity", "seed", "future events", "terminal outcome", "private opponent state"],
        "base_agent": str(base_path),
        "base_agent_sha256": sha256(base_path),
        "alternate_agent": str(alternate_path),
        "alternate_agent_sha256": sha256(alternate_path),
        "output_agent": str(output_path),
        "output_agent_sha256": sha256(output_path),
        "output_bytes": output_path.stat().st_size,
        "truth_boundary": "Public-state candidate; independent official Python 1.32.7 holdout is mandatory.",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

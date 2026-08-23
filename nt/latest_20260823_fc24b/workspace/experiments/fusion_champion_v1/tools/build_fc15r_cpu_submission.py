#!/usr/bin/env python3
"""Build FC15R: accepted FC15 plus a narrow public opening hedge."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import zlib

from build_fc15_cpu_submission import ROOT, render_fc15_cpu_source


TEMPLATE = '''from __future__ import annotations

import base64
import inspect
import types
import zlib


def _load_frozen(name, payload):
    module = types.ModuleType(name)
    module.__dict__["__name__"] = name
    source = zlib.decompress(base64.b85decode(payload)).decode("utf-8")
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


_BASE = _load_frozen("_fc15r_base", __BASE__)
_UEDDY = _load_frozen("_fc15r_ueddy", __UEDDY__)
_WITH_CONFIGURATION = {
    id(module): len(inspect.signature(module.__dict__["agent"]).parameters) >= 2
    for module in (_BASE, _UEDDY)
}
_STATE = {
    0: {"last_step": -1, "opening_match": False},
    1: {"last_step": -1, "opening_match": False},
}


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _seat(obs):
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs, seat):
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _call(module, obs, configuration):
    function = module.__dict__["agent"]
    if _WITH_CONFIGURATION[id(module)]:
        return function(obs, configuration)
    return function(obs)


def _opening_signature(obs, seat, step):
    if step != 1:
        return False
    own = _farm(obs, seat)
    rival = _farm(obs, 1 - seat)
    return (
        int(_get(own, "money", 0) or 0) == 22
        and int(_get(rival, "money", 0) or 0) == 140
        and 1 + len(list(_get(own, "hands", []) or [])) == 6
        and 1 + len(list(_get(rival, "hands", []) or [])) == 6
    )


def agent(obs, configuration=None):
    step = min(max(0, int(_get(obs, "step", 0) or 0)), 718)
    seat = _seat(obs)
    state = _STATE[seat]
    if step == 0 or step < int(state["last_step"]):
        state.update(last_step=step, opening_match=False)
    else:
        state["last_step"] = step

    base_action = _call(_BASE, obs, configuration)
    # Shadow from step zero so the route's weed, sale-debt and task ledgers are
    # valid before the step-192 handoff.
    ueddy_action = _call(_UEDDY, obs, configuration)
    if _opening_signature(obs, seat, step):
        state["opening_match"] = True
    if state["opening_match"] and step >= 192:
        return ueddy_action
    return base_action


__version__ = "FC15R-public-opening-hedge-ueddy-s192-cpu-v1"
'''


def encode_text(source: str) -> str:
    return base64.b85encode(zlib.compress(source.encode("utf-8"), 9)).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = encode_text(render_fc15_cpu_source())
    ueddy_path = (
        ROOT
        / "experiments/gold_adaptive_rule_v2/agents/"
        "gold_imitations_current_20260816_0955_v2/rank02_ueddy/main.py"
    )
    ueddy = encode_text(ueddy_path.read_text(encoding="utf-8"))
    rendered = TEMPLATE.replace("__BASE__", repr(base)).replace(
        "__UEDDY__", repr(ueddy)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

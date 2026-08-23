#!/usr/bin/env python3
"""Build the CPU-only single-file FC2A public submission candidate.

The submitted entrypoint keeps four frozen, independently stateful Python
policies.  Its router only uses actor-visible public state and mirrors the
validated JAX FC2A branch structure:

* K320 with the frozen four-step/cap-32 anti-mirror parameters by default;
* sticky public opening-sheep pressure detection;
* a stable no-Yarn X562 route or forced legacy 6C8S route at step 168;
* the validated PRT 9C5S suffix selector at step 288.

No source agent is imported at runtime; each frozen source is compressed into
the generated main.py so Kaggle receives one self-contained CPU file.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[3]


def encode_source(path: Path, replacements: tuple[tuple[str, str], ...] = ()) -> str:
    source = path.read_text(encoding="utf-8")
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise AssertionError(f"expected one replacement in {path}: {old!r}, got {count}")
        source = source.replace(old, new)
    return base64.b85encode(zlib.compress(source.encode("utf-8"), 9)).decode("ascii")


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


_K320 = _load_frozen("_fc2a_k320", __K320__)
_X562 = _load_frozen("_fc2a_x562", __X562__)
_LEGACY = _load_frozen("_fc2a_legacy_6c8s", __LEGACY__)
_PRT = _load_frozen("_fc2a_prt_9c5s", __PRT__)

# Force only the route identity.  The frozen source still owns legality,
# weed recovery, task execution, market ordering and terminal liquidation.
_LEGACY.__dict__["_kawa_route_label"] = lambda obs: "6c8s_3q"
_LEGACY.__dict__["_kawa_use_legacy_layout"] = lambda obs: True
_PRT.__dict__["_cgr_choose_route"] = lambda obs: "9C-5S-75L"
_WITH_CONFIGURATION = {
    id(module): len(inspect.signature(module.__dict__["agent"]).parameters) >= 2
    for module in (_K320, _X562, _LEGACY, _PRT)
}


_STATE = {
    0: {"last_step": -1, "sheep_pressure": False, "route": None, "prt": False},
    1: {"last_step": -1, "sheep_pressure": False, "route": None, "prt": False},
}


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _seat(obs):
    return int(_get(obs, "player", 0) or 0)


def _farm(obs, seat):
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if 0 <= seat < len(farms) else {}


def _animal_counts(farm):
    cows = sheep = 0
    for row in list(_get(farm, "tiles", []) or []):
        for tile in list(row or []):
            if not isinstance(tile, dict):
                continue
            animal = str(tile.get("animal") or "")
            cows += animal == "COW"
            sheep += animal == "SHEEP"
    return cows, sheep


def _shops(obs):
    town = _get(obs, "town", {}) or {}
    return [str(value) for value in list(_get(town, "unlocked_shops", []) or [])]


def _market_inventory(obs, product):
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    return int(_get(inventory, product, 0) or 0)


def _reset_if_needed(seat, step):
    state = _STATE[seat]
    if step == 0 or step < int(state["last_step"]):
        state.update(last_step=step, sheep_pressure=False, route=None, prt=False)
    else:
        state["last_step"] = step
    return state


def _visible_opening_sheep_pressure(obs, seat, step):
    if not 24 <= step < 96:
        return False
    rival = _farm(obs, 1 - seat)
    cows, sheep = _animal_counts(rival)
    rival_money = int(_get(rival, "money", 0) or 0)
    return sheep >= 4 and cows <= 2 and rival_money <= 1000


def _call(module, obs, configuration):
    function = module.__dict__["agent"]
    if _WITH_CONFIGURATION[id(module)]:
        return function(obs, configuration)
    return function(obs)


def agent(obs, configuration=None):
    step = min(max(0, int(_get(obs, "step", 0) or 0)), 718)
    seat = _seat(obs)
    state = _reset_if_needed(seat, step)

    k320_action = _call(_K320, obs, configuration)

    if _visible_opening_sheep_pressure(obs, seat, step):
        state["sheep_pressure"] = True

    # Before the opening detector closes, keep both pressure branches aligned
    # with the real game state.  Once step 96 is reached, ordinary K320 games
    # no longer execute the two unused policies.
    if not state["sheep_pressure"] and step >= 96:
        return k320_action

    # PRT is stateful.  Shadow it from the opening, just like the two pressure
    # prefixes, so a step-288 handoff does not enter the suffix with an empty
    # route state and silently lose the new day's HIRE transaction.
    prt_action = _call(_PRT, obs, configuration)
    if state["route"] is None:
        x562_action = _call(_X562, obs, configuration)
        legacy_action = _call(_LEGACY, obs, configuration)
    else:
        x562_action = legacy_action = None

    if not state["sheep_pressure"]:
        return k320_action

    if state["route"] is None and step >= 168:
        state["route"] = "legacy" if "YARN_STORE" in _shops(obs) else "x562"

    if state["route"] == "legacy":
        branch_action = legacy_action if legacy_action is not None else _call(_LEGACY, obs, configuration)
    else:
        branch_action = x562_action if x562_action is not None else _call(_X562, obs, configuration)

    if step == 288 and not state["prt"]:
        shops = _shops(obs)
        own_money = int(_get(_farm(obs, seat), "money", 0) or 0)
        state["prt"] = (
            own_money <= 13376
            and sum(value == "YARN_STORE" for value in shops) <= 1
            and _market_inventory(obs, "STRAWBERRY") <= 9979
        )
    if state["prt"]:
        return prt_action
    return branch_action


__version__ = "FC2A-rank14-plus-anti-mirror-cpu-v2-shadow-prt"
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    k320 = encode_source(
        ROOT / "submission/55655402_rayk_k320_adaptive_rank1/main.py",
        (
            ("_PREEMPT_MAX_BATCH = 12", "_PREEMPT_MAX_BATCH = 32"),
            (
                '_PREMIUM = ("STRAWBERRY", "MELON", "MILK", "WOOL")',
                '_PREMIUM = ("STRAWBERRY", "MILK", "WOOL")',
            ),
            ("elif clone <= 6:\n        horizon = 2", "elif clone <= 6:\n        horizon = 4"),
        ),
    )
    x562 = encode_source(ROOT / "references/public_latest8_20260820/x562_latest/main.py")
    legacy = encode_source(ROOT / "references/public_high_potential_20260819/boatlee_v20_multi_route/main.py")
    prt = encode_source(ROOT / "submission/55541953_prt_v6/main.py")
    rendered = (
        TEMPLATE.replace("__K320__", repr(k320))
        .replace("__X562__", repr(x562))
        .replace("__LEGACY__", repr(legacy))
        .replace("__PRT__", repr(prt))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

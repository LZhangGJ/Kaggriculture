"""Decouple the late source animal from the latent-pasture animal."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "agents"
if str(AGENTS) not in sys.path:
    sys.path.insert(0, str(AGENTS))

import e750a_place_funding_repair as FUNDING
import e776a_engine_exact_latent_pasture as PARENT

ENGINE = PARENT.ENGINE
ACTIVATION_STEP = 313
DELIVERY_STEPS = (314, 315, 316)
EXTRA_ANIMAL = "SHEEP"
SOURCE_ANIMAL = "COW"
TARGET_TILE = (5, 3)

_STATE: dict[int, dict[str, Any]] = {0: {}, 1: {}}
_LAST_DIAGNOSTIC: dict[int, dict[str, Any]] = {0: {}, 1: {}}


def _reset(seat: int) -> dict[str, Any]:
    state = {
        "last_step": -1,
        "active": False,
        "cancelled": False,
        "cancel_reason": None,
        "intervention_steps": [],
    }
    _STATE[seat] = state
    return state


def _farm(obs: Any, seat: int) -> Any:
    farms = list(ENGINE._get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _tile(farm: Any, x: int, y: int) -> Any:
    tiles = list(ENGINE._get(farm, "tiles", []) or [])
    if not (0 <= y < len(tiles)):
        return None
    row = list(tiles[y] or [])
    return row[x] if 0 <= x < len(row) else None


def _empty_pasture(tile: Any) -> bool:
    return isinstance(tile, dict) and tile.get("kind") == "PASTURE" and "animal" not in tile


def _split_purchase(
    obs: Any,
    seat: int,
    action: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    parent_state = PARENT._STATE.get(seat, {})
    if not (
        bool(parent_state.get("active"))
        and str(parent_state.get("animal")) == SOURCE_ANIMAL
        and len(action.get("market") or []) < ENGINE.MAX_MARKET_ORDERS
    ):
        return action
    positions = [
        index
        for index, order in enumerate(action.get("market") or [])
        if len(order) >= 3
        and order[0] == "BUY_ANIMAL"
        and str(order[1]) == SOURCE_ANIMAL
        and int(order[2] or 0) == 2
    ]
    if len(positions) != 1:
        return action

    changed = ENGINE._copy_action(action)
    changed["market"][positions[0]][2] = 1
    changed["market"].append(["BUY_ANIMAL", EXTRA_ANIMAL, 1])
    funded, _stopped = FUNDING._fund_market(obs, seat, changed)
    if funded.get("market") != changed.get("market"):
        return action

    state["active"] = True
    state["intervention_steps"].append(ACTIVATION_STEP)
    return changed


def _delivery(
    obs: Any,
    seat: int,
    step: int,
    action: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    if not state.get("active") or step not in DELIVERY_STEPS:
        return action
    farm = _farm(obs, seat)
    private = ENGINE._get(obs, "private", {}) or {}
    hands = list(ENGINE._get(farm, "hands", []) or [])
    inventories = list(ENGINE._get(private, "inventories", []) or [])
    commands = list(action.get("hands") or [])
    if len(hands) != 11 or len(commands) != 11 or len(inventories) < 12:
        state["active"] = False
        state["cancelled"] = True
        state["cancel_reason"] = "missing appended hand"
        return action

    position = tuple(hands[-1] or ())
    inventory = dict(inventories[11] or {})
    expected = {
        314: ((5, 4), {}, ["PICKUP", EXTRA_ANIMAL, 1]),
        315: ((5, 4), {EXTRA_ANIMAL: 1}, ["NORTH"]),
        316: ((5, 3), {EXTRA_ANIMAL: 1}, ["PLACE", EXTRA_ANIMAL]),
    }
    expected_position, expected_inventory, command = expected[step]
    if position != expected_position or inventory != expected_inventory:
        state["active"] = False
        state["cancelled"] = True
        state["cancel_reason"] = f"visible diversified-delivery mismatch at step {step}"
        return action
    if step == 316 and not _empty_pasture(_tile(farm, *TARGET_TILE)):
        state["active"] = False
        state["cancelled"] = True
        state["cancel_reason"] = "target pasture no longer empty"
        return action

    changed = ENGINE._copy_action(action)
    changed["hands"][-1] = command
    state["intervention_steps"].append(step)
    return changed


def agent(obs: Any, configuration: Any = None) -> dict[str, Any]:
    seat = ENGINE._seat(obs)
    step = max(0, min(ENGINE.MAX_STEPS - 1, int(ENGINE._get(obs, "step", 0) or 0)))
    state = _STATE[seat]
    if not state or step == 0 or step <= int(state.get("last_step", -1)):
        state = _reset(seat)
    parent_action = PARENT.agent(obs, configuration)
    if step == ACTIVATION_STEP:
        action = _split_purchase(obs, seat, parent_action, state)
    else:
        action = _delivery(obs, seat, step, parent_action, state)
    state["last_step"] = step
    _LAST_DIAGNOSTIC[seat] = {
        "step": step,
        "active": bool(state["active"]),
        "cancelled": bool(state["cancelled"]),
        "cancel_reason": state["cancel_reason"],
        "intervention_steps": list(state["intervention_steps"]),
        "parent": copy.deepcopy(PARENT._LAST_DIAGNOSTIC.get(seat, {})),
    }
    return action


diversified_agent = agent

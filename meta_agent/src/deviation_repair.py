"""Causal repairs for replay-route execution deviations.

The weed transaction follows the public Kaito v35 implementation: when a
planned PLANT/BUILD_PASTURE collides with a WEED tile, the affected actor DIGs,
retries the intended action one turn later, then replays that actor's planned
actions with a one-turn delay for a bounded window.  Other actors and all
market orders remain untouched.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any


Action = dict[str, Any]


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _seat(observation: Any) -> int:
    try:
        return 1 if int(_get(observation, "player", 0) or 0) == 1 else 0
    except (TypeError, ValueError):
        return 0


def _step(observation: Any) -> int:
    try:
        explicit = _get(observation, "step", None)
        if explicit is not None:
            return max(0, int(explicit))
        return max(
            0,
            int(_get(observation, "day", 0) or 0) * 24
            + int(_get(observation, "hour", 0) or 0),
        )
    except (TypeError, ValueError):
        return 0


def _tile(farm: Any, position: Any) -> Any:
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _actor_action(action: Action, actor: str | int) -> list[Any]:
    if actor == "farmer":
        return copy.deepcopy(action.get("farmer", ["PASS"]) or ["PASS"])
    hands = action.get("hands", []) or []
    index = int(actor)
    return copy.deepcopy(hands[index] if index < len(hands) else ["PASS"])


class WeedTransactionRepair:
    """Repair a weed collision without shifting unaffected route actors."""

    def __init__(
        self,
        planned_action: Callable[[int], Action],
        *,
        replay_steps: int = 8,
        stop_on_match: bool = False,
    ) -> None:
        self.planned_action = planned_action
        self.replay_steps = max(0, int(replay_steps))
        self.stop_on_match = bool(stop_on_match)
        self.states: dict[int, dict[str, Any]] = {0: {}, 1: {}}
        self.telemetry: dict[str, Any] = {
            "games": 0,
            "collisions": 0,
            "dig_actions": 0,
            "retry_actions": 0,
            "replay_actions": 0,
            "early_matches": 0,
            "expired_transactions": 0,
            "by_operation": {},
        }

    def reset(self) -> None:
        self.states = {0: {}, 1: {}}

    def _planned_actor_action(self, step: int, actor: str | int) -> list[Any]:
        if step < 0:
            return ["PASS"]
        try:
            return _actor_action(self.planned_action(step), actor)
        except (IndexError, KeyError, RuntimeError, TypeError, ValueError):
            return ["PASS"]

    def apply(self, observation: Any, action: Action) -> Action:
        step = _step(observation)
        seat = _seat(observation)
        game = self.states[seat]
        if step == 0 or step < int(game.get("last_step", -1)):
            game = {"last_step": step, "active": {}}
            self.states[seat] = game
            self.telemetry["games"] += 1
        game["last_step"] = step

        result = copy.deepcopy(action or {})
        result.setdefault("farmer", ["PASS"])
        result.setdefault("hands", [])
        result.setdefault("market", [])
        farms = list(_get(observation, "farms", []) or [])
        farm = farms[seat] if seat < len(farms) else {}
        positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
        unit_actions = [result.get("farmer", ["PASS"]), *list(result.get("hands", []) or [])]
        active = game["active"]

        for actor, transaction in list(active.items()):
            index = 0 if actor == "farmer" else int(actor) + 1
            if index >= len(unit_actions):
                active.pop(actor, None)
                self.telemetry["expired_transactions"] += 1
                continue
            age = step - int(transaction["start"])
            if age == 1:
                unit_actions[index] = copy.deepcopy(transaction["intended"])
                self.telemetry["retry_actions"] += 1
                continue
            if 2 <= age <= 1 + self.replay_steps:
                previous = self._planned_actor_action(step - 1, actor)
                if self.stop_on_match and unit_actions[index] == previous:
                    active.pop(actor, None)
                    self.telemetry["early_matches"] += 1
                    continue
                unit_actions[index] = previous
                self.telemetry["replay_actions"] += 1
                continue
            active.pop(actor, None)
            self.telemetry["expired_transactions"] += 1

        for index, (position, intended) in enumerate(zip(positions, unit_actions)):
            actor: str | int = "farmer" if index == 0 else index - 1
            if actor in active or not isinstance(intended, list) or not intended:
                continue
            operation = intended[0]
            if operation not in ("BUILD_COOP", "BUILD_PASTURE", "PLANT"):
                continue
            tile = _tile(farm, position)
            if not isinstance(tile, dict) or tile.get("kind") != "WEED":
                continue
            active[actor] = {"start": step, "intended": copy.deepcopy(intended)}
            unit_actions[index] = ["DIG"]
            self.telemetry["collisions"] += 1
            self.telemetry["dig_actions"] += 1
            counts = self.telemetry["by_operation"]
            counts[operation] = counts.get(operation, 0) + 1

        result["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
        result["hands"] = unit_actions[1:]
        return result

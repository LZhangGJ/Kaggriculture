"""Official-engine adapter for NT's reconstructed Front-40 trace banks.

The NT archive stores replay-derived actions as compact NumPy arrays.  This
module decodes only the routes reachable from the archive's frozen public-shop
router, then applies the same class of causal execution repairs used by the
local replay agents.  It is an evaluation adapter, not a claim that the
reconstructed policy is the teams' private submission source.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deviation_repair import WeedTransactionRepair
from .market_manager import StorageMarketManager


SHOP_ORDER = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)
UNIT_OPS = (
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "DROP",
    "PICKUP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
)
MARKET_OPS = (
    "NONE",
    "HIRE",
    "BUY_LAND",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
)
PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
SHED_ITEMS = PRODUCTS + ("GOOSE", "COW", "SHEEP")


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _step(observation: Any) -> int:
    explicit = _get(observation, "step", None)
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except (TypeError, ValueError):
            return 0
    try:
        return max(
            0,
            int(_get(observation, "day", 0) or 0) * 24
            + int(_get(observation, "hour", 0) or 0),
        )
    except (TypeError, ValueError):
        return 0


def _visible_shops(observation: Any) -> tuple[str, ...]:
    town = _get(observation, "town", {}) or {}
    values = _get(town, "unlocked_shops", None)
    if values is None:
        values = _get(observation, "town_shops", ())
    return tuple(str(value) for value in (values or ()))


def _unit_action(op: int, item: int, amount: int) -> list[Any]:
    if not 0 <= op < len(UNIT_OPS):
        return ["PASS"]
    name = UNIT_OPS[op]
    if name == "PLANT":
        return [name, PRODUCTS[item]] if 0 <= item < 5 else ["PASS"]
    if name in {"PICKUP", "PLACE"}:
        if not 0 <= item < len(SHED_ITEMS):
            return ["PASS"]
        return [name, SHED_ITEMS[item], max(1, int(amount))]
    return [name]


def _market_action(op: int, item: int, amount: int) -> list[Any] | None:
    if not 0 <= op < len(MARKET_OPS):
        return None
    name = MARKET_OPS[op]
    if name == "NONE":
        return None
    if name in {"HIRE", "BUY_LAND"}:
        return [name]
    if name == "BUY_SEED":
        valid = 0 <= item < 5
    elif name == "BUY_ANIMAL":
        valid = 9 <= item < len(SHED_ITEMS)
    else:
        valid = 0 <= item < len(PRODUCTS)
    if not valid:
        return None
    return [name, SHED_ITEMS[item], max(1, int(amount))]


def _decode_routes(
    path: str | Path,
    route_ids: Sequence[int],
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    wanted = tuple(sorted(set(int(value) for value in route_ids)))
    with np.load(Path(path), allow_pickle=False) as bank:
        route_count = int(bank["unit_op"].shape[0])
        invalid = [value for value in wanted if not 0 <= value < route_count]
        if invalid:
            raise ValueError(f"route IDs outside {path}: {invalid}")
        bootstrap = int(np.asarray(bank["bootstrap_route_id"]))
        unit_op = bank["unit_op"][list(wanted)]
        unit_item = bank["unit_item"][list(wanted)]
        unit_amount = bank["unit_amount"][list(wanted)]
        unit_count = bank["unit_count"][list(wanted)]
        market_op = bank["market_op"][list(wanted)]
        market_item = bank["market_item"][list(wanted)]
        market_amount = bank["market_amount"][list(wanted)]
        market_count = bank["market_count"][list(wanted)]

    decoded: dict[int, list[dict[str, Any]]] = {}
    for local, route_id in enumerate(wanted):
        steps: list[dict[str, Any]] = []
        for step in range(unit_op.shape[1]):
            count = max(1, int(unit_count[local, step]))
            units = [
                _unit_action(
                    int(unit_op[local, step, index]),
                    int(unit_item[local, step, index]),
                    int(unit_amount[local, step, index]),
                )
                for index in range(count)
            ]
            orders: list[list[Any]] = []
            for index in range(max(0, int(market_count[local, step]))):
                order = _market_action(
                    int(market_op[local, step, index]),
                    int(market_item[local, step, index]),
                    int(market_amount[local, step, index]),
                )
                if order is not None:
                    orders.append(order)
            steps.append({"farmer": units[0], "hands": units[1:], "market": orders})
        decoded[route_id] = steps
    return decoded, bootstrap


class NtFront40TraceAgent:
    """Execute one NT reconstructed route bank under its public-shop router."""

    def __init__(
        self,
        bank: str | Path,
        first_route_ids: Sequence[int],
        *,
        second_route_ids: Sequence[Sequence[int]] | None = None,
        manage_market: bool = True,
    ) -> None:
        if len(first_route_ids) != len(SHOP_ORDER):
            raise ValueError("first_route_ids must contain eight shop entries")
        self.first_route_ids = tuple(int(value) for value in first_route_ids)
        self.second_route_ids = (
            tuple(tuple(int(value) for value in row) for row in second_route_ids)
            if second_route_ids is not None
            else None
        )
        wanted = set(self.first_route_ids)
        if self.second_route_ids is not None:
            # Maps are stored in two shapes.  Compact maps are 8x8 and index
            # rows by first-shop ID.  Capacity-shaped maps index rows by route
            # ID and commonly contain padded identity rows beyond the bank's
            # real route count; only first-stage reachable rows are relevant.
            if len(self.second_route_ids) == 8 and all(
                len(row) == 8 for row in self.second_route_ids
            ):
                for row in self.second_route_ids:
                    wanted.update(row)
            else:
                for route_id in self.first_route_ids:
                    if 0 <= route_id < len(self.second_route_ids):
                        wanted.update(self.second_route_ids[route_id])
        # The bootstrap is not known until the bank is opened.  Decode the
        # selected routes first, then add it only when it is genuinely distinct.
        with np.load(Path(bank), allow_pickle=False) as payload:
            bootstrap = int(np.asarray(payload["bootstrap_route_id"]))
        wanted.add(bootstrap)
        self.actions, self.bootstrap = _decode_routes(bank, sorted(wanted))
        self.current_route = self.bootstrap
        self.first_shop_id: int | None = None
        self.selected_shop_count = 0
        self.repair = WeedTransactionRepair(self._planned_action, replay_steps=8)
        self.market = StorageMarketManager() if manage_market else None

    def reset(self) -> None:
        self.current_route = self.bootstrap
        self.first_shop_id = None
        self.selected_shop_count = 0
        self.repair.reset()

    def _planned_action(self, step: int) -> dict[str, Any]:
        tape = self.actions[self.current_route]
        if not 0 <= step < len(tape):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return tape[step]

    def _select_route(self, observation: Any) -> None:
        shops = _visible_shops(observation)
        if shops and self.selected_shop_count < 1:
            try:
                first = SHOP_ORDER.index(shops[0])
            except ValueError:
                first = 0
            self.first_shop_id = first
            self.current_route = self.first_route_ids[first]
            self.selected_shop_count = 1
        if (
            len(shops) > 1
            and self.selected_shop_count < 2
            and self.second_route_ids is not None
        ):
            try:
                second = SHOP_ORDER.index(shops[1])
            except ValueError:
                second = 0
            rows = self.second_route_ids
            if len(rows) == 8 and all(len(row) == 8 for row in rows):
                row = self.first_shop_id if self.first_shop_id is not None else 0
            else:
                row = self.current_route
            if 0 <= row < len(rows) and 0 <= second < len(rows[row]):
                selected = rows[row][second]
                if selected in self.actions and selected != self.current_route:
                    self.current_route = selected
            self.selected_shop_count = 2

    def __call__(self, observation: Any, configuration: Any = None) -> dict[str, Any]:
        step = _step(observation)
        if step == 0:
            self.reset()
        self._select_route(observation)
        action = copy.deepcopy(self._planned_action(step))
        action = self.repair.apply(observation, action)
        try:
            seat = int(_get(observation, "player", 0) or 0)
            farms = _get(observation, "farms", ()) or ()
            hand_count = len(_get(farms[seat], "hands", ()) or ())
        except (IndexError, TypeError, ValueError):
            hand_count = 0
        hands = list(action.get("hands", ()) or ())
        action["hands"] = (hands + [["PASS"]] * hand_count)[:hand_count]
        if self.market is not None:
            action = self.market.apply(observation, action, configuration)
        return action


__all__ = ["NtFront40TraceAgent", "SHOP_ORDER"]

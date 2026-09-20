"""Observation-driven SELL execution and shed-capacity protection.

Production routes still decide *when* and *what* they intend to sell.  This
module turns those intents into executable orders using the live shed, preserves
all non-SELL market decisions, and adds only rule-derived capacity/liquidation
sales.  It never reads opponent-private information or future observations.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping


# Synchronized with kaggle-environments 1.32.7.  Keep the official field names
# so sparse ``marketParams`` overrides can be applied without translation.
MARKET_PARAMS = {
    "WHEAT": {"base": 25, "I0": 10000, "T": 400, "below_func": "sqrt", "below_target": 0.8, "above_func": "log", "above_target": 0.2},
    "CARROT": {"base": 35, "I0": 10000, "T": 450, "below_func": "hinge", "below_target": 1.0, "above_func": "sqrt", "above_target": 0.7},
    "TOMATO": {"base": 60, "I0": 10000, "T": 200, "below_func": "hinge", "below_target": 0.4, "above_func": "sqrt", "above_target": 0.6},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt", "below_target": 0.7, "above_func": "linear", "above_target": 1.6},
    "MELON": {"base": 250, "I0": 10000, "T": 300, "below_func": "log", "below_target": 0.2, "above_func": "sq", "above_target": 3.6},
    "EGG": {"base": 50, "I0": 10000, "T": 332, "below_func": "hinge", "below_target": 0.4, "above_func": "log", "above_target": 0.2},
    "MILK": {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt", "below_target": 0.6, "above_func": "linear", "above_target": 1.6},
    "WOOL": {"base": 200, "I0": 10000, "T": 105, "below_func": "log", "below_target": 0.2, "above_func": "sq", "above_target": 3.2},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.4, "above_func": "linear", "above_target": 0.4},
}
SEED_COSTS = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COSTS = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
ANIMAL_STRUCTURES = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
ANIMAL_PRODUCTS = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
CROP_FIRST_YIELD_DAYS = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
LAND_PRICES = (1000, 2000, 4000)
HINGE_GAIN = 8.0
SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


@dataclass(frozen=True, slots=True)
class StorageMarketConfig:
    shed_capacity: int = 100
    turns_per_day: int = 24
    episode_steps: int = 720
    max_market_slots: int = 10
    demand_alpha: float = 0.25
    # Finished animal products and sale crops go first; WHEAT is deliberately
    # last because it is also an animal input.
    overflow_priority: tuple[str, ...] = (
        "WOOL", "MILK", "EGG", "MELON", "STRAWBERRY",
        "TOMATO", "CARROT", "FERTILIZER", "WHEAT",
    )
    liquidation_priority: tuple[str, ...] = (
        "CARROT", "EGG", "FERTILIZER", "MELON", "MILK",
        "STRAWBERRY", "TOMATO", "WHEAT", "WOOL",
    )


class StorageMarketManager:
    """Convert expert SELL intents into live-state-aware market orders."""

    def __init__(self, config: StorageMarketConfig | None = None) -> None:
        self.config = config or StorageMarketConfig()

    def apply(
        self,
        observation: Any,
        route_action: Mapping[str, Any],
        configuration: Any = None,
    ) -> dict[str, Any]:
        cfg = self._runtime_config(configuration)
        action = _canonical_action(route_action)
        step = _int(_get(observation, "step", 0))
        hour = _int(_get(observation, "hour", step % cfg.turns_per_day))

        # A lossy DROP executes before market orders in the official engine.  On
        # the last turn of a day, defer only those DROP actions to the automatic
        # post-market deposit so SELL orders can make room before anything is lost.
        shed, inventories, lossy_drops = _project_unit_storage(
            observation, action, cfg.shed_capacity
        )
        while hour == cfg.turns_per_day - 1 and lossy_drops:
            # Defer the earliest lossy DROP, then recompute.  Its removal may
            # make a later DROP safe, so deferring the whole first-pass set
            # would unnecessarily hide sellable goods from the market queue.
            _defer_unit_drops(action, {min(lossy_drops)})
            shed, inventories, lossy_drops = _project_unit_storage(
                observation, action, cfg.shed_capacity
            )

        # The official market is an ordered queue.  Earlier buys, sales and
        # atomic purchases affect the executability of every later order.
        market, remaining = _project_market_queue(
            observation,
            action["market"],
            shed,
            cfg,
            configuration,
        )
        if hour == cfg.turns_per_day - 1:
            overflow = max(
                0,
                sum(remaining.values())
                + sum(sum(inventory.values()) for inventory in inventories)
                - cfg.shed_capacity,
            )
            market = _add_sales(
                market,
                remaining,
                overflow,
                cfg.overflow_priority,
                cfg.max_market_slots,
            )

        if step >= cfg.episode_steps - 4:
            market = _add_sales(
                market,
                remaining,
                sum(remaining.values()),
                cfg.liquidation_priority,
                cfg.max_market_slots,
            )

        action["market"] = _rank_sell_slots(
            observation,
            market,
            configuration,
            demand_alpha=cfg.demand_alpha,
        )[: cfg.max_market_slots]
        return action

    def _runtime_config(self, configuration: Any) -> StorageMarketConfig:
        base = self.config
        return StorageMarketConfig(
            shed_capacity=max(1, _int(_get(configuration, "shedCapacity", base.shed_capacity))),
            turns_per_day=max(1, _int(_get(configuration, "turnsPerDay", base.turns_per_day))),
            episode_steps=max(4, _int(_get(configuration, "episodeSteps", base.episode_steps))),
            max_market_slots=max(
                1,
                _int(
                    _get(
                        configuration,
                        "maxMarketOrdersPerTurn",
                        base.max_market_slots,
                    )
                ),
            ),
            demand_alpha=base.demand_alpha,
            overflow_priority=base.overflow_priority,
            liquidation_priority=base.liquidation_priority,
        )


@dataclass(frozen=True, slots=True)
class LeadSaleConfig:
    """Generic causal sale lead for publicly similar production programs."""

    lead_turns: int = 5
    max_batch: int = 20
    max_public_distance: int = 8
    min_planned_quantity: int = 4
    start_step: int = 120
    stop_step: int = 680
    max_market_slots: int = 10
    premium_items: tuple[str, ...] = ("STRAWBERRY", "MELON", "MILK", "WOOL")


class CausalLeadSaleManager:
    """Move the selected route's own future SELL intent earlier, then repay it.

    The manager sees no future environment state and no opponent-private data.
    Its only look-ahead is the already selected fixed route's future action.
    """

    def __init__(self, config: LeadSaleConfig | None = None) -> None:
        self.config = config or LeadSaleConfig()
        self.due: dict[int, dict[str, int]] = {}

    def reset(self) -> None:
        self.due.clear()

    def repay(self, step: int, route_action: Mapping[str, Any]) -> dict[str, Any]:
        action = _canonical_action(route_action)
        remaining = self.due.pop(int(step), {})
        if not remaining:
            return action
        market: list[list[Any]] = []
        for raw in action["market"]:
            order = list(raw)
            if _is_sell(order) and remaining.get(str(order[1]), 0) > 0:
                item = str(order[1])
                reduction = min(_positive_int(order[2]), remaining[item])
                quantity = _positive_int(order[2]) - reduction
                remaining[item] -= reduction
                if quantity <= 0:
                    continue
                order[2] = quantity
            market.append(order)
        action["market"] = market
        return action

    def advance(
        self,
        observation: Any,
        current_action: Mapping[str, Any],
        future_route_action: Mapping[str, Any],
        configuration: Any = None,
    ) -> dict[str, Any]:
        cfg = self.config
        step = _int(_get(observation, "step", 0))
        if not cfg.start_step <= step < cfg.stop_step:
            return _canonical_action(current_action)
        if _public_farm_distance(observation) > cfg.max_public_distance:
            return _canonical_action(current_action)
        planned: dict[str, int] = {}
        for order in _canonical_action(future_route_action)["market"]:
            if _is_sell(order) and str(order[1]) in cfg.premium_items:
                item = str(order[1])
                planned[item] = planned.get(item, 0) + _positive_int(order[2])
        if not any(quantity >= cfg.min_planned_quantity for quantity in planned.values()):
            return _canonical_action(current_action)

        action = _canonical_action(current_action)
        capacity = max(1, _int(_get(configuration, "shedCapacity", 100)))
        max_market_slots = max(
            1,
            _int(
                _get(
                    configuration,
                    "maxMarketOrdersPerTurn",
                    cfg.max_market_slots,
                )
            ),
        )
        remaining = _project_market_shed(observation, action, capacity)
        market = [list(order) for order in action["market"]]
        for order in market:
            if _is_sell(order):
                item = str(order[1])
                remaining[item] = max(0, remaining.get(item, 0) - _positive_int(order[2]))

        shifted: dict[str, int] = {}
        for item in cfg.premium_items:
            target = planned.get(item, 0)
            if target < cfg.min_planned_quantity:
                continue
            quantity = min(max(0, remaining.get(item, 0)), cfg.max_batch, target)
            if quantity <= 0:
                continue
            if not _merge_or_insert_sell(market, item, quantity, max_market_slots):
                continue
            remaining[item] = max(0, remaining.get(item, 0) - quantity)
            shifted[item] = shifted.get(item, 0) + quantity
        if shifted:
            due = self.due.setdefault(step + cfg.lead_turns, {})
            for item, quantity in shifted.items():
                due[item] = due.get(item, 0) + quantity
            action["market"] = _rank_sell_slots(
                observation,
                market,
                configuration,
                demand_alpha=0.25,
            )[:max_market_slots]
        return action


def _merge_or_insert_sell(
    market: list[list[Any]], item: str, quantity: int, max_slots: int
) -> bool:
    existing = next((order for order in market if _is_sell(order) and order[1] == item), None)
    if existing is not None:
        existing[2] = _positive_int(existing[2]) + quantity
        return True
    replacement = next(
        (index for index, order in enumerate(market) if not order or order[0] in {"PASS", "NONE"}),
        None,
    )
    if replacement is not None:
        market[replacement] = ["SELL", item, quantity]
        return True
    if len(market) < max_slots:
        market.append(["SELL", item, quantity])
        return True
    return False


def _public_farm_distance(observation: Any) -> int:
    farms = list(_get(observation, "farms", []) or [])
    seat = _int(_get(observation, "player", 0))
    if len(farms) < 2 or not 0 <= seat < len(farms):
        return 10**9

    def signature(farm: Any) -> tuple[int, dict[str, int]]:
        counts: dict[str, int] = {}
        for row in list(_get(farm, "tiles", []) or []):
            for tile in list(row or []):
                if not isinstance(tile, dict):
                    continue
                for prefix, value in (
                    ("crop", tile.get("crop")),
                    ("animal", tile.get("animal")),
                    ("kind", tile.get("kind")),
                ):
                    if value:
                        key = f"{prefix}:{value}"
                        counts[key] = counts.get(key, 0) + 1
        quadrants = len(_get(farm, "unlocked_quadrants", []) or [])
        return quadrants, counts

    own_q, own_counts = signature(farms[seat])
    opp_q, opp_counts = signature(farms[1 - seat])
    keys = own_counts.keys() | opp_counts.keys()
    return 3 * abs(own_q - opp_q) + sum(
        abs(own_counts.get(key, 0) - opp_counts.get(key, 0)) for key in keys
    )


def _canonical_action(value: Mapping[str, Any]) -> dict[str, Any]:
    action = dict(value or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order or ["PASS"]) for order in (action.get("market") or [])],
    }


def _project_unit_storage(
    observation: Any,
    action: Mapping[str, Any],
    capacity: int,
) -> tuple[dict[str, int], list[dict[str, int]], set[int]]:
    """Mirror storage-changing unit actions from kaggle-environments 1.32.7."""
    private = _get(observation, "private", {}) or {}
    shed = {str(item): max(0, _int(value)) for item, value in dict(_get(private, "shed", {}) or {}).items()}
    inventories = [dict(value or {}) for value in list(_get(private, "inventories", []) or [])]
    farm = copy.deepcopy(_own_farm(observation))
    positions = [_get(farm, "farmer", [0, 0]), *list(_get(farm, "hands", []) or [])]
    orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    access = _shed_access(len(_get(farm, "tiles", []) or []) or 10)
    lossy_drops: set[int] = set()
    day = _int(_get(observation, "day", _int(_get(observation, "step", 0)) // 24))
    for index, order in enumerate(orders):
        if index >= len(positions) or index >= len(inventories) or not order:
            continue
        try:
            position = (int(positions[index][0]), int(positions[index][1]))
        except (IndexError, TypeError, ValueError):
            continue
        inventory = inventories[index]
        op = str(order[0])
        if op == "PICKUP" and len(order) >= 2:
            if position not in access:
                continue
            item = str(order[1])
            quantity = min(
                shed.get(item, 0),
                _positive_int(order[2]) if len(order) >= 3 else 1,
            )
            shed[item] = max(0, shed.get(item, 0) - quantity)
            inventory[item] = inventory.get(item, 0) + quantity
        elif op == "DROP":
            if position not in access:
                continue
            room = max(0, capacity - sum(shed.values()))
            if sum(max(0, _int(value)) for value in inventory.values()) > room:
                lossy_drops.add(index)
            _deposit(shed, list(inventory.items()), capacity)
            inventory.clear()
        elif op == "PLACE" and len(order) >= 2:
            item = str(order[1])
            # Animal placement onto a matching structure is not a shed deposit.
            tile = _tile_at(farm, position)
            structure = ANIMAL_STRUCTURES.get(item)
            if (
                structure
                and isinstance(tile, dict)
                and tile.get("kind") == structure
                and "animal" not in tile
                and _take_inventory(inventory, item, 1)
            ):
                tile["animal"] = item
                continue
            if position not in access:
                continue
            requested = _positive_int(order[2]) if len(order) >= 3 else 1
            quantity = min(
                requested,
                max(0, _int(inventory.get(item, 0))),
                max(0, capacity - sum(shed.values())),
            )
            if quantity > 0:
                _take_inventory(inventory, item, quantity)
                shed[item] = shed.get(item, 0) + quantity
        else:
            tile = _tile_at(farm, position)
            if not isinstance(tile, dict):
                continue
            if op == "HARVEST" and _int(tile.get("yield_units", 0)) > 0:
                if tile.get("kind") == "PLANT":
                    crop = str(tile.get("crop") or "")
                    first_yield = CROP_FIRST_YIELD_DAYS.get(crop)
                    if first_yield is None or day - _int(tile.get("planted_day", 0)) < first_yield:
                        continue
                    quantity = _int(tile.get("yield_units", 0))
                    inventory[crop] = inventory.get(crop, 0) + quantity
                    tile["yield_units"] = 0
                elif str(tile.get("animal") or "") in ANIMAL_PRODUCTS:
                    product = ANIMAL_PRODUCTS[str(tile["animal"])]
                    quantity = _int(tile.get("yield_units", 0))
                    inventory[product] = inventory.get(product, 0) + quantity
                    tile["yield_units"] = 0
            elif op == "FERTILIZE" and tile.get("kind") == "PLANT":
                _take_inventory(inventory, "FERTILIZER", 1)
            elif op == "FEED" and "animal" in tile and not tile.get("fed_today", False):
                if _take_inventory(inventory, "WHEAT", 1):
                    tile["fed_today"] = True
            elif (
                op == "COLLECT_FERTILIZER"
                and "animal" in tile
                and tile.get("fertilizer_available", False)
            ):
                inventory["FERTILIZER"] = inventory.get("FERTILIZER", 0) + 1
                tile["fertilizer_available"] = False
    return shed, inventories, lossy_drops


def _take_inventory(inventory: dict[str, int], item: str, quantity: int) -> bool:
    if inventory.get(item, 0) < quantity:
        return False
    inventory[item] -= quantity
    if inventory[item] == 0:
        inventory.pop(item, None)
    return True


def _defer_unit_drops(action: dict[str, Any], actors: set[int]) -> None:
    if 0 in actors:
        action["farmer"] = ["PASS"]
    for index in sorted(value - 1 for value in actors if value > 0):
        if index < len(action["hands"]):
            action["hands"][index] = ["PASS"]


def _project_market_shed(observation: Any, action: Mapping[str, Any], capacity: int) -> dict[str, int]:
    shed, _, _ = _project_unit_storage(observation, action, capacity)
    return shed


def _project_market_queue(
    observation: Any,
    raw_market: list[list[Any]],
    initial_shed: dict[str, int],
    cfg: StorageMarketConfig,
    configuration: Any = None,
) -> tuple[list[list[Any]], dict[str, int]]:
    """Project this player's ordered market queue using official per-unit rules."""
    shed = dict(initial_shed)
    farm = _own_farm(observation)
    money = float(_get(farm, "money", 0.0) or 0.0)
    hires_today = _int(_get(farm, "hires_today", 0))
    unlocked = len(_get(farm, "unlocked_quadrants", []) or [])
    hire_multiplier = _int(_get(configuration, "farmHandCostMult", 1))
    market_state = _get(observation, "market", {}) or {}
    params = _resolved_market_params(market_state)
    inventory = {
        item: _int(_get(_get(market_state, "inventory", {}) or {}, item, row["I0"]))
        for item, row in params.items()
    }
    result: list[list[Any]] = []
    for raw in list(raw_market or [])[: cfg.max_market_slots]:
        order = list(raw or ["PASS"])
        op = str(order[0])
        if op == "HIRE":
            cost = hire_multiplier * _fib(hires_today)
            if money >= cost:
                money -= cost
                hires_today += 1
            result.append(order)
            continue
        if op == "BUY_LAND":
            extra = unlocked - 1
            if 0 <= extra < len(LAND_PRICES) and money >= LAND_PRICES[extra]:
                money -= LAND_PRICES[extra]
                unlocked += 1
            result.append(order)
            continue
        if len(order) < 3:
            result.append(order)
            continue
        quantity = _positive_int(order[2])
        item = str(order[1]) if len(order) >= 2 else ""
        if quantity <= 0:
            result.append(order)
            continue
        if op == "SELL" and item in MARKET_PARAMS:
            executed = min(quantity, max(0, shed.get(item, 0)))
            for _ in range(executed):
                price = _market_price(item, inventory[item], params)
                shed[item] -= 1
                money += price
                if price > 1:
                    inventory[item] += 1
            result.append(["SELL", item, executed] if executed > 0 else ["PASS"])
            continue
        if op == "BUY_PRODUCT" and item in {"WHEAT", "FERTILIZER"}:
            for _ in range(quantity):
                price = _market_price(item, inventory[item] - 1, params)
                if money < price or sum(shed.values()) >= cfg.shed_capacity:
                    break
                money -= price
                shed[item] = shed.get(item, 0) + 1
                inventory[item] -= 1
        elif op == "BUY_SEED" and item in SEED_COSTS:
            price = SEED_COSTS[item]
            filled = min(quantity, int(money // price))
            money -= filled * price
        elif op == "BUY_ANIMAL" and item in ANIMAL_COSTS:
            price = ANIMAL_COSTS[item]
            for _ in range(quantity):
                if money < price or sum(shed.values()) >= cfg.shed_capacity:
                    break
                money -= price
                shed[item] = shed.get(item, 0) + 1
        result.append(order)
    return result, shed


def _fib(value: int) -> int:
    left, right = 1, 1
    for _ in range(max(0, int(value))):
        left, right = right, left + right
    return left


def executable_sell_labels(
    observation: Any,
    action: Mapping[str, Any],
    items: tuple[str, ...] | list[str],
    capacity: int = 100,
) -> tuple[dict[str, int], dict[str, int], list[str], dict[str, int]]:
    """Recover executable expert sales instead of requested ``SELL`` orders.

    Kaggriculture silently turns a SELL into a no-op once the player's shed is
    empty.  Public replay actions nevertheless retain the original requested
    quantity (often a large sentinel such as 999).  Those requests are not
    demonstrations of a trading decision and must not become positive BC
    labels.

    Unit actions are applied before market actions by the official interpreter,
    so ``available`` includes same-turn DROP/PICKUP/PLACE effects.  Market SELLs
    are then consumed sequentially, matching the official per-order semantics.
    Prices and the opponent can change sale revenue, but cannot make a valid
    own SELL fail while own stock remains, so this reconstruction is exact for
    the sale quantity represented by the BC target.
    """
    tracked = tuple(str(item) for item in items)
    available = _project_market_shed(observation, action, max(1, int(capacity)))
    cfg = StorageMarketConfig(
        shed_capacity=max(1, int(capacity)),
        max_market_slots=10,
    )
    projected_market, _ = _project_market_queue(
        observation,
        list(_get(action, "market", []) or []),
        available,
        cfg,
    )
    sold = {item: 0 for item in tracked}
    sell_order: list[str] = []
    stats = {
        "requested_sell_orders": 0,
        "executable_sell_orders": 0,
        "empty_sell_orders": 0,
        "partially_filled_sell_orders": 0,
        "requested_sell_units": 0,
        "executed_sell_units": 0,
    }
    for raw, projected in zip(
        list(_get(action, "market", []) or [])[: cfg.max_market_slots],
        projected_market,
    ):
        order = list(raw or [])
        if len(order) < 2 or str(order[0]) != "SELL":
            continue
        item = str(order[1])
        if item not in sold:
            continue
        requested = _positive_int(order[2]) if len(order) >= 3 else 1
        quantity = (
            _positive_int(projected[2])
            if _is_sell(projected) and str(projected[1]) == item
            else 0
        )
        stats["requested_sell_orders"] += 1
        # Cap only the diagnostic counter so sentinel 999 orders do not make
        # the audit look like useful economic volume.
        stats["requested_sell_units"] += min(requested, max(1, int(capacity)))
        if quantity <= 0:
            stats["empty_sell_orders"] += 1
            continue
        stats["executable_sell_orders"] += 1
        stats["executed_sell_units"] += quantity
        if quantity < requested:
            stats["partially_filled_sell_orders"] += 1
        sold[item] += quantity
        if item not in sell_order:
            sell_order.append(item)
    return available, sold, sell_order, stats


def _deposit(shed: dict[str, int], items: Any, capacity: int) -> None:
    for item, raw_quantity in items:
        room = max(0, capacity - sum(shed.values()))
        quantity = min(room, max(0, _int(raw_quantity)))
        if quantity:
            shed[str(item)] = shed.get(str(item), 0) + quantity


def _end_of_day_overflow(
    observation: Any,
    action: Mapping[str, Any],
    normalized_market: list[list[Any]],
    capacity: int,
) -> int:
    capacity = max(1, int(capacity))
    shed, inventories, _ = _project_unit_storage(observation, action, capacity)
    cfg = StorageMarketConfig(
        shed_capacity=capacity,
        max_market_slots=max(10, len(normalized_market)),
    )
    _, remaining = _project_market_queue(
        observation,
        normalized_market,
        shed,
        cfg,
    )
    carried = sum(sum(inventory.values()) for inventory in inventories)
    total = sum(remaining.values()) + carried
    return max(0, total - capacity)


def _add_sales(
    market: list[list[Any]],
    remaining: dict[str, int],
    required: int,
    priority: tuple[str, ...],
    max_slots: int,
) -> list[list[Any]]:
    required = max(0, int(required))
    if required <= 0:
        return market
    while market and (not market[-1] or market[-1][0] in {"PASS", "NONE"}):
        market.pop()
    for item in priority:
        quantity = min(required, max(0, remaining.get(item, 0)))
        if quantity <= 0:
            continue
        if len(market) < max_slots:
            market.append(["SELL", item, quantity])
        else:
            continue
        remaining[item] = max(0, remaining.get(item, 0) - quantity)
        required -= quantity
        if required <= 0:
            break
    return market


def _rank_sell_slots(
    observation: Any,
    market: list[list[Any]],
    configuration: Any,
    *,
    demand_alpha: float,
) -> list[list[Any]]:
    result = [list(order) for order in market]
    start = 0
    while start < len(result):
        if not _is_sell(result[start]):
            start += 1
            continue
        stop = start + 1
        while stop < len(result) and _is_sell(result[stop]):
            stop += 1
        result[start:stop] = sorted(
            result[start:stop],
            key=lambda order: (
                -_sell_score(observation, configuration, order, demand_alpha),
                str(order[1]),
            ),
        )
        start = stop
    return result


def _sell_score(observation: Any, configuration: Any, order: list[Any], demand_alpha: float) -> float:
    item = str(order[1])
    quantity = _positive_int(order[2])
    market = _get(observation, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    params = _resolved_market_params(market)
    current_inventory = _int(_get(inventory, item, params[item]["I0"]))
    current_quote = float(
        _get(prices, item, _market_price(item, current_inventory, params)) or 0
    )
    impact = quantity * max(
        0.0,
        current_quote - _market_price(item, current_inventory + quantity, params),
    )
    shops = list(_get(_get(observation, "town", {}) or {}, "unlocked_shops", []) or [])
    turns = max(1, _int(_get(configuration, "turnsPerDay", 24)))
    interval = max(1, _int(_get(configuration, "townShopSellInterval", 4)))
    demand = sum(
        (turns / interval) * (2 if len(SHOP_PRODUCTS.get(shop, ())) == 1 else 1)
        for shop in shops
        if item in SHOP_PRODUCTS.get(shop, ())
    )
    if item != "FERTILIZER":
        demand += turns / max(1, _int(_get(configuration, "townCenterSellInterval", 24)))
    excess = max(0.0, current_inventory + quantity - params[item]["I0"])
    urgency = min(1.0, (excess / max(0.25, demand)) / 10.0)
    return impact * (1.0 + demand_alpha * urgency)


def _resolved_market_params(market: Any) -> dict[str, dict[str, Any]]:
    result = {item: dict(values) for item, values in MARKET_PARAMS.items()}
    overrides = _get(market, "params", {}) or {}
    for item, patch in dict(overrides).items():
        if item in result and isinstance(patch, Mapping):
            result[item].update(dict(patch))
    return result


def _market_price(
    item: str,
    inventory: int,
    params: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    row = dict((params or MARKET_PARAMS)[item])
    base = float(row["base"])
    equilibrium = int(row["I0"])
    scale = float(row["T"])
    if inventory < equilibrium:
        shape = str(row["below_func"])
        amplitude = float(row["below_target"]) * base / _shape(shape, scale, scale)
        price = base + amplitude * _shape(shape, equilibrium - inventory, scale)
    else:
        shape = str(row["above_func"])
        amplitude = float(row["above_target"]) * base / _shape(shape, scale, scale)
        price = base - amplitude * _shape(shape, inventory - equilibrium, scale)
    return max(1, int(round(price)))


def _shape(name: str, value: float, scale: float | None = None) -> float:
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    if name == "hinge":
        if scale is None or scale <= 0:
            return value
        ratio = value / scale
        return ratio + HINGE_GAIN * max(0.0, ratio - 1.0) ** 2
    return value


def _is_sell(order: Any) -> bool:
    return isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] == "SELL" and order[1] in MARKET_PARAMS


def _own_farm(observation: Any) -> Any:
    farms = list(_get(observation, "farms", []) or [])
    seat = _int(_get(observation, "player", 0))
    return farms[seat] if 0 <= seat < len(farms) else {}


def _shed_access(size: int) -> set[tuple[int, int]]:
    half = size // 2
    return {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}


def _tile_at(farm: Any, position: Any) -> Any:
    try:
        return (_get(farm, "tiles", []) or [])[int(position[1])][int(position[0])]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _positive_int(value: Any) -> int:
    return max(0, _int(value))

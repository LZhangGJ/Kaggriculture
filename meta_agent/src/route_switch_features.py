"""Compact, interpretable state features for searched route switching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
)
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
TRADE_ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
OPPONENT_SALE_ITEMS = (
    "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
)


def observation_step(observation: Mapping[str, Any], default: int = 0) -> int:
    """Game step from the public clock.

    The official interpreter sets ``step`` on player 0's observation only;
    player 1 receives ``day`` and ``hour`` alone.  The step is part of the
    feature vector and gates the per-day tile sampling, so a missing key must
    not silently collapse a whole seat onto step zero.
    """
    day = observation.get("day")
    hour = observation.get("hour")
    if day is not None and hour is not None:
        return int(day) * 24 + int(hour)
    step = observation.get("step")
    try:
        return default if step is None else int(step)
    except (TypeError, ValueError):
        return default


SHOPS = (
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
)
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
LAND_COST = (1000, 2000, 4000)
HORIZONS = (24, 48, 72)


def _farm(observation: Mapping[str, Any], player: int) -> Mapping[str, Any]:
    farms = list(observation.get("farms", []) or [])
    return farms[player] if 0 <= player < len(farms) else {}


def _tile_counts(farm: Mapping[str, Any]) -> dict[str, Any]:
    crops = {item: 0 for item in CROPS}
    crop_yield = {item: 0.0 for item in CROPS}
    animals = {item: 0 for item in ANIMALS}
    animal_yield = {item: 0.0 for item in ANIMALS}
    structures = {item: 0 for item in ("SOIL", "COOP", "PASTURE")}
    weeds = empty = crop_stress = animal_stress = 0
    for row in list(farm.get("tiles", []) or [])[:10]:
        for tile in list(row or [])[:10]:
            if tile is None:
                empty += 1
                continue
            if not isinstance(tile, Mapping):
                continue
            kind = str(tile.get("kind") or "")
            if kind == "WEED":
                weeds += 1
            if kind in structures:
                structures[kind] += 1
            crop = str(tile.get("crop") or "")
            if crop in crops:
                crops[crop] += 1
                crop_yield[crop] += float(tile.get("yield_units", 0) or 0)
                crop_stress += int(not bool(tile.get("watered_today", False)))
                crop_stress += int(tile.get("consecutive_unwatered", 0) or 0)
            animal = str(tile.get("animal") or "")
            if animal in animals:
                animals[animal] += 1
                animal_yield[animal] += float(tile.get("yield_units", 0) or 0)
                animal_stress += int(not bool(tile.get("fed_today", False)))
                animal_stress += int(tile.get("consecutive_unfed", 0) or 0)
    return {
        "crops": crops, "crop_yield": crop_yield,
        "animals": animals, "animal_yield": animal_yield,
        "structures": structures, "weeds": weeds, "empty": empty,
        "crop_stress": crop_stress, "animal_stress": animal_stress,
    }


def _farm_vector(farm: Mapping[str, Any], counts: Mapping[str, Any]) -> list[float]:
    return [
        float(farm.get("money", 0) or 0),
        float(len(farm.get("hands", []) or [])),
        float(len(farm.get("unlocked_quadrants", []) or [])),
        float(farm.get("hires_today", 0) or 0),
        float(counts["weeds"]), float(counts["empty"]),
        *[float(counts["crops"][item]) for item in CROPS],
        *[float(counts["animals"][item]) for item in ANIMALS],
        *[float(counts["structures"][item]) for item in ("SOIL", "COOP", "PASTURE")],
        *[float(counts["crop_yield"][item]) for item in CROPS],
        *[float(counts["animal_yield"][item]) for item in ANIMALS],
        float(counts["crop_stress"]), float(counts["animal_stress"]),
    ]


def _farm_names(prefix: str) -> list[str]:
    return [
        f"{prefix}_money", f"{prefix}_hands", f"{prefix}_land",
        f"{prefix}_hires_today", f"{prefix}_weeds", f"{prefix}_empty_tiles",
        *[f"{prefix}_{item.lower()}_count" for item in CROPS],
        *[f"{prefix}_{item.lower()}_count" for item in ANIMALS],
        *[f"{prefix}_{item.lower()}_count" for item in ("SOIL", "COOP", "PASTURE")],
        *[f"{prefix}_{item.lower()}_yield" for item in CROPS],
        *[f"{prefix}_{item.lower()}_yield" for item in ANIMALS],
        f"{prefix}_crop_stress", f"{prefix}_animal_stress",
    ]


@dataclass
class RouteSwitchHistory:
    steps: int = 0
    tile_samples: int = 0
    min_money: list[float] = field(default_factory=lambda: [float("inf"), float("inf")])
    low100: list[int] = field(default_factory=lambda: [0, 0])
    low300: list[int] = field(default_factory=lambda: [0, 0])
    max_weeds: list[int] = field(default_factory=lambda: [0, 0])
    weed_area: list[int] = field(default_factory=lambda: [0, 0])
    max_crops: list[int] = field(default_factory=lambda: [0, 0])
    max_animals: list[int] = field(default_factory=lambda: [0, 0])

    def reset(self) -> None:
        self.steps = 0
        self.tile_samples = 0
        self.min_money[:] = [float("inf"), float("inf")]
        self.low100[:] = [0, 0]
        self.low300[:] = [0, 0]
        self.max_weeds[:] = [0, 0]
        self.weed_area[:] = [0, 0]
        self.max_crops[:] = [0, 0]
        self.max_animals[:] = [0, 0]

    def update(self, observation: Mapping[str, Any]) -> None:
        step = observation_step(observation, self.steps)
        sample_tiles = self.steps == 0 or step % 24 == 0
        for player in (0, 1):
            farm = _farm(observation, player)
            money = float(farm.get("money", 0) or 0)
            self.min_money[player] = min(self.min_money[player], money)
            self.low100[player] += int(money < 100)
            self.low300[player] += int(money < 300)
            if sample_tiles:
                counts = _tile_counts(farm)
                weeds = int(counts["weeds"])
                crops = sum(counts["crops"].values())
                animals = sum(counts["animals"].values())
                self.max_weeds[player] = max(self.max_weeds[player], weeds)
                self.weed_area[player] += weeds
                self.max_crops[player] = max(self.max_crops[player], crops)
                self.max_animals[player] = max(self.max_animals[player], animals)
        self.tile_samples += int(sample_tiles)
        self.steps += 1

    def vector(self, observation: Mapping[str, Any], own: int) -> list[float]:
        result = []
        denominator = max(1, self.steps)
        for player in (own, 1 - own):
            counts = _tile_counts(_farm(observation, player))
            crops = sum(counts["crops"].values())
            animals = sum(counts["animals"].values())
            result.extend([
                self.min_money[player] if math_isfinite(self.min_money[player]) else 0.0,
                self.low100[player] / denominator,
                self.low300[player] / denominator,
                float(self.max_weeds[player]),
                self.weed_area[player] / max(1, self.tile_samples),
                float(max(0, self.max_crops[player] - crops)),
                float(max(0, self.max_animals[player] - animals)),
            ])
        return result


def math_isfinite(value: float) -> bool:
    return value != float("inf") and value != float("-inf") and value == value


@dataclass
class OpponentSaleHistory:
    """Exact public-flow ledger for products neither player can buy."""

    cumulative: list[int] = field(default_factory=lambda: [0] * len(OPPONENT_SALE_ITEMS))
    previous_step: int | None = None
    previous_inventory: list[int] = field(default_factory=list)
    previous_shops: tuple[str, ...] = ()
    own_sales: list[int] = field(default_factory=lambda: [0] * len(OPPONENT_SALE_ITEMS))
    invalid_ticks: int = 0

    def update(self, observation: Mapping[str, Any], configuration: Any = None) -> None:
        from .market_manager import SHOP_PRODUCTS, _get, _market_price, _resolved_market_params

        step = observation_step(observation)
        market = dict(observation.get("market", {}) or {})
        inventory = dict(market.get("inventory", {}) or {})
        current = [int(inventory.get(item, 10_000) or 0) for item in OPPONENT_SALE_ITEMS]
        if self.previous_step is not None:
            if step != self.previous_step + 1:
                raise ValueError(f"non-contiguous market observations: {self.previous_step} -> {step}")
            shop_interval = max(1, int(_get(configuration, "townShopSellInterval", 4)))
            center_interval = max(1, int(_get(configuration, "townCenterSellInterval", 24)))
            demand = [0] * len(OPPONENT_SALE_ITEMS)
            if self.previous_step % shop_interval == 0:
                for shop in self.previous_shops:
                    products = SHOP_PRODUCTS.get(shop, ())
                    quantity = 2 if len(products) == 1 else 1
                    for item in products:
                        if item in OPPONENT_SALE_ITEMS:
                            demand[OPPONENT_SALE_ITEMS.index(item)] += quantity
            if self.previous_step % center_interval == 0:
                demand = [value + 1 for value in demand]
            params = _resolved_market_params({"params": market.get("params", {})})
            inferred = []
            exact = True
            for index, item in enumerate(OPPONENT_SALE_ITEMS):
                exact &= (
                    _market_price(item, self.previous_inventory[index], params) > 1
                    and _market_price(item, current[index] + demand[index], params) > 1
                )
                inferred.append(
                    current[index] - self.previous_inventory[index]
                    + demand[index] - self.own_sales[index]
                )
            if not exact or any(value < 0 for value in inferred):
                self.invalid_ticks += 1
            else:
                self.cumulative[:] = [a + b for a, b in zip(self.cumulative, inferred)]
        self.previous_step = step
        self.previous_inventory = current
        self.previous_shops = tuple(
            str(value) for value in (observation.get("town", {}) or {}).get("unlocked_shops", []) or []
        )
        self.own_sales[:] = [0] * len(OPPONENT_SALE_ITEMS)

    def record_action(
        self, observation: Mapping[str, Any], action: Mapping[str, Any], configuration: Any = None
    ) -> None:
        from .market_manager import _get, executable_sell_labels

        if observation_step(observation) != self.previous_step:
            raise ValueError("record_action must follow update for the same observation")
        capacity = max(1, int(_get(configuration, "shedCapacity", 100)))
        _, sold, _, _ = executable_sell_labels(
            observation, action, OPPONENT_SALE_ITEMS, capacity
        )
        self.own_sales[:] = [int(sold[item]) for item in OPPONENT_SALE_ITEMS]

    def vector(self) -> np.ndarray:
        if self.invalid_ticks:
            raise ValueError(f"opponent-sale ledger has {self.invalid_ticks} inexact ticks")
        return np.asarray(self.cumulative, dtype=np.float32)


def opponent_sale_feature_names() -> list[str]:
    return [f"opponent_cumulative_sell_{item.lower()}" for item in OPPONENT_SALE_ITEMS]


def _fib(index: int) -> int:
    left = right = 1
    for _ in range(max(0, index)):
        left, right = right, left + right
    return left


def _order_flow(
    order: Sequence[Any], prices: Mapping[str, Any], hires: int, land: int
) -> tuple[float, float, int, int, int, int, int]:
    if not order:
        return 0.0, 0.0, hires, land, 0, 0, 0
    op = str(order[0])
    item = str(order[1]) if len(order) >= 2 else ""
    quantity = max(1, int(order[2] or 0)) if len(order) >= 3 else 1
    spend = revenue = 0.0
    hire_count = land_count = animal_count = 0
    if op == "HIRE":
        spend = float(_fib(hires)); hires += 1; hire_count = 1
    elif op == "BUY_LAND":
        extra = max(0, land - 1)
        if extra < len(LAND_COST):
            spend = float(LAND_COST[extra]); land += 1; land_count = 1
    elif op == "BUY_SEED" and item in SEED_COST:
        spend = float(quantity * SEED_COST[item])
    elif op == "BUY_ANIMAL" and item in ANIMAL_COST:
        spend = float(quantity * ANIMAL_COST[item]); animal_count = quantity
    elif op == "BUY_PRODUCT":
        spend = float(quantity * float(prices.get(item, 0) or 0))
    elif op == "SELL":
        revenue = float(quantity * float(prices.get(item, 0) or 0))
    return spend, revenue, hires, land, hire_count, land_count, animal_count


def _plan_vector(
    observation: Mapping[str, Any], actions: Sequence[Mapping[str, Any]] | None
) -> list[float]:
    step = observation_step(observation)
    player = int(observation.get("player", 0) or 0)
    farm = _farm(observation, player)
    money = float(farm.get("money", 0) or 0)
    prices = dict((observation.get("market", {}) or {}).get("prices", {}) or {})
    route = list(actions or [])
    result = []
    for horizon in HORIZONS:
        cumulative = requirement = expense = 0.0
        hires = int(farm.get("hires_today", 0) or 0)
        land = len(farm.get("unlocked_quadrants", []) or [])
        hires_planned = lands_planned = animals_planned = 0
        previous_day = step // 24
        for future in range(step, min(len(route), step + horizon)):
            day = future // 24
            if day != previous_day:
                hires = 0
                previous_day = day
            for raw in list((route[future] or {}).get("market", []) or []):
                spend, revenue, hires, land, hc, lc, ac = _order_flow(
                    list(raw or []), prices, hires, land
                )
                expense += spend
                cumulative += spend - revenue
                requirement = max(requirement, cumulative)
                hires_planned += hc
                lands_planned += lc
                animals_planned += ac
        result.extend([
            expense, requirement, money - requirement,
            float(hires_planned), float(lands_planned), float(animals_planned),
        ])
    return result


def route_switch_feature_names() -> list[str]:
    names = [*_farm_names("self"), *_farm_names("opponent")]
    names.extend(f"self_shed_{item.lower()}" for item in ITEMS)
    names.extend(f"self_seed_{item.lower()}" for item in CROPS)
    names.extend(f"self_carried_{item.lower()}" for item in ITEMS)
    names.extend(("self_shed_total", "self_carried_total", "self_shed_free"))
    for item in TRADE_ITEMS:
        names.extend((f"market_{item.lower()}_inventory", f"market_{item.lower()}_price"))
    names.extend(f"shop_{shop.lower()}" for shop in SHOPS)
    names.extend(("step", "day", "hour"))
    for prefix in ("self_history", "opponent_history"):
        names.extend((
            f"{prefix}_min_money", f"{prefix}_cash_below_100_fraction",
            f"{prefix}_cash_below_300_fraction", f"{prefix}_max_weeds",
            f"{prefix}_mean_weeds", f"{prefix}_lost_crops", f"{prefix}_lost_animals",
        ))
    for horizon in HORIZONS:
        names.extend((
            f"plan_{horizon}_expense", f"plan_{horizon}_capital_requirement",
            f"plan_{horizon}_cash_slack", f"plan_{horizon}_hires",
            f"plan_{horizon}_lands", f"plan_{horizon}_animals",
        ))
    return names


def route_switch_vector(
    observation: Mapping[str, Any],
    history: RouteSwitchHistory,
    route_actions: Sequence[Mapping[str, Any]] | None,
) -> np.ndarray:
    player = int(observation.get("player", 0) or 0)
    own = _farm(observation, player)
    opponent = _farm(observation, 1 - player)
    private = dict(observation.get("private", {}) or {})
    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    carried = {item: 0.0 for item in ITEMS}
    for inventory in list(private.get("inventories", []) or []):
        for item, quantity in dict(inventory or {}).items():
            if str(item) in carried:
                carried[str(item)] += float(quantity or 0)
    market = dict(observation.get("market", {}) or {})
    inventory = dict(market.get("inventory", {}) or {})
    prices = dict(market.get("prices", {}) or {})
    shops = set((observation.get("town", {}) or {}).get("unlocked_shops", []) or [])
    values = [
        *_farm_vector(own, _tile_counts(own)),
        *_farm_vector(opponent, _tile_counts(opponent)),
        *[float(shed.get(item, 0) or 0) for item in ITEMS],
        *[float(seeds.get(item, 0) or 0) for item in CROPS],
        *[float(carried[item]) for item in ITEMS],
        float(sum(float(value or 0) for value in shed.values())),
        float(sum(carried.values())),
        100.0 - float(sum(float(value or 0) for value in shed.values())),
    ]
    for item in TRADE_ITEMS:
        values.extend((float(inventory.get(item, 0) or 0), float(prices.get(item, 0) or 0)))
    values.extend(float(shop in shops) for shop in SHOPS)
    step = observation_step(observation)
    # Derive the whole clock triple from one step so the vector can never carry
    # a step/day/hour combination the tree was not trained on.
    values.extend((float(step), float(step // 24), float(step % 24)))
    values.extend(history.vector(observation, player))
    values.extend(_plan_vector(observation, route_actions))
    result = np.asarray(values, dtype=np.float32)
    expected = len(route_switch_feature_names())
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise ValueError(f"route-switch feature shape {result.shape}, expected {(expected,)}")
    return result


ROUTE_SWITCH_DIM = len(route_switch_feature_names())

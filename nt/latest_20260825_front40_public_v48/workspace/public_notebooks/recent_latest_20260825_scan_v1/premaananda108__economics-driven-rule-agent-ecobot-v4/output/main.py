"""Kaggle Submission — Single File Bundle."""

from __future__ import annotations

from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Any
from typing import Any, Callable
import math

# ---------------------------------------------------------------------------
# MODULE: constants.py
# ---------------------------------------------------------------------------

TOTAL_DAYS: int = 30
TURNS_PER_DAY: int = 24
SHED_CAPACITY: int = 100
MAX_MARKET_ORDERS: int = 10
BOARD_SIZE: int = 10
I0: int = 10_000

# Interleaved by distance rings from shed junction (4.5, 4.5) across all quadrants
PASTURE_CLUSTER: list[tuple[int, int]] = [
    # Ring 0: Shed corner tiles (dist 0)
    (4, 4), (5, 4), (4, 5), (5, 5),
    # Ring 1: Immediate shed neighbors (dist 1)
    (4, 3), (3, 4), (5, 3), (6, 4), (3, 5), (4, 6), (6, 5), (5, 6),
    # Ring 2: Secondary cluster tiles (dist 2)
    (3, 3), (4, 2), (2, 4), (6, 3), (5, 2), (7, 4), (2, 5), (3, 6), (4, 7), (7, 5), (6, 6), (5, 7),
]
SHED_TILES: list[tuple[int, int]] = [(4, 4), (5, 4), (4, 5), (5, 5)]
SHED_TILES_SET: frozenset[tuple[int, int]] = frozenset(SHED_TILES)

COOP_CLUSTER: list[tuple[int, int]] = [
    (2, 2), (3, 2), (2, 3),
    (7, 2), (7, 3), (8, 2),
    (2, 6), (2, 7),
    (7, 6), (7, 7),
]

MILK_SUPPORT_SHOPS: tuple[str, ...] = ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP")

SELLABLE_ITEMS: tuple[str, ...] = (
    "FERTILIZER", "MILK", "WOOL", "EGG",
    "MELON", "STRAWBERRY", "CARROT", "TOMATO", "WHEAT",
)

SHOP_DEMANDS: dict[str, dict[str, float]] = {
    "BAKERY":         {"EGG": 6.0, "WHEAT": 6.0},
    "PIZZA_SHOP":     {"MILK": 6.0, "TOMATO": 6.0, "WHEAT": 6.0},
    "BRUNCH_SPOT":    {"EGG": 6.0, "WHEAT": 6.0, "STRAWBERRY": 6.0},
    "YARN_STORE":     {"WOOL": 12.0},
    "ICE_CREAM_SHOP": {"STRAWBERRY": 6.0, "MILK": 6.0, "WHEAT": 6.0},
    "PET_CAFE":       {"CARROT": 12.0},
    "SMOOTHIE_SHOP":  {"STRAWBERRY": 6.0, "MILK": 6.0},
    "FARMERS_MARKET": {"WHEAT": 6.0, "CARROT": 6.0, "TOMATO": 6.0, "STRAWBERRY": 6.0},
}
WHEAT_DEMAND_SHOPS: frozenset[str] = frozenset(
    shop for shop, demands in SHOP_DEMANDS.items() if "WHEAT" in demands
)

TOWN_CENTER_FLAT: float = 1.0
SELL_FLOOR_PCT: float = 0.30
SHED_SELL_PRESSURE: int = 85
LIQUIDATION_DAY: int = 28
MELON_LAST_PLANT_DAY: int = 18
STRAWBERRY_LAST_PLANT_DAY: int = 14
WHEAT_LAST_PLANT_DAY: int = 26
WHEAT_SURPLUS_MIN_SHOPS: int = 2
WHEAT_SURPLUS_SEED_RESERVE: int = 35
STRAWBERRY_TARGET_BUSHES: int = 32
MAX_TOTAL_HERD: int = 18
MIN_PAYBACK_DAYS: int = 8
MIN_ANIMAL_PAYBACK_DAYS: int = 15
HAND_COST_PER_ANIMAL_DAY: float = 2.0
SEED_PURCHASE_CASH_RESERVE: float = 200.0
LAND_RESERVE_NE: float = 350.0
LAND_RESERVE_SW_SE: float = 400.0
FEED_WHEAT_RESERVE: int = 4
FEED_WHEAT_RESERVE_MIN_RATIO: int = 2
N_FERTILIZER_HANDS: int = 2
FERTILIZER_PICKUP_BATCH: int = 4

CROP_PIVOT_MAX_TILES: dict[str, int] = {"CARROT": 8, "TOMATO": 6}

CROPS: dict[str, dict[str, Any]] = {
    "WHEAT": {
        "seed_cost": 10, "base_price": 25,
        "first_yield_day": 2, "max_yield_day": 4,
        "max_units_base": 4, "max_units_fert": 6,
        "bonus_start_day": 2, "is_ongoing": False,
    },
    "CARROT": {
        "seed_cost": 20, "base_price": 35,
        "first_yield_day": 2, "max_yield_day": 3,
        "max_units_base": 3, "max_units_fert": 4,
        "bonus_start_day": 2, "is_ongoing": False,
    },
    "TOMATO": {
        "seed_cost": 50, "base_price": 60,
        "first_yield_day": 8, "max_yield_day": 11,
        "is_ongoing": True,
        "bonus_start_day": 8, "yield_days": [8, 9, 10, 11],
    },
    "STRAWBERRY": {
        "seed_cost": 100, "base_price": 120,
        "first_yield_day": 10, "max_yield_day": 16,
        "is_ongoing": True,
        "bonus_start_day": 10, "yield_days": [10, 12, 14, 16],
    },
    "MELON": {
        "seed_cost": 80, "base_price": 250,
        "first_yield_day": 10, "max_yield_day": 10,
        "is_ongoing": False,
        "bonus_start_day": 6, "max_units_base": 6, "max_units_fert": 6,
    },
}

ANIMALS: dict[str, dict[str, Any]] = {
    "GOOSE": {"cost": 300, "base_price": 50, "structure": "COOP", "first_yield_day": 4, "interval": 1, "product": "EGG"},
    "COW":   {"cost": 400, "base_price": 160, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "product": "MILK"},
    "SHEEP": {"cost": 500, "base_price": 200, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "product": "WOOL"},
}

MARKET_PARAMS: dict[str, dict[str, Any]] = {
    "WHEAT":       {"base": 25,  "T": 400, "below_func": "sqrt",  "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":      {"base": 35,  "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":      {"base": 60,  "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY":  {"base": 120, "T": 100, "below_func": "sqrt",  "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":       {"base": 250, "T": 300, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":         {"base": 50,  "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":        {"base": 160, "T": 122, "below_func": "sqrt",  "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":        {"base": 200, "T": 105, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER":  {"base": 100, "T": 200, "below_func": "linear","below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

_FIB: list[int] = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


def fib_cost(n: int) -> int:
    if n < 0:
        raise ValueError(f"Hire index cannot be negative: {n}")
    if n < len(_FIB):
        return _FIB[n]
    a, b = _FIB[-2], _FIB[-1]
    for _ in range(n - len(_FIB) + 1):
        a, b = b, a + b
    return b


def total_hire_cost(count: int, already_hired: int) -> int:
    return sum(fib_cost(already_hired + i) for i in range(count))


def get_target_animals(unlocked_shops: list[str]) -> tuple[int, int, int]:
    if "YARN_STORE" in unlocked_shops[:4]:
        return 6, 12, 4
    if any(s in MILK_SUPPORT_SHOPS for s in unlocked_shops[:4]):
        return 10, 4, 3
    return 8, 6, 3


# ---------------------------------------------------------------------------
# MODULE: market.py
# ---------------------------------------------------------------------------

def _shape(func: str, x: float, t: float) -> float:
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(max(0.0, x))
    if func == "log":
        return math.log(1.0 + max(0.0, x))
    if func == "hinge":
        u = x / max(1.0, t)
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    raise ValueError(f"Unknown shape function: {func!r}")


def market_price(item: str, inv: int) -> int:
    if item not in MARKET_PARAMS:
        raise ValueError(f"Unknown market item: {item!r}")
    p = MARKET_PARAMS[item]
    base = float(p["base"])
    t = float(p["T"])
    if inv == I0:
        return int(round(base))
    if inv < I0:
        x = float(I0 - inv)
        denom = _shape(p["below_func"], t, t)
        amp = (p["below_target"] * base) / denom if denom > 0 else 0.0
        return max(1, int(round(base + amp * _shape(p["below_func"], x, t))))
    x = float(inv - I0)
    denom = _shape(p["above_func"], t, t)
    amp = (p["above_target"] * base) / denom if denom > 0 else 0.0
    return max(1, int(round(base - amp * _shape(p["above_func"], x, t))))


def get_price(item: str, market_inv: dict[str, int]) -> int:
    return market_price(item, market_inv.get(item, I0))


# ---------------------------------------------------------------------------
# MODULE: navigation.py
# ---------------------------------------------------------------------------

_DIRS: tuple[tuple[int, int, str], ...] = (
    (0, -1, "NORTH"), (0, 1, "SOUTH"), (1, 0, "EAST"), (-1, 0, "WEST")
)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def quad_of(pos: tuple[int, int]) -> str:
    half = BOARD_SIZE // 2
    if pos[1] < half:
        return "NE" if pos[0] >= half else "NW"
    return "SE" if pos[0] >= half else "SW"


def is_on_shed_tile(pos: tuple[int, int]) -> bool:
    return pos in SHED_TILES_SET


def closest_shed_pos(pos: tuple[int, int]) -> tuple[int, int]:
    return min(SHED_TILES, key=lambda s: manhattan(pos, s))


def bfs_step(start: tuple[int, int], goal: tuple[int, int]) -> str | None:
    if start == goal:
        return None
    visited: set[tuple[int, int]] = {start}
    q: deque[tuple[tuple[int, int], str]] = deque()
    for dx, dy, move in _DIRS:
        nx, ny = start[0] + dx, start[1] + dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if (nx, ny) == goal:
                return move
            visited.add((nx, ny))
            q.append(((nx, ny), move))
    while q:
        (cx, cy), first = q.popleft()
        for dx, dy, _ in _DIRS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and (nx, ny) not in visited:
                if (nx, ny) == goal:
                    return first
                visited.add((nx, ny))
                q.append(((nx, ny), first))
    return None


def go_shed_or_drop(pos: tuple[int, int]) -> list[Any]:
    if is_on_shed_tile(pos):
        return ["DROP"]
    return [bfs_step(pos, closest_shed_pos(pos)) or "PASS"]


# ---------------------------------------------------------------------------
# MODULE: farm.py
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class AnimalCensus:
    field_cows: int
    field_sheep: int
    field_geese: int
    shed_cows: int
    shed_sheep: int
    shed_geese: int
    carried_cows: int
    carried_sheep: int
    carried_geese: int

    @property
    def total_cows(self) -> int:
        return self.field_cows + self.shed_cows + self.carried_cows

    @property
    def total_sheep(self) -> int:
        return self.field_sheep + self.shed_sheep + self.carried_sheep

    @property
    def total_geese(self) -> int:
        return self.field_geese + self.shed_geese + self.carried_geese

    @property
    def total(self) -> int:
        return self.total_cows + self.total_sheep

    @property
    def total_all(self) -> int:
        return self.total + self.total_geese

    @property
    def on_field(self) -> int:
        return self.field_cows + self.field_sheep + self.field_geese

    @property
    def in_shed(self) -> int:
        return self.shed_cows + self.shed_sheep + self.shed_geese

    @property
    def carried(self) -> int:
        return self.carried_cows + self.carried_sheep + self.carried_geese

    @property
    def n_feedable(self) -> int:
        return self.on_field + self.in_shed


def count_animal_census(
    farm_state: dict[str, Any],
    shed: dict[str, int],
    inventories: list[dict[str, int]],
) -> AnimalCensus:
    return AnimalCensus(
        field_cows=sum(1 for a in farm_state["animals"] if a["animal"] == "COW"),
        field_sheep=sum(1 for a in farm_state["animals"] if a["animal"] == "SHEEP"),
        field_geese=sum(1 for a in farm_state["animals"] if a["animal"] == "GOOSE"),
        shed_cows=shed.get("COW", 0),
        shed_sheep=shed.get("SHEEP", 0),
        shed_geese=shed.get("GOOSE", 0),
        carried_cows=sum(inv.get("COW", 0) for inv in inventories),
        carried_sheep=sum(inv.get("SHEEP", 0) for inv in inventories),
        carried_geese=sum(inv.get("GOOSE", 0) for inv in inventories),
    )


def parse_farm_state(tiles: list[list[Any]], day: int) -> dict[str, Any]:
    state: dict[str, Any] = {
        "animals": [],
        "empty_pastures": [],
        "empty_coops": [],
        "plants": [],
        "weeds": [],
        "empty_tiles": [],
        "unlocked_count": 0,
    }

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            tile = tiles[y][x]
            pos = (x, y)

            if tile == "LOCKED":
                continue
            state["unlocked_count"] += 1

            if tile is None:
                state["empty_tiles"].append(pos)
                continue

            if not isinstance(tile, dict):
                raise TypeError(f"Unexpected tile structure at {pos}: {tile!r}")

            kind = tile.get("kind")
            if kind == "WEED":
                state["weeds"].append(pos)
            elif kind in ("PASTURE", "COOP"):
                animal = tile.get("animal")
                if animal is None:
                    if kind == "COOP":
                        state["empty_coops"].append(pos)
                    else:
                        state["empty_pastures"].append(pos)
                else:
                    state["animals"].append({
                        "pos": pos,
                        "animal": animal,
                        "fed_today": tile.get("fed_today", False),
                        "cared_today": tile.get("cared_today", False),
                        "fertilizer_available": tile.get("fertilizer_available", False),
                        "yield_units": tile.get("yield_units", 0),
                    })
            elif kind == "PLANT":
                crop = tile.get("crop")
                if crop is None or crop not in CROPS:
                    raise ValueError(f"Unknown or missing crop {crop!r} at {pos}: {tile}")
                planted_day = tile.get("planted_day", day)
                age = day - planted_day
                spec = CROPS[crop]
                yu = tile.get("yield_units", 0)
                watered = tile.get("watered_today", False)
                fert_until = tile.get("fertilized_until_day", -1)

                is_expired = False
                if spec["is_ongoing"]:
                    last_yield_age = spec["yield_days"][-1]
                    if age >= last_yield_age + 1:
                        is_expired = True

                if crop == "STRAWBERRY":
                    upcoming_ages = [d for d in spec["yield_days"] if d >= age]
                    if upcoming_ages:
                        days_to_yield = upcoming_ages[0] - age
                        yield_day = day + days_to_yield
                        fert_due = days_to_yield <= 1 and fert_until < yield_day
                    else:
                        fert_due = False
                else:
                    fert_due = False

                state["plants"].append({
                    "pos": pos,
                    "crop": crop,
                    "age": age,
                    "watered_today": watered,
                    "yield_units": yu,
                    "fertilize_due": fert_due,
                    "is_expired": is_expired,
                    "is_ongoing": spec["is_ongoing"],
                    "first_yield_day": spec["first_yield_day"],
                    "max_yield_day": spec["max_yield_day"],
                })
            else:
                raise ValueError(f"Unknown tile kind {kind!r} at {pos}: {tile}")

    return state


def planted_count(farm_state: dict[str, Any], crop: str) -> int:
    return sum(1 for p in farm_state["plants"] if p["crop"] == crop)


def committed_count(farm_state: dict[str, Any], seeds: dict[str, int], crop: str) -> int:
    return planted_count(farm_state, crop) + seeds.get(crop, 0)


def count_fertilize_due(farm_state: dict[str, Any]) -> int:
    return sum(1 for p in farm_state["plants"] if p["fertilize_due"])


# ---------------------------------------------------------------------------
# MODULE: planner.py
# ---------------------------------------------------------------------------

_ANIMAL_PROFIT_SPEC: dict[str, tuple[str, int, float]] = {
    "GOOSE_COOP": ("EGG", 4, 1.0),
    "COW": ("MILK", 8, 0.5),
    "SHEEP": ("WOOL", 6, 1.0 / 3.0),
}

_KIND_SPEC: dict[str, tuple[str, float, int]] = {
    "COW":        ("MILK", 0.5,        ANIMALS["COW"]["first_yield_day"]),
    "SHEEP":      ("WOOL", 1.0 / 3.0,   ANIMALS["SHEEP"]["first_yield_day"]),
    "GOOSE_COOP": ("EGG",  1.0,         ANIMALS["GOOSE"]["first_yield_day"]),
}
_KIND_COST: dict[str, float] = {
    "COW": float(ANIMALS["COW"]["cost"]),
    "SHEEP": float(ANIMALS["SHEEP"]["cost"]),
    "GOOSE_COOP": float(ANIMALS["GOOSE"]["cost"]),
}


@dataclass(slots=True)
class DevelopmentPlan:
    geese: int = 0
    cows: int = 0
    sheep: int = 0
    target_quads: int = 1
    crop_mix: dict[str, int] = field(default_factory=dict)
    strawberry_target: int = STRAWBERRY_TARGET_BUSHES

    @property
    def total_herd(self) -> int:
        return self.cows + self.sheep + self.geese

    @property
    def total_grazers(self) -> int:
        return self.cows + self.sheep


@dataclass(slots=True)
class ReplanState:
    day: int = -1
    shop_count: int = -1
    quad_count: int = -1
    prices: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class DriftState:
    prev_inv: dict[str, int] = field(default_factory=dict)
    prev_day: int = -1
    observed: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class BuildCandidate:
    kind: str
    item: str
    cost: float
    daily_profit: float
    first_yield_lag: int


def compute_daily_demand(unlocked_shops: list[str]) -> dict[str, float]:
    demand: dict[str, float] = {item: TOWN_CENTER_FLAT for item in SELLABLE_ITEMS}
    demand.pop("FERTILIZER", None)
    for shop in unlocked_shops:
        for item, qty in SHOP_DEMANDS.get(shop, {}).items():
            demand[item] = demand.get(item, 0.0) + qty
    return demand


def update_drift_tracker(day: int, market_inv: dict[str, int], state: DriftState) -> dict[str, float]:
    if day != state.prev_day:
        if state.prev_day >= 0:
            for item in SELLABLE_ITEMS:
                if item in state.prev_inv and item in market_inv:
                    state.observed[item] = float(state.prev_inv[item] - market_inv[item])
        state.prev_inv = dict(market_inv)
        state.prev_day = day
    return state.observed


def effective_drift(demand: dict[str, float], observed: dict[str, float]) -> dict[str, float]:
    return {
        item: max(demand.get(item, 0.0), 0.6 * observed.get(item, 0.0))
        for item in SELLABLE_ITEMS
    }


def herd_feed_cost(
    herd: int, drift: dict[str, float], market_inv: dict[str, int], horizon: int = 8
) -> float:
    w_drift = drift.get("WHEAT", 0.0) + float(herd)
    projected = max(1, market_inv.get("WHEAT", I0) - int(w_drift * horizon))
    return float(market_price("WHEAT", projected))


def expected_price_for_kind(
    item: str,
    lag: int,
    market_inv: dict[str, int],
    drift: dict[str, float],
    own_supply_units_per_day: float,
) -> int:
    net_drift = drift.get(item, 0.0) - own_supply_units_per_day
    projected_inv = market_inv.get(item, I0) - int(net_drift * lag)
    return market_price(item, max(1, projected_inv))


def demand_multiplier(item: str, demand: dict[str, float], market_inv: dict[str, int]) -> float:
    d = demand.get(item, 0.0)
    scarcity = I0 - market_inv.get(item, I0)
    if d >= 12.0:
        return 1.6
    if d >= 6.0:
        return 1.3
    if scarcity > 0.6 * MARKET_PARAMS[item]["T"]:
        return 1.4
    return 1.0


def _marginal_profit(
    kind: str,
    plan: DevelopmentPlan,
    demand: dict[str, float],
    drift: dict[str, float],
    market_inv: dict[str, int],
) -> float:
    item, rate, lag = _KIND_SPEC[kind]
    feed = herd_feed_cost(plan.total_herd + 1, drift, market_inv)
    own_count = plan.cows if kind == "COW" else (plan.sheep if kind == "SHEEP" else plan.geese)
    own_supply = own_count * rate
    p = expected_price_for_kind(item, lag, market_inv, drift, own_supply)
    return (p * rate - feed - HAND_COST_PER_ANIMAL_DAY) * demand_multiplier(item, demand, market_inv)


def daily_profit_of(kind: str, drift: dict[str, float], market_inv: dict[str, int]) -> float:
    if kind in _ANIMAL_PROFIT_SPEC:
        item, lag, rate = _ANIMAL_PROFIT_SPEC[kind]
        projected = max(1, market_inv.get(item, I0) - int(drift.get(item, 0.0) * lag))
        return market_price(item, projected) * rate - float(get_price("WHEAT", market_inv)) - HAND_COST_PER_ANIMAL_DAY
    if kind == "CARROT_PIVOT":
        projected = max(1, market_inv.get("CARROT", I0) - int(drift.get("CARROT", 0.0) * 3))
        return (3.5 * market_price("CARROT", projected) - CROPS["CARROT"]["seed_cost"]) / 3.0
    if kind == "TOMATO_PIVOT":
        projected = max(1, market_inv.get("TOMATO", I0) - int(drift.get("TOMATO", 0.0) * 8))
        return (6.0 * market_price("TOMATO", projected) - CROPS["TOMATO"]["seed_cost"]) / 11.0
    return 0.0


def enumerate_candidates(
    demand: dict[str, float],
    drift: dict[str, float],
    market_inv: dict[str, int],
) -> list[BuildCandidate]:
    specs = (
        ("GOOSE_COOP", "EGG", float(ANIMALS["GOOSE"]["cost"]), ANIMALS["GOOSE"]["first_yield_day"]),
        ("COW", "MILK", float(ANIMALS["COW"]["cost"]), ANIMALS["COW"]["first_yield_day"]),
        ("SHEEP", "WOOL", float(ANIMALS["SHEEP"]["cost"]), ANIMALS["SHEEP"]["first_yield_day"]),
        ("CARROT_PIVOT", "CARROT", float(CROPS["CARROT"]["seed_cost"]), CROPS["CARROT"]["max_yield_day"]),
        ("TOMATO_PIVOT", "TOMATO", float(CROPS["TOMATO"]["seed_cost"]), CROPS["TOMATO"]["first_yield_day"]),
    )
    out: list[BuildCandidate] = []
    for kind, item, cost, lag in specs:
        if kind.endswith("_PIVOT") and demand.get(item, 0.0) < 6.0:
            continue
        out.append(BuildCandidate(kind, item, cost, daily_profit_of(kind, drift, market_inv), lag))
    return out


def _unlocked_cluster_tiles(cluster: list[tuple[int, int]], unlocked_quads: list[str]) -> list[tuple[int, int]]:
    return [p for p in cluster if quad_of(p) in unlocked_quads]


def reserve_for_tomorrow(day: int, census: AnimalCensus, market_inv: dict[str, int]) -> float:
    if day >= TOTAL_DAYS - 2:
        return 0.0
    hires = float(total_hire_cost(6, 0))
    feed = (census.on_field + census.in_shed + 2) * float(get_price("WHEAT", market_inv))
    return hires + feed + 50.0


def build_development_plan(
    day: int,
    money: float,
    demand: dict[str, float],
    drift: dict[str, float],
    market_inv: dict[str, int],
    farm_state: dict[str, Any],
    census: AnimalCensus,
    unlocked_quads: list[str],
    unlocked_shops: list[str],
) -> DevelopmentPlan:
    cap_cows, cap_sheep, _ = get_target_animals(unlocked_shops)
    target_cows = cap_cows if day <= 14 else census.total_cows
    target_sheep = cap_sheep if day <= 14 else census.total_sheep

    plan = DevelopmentPlan(
        geese=census.total_geese,
        cows=max(census.total_cows, target_cows),
        sheep=max(census.total_sheep, target_sheep),
        target_quads=len(unlocked_quads),
        crop_mix={},
        strawberry_target=STRAWBERRY_TARGET_BUSHES,
    )
    pasture_cap = len(_unlocked_cluster_tiles(PASTURE_CLUSTER, unlocked_quads))
    coop_cap = len(_unlocked_cluster_tiles(COOP_CLUSTER, unlocked_quads))

    candidates = enumerate_candidates(demand, drift, market_inv)
    for c in candidates:
        c.daily_profit *= demand_multiplier(c.item, demand, market_inv)

    budget = money - reserve_for_tomorrow(day, census, market_inv)

    for c in sorted(
        (c for c in candidates if c.kind in ("CARROT_PIVOT", "TOMATO_PIVOT")),
        key=lambda x: -(x.daily_profit / x.cost),
    ):
        crop = "CARROT" if c.kind == "CARROT_PIVOT" else "TOMATO"
        while budget >= c.cost:
            cur = plan.crop_mix.get(crop, 0)
            if cur >= CROP_PIVOT_MAX_TILES[crop]:
                break
            plan.crop_mix[crop] = cur + 1
            budget -= c.cost

    has_yarn = "YARN_STORE" in unlocked_shops[:4]
    quad_limit = 4 if has_yarn else 3

    if (plan.total_grazers > pasture_cap or plan.geese > coop_cap) and len(unlocked_quads) < quad_limit:
        plan.target_quads = len(unlocked_quads) + 1

    return plan


def needs_replan(
    day: int,
    unlocked_shops: list[str],
    unlocked_quads: list[str],
    market_inv: dict[str, int],
    replan: ReplanState,
) -> bool:
    if (
        day != replan.day
        or len(unlocked_shops) != replan.shop_count
        or len(unlocked_quads) != replan.quad_count
    ):
        return True
    for item in SELLABLE_ITEMS:
        p_old = replan.prices.get(item, 0)
        p_new = get_price(item, market_inv)
        if p_old > 0 and abs(p_new - p_old) / p_old > 0.25:
            return True
    return False


def mark_replanned(
    day: int,
    unlocked_shops: list[str],
    unlocked_quads: list[str],
    market_inv: dict[str, int],
    replan: ReplanState,
) -> None:
    replan.day = day
    replan.shop_count = len(unlocked_shops)
    replan.quad_count = len(unlocked_quads)
    replan.prices = {item: get_price(item, market_inv) for item in SELLABLE_ITEMS}


# ---------------------------------------------------------------------------
# MODULE: market_orders.py
# ---------------------------------------------------------------------------

_WORK_PER_ANIMAL: float = 4.0
_WORK_PER_PLANT: float = 2.5
_WORK_PER_PLANTING: float = 3.5
_WORK_PER_DEPLOY: float = 7.0
_WORK_PER_PENDING_HARVEST: float = 1.5

_ABOVE_SQ_ITEMS: frozenset[str] = frozenset(("MELON", "WOOL"))
_SELL_PRICE_DROP_LIMIT: float = 0.15


def _throttled_qty(item: str, qty: int, market_inv: dict[str, int], day: int) -> int:
    if day >= LIQUIDATION_DAY:
        return qty
    if item not in _ABOVE_SQ_ITEMS:
        return qty
    p_now = get_price(item, market_inv)
    if p_now <= 1:
        return qty
    inv_now = market_inv.get(item, I0)
    lo, hi, best = 1, qty, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        p_after = get_price(item, {**market_inv, item: inv_now + mid})
        if (p_now - p_after) / p_now <= _SELL_PRICE_DROP_LIMIT:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def get_desired_hires(
    day: int,
    n_animals: int,
    n_plants: int,
    seeds_to_plant: int,
    animals_to_deploy: int,
    ready_harvest_tiles: int,
    unlocked_quads_count: int,
) -> int:
    """Calculate workforce needed today from physical task volume and farm layout size."""
    if day < 0 or day >= TOTAL_DAYS:
        raise ValueError(f"Invalid day: {day}")
    if day == 0:
        return 5

    work = (
        _WORK_PER_ANIMAL * n_animals
        + _WORK_PER_PLANT * n_plants
        + _WORK_PER_PLANTING * seeds_to_plant
        + _WORK_PER_DEPLOY * animals_to_deploy
        + _WORK_PER_PENDING_HARVEST * ready_harvest_tiles
    )

    efficiency = 11.0 if unlocked_quads_count == 1 else (9.0 if unlocked_quads_count == 2 else 7.5)
    desired = int(math.ceil(work / efficiency))

    if day <= 2:
        desired = max(2, min(desired, 4))
    elif day == LIQUIDATION_DAY:
        desired = min(desired, 9)
    elif day > LIQUIDATION_DAY:
        desired = min(desired, 8)

    return max(2, min(10, desired))


def _plan_sells(
    day: int,
    hour: int,
    shed: dict[str, int],
    farm_state: dict[str, Any],
    market_inv: dict[str, int],
    inventories: list[dict[str, int]],
    census: AnimalCensus,
) -> tuple[list[list[Any]], float]:
    orders: list[list[Any]] = []
    revenue: float = 0.0

    carried_sells: dict[str, int] = defaultdict(int)
    if day >= 29 and hour >= 15:
        for inv in inventories:
            for item, qty in inv.items():
                if qty > 0 and item not in ("COW", "SHEEP", "GOOSE"):
                    carried_sells[item] += qty

    fert_due_count = count_fertilize_due(farm_state)
    fert_in_shed = shed.get("FERTILIZER", 0) + carried_sells.get("FERTILIZER", 0)
    fert_capacity = N_FERTILIZER_HANDS * FERTILIZER_PICKUP_BATCH
    fert_reserve = 0 if day >= LIQUIDATION_DAY else min(fert_due_count, fert_capacity)
    fert_to_sell = max(0, fert_in_shed - fert_reserve)
    if fert_to_sell > 0:
        orders.append(["SELL", "FERTILIZER", fert_to_sell])
        revenue += fert_to_sell * get_price("FERTILIZER", market_inv)

    for item in ("MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO"):
        qty = shed.get(item, 0) + carried_sells.get(item, 0)
        if qty > 0:
            sell_qty = _throttled_qty(item, qty, market_inv, day)
            if sell_qty > 0:
                orders.append(["SELL", item, sell_qty])
                revenue += sell_qty * get_price(item, market_inv)

    wheat_in_shed = shed.get("WHEAT", 0) + carried_sells.get("WHEAT", 0)
    feed_buffer = 0 if day >= LIQUIDATION_DAY else max(4, census.n_feedable)
    if wheat_in_shed > feed_buffer and day > 0:
        excess = wheat_in_shed - feed_buffer
        orders.append(["SELL", "WHEAT", excess])
        revenue += excess * get_price("WHEAT", market_inv)

    return orders, revenue


def _plan_seed_buy(
    orders: list[list[Any]], crop: str, want: int, budget: float, cost: float
) -> tuple[float, float, int]:
    if want <= 0:
        return budget, cost, 0
    price = CROPS[crop]["seed_cost"]
    bought = min(want, int(budget // price))
    if bought <= 0:
        return budget, cost, 0
    orders.append(["BUY_SEED", crop, bought])
    spend = bought * price
    return budget - spend, cost + spend, bought


def plan_market_orders(
    day: int,
    hour: int,
    money: float,
    shed: dict[str, int],
    seeds: dict[str, int],
    market_inv: dict[str, int],
    unlocked_quads: list[str],
    unlocked_shops: list[str],
    farm_state: dict[str, Any],
    hires_today: int,
    inventories: list[dict[str, int]],
    needed_pastures: list[tuple[int, int]],
    census: AnimalCensus,
    plan: DevelopmentPlan,
    target_crops_today: list[str],
) -> list[list[Any]]:
    orders: list[list[Any]] = []
    budget = money

    shops_stable = len(unlocked_shops) >= 3
    if shops_stable:
        target_quadrants = plan.target_quads
    else:
        _, _, legacy_quads = get_target_animals(unlocked_shops)
        target_quadrants = max(plan.target_quads, legacy_quads)

    if day == 0 and hour <= 1:
        if hires_today < 5:
            for _ in range(5 - hires_today):
                orders.append(["HIRE"])
        if shed.get("SHEEP", 0) == 0 and len(farm_state["animals"]) == 0:
            orders.append(["BUY_ANIMAL", "SHEEP", 2])
            orders.append(["BUY_ANIMAL", "COW", 2])
            orders.append(["BUY_SEED", "MELON", 12])
            orders.append(["BUY_SEED", "WHEAT", 7])
            orders.append(["BUY_PRODUCT", "WHEAT", 4])
        return orders[:MAX_MARKET_ORDERS]

    sell_orders, revenue = _plan_sells(day, hour, shed, farm_state, market_inv, inventories, census)
    orders.extend(sell_orders)
    budget += revenue

    # Feed backup
    if census.n_feedable > 0:
        base_need = int(math.ceil(census.n_feedable / 4.0) * 4)
        reserve = FEED_WHEAT_RESERVE if base_need >= FEED_WHEAT_RESERVE * FEED_WHEAT_RESERVE_MIN_RATIO else 0
        needed_wheat = base_need + reserve
        wheat_avail = shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for inv in inventories)
        if day < LIQUIDATION_DAY and wheat_avail < needed_wheat:
            wp = get_price("WHEAT", market_inv)
            can_buy = min(needed_wheat - wheat_avail, int(budget // max(1, wp)))
            if can_buy > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", can_buy])
                budget -= float(wp * can_buy)

    # Hires
    ready_harvest = sum(1 for p in farm_state["plants"] if p["yield_units"] > 0)
    ready_harvest += sum(1 for a in farm_state["animals"] if a["yield_units"] > 0)
    seeds_to_plant = min(
        sum(seeds.get(c, 0) for c in target_crops_today), len(farm_state["empty_tiles"])
    )
    desired = get_desired_hires(
        day,
        len(farm_state["animals"]),
        len(farm_state["plants"]),
        seeds_to_plant,
        census.in_shed + len(needed_pastures),
        ready_harvest,
        len(unlocked_quads),
    )
    new_hires = max(0, desired - hires_today)
    while new_hires > 0:
        hcost = total_hire_cost(new_hires, hires_today)
        if budget >= hcost:
            orders.extend([["HIRE"] for _ in range(new_hires)])
            budget -= float(hcost)
            break
        new_hires -= 1

    # Seeds
    cash_reserve = SEED_PURCHASE_CASH_RESERVE if 4 <= day < LIQUIDATION_DAY else 0.0
    spendable = max(0.0, budget - cash_reserve)
    seed_cost = 0.0
    empty_count = len(farm_state["empty_tiles"])

    if shops_stable and plan.crop_mix:
        straw_done = committed_count(farm_state, seeds, "STRAWBERRY")
        for crop, target in plan.crop_mix.items():
            if crop not in CROPS or target <= 0:
                continue
            early_ok = 4 <= day <= 10 and straw_done >= plan.strawberry_target
            late_ok = STRAWBERRY_LAST_PLANT_DAY < day <= TOTAL_DAYS - 5 and crop == "CARROT"
            if not (early_ok or late_ok):
                continue
            spec = CROPS[crop]
            lag = spec["first_yield_day"] if spec["is_ongoing"] else spec["max_yield_day"]
            if day > TOTAL_DAYS - lag - 2:
                continue
            curr = committed_count(farm_state, seeds, crop)
            spendable, seed_cost, _ = _plan_seed_buy(orders, crop, target - curr, spendable, seed_cost)

    if 4 <= day <= WHEAT_LAST_PLANT_DAY:
        curr_wheat = committed_count(farm_state, seeds, "WHEAT")
        if curr_wheat < census.n_feedable and empty_count > 0:
            spendable, seed_cost, bought = _plan_seed_buy(
                orders, "WHEAT", min(census.n_feedable - curr_wheat, empty_count), spendable, seed_cost
            )
            empty_count = max(0, empty_count - bought)

        total_straw = committed_count(farm_state, seeds, "STRAWBERRY")
        if day <= STRAWBERRY_LAST_PLANT_DAY and total_straw < plan.strawberry_target and empty_count > 0:
            spendable, seed_cost, bought = _plan_seed_buy(
                orders, "STRAWBERRY", min(plan.strawberry_target - total_straw, empty_count), spendable, seed_cost
            )
            empty_count = max(0, empty_count - bought)

        # Dynamic wheat seed buffer: continuous flow on days 4..23, zero on days 24..26
        wheat_buffer = 15 if 4 <= day <= 23 else 0

        # On the final planting day (day 26), restrict purchases to the first half of the day
        if day < WHEAT_LAST_PLANT_DAY or (day == WHEAT_LAST_PLANT_DAY and hour < 12):
            want_wheat = max(0, empty_count + wheat_buffer - seeds.get("WHEAT", 0))
            spendable, seed_cost, _ = _plan_seed_buy(orders, "WHEAT", want_wheat, spendable, seed_cost)
    elif 1 <= day <= 3:
        want_wheat = max(0, empty_count - seeds.get("WHEAT", 0))
        spendable, seed_cost, _ = _plan_seed_buy(orders, "WHEAT", want_wheat, spendable, seed_cost)

    budget -= seed_cost

    # --- Animals: gated until strawberry allocation is filled, not by a fixed day ---
    animal_orders: list[list[Any]] = []
    straw_ready = (
        committed_count(farm_state, seeds, "STRAWBERRY") >= plan.strawberry_target
        or day > STRAWBERRY_LAST_PLANT_DAY
    )
    if day >= 4 and straw_ready:
        animal_targets = (("GOOSE", plan.geese), ("SHEEP", plan.sheep), ("COW", plan.cows))
        for species, target in animal_targets:
            have = census.total_geese if species == "GOOSE" else (census.total_sheep if species == "SHEEP" else census.total_cows)
            deficit = target - have
            if deficit <= 0:
                continue
            price = ANIMALS[species]["cost"]
            if day > TOTAL_DAYS - ANIMALS[species]["first_yield_day"] - MIN_PAYBACK_DAYS:
                continue
            want = min(2, deficit, int((budget - 100) // price))
            if want > 0:
                animal_orders.append(["BUY_ANIMAL", species, want])
                budget -= float(want * price)

    # --- Land: calculate on REMAINING budget (after animals) ---
    land_orders: list[list[Any]] = []
    want_land = target_quadrants > len(unlocked_quads)
    if "NE" not in unlocked_quads and day >= 5 and (want_land and budget >= 1000 or budget >= 1000 + LAND_RESERVE_NE):
        land_orders.append(["BUY_LAND"])
        budget -= 1000.0
    if "SW" not in unlocked_quads and "NE" in unlocked_quads and day >= 9 and (want_land and budget >= 2000 or budget >= 2000 + LAND_RESERVE_SW_SE):
        land_orders.append(["BUY_LAND"])
        budget -= 2000.0
    if "SE" not in unlocked_quads and "SW" in unlocked_quads and day >= 12 and target_quadrants >= 4 and budget >= 4000 + LAND_RESERVE_SW_SE:
        land_orders.append(["BUY_LAND"])
        budget -= 4000.0

    # --- Order Queue: Land appended before Animals for truncation priority ---
    orders.extend(land_orders)
    orders.extend(animal_orders)

    return orders[:MAX_MARKET_ORDERS]


# ---------------------------------------------------------------------------
# MODULE: dispatch.py
# ---------------------------------------------------------------------------

_VALUABLE_ITEMS: frozenset[str] = frozenset((
    "MELON", "STRAWBERRY", "WOOL", "MILK", "EGG", "TOMATO", "CARROT"
))


def _has_sellable_cargo(inv: dict[str, int], animals_unfed: bool, fert_due: bool) -> bool:
    if any(inv.get(item, 0) > 0 for item in _VALUABLE_ITEMS):
        return True
    if not fert_due and inv.get("FERTILIZER", 0) > 0:
        return True
    if not animals_unfed and inv.get("WHEAT", 0) > 0:
        return True
    return False


@dataclass(slots=True)
class Task:
    priority: int
    pos: tuple[int, int]
    action: list[Any]
    need: tuple[str, ...] | None = None


def select_target_crops(day: int, seeds_stock: dict[str, int]) -> list[str]:
    if day > WHEAT_LAST_PLANT_DAY:
        return []
    if 21 <= day <= WHEAT_LAST_PLANT_DAY:
        return ["WHEAT"]
    crops: list[str] = []
    if day <= 12 and seeds_stock.get("MELON", 0) > 0:
        crops.append("MELON")
    if day <= STRAWBERRY_LAST_PLANT_DAY and seeds_stock.get("STRAWBERRY", 0) > 0:
        crops.append("STRAWBERRY")
    if 13 <= day <= MELON_LAST_PLANT_DAY and seeds_stock.get("MELON", 0) > 0:
        crops.append("MELON")
    crops.append("WHEAT")
    return crops


def _live_reserved_tiles(
    cluster: list[tuple[int, int]],
    farm_state: dict[str, Any],
    unlocked_quads: list[str],
    target_slots: int,
    kind: str,
) -> frozenset[tuple[int, int]]:
    empty_key = "empty_pastures" if kind == "PASTURE" else "empty_coops"
    built: list[tuple[int, int]] = [
        a["pos"] for a in farm_state["animals"]
        if a["pos"] in cluster and (
            (kind == "PASTURE" and a["animal"] in ("COW", "SHEEP"))
            or (kind == "COOP" and a["animal"] == "GOOSE")
        )
    ]
    built += [p for p in farm_state[empty_key] if p in cluster]
    reserved: set[tuple[int, int]] = set(built)

    if target_slots <= 0 and not reserved:
        return frozenset()

    # Limit: on a single unlocked quadrant reserve at most 4 tiles for animals
    max_cap = 4 if len(unlocked_quads) == 1 else (MAX_TOTAL_HERD if kind == "PASTURE" else target_slots)
    effective_target = min(max_cap, max(target_slots, len(reserved)))

    for pos in cluster:
        if len(reserved) >= effective_target:
            break
        if quad_of(pos) in unlocked_quads:
            reserved.add(pos)

    return frozenset(reserved)


def compute_needed_pastures(
    farm_state: dict[str, Any],
    census: AnimalCensus,
) -> list[tuple[int, int]]:
    needed: list[tuple[int, int]] = []
    planted_positions = {p["pos"]: p for p in farm_state["plants"]}
    occupied_pastures = census.field_cows + census.field_sheep
    for p_pos in PASTURE_CLUSTER:
        if len(needed) + occupied_pastures + len(farm_state["empty_pastures"]) >= census.total:
            break
        if (
            p_pos in farm_state["empty_tiles"]
            or p_pos in farm_state["weeds"]
            or p_pos in planted_positions
        ):
            needed.append(p_pos)
    return needed


def _add_pickup_tasks(
    tasks: list[Task],
    priority: int,
    item: str,
    qty: int,
    shed_subset: list[tuple[int, int]] | None = None,
) -> None:
    for shed_pos in (shed_subset if shed_subset is not None else SHED_TILES):
        tasks.append(Task(priority, shed_pos, ["PICKUP", item, qty]))


def _resolve_structure_action(
    pos: tuple[int, int],
    farm_state: dict[str, Any],
    planted_positions: dict[tuple[int, int], dict[str, Any]],
    build_action: str,
) -> list[Any] | None:
    if pos in farm_state["weeds"]:
        return ["DIG"]
    if pos in planted_positions:
        p = planted_positions[pos]
        if p["yield_units"] > 0 or (not p["is_ongoing"] and p["age"] >= p["first_yield_day"] - 1):
            return None
        return ["DIG"]
    if pos in farm_state["empty_tiles"]:
        return [build_action]
    return None


def _build_tasks(
    farm_state: dict[str, Any],
    shed: dict[str, int],
    seeds_stock: dict[str, int],
    needed_pastures: list[tuple[int, int]],
    all_units: list[tuple[tuple[int, int], dict[str, int]]],
    day: int,
    hour: int,
    plan: DevelopmentPlan,
    unlocked_quads: list[str],
) -> list[Task]:
    tasks: list[Task] = []

    # 1. Animal care
    for a in farm_state["animals"]:
        pos = a["pos"]
        if not a["fed_today"]:
            feed_pri = 350 if a.get("consecutive_unfed", 0) >= 1 else 260
            tasks.append(Task(feed_pri, pos, ["FEED"], need=("WHEAT",)))
        if not a["cared_today"]:
            tasks.append(Task(230, pos, ["CARE"]))
        if a["fertilizer_available"]:
            tasks.append(Task(210, pos, ["COLLECT_FERTILIZER"]))
        if a["yield_units"] > 0:
            tasks.append(Task(205, pos, ["HARVEST"]))

    # 2. Animal placement & pickups
    n_cow_shed, n_sheep_shed = shed.get("COW", 0), shed.get("SHEEP", 0)
    n_goose_shed = shed.get("GOOSE", 0)
    has_carried_cow = any(inv.get("COW", 0) > 0 for _, inv in all_units)
    has_carried_sheep = any(inv.get("SHEEP", 0) > 0 for _, inv in all_units)
    carried_goose_count = sum(1 for _, inv in all_units if inv.get("GOOSE", 0) > 0)

    if (n_cow_shed > 0 or n_sheep_shed > 0 or has_carried_cow or has_carried_sheep) and farm_state["empty_pastures"]:
        for pos in farm_state["empty_pastures"]:
            tasks.append(Task(300, pos, ["PLACE"], need=("COW", "SHEEP")))

    if (n_goose_shed > 0 or carried_goose_count > 0) and farm_state["empty_coops"]:
        for pos in farm_state["empty_coops"]:
            tasks.append(Task(300, pos, ["PLACE"], need=("GOOSE",)))

    if (n_cow_shed > 0 or n_sheep_shed > 0) and farm_state["empty_pastures"]:
        _add_pickup_tasks(tasks, 185, "COW" if n_cow_shed > 0 else "SHEEP", 1)

    if n_goose_shed > 0 and farm_state["empty_coops"]:
        _add_pickup_tasks(tasks, 185, "GOOSE", 1)

    # 3. Crops Harvesting (Priority 205 — same as animal harvest)
    for p in farm_state["plants"]:
        if p["is_expired"]:
            tasks.append(Task(150, p["pos"], ["DIG"]))
        ready_to_harvest = p["yield_units"] > 0 and (
            p["is_ongoing"]
            or p["age"] >= p["max_yield_day"]
            or day >= 28
        )
        if ready_to_harvest:
            tasks.append(Task(205, p["pos"], ["HARVEST"]))
        if not p["watered_today"] and not p["is_expired"]:
            tasks.append(Task(200, p["pos"], ["WATER"]))
        if p["fertilize_due"]:
            tasks.append(Task(180, p["pos"], ["FERTILIZE"], need=("FERTILIZER",)))

    # 4. Shed Drop Tasks (Priority 160 for travel; Priority 210 if worker is already standing on a shed tile)
    standing_shed_positions = {pos for pos, _ in all_units if pos in SHED_TILES_SET}
    for shed_pos in SHED_TILES:
        tasks.append(Task(160, shed_pos, ["DROP"]))
        if shed_pos in standing_shed_positions:
            tasks.append(Task(210, shed_pos, ["DROP"]))

    # 5. Fertilizer Pickup (targeted to needed carrier count only)
    fert_due_count = count_fertilize_due(farm_state)
    if fert_due_count > 0 and shed.get("FERTILIZER", 0) > 0:
        carried_fert = sum(inv.get("FERTILIZER", 0) for _, inv in all_units)
        if carried_fert < fert_due_count:
            needed_carriers = min(4, max(1, math.ceil((fert_due_count - carried_fert) / 4.0)))
            _add_pickup_tasks(tasks, 178, "FERTILIZER", 4, shed_subset=SHED_TILES[:needed_carriers])

    # 6. Weeds
    if day <= WHEAT_LAST_PLANT_DAY:
        for pos in farm_state["weeds"]:
            tasks.append(Task(150, pos, ["DIG"]))

    # 7. Crop planting
    if hour < TURNS_PER_DAY - 1:
        res_pasture = _live_reserved_tiles(PASTURE_CLUSTER, farm_state, unlocked_quads, plan.total_grazers, "PASTURE")
        res_coop = _live_reserved_tiles(COOP_CLUSTER, farm_state, unlocked_quads, plan.geese, "COOP")
        needed_structs = set(needed_pastures)
        plantable = [
            pos for pos in farm_state["empty_tiles"]
            if pos not in res_pasture and pos not in res_coop and pos not in needed_structs
        ]
        plantable.sort(key=lambda p: (manhattan(p, (4, 4)), p[1], p[0]))

        if plan.crop_mix:
            for pivot_crop in ("CARROT", "TOMATO"):
                if not plantable:
                    break
                target = plan.crop_mix.get(pivot_crop, 0)
                available = seeds_stock.get(pivot_crop, 0)
                if target <= 0 or available <= 0:
                    continue
                n_pivot = min(len(plantable), available, target)
                for pos in plantable[:n_pivot]:
                    tasks.append(Task(138, pos, ["PLANT", pivot_crop]))
                plantable = plantable[n_pivot:]

        for crop in select_target_crops(day, seeds_stock):
            if not plantable:
                break
            n_plant = min(len(plantable), seeds_stock.get(crop, 0))
            if n_plant <= 0:
                continue
            for pos in plantable[:n_plant]:
                tasks.append(Task(140, pos, ["PLANT", crop]))
            plantable = plantable[n_plant:]

    # 8. Pastures & Coops construction
    planted_positions = {p["pos"]: p for p in farm_state["plants"]}
    p_pri = 200 if (n_cow_shed > 0 or n_sheep_shed > 0) else 120
    for pos in needed_pastures:
        act = _resolve_structure_action(pos, farm_state, planted_positions, "BUILD_PASTURE")
        if act is not None:
            tasks.append(Task(p_pri, pos, act))

    if plan.geese > 0:
        geese_waiting = n_goose_shed + carried_goose_count
        if geese_waiting > 0 or plan.geese > len(farm_state["empty_coops"]):
            c_pri = 200 if geese_waiting > 0 else 125
            for pos in _live_reserved_tiles(COOP_CLUSTER, farm_state, unlocked_quads, plan.geese, "COOP"):
                act = _resolve_structure_action(pos, farm_state, planted_positions, "BUILD_COOP")
                if act is not None:
                    tasks.append(Task(c_pri, pos, act))

    # 9. Feed pickup
    unfed = sum(1 for a in farm_state["animals"] if not a["fed_today"])
    if unfed > 0 and shed.get("WHEAT", 0) > 0:
        carried_feeders = sum(1 for _, inv in all_units if inv.get("WHEAT", 0) > 0)
        needed_feeders = min(4, max(1, math.ceil(unfed / 4.0)))
        if carried_feeders < needed_feeders:
            _add_pickup_tasks(tasks, 250, "WHEAT", 4, shed_subset=SHED_TILES[: needed_feeders - carried_feeders])

    return tasks


def _assign_tasks(
    active_units: list[tuple[int, tuple[int, int], dict[str, int]]],
    tasks: list[Task],
    animals_unfed: bool,
    fert_due: bool,
) -> dict[int, Task]:
    def eligible(inv: dict[str, int], task: Task) -> bool:
        if task.action[0] == "DROP":
            return _has_sellable_cargo(inv, animals_unfed, fert_due)
        if task.action[0] == "PICKUP" and len(task.action) >= 2:
            item = task.action[1]
            if inv.get(item, 0) > 0:
                return False
            if item in ("COW", "SHEEP", "GOOSE") and any(inv.get(a, 0) > 0 for a in ("COW", "SHEEP", "GOOSE")):
                return False
        return task.need is None or any(inv.get(item, 0) > 0 for item in task.need)

    assigned: dict[int, Task] = {}
    claimed_units: set[int] = set()
    claimed_pos: set[tuple[int, int]] = set()

    for ui, pos, inv in active_units:
        standing = [t for t in tasks if t.pos == pos and t.pos not in claimed_pos and eligible(inv, t)]
        if standing:
            best_t = max(standing, key=lambda t: t.priority)
            assigned[ui] = best_t
            claimed_units.add(ui)
            claimed_pos.add(pos)

    for ui, pos, inv in active_units:
        if ui in claimed_units:
            continue
        available = [t for t in tasks if t.pos not in claimed_pos and eligible(inv, t)]
        if not available:
            continue
        best_t = max(available, key=lambda t: (t.priority // 20, -manhattan(pos, t.pos), t.priority))
        assigned[ui] = best_t
        claimed_units.add(ui)
        claimed_pos.add(best_t.pos)

    return assigned


def _to_action(
    pos: tuple[int, int],
    inv: dict[str, int],
    task: Task,
    step_to: Callable[[tuple[int, int]], list[Any]],
) -> list[Any]:
    if pos != task.pos:
        return step_to(task.pos)
    if task.action[0] == "PLACE":
        for animal in ("COW", "SHEEP", "GOOSE"):
            if inv.get(animal, 0) > 0:
                return ["PLACE", animal]
        return ["PASS"]
    return task.action


def dispatch_units(
    all_units: list[tuple[tuple[int, int], dict[str, int]]],
    farm_state: dict[str, Any],
    shed: dict[str, int],
    seeds: dict[str, int],
    day: int,
    hour: int,
    needed_pastures: list[tuple[int, int]],
    plan: DevelopmentPlan,
    unlocked_quads: list[str],
) -> list[list[Any]]:
    evac_actions: dict[int, list[Any]] = {}
    if day >= 29 and hour >= 15:
        for ui, (pos, inv) in enumerate(all_units):
            if any(inv.get(item, 0) > 0 for item in SELLABLE_ITEMS):
                evac_actions[ui] = go_shed_or_drop(pos)

    tasks = _build_tasks(
        farm_state, shed, dict(seeds), needed_pastures,
        all_units, day, hour, plan=plan, unlocked_quads=unlocked_quads,
    )
    active_units = [
        (ui, pos, inv)
        for ui, (pos, inv) in enumerate(all_units)
        if ui not in evac_actions
    ]

    animals_unfed = any(not a["fed_today"] for a in farm_state["animals"])
    fert_due = count_fertilize_due(farm_state) > 0

    assigned = _assign_tasks(active_units, tasks, animals_unfed, fert_due)

    actions: list[list[Any]] = []
    for ui, (pos, inv) in enumerate(all_units):
        if ui in evac_actions:
            actions.append(evac_actions[ui])
            continue

        task = assigned.get(ui)
        if task is not None:
            step_to = lambda target, _p=pos: [bfs_step(_p, target) or "PASS"]
            actions.append(_to_action(pos, inv, task, step_to))
            continue

        inv_total = sum(inv.values())
        if inv_total > 0:
            feed_only = inv.get("WHEAT", 0) == inv_total
            if not (feed_only and any(not a["fed_today"] for a in farm_state["animals"])):
                actions.append(go_shed_or_drop(pos))
                continue
        actions.append(["PASS"])

    return actions


# ---------------------------------------------------------------------------
# MODULE: main.py
# ---------------------------------------------------------------------------

"""Kaggle Agent Entry Point — EcoBot v3 Modular Runner."""





@dataclass(slots=True)
class AgentMemory:
    plan: DevelopmentPlan = field(default_factory=DevelopmentPlan)
    replan: ReplanState = field(default_factory=ReplanState)
    drift: DriftState = field(default_factory=DriftState)


_MEMORY = AgentMemory()


def agent(obs: dict[str, Any]) -> dict[str, Any]:
    global _MEMORY
    if obs.get("step", 0) == 0:
        _MEMORY = AgentMemory()

    player = obs["player"]
    day = obs["day"]
    hour = obs["hour"]

    me = obs["farms"][player]
    private = obs["private"]
    market = obs["market"]
    unlocked_shops = obs["town"]["unlocked_shops"]

    money = float(me["money"])
    tiles = me["tiles"]
    unlocked_quads = me["unlocked_quadrants"]
    shed = private["shed"]
    seeds = private["seeds"]
    market_inv = market["inventory"]
    inventories = private["inventories"]
    hires_today = me["hires_today"]

    # 1. State parsing & census
    farm_state = parse_farm_state(tiles, day)
    census = count_animal_census(farm_state, shed, inventories)

    # 2. Economy & Replan (Pure functions with explicit state tracking)
    demand = compute_daily_demand(unlocked_shops)
    observed = update_drift_tracker(day, market_inv, _MEMORY.drift)
    drift = effective_drift(demand, observed)

    if needs_replan(day, unlocked_shops, unlocked_quads, market_inv, _MEMORY.replan):
        _MEMORY.plan = build_development_plan(
            day, money, demand, drift, market_inv, farm_state, census,
            unlocked_quads, unlocked_shops,
        )
        mark_replanned(day, unlocked_shops, unlocked_quads, market_inv, _MEMORY.replan)

    # 3. Market Orders
    needed_pastures = compute_needed_pastures(farm_state, census)
    target_crops_today = select_target_crops(day, seeds)

    market_orders = plan_market_orders(
        day, hour, money, shed, seeds, market_inv,
        unlocked_quads, unlocked_shops, farm_state, hires_today, inventories,
        needed_pastures, census, _MEMORY.plan, target_crops_today,
    )

    # 4. Units & Field Actions
    farmer_pos = (me["farmer"][0], me["farmer"][1])
    all_units = [(farmer_pos, inventories[0])]
    all_units.extend(((h[0], h[1]), inventories[idx + 1]) for idx, h in enumerate(me["hands"]))

    actions = dispatch_units(
        all_units, farm_state, shed, seeds, day, hour, needed_pastures,
        plan=_MEMORY.plan, unlocked_quads=unlocked_quads,
    )

    return {
        "farmer": actions[0],
        "hands": actions[1:],
        "market": market_orders,
    }

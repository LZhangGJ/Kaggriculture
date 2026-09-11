"""Kaggle Submission — Single File Bundle."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# MODULE: constants.py
# ---------------------------------------------------------------------------

TOTAL_DAYS: int = 30
TURNS_PER_DAY: int = 24
PURCHASE_HOUR: int = 21   # Single daily window for BUY_SEED / BUY_ANIMAL / BUY_LAND
MAX_MARKET_ORDERS: int = 10
MAX_HANDS: int = 13
BOARD_SIZE: int = 10
I0: int = 10_000

# Interleaved by distance rings from shed junction across NW, NE, SW (total 18 pasture slots)
PASTURE_CLUSTER: list[tuple[int, int]] = [
    # Ring 0: Shed corner tiles (dist 0)
    (4, 4), (5, 4), (4, 5),
    # Ring 1: Immediate shed neighbors (dist 1)
    (4, 3), (3, 4), (5, 3), (6, 4), (3, 5), (4, 6),
    # Ring 2: Secondary cluster tiles (dist 2)
    (3, 3), (4, 2), (2, 4), (6, 3), (5, 2), (7, 4), (2, 5), (3, 6), (4, 7),
]
SHED_TILES: list[tuple[int, int]] = [(4, 4), (5, 4), (4, 5), (5, 5)]
SHED_TILES_SET: frozenset[tuple[int, int]] = frozenset(SHED_TILES)

# Furthest corner tiles in each quadrant from the shed — unconditionally excluded
# from planting and weed digging (see dispatch.py:build_task_catalog) so workers never
# make long detours across the entire map for low-value tiles.
BLOCKED_CLUSTER: list[tuple[int, int]] = [
    (0, 0), (1, 0), (0, 1),                                                     # NW far corner
    (BOARD_SIZE - 1, 0), (BOARD_SIZE - 2, 0), (BOARD_SIZE - 1, 1),               # NE far corner
    (0, BOARD_SIZE - 1), (1, BOARD_SIZE - 1), (0, BOARD_SIZE - 2),               # SW far corner
    (BOARD_SIZE - 1, BOARD_SIZE - 1), (BOARD_SIZE - 2, BOARD_SIZE - 1),          # SE far corner
    (BOARD_SIZE - 1, BOARD_SIZE - 2),
]
BLOCKED_CLUSTER_SET: frozenset[tuple[int, int]] = frozenset(BLOCKED_CLUSTER)

MILK_SUPPORT_SHOPS: tuple[str, ...] = ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP")
EGG_SUPPORT_SHOPS: tuple[str, ...] = ("BAKERY", "BRUNCH_SPOT")
WOOL_SUPPORT_SHOPS: tuple[str, ...] = ("YARN_STORE",)

# Two-stage demand model (Stage 1): shop draw parameters and probability threshold
# where the chance of a relevant shop opening is sufficient to expand the herd ceiling
# in advance, without waiting for its actual appearance in unlocked_shops.
TOWN_SHOP_UNLOCK_INTERVAL: int = 3   # Town shop unlock interval in days
MAX_SHOP_UNLOCKS: int = 8            # Maximum number of town shop draws
RECOURSE_PROBABILITY_THRESHOLD: float = 0.5

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

TOWN_CENTER_FLAT: float = 1.0
LIQUIDATION_DAY: int = 28
LAST_DAY: int = TOTAL_DAYS - 1
MELON_LAST_PLANT_DAY: int = 18
STRAWBERRY_LAST_PLANT_DAY: int = 14
WHEAT_LAST_PLANT_DAY: int = 26
PASTURE_RESERVATION_LAST_DAY: int = 16
MAX_TOTAL_HERD: int = 18
MIN_PAYBACK_DAYS: int = 8
HAND_COST_PER_ANIMAL_DAY: float = 2.0
SEED_PURCHASE_CASH_RESERVE: float = 200.0
FEED_WHEAT_RESERVE: int = 4
FEED_WHEAT_RESERVE_MIN_RATIO: int = 2
FEED_WHEAT_SELL_RATIO: int = 4


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

FIBONACCI_HIRE_COSTS: tuple[int, ...] = (
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765,
)

class _Constants:
    def __getattr__(self, name: str) -> Any:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f'Constant {name!r} is not defined') from None

c = _Constants()


# ---------------------------------------------------------------------------
# MODULE: market.py
# ---------------------------------------------------------------------------

def total_hire_cost(count: int, already_hired: int) -> int:
    if count < 0 or already_hired < 0 or already_hired + count > len(c.FIBONACCI_HIRE_COSTS):
        raise ValueError(
            f"Invalid hire range: count={count}, already_hired={already_hired}, max_limit={len(c.FIBONACCI_HIRE_COSTS)}"
        )
    return sum(c.FIBONACCI_HIRE_COSTS[already_hired : already_hired + count])


def _hinge(x: float, t: float) -> float:
    u = x / max(1.0, t)
    return u + 8.0 * max(0.0, u - 1.0) ** 2


_SHAPE_FUNCS: dict[str, Callable[[float, float], float]] = {
    "linear": lambda x, t: x,
    "sq": lambda x, t: x * x,
    "sqrt": lambda x, t: math.sqrt(max(0.0, x)),
    "log": lambda x, t: math.log(1.0 + max(0.0, x)),
    "hinge": _hinge,
}


def _shape(func: str, x: float, t: float) -> float:
    if func not in _SHAPE_FUNCS:
        raise ValueError(f"Unknown shape function: {func!r}")
    return _SHAPE_FUNCS[func](x, t)


def market_price(item: str, inv: int) -> int:
    if item not in c.MARKET_PARAMS:
        raise ValueError(f"Unknown market item: {item!r}")
    p = c.MARKET_PARAMS[item]
    base, t = float(p["base"]), float(p["T"])
    if inv == c.I0:
        return int(round(base))

    below = inv < c.I0
    func = p["below_func"] if below else p["above_func"]
    target = p["below_target"] if below else p["above_target"]
    sign = 1 if below else -1
    x = float(c.I0 - inv) if below else float(inv - c.I0)

    denom = _shape(func, t, t)
    amp = (target * base) / denom if denom > 0 else 0.0
    return max(1, int(round(base + sign * amp * _shape(func, x, t))))


def get_price(item: str, market_inv: dict[str, int]) -> int:
    return market_price(item, market_inv.get(item, c.I0))


# ---------------------------------------------------------------------------
# MODULE: navigation.py
# ---------------------------------------------------------------------------

def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def quad_of(pos: tuple[int, int]) -> str:
    half = c.BOARD_SIZE // 2
    if pos[1] < half:
        return "NE" if pos[0] >= half else "NW"
    return "SE" if pos[0] >= half else "SW"


def bfs_step(start: tuple[int, int], goal: tuple[int, int]) -> str | None:
    if start == goal:
        return None
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    return "SOUTH" if dy > 0 else "NORTH"


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
    def n_feedable(self) -> int:
        return self.on_field + self.in_shed


def count_animal_census(
    farm_state: dict[str, Any],
    shed: dict[str, int],
    inventories: list[dict[str, int]],
) -> AnimalCensus:
    field = Counter(a["animal"] for a in farm_state["animals"])
    carried = Counter()
    for inv in inventories:
        for species in ("COW", "SHEEP", "GOOSE"):
            carried[species] += inv.get(species, 0)
    return AnimalCensus(
        field_cows=field["COW"],
        field_sheep=field["SHEEP"],
        field_geese=field["GOOSE"],
        shed_cows=shed.get("COW", 0),
        shed_sheep=shed.get("SHEEP", 0),
        shed_geese=shed.get("GOOSE", 0),
        carried_cows=carried["COW"],
        carried_sheep=carried["SHEEP"],
        carried_geese=carried["GOOSE"],
    )


def kept_feedable_count(census: AnimalCensus, retained_caps: dict[str, int] | None) -> int:
    """On-field + in-shed animals that survive a cull's retained_caps, per species."""
    caps = retained_caps if retained_caps is not None else {}
    return (
        min(census.field_cows + census.shed_cows, caps.get("COW", 999))
        + min(census.field_sheep + census.shed_sheep, caps.get("SHEEP", 999))
        + min(census.field_geese + census.shed_geese, caps.get("GOOSE", 999))
    )


def wheat_feed_thresholds(
    census: AnimalCensus, retained_caps: dict[str, int] | None, day: int
) -> tuple[int, int]:
    """Buy floor and sell ceiling for shed wheat, with a deliberate dead zone."""
    if day >= c.LIQUIDATION_DAY:
        return 0, 0
    n_kept = kept_feedable_count(census, retained_caps)
    floor = max(c.FEED_WHEAT_RESERVE, n_kept * c.FEED_WHEAT_RESERVE_MIN_RATIO)
    ceiling = max(c.FEED_WHEAT_RESERVE * c.FEED_WHEAT_SELL_RATIO, n_kept * c.FEED_WHEAT_SELL_RATIO)
    return floor, ceiling


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

    for y in range(c.BOARD_SIZE):
        for x in range(c.BOARD_SIZE):
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
                        "consecutive_unfed": tile.get("consecutive_unfed", 0),
                        "cared_today": tile.get("cared_today", False),
                        "fertilizer_available": tile.get("fertilizer_available", False),
                        "yield_units": tile.get("yield_units", 0),
                    })
            elif kind == "PLANT":
                crop = tile.get("crop")
                if crop is None or crop not in c.CROPS:
                    raise ValueError(f"Unknown or missing crop {crop!r} at {pos}: {tile}")
                planted_day = tile.get("planted_day", day)
                age = day - planted_day
                spec = c.CROPS[crop]
                yu = tile.get("yield_units", 0)
                watered = tile.get("watered_today", False)
                fert_until = tile.get("fertilized_until_day", -1)
                consecutive_unwatered = tile.get("consecutive_unwatered")
                if consecutive_unwatered is None:
                    raise KeyError(f"Missing consecutive_unwatered at {pos}: {tile}")

                if spec["is_ongoing"]:
                    is_expired = age >= spec["yield_days"][-1] + 1
                else:
                    is_expired = age >= spec["max_yield_day"] + 1

                fert_due = False
                bonus_active = False
                if spec["is_ongoing"]:
                    upcoming_ages = [d for d in spec["yield_days"] if d >= age + 1]
                    if upcoming_ages:
                        next_yield_age = upcoming_ages[0]
                        # Overnight refresh day -> day + 1 advances age to age + 1.
                        # Payout for next day is reachable if day + 1 < c.LAST_DAY.
                        payoff_reachable = (day + 1) < c.LAST_DAY
                        is_next_yield_tomorrow = (next_yield_age == age + 1)
                        fert_due = payoff_reachable and is_next_yield_tomorrow and fert_until < (day + 1)
                        # Bonus requires both watering and fertilizer for the overnight refresh.
                        bonus_active = payoff_reachable and is_next_yield_tomorrow and (fert_until >= (day + 1) or fert_due)
                else:
                    # One-time crops: every watered day inside [bonus_start_day, max_yield_day]
                    # adds yield via that day's refresh, so watering there is never optional.
                    bonus_active = spec["bonus_start_day"] <= age <= spec["max_yield_day"]

                water_needed = not watered and day < c.LAST_DAY and (
                    consecutive_unwatered >= 1 or bonus_active
                )

                state["plants"].append({
                    "pos": pos,
                    "crop": crop,
                    "age": age,
                    "watered_today": watered,
                    "water_needed": water_needed,
                    "yield_units": yu,
                    "fertilize_due": fert_due,
                    "fert_until_day": fert_until,
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


def count_fertilize_due_tomorrow(farm_state: dict[str, Any], day: int) -> int:
    """Ongoing plants that will become fertilize_due tomorrow (age + 2 is a yield day)
    and are not already covered by current fertilization. Used to set the fertilizer
    sell reserve one day ahead."""
    count = 0
    payoff_reachable = (day + 2) < c.LAST_DAY
    if not payoff_reachable:
        return 0
    for p in farm_state["plants"]:
        if not p["is_ongoing"] or p["is_expired"]:
            continue
        spec = c.CROPS[p["crop"]]
        upcoming_ages = [d for d in spec["yield_days"] if d >= p["age"] + 2]
        if not upcoming_ages or upcoming_ages[0] != p["age"] + 2:
            continue
        # If already fertilized through day + 2, no reserve needed.
        if p["fert_until_day"] >= (day + 2):
            continue
        count += 1
    return count


def count_maturing_tomorrow(farm_state: dict[str, Any], crop: str) -> int:
    """One-time `crop` tiles reaching max yield by tomorrow — about to be
    harvested and freed, so seed should be bought a day ahead of need."""
    if crop not in c.CROPS or c.CROPS[crop]["is_ongoing"]:
        raise ValueError(f"count_maturing_tomorrow expects a one-time crop, got {crop!r}")
    return sum(
        1 for p in farm_state["plants"]
        if p["crop"] == crop and p["age"] + 1 >= p["max_yield_day"]
    )


def coop_capacity(farm_state: dict[str, Any]) -> int:
    """Geese that can already be housed by structures standing on the
    ground: built-and-empty coops, plus coops a goose already occupies."""
    return len(farm_state["empty_coops"]) + sum(
        1 for a in farm_state["animals"] if a["animal"] == "GOOSE"
    )


def live_reserved_structures(
    farm_state: dict[str, Any],
    unlocked_quads: list[str],
    grazer_target: int,
    goose_target: int,
    day: int,
) -> frozenset[tuple[int, int]]:
    """Shared PASTURE_CLUSTER reservation for future pasture (COW/SHEEP) and
    coop (GOOSE) structures. Each species' gate (day window, quad count) is
    evaluated independently; the combined demand then draws from one
    unprioritized queue over the cluster's fixed distance order — overflow
    past the cluster size is simply not reserved, never spills outside it."""
    grazer_need = 0
    if day <= c.PASTURE_RESERVATION_LAST_DAY:
        quad_cap = 4 if len(unlocked_quads) == 1 else min(c.MAX_TOTAL_HERD, len(unlocked_quads) * 6)
        grazer_need = max(grazer_target, quad_cap)

    goose_need = 0
    if day < c.LIQUIDATION_DAY and len(unlocked_quads) >= 2:
        goose_need = max(0, goose_target - coop_capacity(farm_state))

    combined_target = grazer_need + goose_need
    reserved: set[tuple[int, int]] = set()
    for pos in c.PASTURE_CLUSTER:
        if len(reserved) >= combined_target:
            break
        if quad_of(pos) in unlocked_quads:
            reserved.add(pos)
    return frozenset(reserved)


# ---------------------------------------------------------------------------
# MODULE: dispatch.py
# ---------------------------------------------------------------------------

"""Workforce dispatch — daily VRP planner (parallel cheapest insertion) + fail-fast executor."""




# ---------------------------------------------------------------------------
# Shed access geometry
# ---------------------------------------------------------------------------
_SHED_ACCESS: tuple[tuple[int, int], ...] = tuple(c.SHED_TILES)
_SHED_ACCESS_SET: frozenset[tuple[int, int]] = c.SHED_TILES_SET
_SHED_CENTER: tuple[int, int] = c.SHED_TILES[0]

_MOVES: dict[str, tuple[int, int]] = {
    "EAST": (1, 0),
    "WEST": (-1, 0),
    "SOUTH": (0, 1),
    "NORTH": (0, -1),
}

# ---------------------------------------------------------------------------
# Urgency scale & Premium delivery config
# ---------------------------------------------------------------------------
_URG_FEED_CRITICAL: int = 100
_URG_PLACE: int = 95
_URG_HARVEST_PREMIUM: int = 85  # priority-tier boundary for premium rounds — see _solve_day
_URG_HARVEST: int = 80
_URG_FERTILIZE: int = 77
_URG_WATER: int = 75
_URG_PLANT_PRIORITY: int = 72
_URG_PLANT: int = 71
_URG_FEED: int = 70
_URG_COLLECT_FERTILIZER: int = 68
_URG_CARE: int = 65
_URG_DIG_EXPIRED: int = 30
_URG_DIG_WEED: int = 20

# Crops that must reach the shed the SAME DAY they are harvested (steep glut curve —
# selling all at once late in the day tanks the price; regular crops ride home for free
# via the engine's end-of-day inventory drop). Mechanism: any route that picks up premium
# cargo pays a projected final-DROP surcharge during insertion (so the detour always fits
# the budget), and `_append_drops` materializes exactly one DROP at the route's end.
# The whole route executes within the day, so premium produce is in the shed by daytime.
_PREMIUM_DROP_CROPS: frozenset[str] = frozenset(("MELON", "STRAWBERRY"))
_ANIMAL_SPECIES: tuple[str, ...] = ("COW", "SHEEP", "GOOSE")
_PASTURE_SPECIES: tuple[str, ...] = ("COW", "SHEEP")


def _plant_ready_to_harvest(p: dict[str, Any], day: int) -> bool:
    return p["yield_units"] > 0 and (
        p["is_ongoing"]
        or p["age"] >= p["max_yield_day"]
        or day >= c.LIQUIDATION_DAY
    )


def _ready_premium_bushes(farm_state: dict[str, Any], day: int) -> list[dict[str, Any]]:
    """Melon/strawberry bushes ready to pick — routed via _plan_premium_rounds, never
    through build_task_catalog (bush-count pairing has no notion of yield_units)."""
    return [
        p for p in farm_state["plants"]
        if not p["is_expired"] and p["crop"] in _PREMIUM_DROP_CROPS and _plant_ready_to_harvest(p, day)
    ]


# ---------------------------------------------------------------------------
# Task catalog & Atomic builders
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class Task:
    urgency: int
    pos: tuple[int, int]
    actions: tuple[list[Any], ...]     # atomic sequence executed at one stop
    need: tuple[str, ...] = ()         # OR-alternatives; solver picks the cheapest available
    produces: tuple[str, int] | None = None  # (item, qty) — set only on HARVEST/COLLECT_FERTILIZER


def _plant_task(urgency: int, pos: tuple[int, int], crop: str) -> Task:
    return Task(urgency, pos, (["PLANT", crop], ["WATER"]))


def _harvest_actions(watered_today: bool, day: int) -> list[list[Any]]:
    """WATER-then-HARVEST if not watered today and there's still a day-refresh left
    to bank it (c.LAST_DAY's own refresh never runs, so watering there is wasted),
    else HARVEST alone. Shared by the regular catalog harvest task and premium
    round blocks."""
    return [["WATER"], ["HARVEST"]] if not watered_today and day < c.LAST_DAY else [["HARVEST"]]


def _build_and_place_task(
    urgency: int, pos: tuple[int, int], build_actions: list[str], species: tuple[str, ...]
) -> Task:
    return Task(urgency, pos, tuple([a] for a in build_actions) + ((["PLACE"],)), need=species)


def select_target_crops(day: int, seeds_stock: dict[str, int]) -> list[str]:
    if day > c.WHEAT_LAST_PLANT_DAY:
        return []
    if day >= 21:
        return ["WHEAT"] if seeds_stock.get("WHEAT", 0) > 0 else []
    # Before day 13 melon goes first; afterwards strawberry's earlier deadline takes priority.
    premium_order = ("MELON", "STRAWBERRY") if day <= 12 else ("STRAWBERRY", "MELON")
    deadlines = {"MELON": c.MELON_LAST_PLANT_DAY, "STRAWBERRY": c.STRAWBERRY_LAST_PLANT_DAY}
    return [
        crop for crop in (*premium_order, "WHEAT")
        if seeds_stock.get(crop, 0) > 0 and day <= deadlines.get(crop, c.WHEAT_LAST_PLANT_DAY)
    ]


def compute_needed_pastures(
    farm_state: dict[str, Any],
    census: AnimalCensus,
) -> list[tuple[int, int]]:
    needed: list[tuple[int, int]] = []
    planted = {p["pos"] for p in farm_state["plants"]}
    occupied_pastures = census.field_cows + census.field_sheep
    for p_pos in c.PASTURE_CLUSTER:
        if len(needed) + occupied_pastures + len(farm_state["empty_pastures"]) >= census.total:
            break
        if p_pos in farm_state["empty_tiles"] or p_pos in farm_state["weeds"] or p_pos in planted:
            needed.append(p_pos)
    return needed


def compute_needed_coops(
    farm_state: dict[str, Any],
    census: AnimalCensus,
    exclude: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Same rationale as compute_needed_pastures, drawing from the same shared
    cluster; `exclude` keeps it from re-claiming positions compute_needed_pastures
    already committed to a pasture this turn."""
    needed: list[tuple[int, int]] = []
    planted = {p["pos"] for p in farm_state["plants"]}
    occupied_coops = census.field_geese
    for pos in c.PASTURE_CLUSTER:
        if pos in exclude:
            continue
        if len(needed) + occupied_coops + len(farm_state["empty_coops"]) >= census.total_geese:
            break
        if pos in farm_state["empty_tiles"] or pos in farm_state["weeds"] or pos in planted:
            needed.append(pos)
    return needed


def _resolve_structure_action(
    pos: tuple[int, int],
    farm_state: dict[str, Any],
    planted_positions: dict[tuple[int, int], dict[str, Any]],
    build_action: str,
) -> list[str] | None:
    if pos in farm_state["weeds"]:
        return ["DIG", build_action]
    if pos in planted_positions:
        p = planted_positions[pos]
        if p["yield_units"] > 0 or (not p["is_ongoing"] and p["age"] >= p["first_yield_day"] - 1):
            return None
        return ["DIG", build_action]
    if pos in farm_state["empty_tiles"]:
        return [build_action]
    return None


def _kept_animal_positions(
    farm_state: dict[str, Any],
    retained_caps: dict[str, int] | None,
) -> set[tuple[int, int]]:
    caps = retained_caps if retained_caps is not None else {}
    kept: set[tuple[int, int]] = set()
    for species in _ANIMAL_SPECIES:
        cap = caps.get(species, 999)
        if cap <= 0:
            continue
        species_animals = [a for a in farm_state["animals"] if a["animal"] == species]
        species_animals.sort(key=lambda a: (a["fed_today"], manhattan(a["pos"], _SHED_CENTER)))
        kept.update(a["pos"] for a in species_animals[:cap])
    return kept


def build_task_catalog(
    farm_state: dict[str, Any],
    shed: dict[str, int],
    seeds_stock: dict[str, int],
    needed_pastures: list[tuple[int, int]],
    needed_coops: list[tuple[int, int]],
    all_units: list[tuple[tuple[int, int], dict[str, int]]],
    day: int,
    hour: int,
    hints: dict[str, Any],
    unlocked_quads: list[str],
    retained_caps: dict[str, int] | None = None,
) -> list[Task]:
    tasks: list[Task] = []
    kept_animals = _kept_animal_positions(farm_state, retained_caps)

    carried: dict[str, int] = {
        sp: sum(inv.get(sp, 0) for _, inv in all_units) for sp in _ANIMAL_SPECIES
    }
    wheat_available = shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for _, inv in all_units)
    fert_available = shed.get("FERTILIZER", 0) + sum(inv.get("FERTILIZER", 0) for _, inv in all_units)

    res_structures = live_reserved_structures(
        farm_state, unlocked_quads, hints["reserved_grazer_slots"], hints["reserved_geese_slots"], day=day
    )
    needed_structs = set(needed_pastures) | set(needed_coops)

    # 1. Animal care
    for a in farm_state["animals"]:
        pos = a["pos"]
        product = c.ANIMALS[a["animal"]]["product"]
        if pos not in kept_animals:
            if a["yield_units"] > 0:
                tasks.append(Task(_URG_HARVEST, pos, (["HARVEST"],), produces=(product, a["yield_units"])))
            if a["fertilizer_available"]:
                tasks.append(Task(_URG_COLLECT_FERTILIZER, pos, (["COLLECT_FERTILIZER"],), produces=("FERTILIZER", 1)))
            continue
        if not a["fed_today"] and wheat_available > 0:
            urg = _URG_FEED_CRITICAL if a["consecutive_unfed"] >= 1 else _URG_FEED
            tasks.append(Task(urg, pos, (["FEED"],), need=("WHEAT",)))
        if not a["cared_today"]:
            tasks.append(Task(_URG_CARE, pos, (["CARE"],)))
        if a["fertilizer_available"]:
            tasks.append(Task(_URG_COLLECT_FERTILIZER, pos, (["COLLECT_FERTILIZER"],), produces=("FERTILIZER", 1)))
        if a["yield_units"] > 0:
            tasks.append(Task(_URG_HARVEST, pos, (["HARVEST"],), produces=(product, a["yield_units"])))

    # 2. Animal placement — waiting = shed stock + carried, only while under the retained cap
    caps = retained_caps if retained_caps is not None else {}
    field_count = {sp: sum(1 for a in farm_state["animals"] if a["animal"] == sp) for sp in _ANIMAL_SPECIES}
    waiting = {
        sp: (shed.get(sp, 0) + carried[sp]) if field_count[sp] < caps.get(sp, 999) else 0
        for sp in _ANIMAL_SPECIES
    }
    pasture_waiting = waiting["COW"] + waiting["SHEEP"]
    geese_waiting = waiting["GOOSE"]

    for spots_key, n_waiting, species in (
        ("empty_pastures", pasture_waiting, _PASTURE_SPECIES),
        ("empty_coops", geese_waiting, ("GOOSE",)),
    ):
        if n_waiting > 0 and farm_state[spots_key]:
            spots = sorted(
                farm_state[spots_key],
                key=lambda p: (manhattan(p, _SHED_CENTER), p[1], p[0]),
            )
            for pos in spots[:n_waiting]:
                tasks.append(Task(_URG_PLACE, pos, (["PLACE"],), need=species))

    # 3. Planting queue — pools together *empty* tiles and tiles that will be
    # *freed this hour* by harvesting a ready one-time non-premium crop
    # (WHEAT, CARROT — MELON/STRAWBERRY are handled entirely by
    # _plan_premium_rounds and never appear here). Both pools compete for the
    # same priority queue (MELON/STRAWBERRY -> CARROT/TOMATO pivot -> rest of
    # target_crops), so a freed tile can come back as a *different* crop than
    # the one just harvested — whichever the farm-wide economics currently
    # favor. A freed tile's winning crop is recorded in freed_crop_assignment
    # and consumed by section 4, which glues PLANT+WATER onto the same stop
    # as the HARVEST — there is never a second visit to the tile.
    target_crops = select_target_crops(day, seeds_stock)
    freed_crop_assignment: dict[tuple[int, int], str] = {}

    if hour < c.TURNS_PER_DAY - 1:
        # Zoning: tiles immediately needed for construction (needed_structs) are
        # excluded from planting/replanting — section 6 handles DIG/BUILD via
        # _resolve_structure_action. Future reserved tiles
        # (res_structures - needed_structs) are zoned exclusively for WHEAT,
        # whether they start empty or are freed by harvest this hour.
        # BLOCKED_CLUSTER tiles (far quadrant corners) are dropped before any other
        # zoning — nothing is ever planted there, so no route ever detours to reach them.
        usable_empty_tiles = [pos for pos in farm_state["empty_tiles"] if pos not in c.BLOCKED_CLUSTER_SET]
        future_reserved = res_structures - needed_structs
        open_plantable = [
            pos for pos in usable_empty_tiles
            if pos not in future_reserved and pos not in needed_structs
        ]
        reserved_plantable = [pos for pos in usable_empty_tiles if pos in future_reserved]

        freed_open: list[tuple[int, int]] = []
        freed_reserved: list[tuple[int, int]] = []
        freed_positions: set[tuple[int, int]] = set()
        for p in farm_state["plants"]:
            if p["is_ongoing"] or p["is_expired"] or p["crop"] in _PREMIUM_DROP_CROPS:
                continue
            if not _plant_ready_to_harvest(p, day):
                continue
            pos = p["pos"]
            if pos in needed_structs:
                continue
            freed_positions.add(pos)
            (freed_reserved if pos in future_reserved else freed_open).append(pos)

        open_plantable = open_plantable + freed_open
        reserved_plantable = reserved_plantable + freed_reserved
        open_plantable.sort(key=lambda p: (manhattan(p, _SHED_CENTER), p[1], p[0]))
        reserved_plantable.sort(key=lambda p: (manhattan(p, _SHED_CENTER), p[1], p[0]))

        plant_queue: list[tuple[str, int, int]] = [
            (crop, _URG_PLANT_PRIORITY, seeds_stock.get(crop, 0))
            for crop in ("MELON", "STRAWBERRY")
            if crop in target_crops
        ]
        crop_limits = hints.get("crop_limits", {})
        plant_queue += [
            (pivot, _URG_PLANT, min(seeds_stock.get(pivot, 0), crop_limits.get(pivot, 0)))
            for pivot in ("CARROT", "TOMATO")
        ]
        plant_queue += [
            (crop, _URG_PLANT, seeds_stock.get(crop, 0))
            for crop in target_crops
            if crop not in _PREMIUM_DROP_CROPS
        ]

        for crop, urg, limit in plant_queue:
            if not open_plantable and not reserved_plantable:
                break
            # Wheat alone may use future reserved tiles; other crops only use open land.
            pools = (open_plantable, reserved_plantable) if crop == "WHEAT" else (open_plantable,)
            remaining = limit
            for pool in pools:
                if remaining <= 0 or not pool:
                    continue
                n = min(len(pool), remaining)
                for pos in pool[:n]:
                    if pos in freed_positions:
                        freed_crop_assignment[pos] = crop
                    else:
                        tasks.append(_plant_task(urg, pos, crop))
                del pool[:n]
                remaining -= n

    # 4. Crops — melon/strawberry excluded entirely; _plan_premium_rounds
    #    handles them as bush-pair-then-DROP blocks, not individual Tasks.
    #    A ready one-time non-premium crop gets PLANT+WATER glued onto its
    #    HARVEST stop iff section 3 assigned it a replant crop — which may
    #    differ from the crop currently occupying the tile.
    for p in farm_state["plants"]:
        if p["is_expired"]:
            tasks.append(Task(_URG_DIG_EXPIRED, p["pos"], (["DIG"],)))
        elif _plant_ready_to_harvest(p, day):
            crop = p["crop"]
            if crop not in _PREMIUM_DROP_CROPS:
                actions: tuple[list[Any], ...] = tuple(_harvest_actions(p["watered_today"], day))
                replant_crop = freed_crop_assignment.get(p["pos"])
                if replant_crop is not None:
                    actions = actions + (["PLANT", replant_crop], ["WATER"])
                need: tuple[str, ...] = ()
                if p["is_ongoing"] and p["fertilize_due"] and fert_available > 0:
                    actions = (["FERTILIZE"],) + actions
                    need = ("FERTILIZER",)
                tasks.append(Task(_URG_HARVEST, p["pos"], actions, need=need, produces=(crop, p["yield_units"])))
        elif p["water_needed"] and p["fertilize_due"] and fert_available > 0:
            # Ongoing crops: fertilizer bonus only fires when BOTH watered and
            # fertilized on the same day's overnight refresh — bundle into one
            # stop so the worker never visits the tile twice.
            tasks.append(Task(_URG_FERTILIZE, p["pos"], (["WATER"], ["FERTILIZE"]), need=("FERTILIZER",)))
        elif p["water_needed"]:
            tasks.append(Task(_URG_WATER, p["pos"], (["WATER"],)))
        elif p["fertilize_due"] and fert_available > 0:
            # Already watered today but not yet fertilized — fertilize alone.
            tasks.append(Task(_URG_FERTILIZE, p["pos"], (["FERTILIZE"],), need=("FERTILIZER",)))

    # 5. Weeds — BLOCKED_CLUSTER tiles skipped: nothing is ever planted there,
    # so clearing weeds brings no value and wastes worker travel time.
    if day <= c.WHEAT_LAST_PLANT_DAY:
        for pos in farm_state["weeds"]:
            if pos in c.BLOCKED_CLUSTER_SET:
                continue
            tasks.append(Task(_URG_DIG_WEED, pos, (["DIG"],)))

    # 6. Pasture & coop construction — both draw from the shared PASTURE_CLUSTER.
    # Structures are only built when waiting animals exist (pasture_waiting / geese_waiting),
    # bundling construction and placement into a single _URG_PLACE task.
    planted_positions = {p["pos"]: p for p in farm_state["plants"]}
    pasture_leftover = max(0, pasture_waiting - len(farm_state["empty_pastures"]))
    n_bundled = 0
    for pos in needed_pastures:
        if n_bundled >= pasture_leftover:
            break
        act = _resolve_structure_action(pos, farm_state, planted_positions, "BUILD_PASTURE")
        if act is None:
            continue
        tasks.append(_build_and_place_task(_URG_PLACE, pos, act, _PASTURE_SPECIES))
        n_bundled += 1

    coop_leftover = max(0, geese_waiting - len(farm_state["empty_coops"]))
    n_coop_bundled = 0
    for pos in needed_coops:
        if n_coop_bundled >= coop_leftover:
            break
        act = _resolve_structure_action(pos, farm_state, planted_positions, "BUILD_COOP")
        if act is None:
            continue
        tasks.append(_build_and_place_task(_URG_PLACE, pos, act, ("GOOSE",)))
        n_coop_bundled += 1

    tasks.sort(key=lambda t: (-t.urgency, manhattan(t.pos, _SHED_CENTER), t.pos[1], t.pos[0], str(t.actions)))
    return tasks


# ---------------------------------------------------------------------------
# VRP solver & Executor
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Stop:
    pos: tuple[int, int]
    actions: list[list[Any]]
    produces: tuple[str, int] | None = None


@dataclass(slots=True)
class _Route:
    start: tuple[int, int]
    stops: list[_Stop] = field(default_factory=list)
    cost: int = 0
    carried: dict[str, int] = field(default_factory=dict)
    pickup_of: dict[str, int] = field(default_factory=dict)
    # Index before which insertion is allowed. None = fully open. Set once a premium
    # block/flush-drop is appended, so later passes can only insert *before* that
    # locked tail — never split a committed bush-pair-then-DROP trip.
    insertable_upto: int | None = None
    # Index at/after which insertion is allowed (0 = fully open). Raised by _plan_routes
    # every time it steps down to a strictly lower urgency tier, so a lower-priority task
    # can never be spliced in ahead of an already-committed higher-priority stop.
    insertable_from: int = 0


def _route_cost(route: _Route) -> int:
    total = 0
    pos = route.start
    for s in route.stops:
        total += manhattan(pos, s.pos) + len(s.actions)
        pos = s.pos
    return total


def _drop_cost(pos: tuple[int, int]) -> int:
    """Turns to reach the nearest shed tile from pos and issue DROP (1 if already there)."""
    _, d = _best_pickup_tile(pos, None)
    return d + 1


def _rebuild_pickups(route: _Route) -> dict[str, int]:
    pickup_of: dict[str, int] = {}
    for idx, stop in enumerate(route.stops):
        for line in stop.actions:
            if line[0] == "PICKUP" and line[1] not in pickup_of:
                pickup_of[line[1]] = idx
    return pickup_of


def _best_pickup_tile(prev: tuple[int, int], nxt: tuple[int, int] | None) -> tuple[tuple[int, int], int]:
    base = manhattan(prev, nxt) if nxt is not None else 0
    best_p: tuple[int, int] = _SHED_ACCESS[0]
    best_d = math.inf
    for p in _SHED_ACCESS:
        d = manhattan(prev, p) + (manhattan(p, nxt) if nxt is not None else 0) - base
        if d < best_d:
            best_d, best_p = d, p
    return best_p, int(best_d)


_Mode = Literal["plain", "carried", "bump", "extend", "new"]


@dataclass(slots=True)
class _Candidate:
    cost: int
    insert_at: int
    item: str | None
    mode: _Mode
    pickup_idx: int = -1
    pickup_pos: tuple[int, int] | None = None


def _eval_route(
    route: _Route,
    task: Task,
    shed_stock: dict[str, int],
) -> _Candidate | None:
    stops = route.stops
    n = len(stops)
    limit = route.insertable_upto if route.insertable_upto is not None else n
    floor = route.insertable_from
    start = route.start

    pref_cost = [math.inf] * (n + 1)
    pref_k = [-1] * (n + 1)
    best = math.inf
    best_k = -1
    for k in range(n):
        if k >= floor:
            prev = start if k == 0 else stops[k - 1].pos
            _, d = _best_pickup_tile(prev, stops[k].pos)
            d += 1
            if d < best:
                best, best_k = d, k
        pref_cost[k + 1] = best
        pref_k[k + 1] = best_k

    earliest_pickup = min(route.pickup_of.values(), default=None)
    items: tuple[str | None, ...] = task.need if task.need else (None,)
    task_cost = len(task.actions)
    best_cand: _Candidate | None = None

    for i in range(floor, limit + 1):
        prev = start if i == 0 else stops[i - 1].pos
        nxt = stops[i].pos if i < n else None

        back_leg = manhattan(task.pos, nxt) - manhattan(prev, nxt) if nxt is not None else 0
        travel = manhattan(prev, task.pos)
        d_task = travel + back_leg + task_cost

        for item in items:
            if item is None:
                cand = _Candidate(d_task, i, None, "plain")
            elif route.carried.get(item, 0) > 0:
                cand = _Candidate(d_task, i, item, "carried")
            elif shed_stock.get(item, 0) <= 0:
                continue
            elif (j := route.pickup_of.get(item)) is not None and j < i:
                cand = _Candidate(d_task, i, item, "bump")
            elif earliest_pickup is not None and earliest_pickup < i:
                cand = _Candidate(d_task + 1, i, item, "extend")
            else:
                d_new = pref_cost[i]
                k_new = pref_k[i]
                p_new: tuple[int, int] | None = None
                p_det, d_det = _best_pickup_tile(prev, task.pos)
                d_det += 1
                if d_det <= d_new:
                    d_new, k_new, p_new = d_det, i, p_det
                if math.isinf(d_new):
                    continue
                cand = _Candidate(d_task + d_new, i, item, "new", pickup_idx=k_new, pickup_pos=p_new)

            if best_cand is None or cand.cost < best_cand.cost:
                best_cand = cand

    return best_cand


def _commit(
    route: _Route,
    task: Task,
    cand: _Candidate,
    shed_stock: dict[str, int],
) -> None:
    stops = route.stops
    n_before = len(stops)
    insert_at = cand.insert_at
    item = cand.item

    if cand.mode == "new":
        assert item is not None, f"mode=new requires item, task={task}"
        pickup_pos = cand.pickup_pos
        if pickup_pos is None:
            prev = route.start if cand.pickup_idx == 0 else stops[cand.pickup_idx - 1].pos
            pickup_pos, _ = _best_pickup_tile(prev, stops[cand.pickup_idx].pos)
        stops.insert(cand.pickup_idx, _Stop(pickup_pos, [["PICKUP", item, 1]]))
        insert_at += 1
        shed_stock[item] -= 1
    elif cand.mode == "bump":
        assert item is not None, f"mode=bump requires item, task={task}"
        j = route.pickup_of[item]
        for line in stops[j].actions:
            if line[0] == "PICKUP" and line[1] == item:
                line[2] += 1
                break
        shed_stock[item] -= 1
    elif cand.mode == "extend":
        assert item is not None, f"mode=extend requires item, task={task}"
        j = min(route.pickup_of.values())
        stops[j].actions.append(["PICKUP", item, 1])
        shed_stock[item] -= 1
    elif cand.mode == "carried":
        assert item is not None, f"mode=carried requires item, task={task}"
        route.carried[item] -= 1

    stop_actions: list[list[Any]] = [
        ["PLACE", item] if a[0] == "PLACE" and len(a) == 1 and item is not None else list(a)
        for a in task.actions
    ]
    stops.insert(insert_at, _Stop(task.pos, stop_actions, produces=task.produces))

    if route.insertable_upto is not None:
        route.insertable_upto += len(stops) - n_before

    route.cost = _route_cost(route)
    route.pickup_of = _rebuild_pickups(route)


def _init_routes(all_units: list[tuple[tuple[int, int], dict[str, int]]]) -> list[_Route]:
    return [
        _Route(start=pos, carried={k: v for k, v in inv.items() if v > 0})
        for pos, inv in all_units
    ]


def _final_drop_reserve(route: _Route, task: Task, cand: _Candidate, enforce: bool) -> int:
    """Final season day: unsold inventory is worth $0. Reserve turn(s) for return-and-DROP
    from the position that becomes the route tail after committing this task, ensuring
    _append_drops never runs out of budget."""
    if not enforce:
        return 0
    tail = task.pos if cand.insert_at >= len(route.stops) else route.stops[-1].pos
    return _drop_cost(tail)


def _plan_routes(
    routes: list[_Route],
    catalog: list[Task],
    shed_stock: dict[str, int],
    budget_list: list[int],
    reserve_final_drop: bool = False,
) -> None:
    prev_urgency: float = math.inf
    for task in catalog:
        assert task.urgency <= prev_urgency, (
            f"catalog must be sorted by descending urgency: {task.urgency} follows "
            f"{prev_urgency} (task={task})"
        )
        if task.urgency < prev_urgency:
            for route in routes:
                route.insertable_from = len(route.stops)
        prev_urgency = task.urgency

        best_cand: tuple[int, int, _Candidate] | None = None
        for r_idx, route in enumerate(routes):
            cand = _eval_route(route, task, shed_stock)
            if cand is None:
                continue
            reserve = _final_drop_reserve(route, task, cand, reserve_final_drop)
            if route.cost + cand.cost + reserve > budget_list[r_idx]:
                continue
            if best_cand is None or cand.cost < best_cand[0]:
                best_cand = (cand.cost, r_idx, cand)
        if best_cand is None:
            continue
        _, r_idx, cand = best_cand
        _commit(routes[r_idx], task, cand, shed_stock)


def _append_drops(routes: list[_Route], budget_list: list[int]) -> None:
    """Opportunistic final DROP for routes carrying sellable cargo that isn't dropped
    yet. Premium cargo is dropped as soon as its harvest round completes (see
    _plan_premium_rounds) — a route whose last stop is
    already a DROP is skipped, so it never gets a second one.
    """
    for r_idx, route in enumerate(routes):
        if not route.stops or route.stops[-1].actions[-1][0] == "DROP":
            continue

        has_cargo = any(
            line[0] in ("HARVEST", "COLLECT_FERTILIZER") for s in route.stops for line in s.actions
        ) or any(route.carried.get(it, 0) > 0 for it in c.SELLABLE_ITEMS)
        if not has_cargo:
            continue

        last = route.stops[-1]
        delta = _drop_cost(last.pos)
        if route.cost + delta > budget_list[r_idx]:
            continue

        if last.pos in _SHED_ACCESS_SET:
            last.actions.append(["DROP"])
        else:
            p, _ = _best_pickup_tile(last.pos, None)
            route.stops.append(_Stop(p, [["DROP"]]))
        route.cost = _route_cost(route)


def _pair_premium_bushes(ready: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Nearest-neighbor pairing: repeatedly take a bush and match it with whichever
    remaining bush is closest. One trailing solo block if the count is odd."""
    pool = list(ready)
    blocks: list[list[dict[str, Any]]] = []
    while pool:
        a = pool.pop()
        if not pool:
            blocks.append([a])
            break
        j = min(range(len(pool)), key=lambda i: manhattan(a["pos"], pool[i]["pos"]))
        blocks.append([a, pool.pop(j)])
    return blocks


def _block_cost(
    tail: tuple[int, int], block: list[dict[str, Any]], day: int
) -> tuple[int, list[dict[str, Any]]]:
    """Cheapest way to run this 1-2 bush block from tail, ending with DROP. For a pair,
    tries both visiting orders and keeps the cheaper one."""
    orders = (block,) if len(block) == 1 else (block, block[::-1])
    best_cost = math.inf
    best_order = block
    for order in orders:
        pos = tail
        cost = 0
        for bush in order:
            cost += manhattan(pos, bush["pos"]) + len(_harvest_actions(bush["watered_today"], day))
            pos = bush["pos"]
        cost += _drop_cost(pos)
        if cost < best_cost:
            best_cost, best_order = cost, order
    return int(best_cost), list(best_order)


def _plan_premium_rounds(
    routes: list[_Route],
    ready_premium: list[dict[str, Any]],
    day: int,
    budget_list: list[int],
) -> None:
    """Assign bush-pair harvest+DROP blocks to whichever route runs them cheapest.
    Bushes are paired by mutual proximity first (independent of route assignment), then
    each block is appended at the tail of the cheapest-fitting route — always exactly
    2 bushes (1 if odd-one-out) per trip, no yield-unit counting. A block that fits no
    route's remaining budget this hour is simply skipped — the bush stays ready and is
    retried next hour, same as any other budget-starved catalog task.
    """
    for block in _pair_premium_bushes(ready_premium):
        best: tuple[int, int, list[dict[str, Any]]] | None = None  # (cost, route_idx, order)
        for r_idx, route in enumerate(routes):
            tail = route.stops[-1].pos if route.stops else route.start
            cost, order = _block_cost(tail, block, day)
            if route.cost + cost > budget_list[r_idx]:
                continue
            if best is None or cost < best[0]:
                best = (cost, r_idx, order)
        if best is None:
            continue
        _, r_idx, order = best
        route = routes[r_idx]
        if route.insertable_upto is None:
            route.insertable_upto = len(route.stops)
        for bush in order:
            route.stops.append(_Stop(
                bush["pos"], list(_harvest_actions(bush["watered_today"], day)),
                produces=(bush["crop"], bush["yield_units"]),
            ))
        drop_pos, _ = _best_pickup_tile(order[-1]["pos"], None)
        route.stops.append(_Stop(drop_pos, [["DROP"]]))
        route.cost = _route_cost(route)


def _solve_day(
    catalog: list[Task],
    ready_premium: list[dict[str, Any]],
    all_units: list[tuple[tuple[int, int], dict[str, int]]],
    shed: dict[str, int],
    budget_list: list[int],
    day: int,
) -> list[_Route]:
    """Three ordered passes: critical tasks (urgency above the premium tier), premium
    bush rounds (same-day melon/strawberry delivery), then the rest of the catalog —
    which now includes atomic harvest(+replant) blocks for WHEAT/CARROT assembled in
    build_task_catalog. There is no separate replant pass anymore: a tile is freed and
    resown within the same Task/stop, so no cross-route timing synchronization is
    needed.
    """
    routes = _init_routes(all_units)
    shed_stock = dict(shed)

    critical = [t for t in catalog if t.urgency > _URG_HARVEST_PREMIUM]
    rest = [t for t in catalog if t.urgency <= _URG_HARVEST_PREMIUM]
    is_final_day = day == c.LAST_DAY

    _plan_routes(routes, critical, shed_stock, budget_list, reserve_final_drop=False)
    _plan_premium_rounds(routes, ready_premium, day, budget_list)
    _plan_routes(routes, rest, shed_stock, budget_list, reserve_final_drop=is_final_day)
    _append_drops(routes, budget_list)
    return routes


def required_hand_count(
    catalog: list[Task],
    ready_premium: list[dict[str, Any]],
    base_units: list[tuple[tuple[int, int], dict[str, int]]],
    shed: dict[str, int],
    max_hands: int,
    day: int,
) -> int:
    if not catalog and not ready_premium:
        return 0

    current_units_count = len(base_units)
    max_team_size = max_hands + 1
    if current_units_count >= max_team_size:
        return 0

    final_buffer = 2 if day == c.LAST_DAY else 0
    budget = max(1, c.TURNS_PER_DAY - 1 - final_buffer)
    target = len(catalog) + len(ready_premium)

    best_extra_hands = 0
    best_inserted = -1
    for team_size in range(current_units_count, max_team_size + 1):
        units = base_units + [
            (_SHED_ACCESS[i % len(_SHED_ACCESS)], {})
            for i in range(current_units_count, team_size)
        ]
        routes = _solve_day(catalog, ready_premium, units, shed, [budget] * len(units), day)
        inserted = sum(
            1 for r in routes for s in r.stops if s.actions and s.actions[0][0] not in ("PICKUP", "DROP")
        )
        if inserted == target:
            return team_size - current_units_count

        active_workers = sum(
            1 for r in routes
            if any(s.actions and s.actions[0][0] not in ("PICKUP", "DROP") for s in r.stops)
        )
        if active_workers == team_size:
            best_inserted = max(best_inserted, inserted)
            best_extra_hands = team_size - current_units_count

    return best_extra_hands


# ---------------------------------------------------------------------------
# Day plan & Fail-fast executor
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DayPlan:
    day: int
    routes: dict[int, list[list[Any]]]


def _materialize(route: _Route) -> list[list[Any]]:
    queue: list[list[Any]] = []
    pos = route.start
    for stop in route.stops:
        while pos != stop.pos:
            step = bfs_step(pos, stop.pos)
            if step is None:
                raise AssertionError(
                    f"bfs_step: no path {pos} -> {stop.pos}, route.start={route.start}"
                )
            queue.append([step])
            dx, dy = _MOVES[step]
            pos = (pos[0] + dx, pos[1] + dy)
        queue.extend(stop.actions)
    return queue


def build_day_plan(
    all_units: list[tuple[tuple[int, int], dict[str, int]]],
    farm_state: dict[str, Any],
    shed: dict[str, int],
    seeds: dict[str, int],
    day: int,
    hour: int,
    needed_pastures: list[tuple[int, int]],
    needed_coops: list[tuple[int, int]],
    hints: dict[str, Any],
    unlocked_quads: list[str],
    retained_caps: dict[str, int] | None = None,
    pending_hires: int = 0,
) -> DayPlan:
    effective_hour = max(1, hour)
    final_buffer = 2 if day == c.LAST_DAY else 0
    budget = max(0, c.TURNS_PER_DAY - effective_hour - final_buffer)

    full_units = list(all_units)
    budgets = [budget] * len(all_units) + [budget - 1] * pending_hires
    for i in range(pending_hires):
        spawn_pos = _SHED_ACCESS[(len(all_units) + i) % len(_SHED_ACCESS)]
        full_units.append((spawn_pos, {}))

    routes = _init_routes(full_units)
    if budget > 0 and full_units:
        catalog = build_task_catalog(
            farm_state, shed, dict(seeds), needed_pastures, needed_coops,
            full_units, day, effective_hour, hints=hints, unlocked_quads=unlocked_quads,
            retained_caps=retained_caps,
        )
        ready_premium = _ready_premium_bushes(farm_state, day)
        routes = _solve_day(catalog, ready_premium, full_units, shed, budgets, day)

    materialized = {ui: _materialize(r) for ui, r in enumerate(routes)}

    for ui, queue in materialized.items():
        if len(queue) > budgets[ui]:
            raise AssertionError(
                f"day={day} hour={hour} unit={ui}: materialized {len(queue)} actions "
                f"exceeds budget={budgets[ui]} (estimated route.cost={routes[ui].cost}), "
                f"stops={routes[ui].stops}"
            )

    return DayPlan(day=day, routes=materialized)


def next_action(unit_idx: int, day: int, plan: DayPlan) -> list[Any]:
    if plan.day != day:
        raise ValueError(f"Stale day plan: built for day {plan.day}, current day is {day}")
    if unit_idx not in plan.routes:
        raise ValueError(
            f"Unit {unit_idx} has no route in the day-{day} plan — a hand was "
            "likely hired after the plan was built."
        )
    queue = plan.routes[unit_idx]
    if not queue:
        return ["PASS"]
    return queue.pop(0)


# ---------------------------------------------------------------------------
# MODULE: evaluator.py
# ---------------------------------------------------------------------------

"""Unified economic evaluator — single decision point for all market actions in eco7.

Combines economic valuation, market order generation, and dispatch hinting
into a single, cohesive pipeline without duplicate guards or hardcoded day limits.
"""





# ---------------------------------------------------------------------------
# Output contract: evaluator → dispatch
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class DispatchHints:
    """Minimal contract evaluator → dispatch."""
    reserved_grazer_slots: int
    reserved_geese_slots: int
    crop_limits: dict[str, int]


_CENSUS_TOTAL_ATTR: dict[str, str] = {
    "COW": "total_cows",
    "SHEEP": "total_sheep",
    "GOOSE": "total_geese",
}


@dataclass(slots=True, frozen=True)
class TurnDecision:
    """Complete evaluator output for one game tick."""
    market_orders: list[list[Any]]
    hints: DispatchHints
    retained_caps: dict[str, int]
    needed_pastures: list[tuple[int, int]]
    needed_coops: list[tuple[int, int]]


# ---------------------------------------------------------------------------
# Evaluator State (Persisted across turns)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DriftState:
    prev_inv: dict[str, int] = field(default_factory=dict)
    prev_day: int = -1
    observed: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CullState:
    last_day: int = -1
    negative_days: dict[str, int] = field(default_factory=dict)
    downsized: set[str] = field(default_factory=set)
    retained_caps: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class EvalState:
    drift: DriftState = field(default_factory=DriftState)
    cull: CullState = field(default_factory=CullState)


# ---------------------------------------------------------------------------
# Candidate Valuation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CandidateScore:
    kind: str
    item: str
    cost: float
    daily_profit: float
    npv: float


def compute_daily_demand(unlocked_shops: list[str]) -> dict[str, float]:
    demand: dict[str, float] = {item: c.TOWN_CENTER_FLAT for item in c.SELLABLE_ITEMS}
    demand.pop("FERTILIZER", None)
    for shop in unlocked_shops:
        for item, qty in c.SHOP_DEMANDS.get(shop, {}).items():
            demand[item] = demand.get(item, 0.0) + qty
    return demand


# ---------------------------------------------------------------------------
# Stage 1: Probabilistic Model of Future Shop Unlocks
# ---------------------------------------------------------------------------

def _remaining_shop_draws(n_unlocked: int, horizon_day: int) -> int:
    """Number of unexecuted shop draws (out of MAX_SHOP_UNLOCKS total) that
    will occur on or before horizon_day, given that draw k occurs on day
    k * TOWN_SHOP_UNLOCK_INTERVAL."""
    return sum(
        1 for draw_idx in range(n_unlocked + 1, c.MAX_SHOP_UNLOCKS + 1)
        if draw_idx * c.TOWN_SHOP_UNLOCK_INTERVAL <= horizon_day
    )


def shop_type_probability(
    unlocked_shops: list[str], horizon_day: int, member_shops: tuple[str, ...]
) -> float:
    """Probability that at least one shop from member_shops unlocks by horizon_day.
    Each future draw independently samples uniformly with replacement from c.SHOP_DEMANDS,
    giving a per-draw success rate of len(member_shops) / len(c.SHOP_DEMANDS)."""
    m = _remaining_shop_draws(len(unlocked_shops), horizon_day)
    if m <= 0:
        return 0.0
    p_hit = len(member_shops) / len(c.SHOP_DEMANDS)
    return 1.0 - (1.0 - p_hit) ** m


def update_drift_tracker(day: int, market_inv: dict[str, int], state: DriftState) -> dict[str, float]:
    if day != state.prev_day:
        if state.prev_day >= 0:
            for item in c.SELLABLE_ITEMS:
                if item in state.prev_inv and item in market_inv:
                    state.observed[item] = float(state.prev_inv[item] - market_inv[item])
        state.prev_inv = dict(market_inv)
        state.prev_day = day
    return state.observed


def effective_drift(demand: dict[str, float], observed: dict[str, float]) -> dict[str, float]:
    return {
        item: max(demand.get(item, 0.0), 0.6 * observed.get(item, 0.0))
        for item in c.SELLABLE_ITEMS
    }


def herd_feed_cost(
    herd: int, drift: dict[str, float], market_inv: dict[str, int], horizon: int = 8
) -> float:
    w_drift = drift.get("WHEAT", 0.0) + float(herd)
    projected = max(1, market_inv.get("WHEAT", c.I0) - int(w_drift * horizon))
    return float(market_price("WHEAT", projected))


def expected_price(
    item: str,
    lag: int,
    market_inv: dict[str, int],
    drift: dict[str, float],
    additional_supply_units: float = 0.0,
) -> int:
    net_drift = drift.get(item, 0.0) - additional_supply_units
    projected_inv = market_inv.get(item, c.I0) - int(net_drift * lag)
    return market_price(item, max(1, projected_inv))


def demand_multiplier(item: str, demand: dict[str, float], market_inv: dict[str, int]) -> float:
    d = demand.get(item, 0.0)
    scarcity = c.I0 - market_inv.get(item, c.I0)
    if d >= 12.0:
        return 1.6
    if d >= 6.0:
        return 1.3
    if scarcity > 0.6 * c.MARKET_PARAMS[item]["T"]:
        return 1.4
    return 1.0


def evaluate_animal(
    species: str,
    day: int,
    total_herd: int,
    current_species_count: int,
    drift: dict[str, float],
    market_inv: dict[str, int],
    demand: dict[str, float],
) -> CandidateScore:
    spec = c.ANIMALS[species]
    cost = float(spec["cost"])
    item = spec["product"]
    lag = spec["first_yield_day"]
    interval = spec["interval"]
    days_left = c.TOTAL_DAYS - day - lag

    if days_left <= 0:
        return CandidateScore(species, item, cost, 0.0, -1.0)

    harvests = days_left // interval
    lifetime_units = harvests * interval
    own_supply = current_species_count * 1.0
    p = expected_price(item, lag, market_inv, drift, own_supply)
    revenue = lifetime_units * p * demand_multiplier(item, demand, market_inv)

    fert_price = float(get_price("FERTILIZER", market_inv))
    fert_revenue = (c.TOTAL_DAYS - day) * fert_price

    feed_price = herd_feed_cost(total_herd + 1, drift, market_inv)
    feed_cost = (c.TOTAL_DAYS - day) * feed_price
    labor_cost = (c.TOTAL_DAYS - day) * c.HAND_COST_PER_ANIMAL_DAY
    npv = revenue + fert_revenue - cost - feed_cost - labor_cost

    daily_profit = npv / max(1, c.TOTAL_DAYS - day)
    return CandidateScore(species, item, cost, daily_profit, npv)


def evaluate_fixed_crop(
    crop: str,
    day: int,
    drift: dict[str, float],
    market_inv: dict[str, int],
    demand: dict[str, float],
    current_crop_units: float = 0.0,
) -> CandidateScore:
    spec = c.CROPS[crop]
    cost = float(spec["seed_cost"])
    yield_days = spec["yield_days"]
    last_plant_day = c.STRAWBERRY_LAST_PLANT_DAY if crop == "STRAWBERRY" else c.TOTAL_DAYS - yield_days[0] - 2

    if day > last_plant_day:
        return CandidateScore(crop, crop, cost, 0.0, -1.0)

    valid_harvests = sum(1 for k in yield_days if day + k <= c.LAST_DAY)
    if valid_harvests <= 0:
        return CandidateScore(crop, crop, cost, 0.0, -1.0)

    expected_units = valid_harvests * 1.5
    lag = yield_days[0]
    p = expected_price(crop, lag, market_inv, drift, current_crop_units)
    revenue = expected_units * p * demand_multiplier(crop, demand, market_inv)
    npv = revenue - cost

    lifespan = min(c.TOTAL_DAYS - day, yield_days[-1] + 1)
    daily_profit = npv / max(1, lifespan)
    return CandidateScore(crop, crop, cost, daily_profit, npv)


def evaluate_replant_crop(
    crop: str,
    day: int,
    drift: dict[str, float],
    market_inv: dict[str, int],
    demand: dict[str, float],
    current_crop_units: float = 0.0,
) -> CandidateScore:
    spec = c.CROPS[crop]
    cost = float(spec["seed_cost"])
    grow_days = spec["max_yield_day"]
    yield_units = float(spec["max_units_base"])

    if crop == "MELON" and day > c.MELON_LAST_PLANT_DAY:
        return CandidateScore(crop, crop, cost, 0.0, -1.0)
    last_plant_day = c.WHEAT_LAST_PLANT_DAY if crop == "WHEAT" else c.TOTAL_DAYS - grow_days - 1
    if day > last_plant_day:
        return CandidateScore(crop, crop, cost, 0.0, -1.0)

    p = expected_price(crop, grow_days, market_inv, drift, current_crop_units)
    revenue = yield_units * p * demand_multiplier(crop, demand, market_inv)
    profit_per_cycle = revenue - cost

    daily_profit = profit_per_cycle / max(1, grow_days)
    npv = daily_profit * (c.TOTAL_DAYS - day)
    return CandidateScore(crop, crop, cost, daily_profit, npv)


_DAILY_AP_USAGE: dict[str, float] = {
    "COW": 3.3,
    "SHEEP": 3.25,
    "GOOSE": 4.0,
    "STRAWBERRY": 1.5,
    "TOMATO": 1.5,
    "CARROT": 1.6,
    "WHEAT": 1.5,
}


def estimate_marginal_ap_cost(
    item_kind: str,
    acc_cows: int,
    acc_sheep: int,
    acc_geese: int,
    acc_crops: dict[str, int],
    day: int,
) -> float:
    """Estimates the marginal daily hiring cost if adding item_kind causes
    the required daily Action Points to cross a worker boundary."""
    current_ap = (
        acc_cows * _DAILY_AP_USAGE["COW"]
        + acc_sheep * _DAILY_AP_USAGE["SHEEP"]
        + acc_geese * _DAILY_AP_USAGE["GOOSE"]
        + sum(count * _DAILY_AP_USAGE.get(crop, 1.5) for crop, count in acc_crops.items())
    )
    wheat_basis = max(15, acc_cows + acc_sheep + acc_geese)
    current_ap += wheat_basis * _DAILY_AP_USAGE["WHEAT"]

    hands_before = max(0, math.ceil(current_ap / 22.0) - 1)

    item_ap = _DAILY_AP_USAGE.get(item_kind, 1.5)
    new_ap = current_ap + item_ap
    hands_after = max(0, math.ceil(new_ap / 22.0) - 1)

    if hands_after > hands_before:
        marginal_hire = total_hire_cost(hands_after, 0) - total_hire_cost(hands_before, 0)
        return float(marginal_hire)

    return 0.0


# ---------------------------------------------------------------------------
# Cull Management
# ---------------------------------------------------------------------------

_CULL_CONSECUTIVE_DAYS: int = 2
_CULL_SPECIES: tuple[str, ...] = ("COW", "SHEEP", "GOOSE")


def update_cull_state(
    day: int,
    drift: dict[str, float],
    market_inv: dict[str, int],
    census: AnimalCensus,
    state: CullState,
) -> dict[str, int]:
    if day == state.last_day:
        return dict(state.retained_caps)
    state.last_day = day

    if day >= c.LIQUIDATION_DAY:
        for species in _CULL_SPECIES:
            state.retained_caps[species] = 0
        return dict(state.retained_caps)

    counts = {
        "COW": census.total_cows,
        "SHEEP": census.total_sheep,
        "GOOSE": census.total_geese,
    }

    for species in _CULL_SPECIES:
        spec = c.ANIMALS[species]
        item = spec["product"]
        lag = spec["first_yield_day"]
        rate = 1.0 / float(spec["interval"])
        projected = max(1, market_inv.get(item, c.I0) - int(drift.get(item, 0.0) * lag))
        fert_price = float(get_price("FERTILIZER", market_inv))
        revenue_day = (market_price(item, projected) * rate) + fert_price
        cost_day = float(get_price("WHEAT", market_inv)) + c.HAND_COST_PER_ANIMAL_DAY
        profit = revenue_day - cost_day

        if species in state.downsized:
            if species not in state.retained_caps:
                state.retained_caps[species] = max(1, math.ceil(counts[species] * 0.5))
            continue

        if counts[species] <= 0:
            state.negative_days[species] = 0
            state.retained_caps[species] = 999
            continue

        if profit < 0.0:
            state.negative_days[species] = state.negative_days.get(species, 0) + 1
            if state.negative_days[species] >= _CULL_CONSECUTIVE_DAYS:
                state.retained_caps[species] = max(1, math.ceil(counts[species] * 0.5))
                state.downsized.add(species)
                state.negative_days[species] = 0
            else:
                state.retained_caps[species] = 999
        else:
            state.negative_days[species] = 0
            state.retained_caps[species] = 999

    return dict(state.retained_caps)


# ---------------------------------------------------------------------------
# Market Trading Logic (Sells, Purchases, Reserves)
# ---------------------------------------------------------------------------

_THROTTLED_SELL_ITEMS: frozenset[str] = frozenset(("MELON", "WOOL", "STRAWBERRY", "WHEAT"))
_SELL_PRICE_DROP_LIMIT: float = 0.15


def _throttled_qty(item: str, qty: int, market_inv: dict[str, int], day: int) -> int:
    if day >= c.LIQUIDATION_DAY:
        return qty
    if item not in _THROTTLED_SELL_ITEMS:
        return qty
    p_now = get_price(item, market_inv)
    if p_now <= 1:
        return qty
    inv_now = market_inv.get(item, c.I0)
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


def _plan_sells(
    day: int,
    hour: int,
    shed: dict[str, int],
    farm_state: dict[str, Any],
    market_inv: dict[str, int],
    inventories: list[dict[str, int]],
    census: AnimalCensus,
    retained_caps: dict[str, int] | None = None,
) -> tuple[list[list[Any]], float]:
    orders: list[list[Any]] = []
    revenue: float = 0.0

    fert_due_count = count_fertilize_due(farm_state)
    fert_in_shed = shed.get("FERTILIZER", 0)
    fert_reserve = fert_due_count + count_fertilize_due_tomorrow(farm_state, day)
    fert_to_sell = max(0, fert_in_shed - fert_reserve)
    if fert_to_sell > 0:
        orders.append(["SELL", "FERTILIZER", fert_to_sell])
        revenue += fert_to_sell * get_price("FERTILIZER", market_inv)

    for item in ("MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO"):
        q = shed.get(item, 0)
        if q > 0:
            sell_qty = _throttled_qty(item, q, market_inv, day)
            if sell_qty > 0:
                orders.append(["SELL", item, sell_qty])
                revenue += sell_qty * get_price(item, market_inv)

    wheat_in_shed = shed.get("WHEAT", 0)
    _, sell_ceiling = wheat_feed_thresholds(census, retained_caps, day)
    if wheat_in_shed > sell_ceiling and day > 0:
        excess = wheat_in_shed - sell_ceiling
        sell_qty = _throttled_qty("WHEAT", excess, market_inv, day)
        if sell_qty > 0:
            orders.append(["SELL", "WHEAT", sell_qty])
            revenue += sell_qty * get_price("WHEAT", market_inv)

    return orders, revenue


# ---------------------------------------------------------------------------
# Cash Settlement — single source of truth for affordability
# ---------------------------------------------------------------------------

_LAND_COST_BY_OWNED_QUADS: dict[int, float] = {1: 1000.0, 2: 2000.0, 3: 4000.0}


def _order_qty_and_unit_cost(order: list[Any], market_inv: dict[str, int]) -> tuple[int, float]:
    kind, item, qty = order
    if kind == "BUY_PRODUCT":
        return qty, float(get_price(item, market_inv))
    if kind == "BUY_SEED":
        return qty, float(c.CROPS[item]["seed_cost"])
    if kind == "BUY_ANIMAL":
        return qty, float(c.ANIMALS[item]["cost"])
    raise ValueError(f"Unknown order kind in settlement: {order!r}")


def settle_orders_against_cash(
    orders: list[list[Any]],
    money: float,
    cash_reserve: float,
    market_inv: dict[str, int],
    hires_today: int,
    unlocked_quads: list[str],
) -> list[list[Any]]:
    """Walks `orders` in engine-execution order against real starting cash.

    HIRE is exempt from `cash_reserve` — workforce is the top priority and is
    allowed to spend into the reserved cash, down to $0 (never negative).
    HIRE stays first in `orders` (see evaluate_turn) so hired hands are ready
    to work this same turn; that also means HIRE never sees this turn's SELL
    revenue, only the starting `money` plus whatever reserve it can draw on.
    Every other category (BUY_LAND, BUY_PRODUCT, BUY_SEED, BUY_ANIMAL) must
    still leave cash >= cash_reserve. SELL always executes (only adds cash).
    """
    settled: list[list[Any]] = []
    cash = money
    hires_committed = hires_today

    for order in orders:
        kind = order[0]

        if kind == "SELL":
            _, item, qty = order
            cash += qty * get_price(item, market_inv)
            settled.append(order)
            continue

        if kind == "HIRE":
            cost = float(total_hire_cost(1, hires_committed))
            if cash - cost < 0.0:
                continue
            cash -= cost
            hires_committed += 1
            settled.append(order)
            continue

        if kind == "BUY_LAND":
            if len(unlocked_quads) not in _LAND_COST_BY_OWNED_QUADS:
                raise ValueError(f"Unexpected unlocked_quads for BUY_LAND: {unlocked_quads}")
            cost = _LAND_COST_BY_OWNED_QUADS[len(unlocked_quads)]
            if cash - cost < cash_reserve:
                continue
            cash -= cost
            settled.append(order)
            continue

        qty, unit_cost = _order_qty_and_unit_cost(order, market_inv)
        affordable_qty = min(qty, int(max(0.0, cash - cash_reserve) // unit_cost))
        if affordable_qty <= 0:
            continue
        cash -= affordable_qty * unit_cost
        settled.append([order[0], order[1], affordable_qty])

    assert cash >= -1e-6, f"Settlement went negative: cash={cash}, starting_money={money}"
    return settled


def get_desired_hires(
    day: int,
    catalog: list[Task],
    ready_premium: list[dict[str, Any]],
    base_units: list[tuple[tuple[int, int], dict[str, int]]],
    shed: dict[str, int],
    unlocked_quads_count: int,
) -> int:
    """Workforce (additional hands) needed today, sized directly off the real task catalog via
    required_hand_count."""
    if day < 0 or day >= c.TOTAL_DAYS:
        raise ValueError(f"Invalid day: {day}")

    max_total_hands = min(c.MAX_HANDS, 11 if unlocked_quads_count < 3 else c.MAX_HANDS)
    return required_hand_count(
        catalog, ready_premium, base_units, shed, max_total_hands, day
    )


def _plan_seed_buy(orders: list[list[Any]], crop: str, want: int) -> int:
    """Emits BUY_SEED for `want` units. Affordability is decided later, in
    settle_orders_against_cash, against real execution-order cash — not here."""
    if want <= 0:
        return 0
    orders.append(["BUY_SEED", crop, want])
    return want


# ---------------------------------------------------------------------------
# Main Evaluator Pipeline
# ---------------------------------------------------------------------------

def evaluate_turn(
    obs: dict[str, Any],
    state: EvalState,
) -> TurnDecision:
    """Single entry point for all economic decision making."""
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

    farmer_pos = (me["farmer"][0], me["farmer"][1])
    all_units: list[tuple[tuple[int, int], dict[str, int]]] = [(farmer_pos, inventories[0])]
    all_units.extend(
        ((h[0], h[1]), inventories[idx + 1]) for idx, h in enumerate(me["hands"])
    )

    # 1. Parsing & Census
    farm_state = parse_farm_state(tiles, day)
    census = count_animal_census(farm_state, shed, inventories)
    needed_pastures = compute_needed_pastures(farm_state, census)
    needed_coops = compute_needed_coops(farm_state, census, exclude=set(needed_pastures))

    # 2. Economy & Demand tracking
    demand = compute_daily_demand(unlocked_shops)
    observed = update_drift_tracker(day, market_inv, state.drift)
    drift = effective_drift(demand, observed)
    retained_caps = update_cull_state(day, drift, market_inv, census, state.cull)
    downsized_species = frozenset(state.cull.downsized)

    orders: list[list[Any]] = []
    budget = money

    # Single cash reserve — computed once, used everywhere
    cash_reserve = c.SEED_PURCHASE_CASH_RESERVE if 4 <= day < c.LIQUIDATION_DAY else 0.0

    # Opening Day Rush (Hour 0)
    if day == 0 and hour == 0:
        if shed.get("SHEEP", 0) == 0 and len(farm_state["animals"]) == 0:
            orders.extend([["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"]])
            orders.append(["BUY_ANIMAL", "SHEEP", 1])
            orders.append(["BUY_ANIMAL", "COW", 3])
            orders.append(["BUY_SEED", "MELON", 10])
            orders.append(["BUY_SEED", "WHEAT", 8])
            orders.append(["BUY_PRODUCT", "WHEAT", 4])
            hints = DispatchHints(
                reserved_grazer_slots=4,
                reserved_geese_slots=0,
                crop_limits={"MELON": 10, "WHEAT": 8},
            )
            settled = settle_orders_against_cash(
                orders, money, cash_reserve, market_inv, hires_today, unlocked_quads
            )
            return TurnDecision(
                market_orders=settled[:c.MAX_MARKET_ORDERS],
                hints=hints,
                retained_caps=retained_caps,
                needed_pastures=needed_pastures,
                needed_coops=needed_coops,
            )

    # 3. Process Sales (Revenue Injection)
    sell_orders, revenue = _plan_sells(
        day, hour, shed, farm_state, market_inv, inventories, census, retained_caps=retained_caps
    )
    orders.extend(sell_orders)
    budget += revenue

    # 4. Emergency Feed Backup (Wheat purchase if under floor)
    n_kept_feedable = kept_feedable_count(census, retained_caps)
    if n_kept_feedable > 0 and day < c.LIQUIDATION_DAY:
        buy_floor, _ = wheat_feed_thresholds(census, retained_caps, day)
        wheat_avail = shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for inv in inventories)
        if wheat_avail < buy_floor:
            want = buy_floor - wheat_avail
            orders.append(["BUY_PRODUCT", "WHEAT", want])
            budget -= float(get_price("WHEAT", market_inv) * want)

    # 5. Greedy Strategic Target Allocation (Planner core merged)
    acc_cows = census.total_cows
    acc_sheep = census.total_sheep
    acc_geese = census.total_geese
    acc_crops: dict[str, int] = {}

    target_quad_count = len(unlocked_quads)
    blocked_in_unlocked = sum(1 for pos in c.BLOCKED_CLUSTER if quad_of(pos) in unlocked_quads)
    planned_pasture_cap = (
        acc_cows + acc_sheep
        if day > c.PASTURE_RESERVATION_LAST_DAY
        else sum(1 for p in c.PASTURE_CLUSTER if quad_of(p) in unlocked_quads)
    )
    planned_coop_cap = target_quad_count * 2
    planned_tile_capacity = target_quad_count * 25 - blocked_in_unlocked

    egg_demand = demand.get("EGG", 0.0)
    wool_demand = demand.get("WOOL", 0.0)
    milk_demand = demand.get("MILK", 0.0)

    # Stage 1: Probability of relevant shop appearance before PASTURE_RESERVATION_LAST_DAY.
    # Used below to avoid prematurely suppressing herd ceilings just because the shop hasn't opened yet.
    milk_shop_p = shop_type_probability(unlocked_shops, c.PASTURE_RESERVATION_LAST_DAY, c.MILK_SUPPORT_SHOPS)
    wool_shop_p = shop_type_probability(unlocked_shops, c.PASTURE_RESERVATION_LAST_DAY, c.WOOL_SUPPORT_SHOPS)
    egg_shop_p = shop_type_probability(unlocked_shops, c.PASTURE_RESERVATION_LAST_DAY, c.EGG_SUPPORT_SHOPS)

    for _ in range(60):
        candidates: list[CandidateScore] = []
        total_herd = acc_cows + acc_sheep + acc_geese
        total_grazers = acc_cows + acc_sheep

        def add_cand(cand: CandidateScore) -> None:
            ap_cost = estimate_marginal_ap_cost(cand.kind, acc_cows, acc_sheep, acc_geese, acc_crops, day)
            if ap_cost > 0.0:
                days_left = max(1, c.TOTAL_DAYS - day)
                cand = CandidateScore(
                    kind=cand.kind,
                    item=cand.item,
                    cost=cand.cost,
                    daily_profit=cand.daily_profit - ap_cost,
                    npv=cand.npv - (ap_cost * days_left),
                )
            if cand.npv > 0.0 and cand.daily_profit > 0.0:
                candidates.append(cand)

        # Evaluate Animals (Unified NPV & Capacity Checks)
        if total_grazers < planned_pasture_cap and total_herd < c.MAX_TOTAL_HERD:
            if "COW" not in downsized_species:
                cow_cap = retained_caps.get("COW", 999) if retained_caps is not None else 999
                if milk_demand < 6.0:
                    # Stage 1: No realized milk demand yet, but if shop probability is high,
                    # expand ceiling to 4 instead of suppressing to 2.
                    max_cows_allowed = 4 if milk_shop_p >= c.RECOURSE_PROBABILITY_THRESHOLD else 2
                elif milk_demand < 12.0:
                    max_cows_allowed = 6
                else:
                    max_cows_allowed = min(14, 2 + int(milk_demand // 6.0) * 4)

                if acc_cows < cow_cap and acc_cows < max_cows_allowed:
                    add_cand(evaluate_animal("COW", day, total_herd, acc_cows, drift, market_inv, demand))

            if "SHEEP" not in downsized_species:
                sheep_cap = retained_caps.get("SHEEP", 999) if retained_caps is not None else 999
                yarn_shops_count = sum(1 for s in unlocked_shops if s == "YARN_STORE")
                if yarn_shops_count == 0:
                    # Stage 1: No yarn store today, but high future probability expands base limit to 4.
                    max_sheep_base = 4 if wool_shop_p >= c.RECOURSE_PROBABILITY_THRESHOLD else 2
                elif yarn_shops_count == 1:
                    max_sheep_base = 4
                else:
                    max_sheep_base = min(8, yarn_shops_count * 4)
                cow_npv_negative = evaluate_animal("COW", day, total_herd, acc_cows, drift, market_inv, demand).npv <= 0.0
                expand_sheep = cow_npv_negative and (yarn_shops_count > 0 or wool_demand >= 6.0)
                max_sheep_allowed = c.MAX_TOTAL_HERD if expand_sheep else max_sheep_base

                if acc_sheep < sheep_cap and acc_sheep < max_sheep_allowed:
                    add_cand(evaluate_animal("SHEEP", day, total_herd, acc_sheep, drift, market_inv, demand))

        egg_shops_count = sum(1 for s in unlocked_shops if s in c.EGG_SUPPORT_SHOPS)
        # Stage 1: Keep goose candidate open even without open egg shops if future probability is high.
        goose_gate_open = egg_shops_count > 0 or egg_shop_p >= c.RECOURSE_PROBABILITY_THRESHOLD
        if goose_gate_open and acc_geese < planned_coop_cap and total_herd < c.MAX_TOTAL_HERD:
            if "GOOSE" not in downsized_species:
                goose_cap = retained_caps.get("GOOSE", 999) if retained_caps is not None else 999
                goose_floor = 2 if egg_shops_count == 0 and egg_shop_p >= c.RECOURSE_PROBABILITY_THRESHOLD else 0
                max_geese_allowed = max(int(egg_demand), goose_floor)
                if acc_geese < goose_cap and acc_geese < max_geese_allowed:
                    add_cand(evaluate_animal("GOOSE", day, total_herd, acc_geese, drift, market_inv, demand))

        # Evaluate Crops
        wheat_tiles_needed = n_kept_feedable
        available_crop_tiles = max(0, planned_tile_capacity - planned_pasture_cap - planned_coop_cap - wheat_tiles_needed)
        total_commercial_crops = sum(acc_crops.values())

        if total_commercial_crops < available_crop_tiles:
            if day <= c.STRAWBERRY_LAST_PLANT_DAY:
                straw_count = acc_crops.get("STRAWBERRY", 0)
                add_cand(evaluate_fixed_crop(
                    "STRAWBERRY", day, drift, market_inv, demand, current_crop_units=float(straw_count) * 0.46
                ))

            if day <= c.MELON_LAST_PLANT_DAY:
                melon_count = acc_crops.get("MELON", 0)
                add_cand(evaluate_replant_crop(
                    "MELON", day, drift, market_inv, demand, current_crop_units=float(melon_count) * 1.0
                ))

            if demand.get("TOMATO", 0.0) >= 6.0:
                tomato_count = acc_crops.get("TOMATO", 0)
                if tomato_count < 8:
                    add_cand(evaluate_fixed_crop(
                        "TOMATO", day, drift, market_inv, demand, current_crop_units=float(tomato_count) * 0.5
                    ))

            if demand.get("CARROT", 0.0) >= 6.0:
                carrot_count = acc_crops.get("CARROT", 0)
                if carrot_count < 8:
                    add_cand(evaluate_replant_crop(
                        "CARROT", day, drift, market_inv, demand, current_crop_units=float(carrot_count) * 1.0
                    ))

            if day <= c.WHEAT_LAST_PLANT_DAY:
                wheat_count = acc_crops.get("WHEAT", 0)
                add_cand(evaluate_replant_crop(
                    "WHEAT", day, drift, market_inv, demand, current_crop_units=float(wheat_count) * 1.0
                ))

        if not candidates:
            break

        best = max(candidates, key=lambda c_cand: c_cand.daily_profit)
        if best.daily_profit <= 0.0:
            break

        if best.kind == "COW":
            acc_cows += 1
        elif best.kind == "SHEEP":
            acc_sheep += 1
        elif best.kind == "GOOSE":
            acc_geese += 1
        else:
            acc_crops[best.kind] = acc_crops.get(best.kind, 0) + 1

    # Land Purchase Target Calculation
    crop_tiles_need = sum(acc_crops.values())
    wheat_basis = n_kept_feedable
    total_needed_tiles = acc_cows + acc_sheep + acc_geese + crop_tiles_need + wheat_basis
    current_tile_capacity = planned_tile_capacity

    target_quads = len(unlocked_quads)
    if len(unlocked_quads) < 3:
        next_cost = 1000.0 if len(unlocked_quads) == 1 else (2000.0 if len(unlocked_quads) == 2 else 4000.0)
        has_land_budget = budget >= next_cost + cash_reserve
        # farm_state["empty_tiles"] is a guaranteed key of parse_farm_state's contract —
        # .get(..., []) would silently mask a broken farm_state instead of failing loud.
        real_empty_tiles = sum(
            1 for pos in farm_state["empty_tiles"] if pos not in c.BLOCKED_CLUSTER_SET
        )
        utilization = 1.0 - (real_empty_tiles / max(1, current_tile_capacity))
        is_capacity_constrained = (
            (total_needed_tiles >= current_tile_capacity and utilization >= 0.8)
            or real_empty_tiles <= 2
            or (acc_cows + acc_sheep) > sum(1 for p in c.PASTURE_CLUSTER if quad_of(p) in unlocked_quads)
            or acc_geese > len(unlocked_quads) * 2
        )
        if has_land_budget and is_capacity_constrained:
            target_quads = len(unlocked_quads) + 1

    # 6. Hiring Decisions (Hour 0 or 1)
    hints = DispatchHints(
        reserved_grazer_slots=acc_cows + acc_sheep,
        reserved_geese_slots=acc_geese,
        crop_limits=dict(acc_crops),
    )
    hints_dict = asdict(hints)

    if hour in (0, 1):
        catalog = build_task_catalog(
            farm_state, shed, dict(seeds), needed_pastures, needed_coops,
            all_units, day, hour, hints=hints_dict, unlocked_quads=unlocked_quads,
            retained_caps=retained_caps,
        )
        ready_premium = _ready_premium_bushes(farm_state, day)
        new_hires = get_desired_hires(
            day, catalog, ready_premium, all_units, shed, len(unlocked_quads)
        )
        hire_orders = [["HIRE"] for _ in range(new_hires)]
        orders = hire_orders + orders

    # 7-9. Purchase window — animals, land, seeds. Gated to a single hour so that
    # decisions are evaluated once daily against revenue accumulated from SELL throughout
    # the day (money peaks by hour 21), rather than smeared across 24 calls with partial cash.
    # Everything purchased here lands in the shed by the end of the tick (market actions execute
    # within the turn), so it will be visible to tomorrow's build_day_plan at hour=1.
    if hour == c.PURCHASE_HOUR:
        res_structures = live_reserved_structures(
            farm_state, unlocked_quads, hints.reserved_grazer_slots, hints.reserved_geese_slots, day=day
        )
        needed_structs = set(needed_pastures) | set(needed_coops)

        # 7. Execute Animal Purchases
        total_bought_this_tick = 0
        total_grazers_bought_this_tick = 0
        current_grazers = census.total_cows + census.total_sheep
        current_total = census.total_all

        animal_targets = (("COW", acc_cows), ("SHEEP", acc_sheep), ("GOOSE", acc_geese))
        animal_orders: list[list[Any]] = []

        for species, target in animal_targets:
            if species in downsized_species:
                continue
            if species == "GOOSE" and len(unlocked_quads) < 2:
                continue
            cap = retained_caps.get(species, 999) if retained_caps is not None else 999
            have = getattr(census, _CENSUS_TOTAL_ATTR[species])
            effective_target = min(target, cap)
            if species == "GOOSE":
                effective_target = min(effective_target, coop_capacity(farm_state) + len(res_structures))
            deficit = effective_target - have
            if deficit <= 0:
                continue

            remaining_herd_slots = max(0, c.MAX_TOTAL_HERD - (current_total + total_bought_this_tick))
            max_buyable = min(remaining_herd_slots, max(0, c.MAX_TOTAL_HERD - (current_grazers + total_grazers_bought_this_tick))) if species in ("COW", "SHEEP") else remaining_herd_slots
            # Single daily purchase attempt — request full deficit without artificial
            # min(2, ...) throttling so the herd reaches target capacity on time.
            want = min(deficit, max_buyable)
            if want > 0:
                animal_orders.append(["BUY_ANIMAL", species, want])
                total_bought_this_tick += want
                if species in ("COW", "SHEEP"):
                    total_grazers_bought_this_tick += want

        # 8. Execute Land Purchases
        land_orders: list[list[Any]] = []
        want_land = target_quads > len(unlocked_quads)
        if "NE" not in unlocked_quads and want_land:
            land_orders.append(["BUY_LAND"])
        elif "SW" not in unlocked_quads and "NE" in unlocked_quads and want_land:
            land_orders.append(["BUY_LAND"])
        elif "SE" not in unlocked_quads and "SW" in unlocked_quads and target_quads >= 4:
            land_orders.append(["BUY_LAND"])

        # 9. Execute Seed Purchases
        wheat_only_land = res_structures - needed_structs
        # BLOCKED_CLUSTER is disjoint from PASTURE_CLUSTER (source of res_structures/
        # needed_structs), so it must be excluded explicitly here — otherwise the
        # evaluator overbuys seeds that dispatch.py will never actually plant.
        open_empty_count = sum(
            1 for pos in farm_state["empty_tiles"]
            if pos not in res_structures and pos not in needed_structs and pos not in c.BLOCKED_CLUSTER_SET
        )
        wheat_reserved_count = sum(
            1 for pos in farm_state["empty_tiles"] if pos in wheat_only_land
        )
        empty_count = open_empty_count
        wheat_committed_this_turn = 0

        if 4 <= day <= c.WHEAT_LAST_PLANT_DAY:
            curr_wheat = committed_count(farm_state, seeds, "WHEAT")
            wheat_capacity = empty_count + wheat_reserved_count
            if curr_wheat < n_kept_feedable and wheat_capacity > 0:
                bought = _plan_seed_buy(orders, "WHEAT", min(n_kept_feedable - curr_wheat, wheat_capacity))
                from_open = min(bought, empty_count)
                empty_count -= from_open
                wheat_reserved_count -= (bought - from_open)
                wheat_committed_this_turn += bought

        straw_target = acc_crops.get("STRAWBERRY", 0)
        total_straw = committed_count(farm_state, seeds, "STRAWBERRY")
        if day <= c.STRAWBERRY_LAST_PLANT_DAY and total_straw < straw_target and empty_count > 0:
            bought = _plan_seed_buy(orders, "STRAWBERRY", min(straw_target - total_straw, empty_count))
            empty_count = max(0, empty_count - bought)

        if acc_crops:
            straw_done = committed_count(farm_state, seeds, "STRAWBERRY")
            for crop, target in acc_crops.items():
                if crop in ("STRAWBERRY", "WHEAT") or crop not in c.CROPS or target <= 0:
                    continue
                if straw_done < straw_target:
                    straw_npv = evaluate_fixed_crop(
                        "STRAWBERRY", day, drift, market_inv, demand,
                        current_crop_units=float(straw_done) * 0.46,
                    ).npv
                    crop_npv = (
                        evaluate_fixed_crop(crop, day, drift, market_inv, demand).npv
                        if c.CROPS[crop].get("is_ongoing") else
                        evaluate_replant_crop(crop, day, drift, market_inv, demand).npv
                    )
                    if straw_npv > crop_npv:
                        continue
                spec = c.CROPS[crop]
                lag = spec["first_yield_day"] if spec["is_ongoing"] else spec["max_yield_day"]
                if day > c.TOTAL_DAYS - lag - 1:
                    continue
                curr = committed_count(farm_state, seeds, crop)
                # CARROT is replanted atomically at the same stop as HARVEST (build_task_catalog),
                # which is planned once daily at hour=1. The seed must be in the shed in advance,
                # requiring the same lookahead as WHEAT below.
                freeing_tomorrow = count_maturing_tomorrow(farm_state, crop) if crop == "CARROT" else 0
                want = (target + freeing_tomorrow) - curr
                if want > 0:
                    bought = _plan_seed_buy(orders, crop, want)
                    empty_count = max(0, empty_count - bought)

        if 1 <= day <= c.WHEAT_LAST_PLANT_DAY and (day < c.WHEAT_LAST_PLANT_DAY or hour < 12):
            wheat_total_target = n_kept_feedable + acc_crops.get("WHEAT", 0)
            curr_wheat_committed = committed_count(farm_state, seeds, "WHEAT") + wheat_committed_this_turn
            wheat_freeing_tomorrow = count_maturing_tomorrow(farm_state, "WHEAT")
            already_have = seeds.get("WHEAT", 0) + wheat_committed_this_turn

            room = wheat_total_target - curr_wheat_committed
            wheat_capacity = empty_count + wheat_reserved_count
            new_planting = min(wheat_capacity, max(0, room))
            replant_qty = max(0, min(wheat_freeing_tomorrow, room + wheat_freeing_tomorrow))
            want_wheat = max(0, new_planting + replant_qty - already_have)
            _plan_seed_buy(orders, "WHEAT", want_wheat)

        orders.extend(land_orders)
        orders.extend(animal_orders)

    settled = settle_orders_against_cash(
        orders, money, cash_reserve, market_inv, hires_today, unlocked_quads
    )

    return TurnDecision(
        market_orders=settled[:c.MAX_MARKET_ORDERS],
        hints=hints,
        retained_caps=retained_caps,
        needed_pastures=needed_pastures,
        needed_coops=needed_coops,
    )


# ---------------------------------------------------------------------------
# MODULE: main.py
# ---------------------------------------------------------------------------

"""Kaggle Agent Entry Point — EcoBot v7 Unified Evaluator."""





@dataclass(slots=True)
class AgentMemory:
    eval_state: EvalState = field(default_factory=EvalState)
    day_plan: DayPlan | None = None


_MEMORIES: dict[int, AgentMemory] = {}


def agent(obs: dict[str, Any]) -> dict[str, Any]:
    global _MEMORIES
    player = obs["player"]
    if obs.get("step") == 0 or (obs.get("day") == 0 and obs.get("hour") == 0) or player not in _MEMORIES:
        _MEMORIES[player] = AgentMemory()

    mem = _MEMORIES[player]
    day = obs["day"]
    hour = obs["hour"]

    me = obs["farms"][player]
    private = obs["private"]
    inventories = private["inventories"]

    farmer_pos = (me["farmer"][0], me["farmer"][1])
    all_units: list[tuple[tuple[int, int], dict[str, int]]] = [(farmer_pos, inventories[0])]
    all_units.extend(((h[0], h[1]), inventories[idx + 1]) for idx, h in enumerate(me["hands"]))

    # 1. Unified evaluator — single call produces market orders + dispatch hints
    decision = evaluate_turn(obs, mem.eval_state)

    # 2. Units & Field Actions
    if hour == 0:
        actions = [["PASS"] for _ in all_units]
    else:
        if mem.day_plan is None or mem.day_plan.day != day:
            farm_state = parse_farm_state(me["tiles"], day)
            pending_hires = max(0, me["hires_today"] - len(me["hands"])) + sum(
                1 for o in decision.market_orders if o[0] == "HIRE"
            )
            hints_dict = asdict(decision.hints)
            mem.day_plan = build_day_plan(
                all_units, farm_state, private["shed"], private["seeds"],
                day, hour, decision.needed_pastures, decision.needed_coops,
                hints=hints_dict,
                unlocked_quads=me["unlocked_quadrants"],
                retained_caps=decision.retained_caps,
                pending_hires=pending_hires,
            )
        actions = [
            next_action(ui, day, mem.day_plan)
            for ui in range(len(all_units))
        ]

    return {
        "farmer": actions[0],
        "hands": actions[1:],
        "market": decision.market_orders,
    }

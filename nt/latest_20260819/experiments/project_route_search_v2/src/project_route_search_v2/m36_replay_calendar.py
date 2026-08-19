"""Compile official gold Replays into complete M3.6 RouteCalendarV3 values.

The compiler is a host-side calibration utility.  It deliberately emits only
daily business targets and low-frequency policy selectors; it never emits or
stores the Replay's raw per-step unit/market actions.  The online controller
must regenerate legal actions from the current state and this calendar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import ANIMALS, CROPS, NUM_ANIMALS, NUM_CROPS

from .m36_calendar import empty_route_calendar_v3
from .m36_schema import (
    M36MarketTemplateV3,
    M36ReleasePolicyV3,
    ROUTE_DAYS_V3,
    RouteCalendarV3,
)


_CROP_ID = {name: index for index, name in enumerate(CROPS)}
_ANIMAL_ID = {name: index for index, name in enumerate(ANIMALS)}


@dataclass(frozen=True)
class ReplayCalendarDiagnosticsV3:
    episode_id: int
    player: int
    team_name: str
    reward: float
    replay_step_count: int
    observed_days: int
    requested_animal_totals: tuple[int, ...]
    filled_animal_additions: tuple[int, ...]
    land_addition_count: int
    day6_crop_peak: tuple[int, ...]
    day6_service_end: tuple[int, ...]
    day6_care_policy: tuple[int, ...]
    planned_ownership_reduction_count: int
    raw_action_payload_stored: bool
    event_overflow_count: int

    @property
    def route_signature(self) -> str:
        animal = "-".join(str(value) for value in self.requested_animal_totals)
        return f"animals={animal};land+={self.land_addition_count}"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["route_signature"] = self.route_signature
        return result


def _quantity(mapping: Mapping[str, Any] | None, item: str) -> int:
    if not isinstance(mapping, Mapping):
        return 0
    return int(mapping.get(item, 0) or 0)


def _inventory_quantity(private: Mapping[str, Any], item: str) -> int:
    total = 0
    for inventory in private.get("inventories", ()) or ():
        total += _quantity(inventory, item)
    return total


def _map_crop_counts(farm: Mapping[str, Any]) -> np.ndarray:
    result = np.zeros((NUM_CROPS,), dtype=np.int16)
    for row in farm.get("tiles", ()) or ():
        for tile in row or ():
            if not isinstance(tile, Mapping) or tile.get("kind") != "PLANT":
                continue
            crop_id = _CROP_ID.get(str(tile.get("crop")))
            if crop_id is not None:
                result[crop_id] += 1
    return result


def _map_animal_counts(farm: Mapping[str, Any]) -> np.ndarray:
    result = np.zeros((NUM_ANIMALS,), dtype=np.int16)
    for row in farm.get("tiles", ()) or ():
        for tile in row or ():
            if not isinstance(tile, Mapping):
                continue
            animal_id = _ANIMAL_ID.get(str(tile.get("animal")))
            if animal_id is not None:
                result[animal_id] += 1
    return result


def _owned_animal_counts(
    farm: Mapping[str, Any], private: Mapping[str, Any]
) -> np.ndarray:
    result = _map_animal_counts(farm).astype(np.int32)
    shed = private.get("shed", {}) or {}
    for name, animal_id in _ANIMAL_ID.items():
        result[animal_id] += _quantity(shed, name)
        result[animal_id] += _inventory_quantity(private, name)
    return result.astype(np.int16)


def _service_animal_counts(
    farm: Mapping[str, Any], private: Mapping[str, Any]
) -> np.ndarray:
    """Animals already in service or carried toward placement, excluding shed."""

    result = _map_animal_counts(farm).astype(np.int32)
    for name, animal_id in _ANIMAL_ID.items():
        result[animal_id] += _inventory_quantity(private, name)
    return result.astype(np.int16)


def _observed_care_policy(farm: Mapping[str, Any]) -> np.ndarray:
    """Compile public CARE state into a daily policy without raw actions.

    A cared tile proves that the route deliberately services that species on
    the current day.  The online controller still chooses the animal, unit,
    path and exact action from live state; the calendar stores only the
    low-frequency business policy.
    """

    result = np.zeros((NUM_ANIMALS,), dtype=np.int8)
    for row in farm.get("tiles", ()) or ():
        for tile in row or ():
            if not isinstance(tile, Mapping) or not bool(tile.get("cared_today", False)):
                continue
            animal_id = _ANIMAL_ID.get(str(tile.get("animal")))
            if animal_id is not None:
                # CAPACITY_AWARE_EVERY_CYCLE in M3AnimalCarePolicyV2.
                result[animal_id] = np.int8(2)
    return result


def _market_template(raw_market: Sequence[Any]) -> int:
    orders = [order for order in (raw_market or ()) if isinstance(order, list) and order]
    if not orders:
        return int(M36MarketTemplateV3.HIRE_ANIMAL_SEED_FEED)
    names = [str(order[0]) for order in orders]
    non_sell = [index for index, name in enumerate(names) if name != "SELL"]
    first_non_sell = min(non_sell) if non_sell else len(names)
    if "SELL" in names[:first_non_sell]:
        return int(M36MarketTemplateV3.SELL_HIRE_ANIMAL_SEED_FEED)
    hire_or_animal = [
        index for index, name in enumerate(names) if name in ("HIRE", "BUY_ANIMAL")
    ]
    first_commitment = min(hire_or_animal) if hire_or_animal else len(names)
    if any(
        name == "BUY_PRODUCT" and index < first_commitment
        for index, name in enumerate(names)
    ):
        return int(M36MarketTemplateV3.CRITICAL_INPUT_HIRE_ANIMAL_SELL)
    return int(M36MarketTemplateV3.HIRE_ANIMAL_SEED_FEED)


def _primary_transaction_template(
    values: Sequence[tuple[int, int]], default: int
) -> int:
    """Return the template of the day's largest non-empty transaction.

    A Replay contains 24 action frames per day, but most market lists are
    empty.  Treating an empty frame as template zero made those 23 no-ops
    outvote the one economically meaningful bundle.  When a route splits its
    opening across two frames, the larger bundle is the best low-frequency
    daily template; ties deliberately retain the earlier bundle.
    """

    if not values:
        return int(default)
    return int(max(values, key=lambda value: value[0])[1])


def compile_gold_replay_calendar_v3(
    replay: Mapping[str, Any], *, player: int, candidate_id: int
) -> tuple[RouteCalendarV3, ReplayCalendarDiagnosticsV3]:
    """Compile one player's official Replay into a full, untruncated calendar."""

    steps = replay.get("steps", ()) or ()
    if not steps:
        raise ValueError("Replay has no steps")
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    hand = np.zeros((ROUTE_DAYS_V3,), dtype=np.int8)
    crop = np.zeros((ROUTE_DAYS_V3, NUM_CROPS), dtype=np.int16)
    animal_additions = np.zeros((ROUTE_DAYS_V3, NUM_ANIMALS), dtype=np.int16)
    animal_service = np.zeros((ROUTE_DAYS_V3, NUM_ANIMALS), dtype=np.int16)
    animal_care = np.zeros((ROUTE_DAYS_V3, NUM_ANIMALS), dtype=np.int8)
    land_additions = np.zeros((ROUTE_DAYS_V3,), dtype=np.int8)
    feed_stock = np.zeros((ROUTE_DAYS_V3,), dtype=np.int16)
    fertilizer_stock = np.zeros((ROUTE_DAYS_V3,), dtype=np.int16)
    market_templates: list[list[tuple[int, int]]] = [
        [] for _ in range(ROUTE_DAYS_V3)
    ]
    requested_animals = np.zeros((NUM_ANIMALS,), dtype=np.int32)
    seen_days: set[int] = set()
    last_owned: np.ndarray | None = None
    last_land: int | None = None
    planned_reductions = 0

    # The target for crops is the peak simultaneous operating scale in a day.
    # Animal service uses the final state of each day so a deliberate terminal
    # release can be represented by a lower next daily target.
    for step in steps:
        row = step[player]
        observation = row.get("observation", {}) or {}
        day = int(observation.get("day", -1))
        if day < 0 or day >= ROUTE_DAYS_V3:
            continue
        seen_days.add(day)
        farms = observation.get("farms", ()) or ()
        if len(farms) <= player:
            raise ValueError(f"Replay observation lacks farm for player {player}")
        farm = farms[player] or {}
        private = observation.get("private", {}) or {}

        hand[day] = max(hand[day], int(farm.get("hires_today", 0) or 0))
        crop[day] = np.maximum(crop[day], _map_crop_counts(farm))

        owned = _owned_animal_counts(farm, private)
        if last_owned is not None:
            delta = owned.astype(np.int32) - last_owned.astype(np.int32)
            animal_additions[day] += np.maximum(delta, 0).astype(np.int16)
            planned_reductions += int(np.maximum(-delta, 0).sum())
        last_owned = owned
        animal_service[day] = _service_animal_counts(farm, private)
        animal_care[day] = np.maximum(
            animal_care[day], _observed_care_policy(farm)
        )

        land = len(farm.get("unlocked_quadrants", ()) or ())
        if last_land is not None and land > last_land:
            land_additions[day] += np.int8(land - last_land)
        last_land = land

        action = row.get("action", {}) or {}
        market = action.get("market", ()) or ()
        if market:
            market_templates[day].append((len(market), _market_template(market)))
        for order in market:
            if not isinstance(order, list) or not order:
                continue
            op = str(order[0])
            item = str(order[1]) if len(order) >= 2 else ""
            quantity = int(order[2]) if len(order) >= 3 else 0
            if op == "BUY_ANIMAL" and item in _ANIMAL_ID:
                requested_animals[_ANIMAL_ID[item]] += quantity
            elif op == "BUY_PRODUCT" and item == "WHEAT":
                feed_stock[day] = max(feed_stock[day], quantity)
            elif op == "BUY_PRODUCT" and item == "FERTILIZER":
                fertilizer_stock[day] = max(fertilizer_stock[day], quantity)

    market_template = np.asarray(
        [
            _primary_transaction_template(
                values, int(M36MarketTemplateV3.HIRE_ANIMAL_SEED_FEED)
            )
            for values in market_templates
        ],
        dtype=np.int8,
    )
    release_policy = np.full(
        (ROUTE_DAYS_V3,),
        int(M36ReleasePolicyV3.LOWEST_FUTURE_NET_VALUE),
        dtype=np.int8,
    )

    calendar = empty_route_calendar_v3(1)._replace(
        candidate_id=jnp.asarray((candidate_id,), dtype=jnp.int32),
        hand_target_by_day=jnp.asarray(hand[None]),
        crop_target_by_day=jnp.asarray(crop[None]),
        animal_purchase_additions_by_day=jnp.asarray(animal_additions[None]),
        animal_service_target_by_day=jnp.asarray(animal_service[None]),
        animal_care_policy_by_day=jnp.asarray(animal_care[None]),
        land_additions_by_day=jnp.asarray(land_additions[None]),
        feed_stock_target_by_day=jnp.asarray(feed_stock[None]),
        fertilizer_safety_stock_by_day=jnp.asarray(fertilizer_stock[None]),
        market_template_by_day=jnp.asarray(market_template[None]),
        planned_release_policy_by_day=jnp.asarray(release_policy[None]),
        event_overflow_count=jnp.zeros((1,), dtype=jnp.int32),
    )

    teams = replay.get("info", {}).get("TeamNames", ()) or ()
    rewards = replay.get("rewards", ()) or ()
    episode_value = replay.get("info", {}).get("EpisodeId", replay.get("id", 0))
    try:
        episode_id = int(episode_value or 0)
    except (TypeError, ValueError):
        episode_id = 0
    diagnostics = ReplayCalendarDiagnosticsV3(
        episode_id=episode_id,
        player=player,
        team_name=str(teams[player]) if len(teams) > player else f"player_{player}",
        reward=float(rewards[player]) if len(rewards) > player else 0.0,
        replay_step_count=len(steps),
        observed_days=len(seen_days),
        requested_animal_totals=tuple(int(value) for value in requested_animals),
        filled_animal_additions=tuple(
            int(value) for value in animal_additions.sum(axis=0)
        ),
        land_addition_count=int(land_additions.sum()),
        day6_crop_peak=tuple(int(value) for value in crop[6]),
        day6_service_end=tuple(int(value) for value in animal_service[6]),
        day6_care_policy=tuple(int(value) for value in animal_care[6]),
        planned_ownership_reduction_count=planned_reductions,
        raw_action_payload_stored=False,
        event_overflow_count=0,
    )
    return calendar, diagnostics


def stack_route_calendars_v3(calendars: Iterable[RouteCalendarV3]) -> RouteCalendarV3:
    values = tuple(calendars)
    if not values:
        raise ValueError("at least one calendar is required")
    return RouteCalendarV3(
        *(jnp.concatenate([getattr(value, name) for value in values], axis=0) for name in RouteCalendarV3._fields)
    )


__all__ = [
    "ReplayCalendarDiagnosticsV3",
    "compile_gold_replay_calendar_v3",
    "stack_route_calendars_v3",
]

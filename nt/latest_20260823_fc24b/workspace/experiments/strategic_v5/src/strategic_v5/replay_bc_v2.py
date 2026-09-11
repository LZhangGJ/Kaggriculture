"""Compile official replays into actor-visible opponent-aware V2 BC tensors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    MAX_UNITS,
    MAX_MARKET_ORDERS,
    NUM_CROPS,
    NUM_PLAYERS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    PRODUCTS,
    SHED_ITEMS,
    SHOP_NAMES,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import State, StaticTables

from replay_task_cards_compiler.compiler import (
    _name_key,
    _runtime_slot,
    extract_bundle,
    normalize_action,
)

from .constants import MAX_CANDIDATES_V1, TaskTypeV1
from .e4_core import (
    BUILD_COOP_SLOT,
    BUILD_PASTURE_SLOT,
    build_full_core_candidates_v1,
    evaluate_full_core_feasibility_v1,
)
from .e5_econ import build_full_econ_features_v1, full_econ_score_v1
from .lifecycle import reset_controller_state_v1
from .opponent_v2 import (
    OpponentHistoryV2,
    build_candidate_features_v2,
    build_global_features_v2,
    build_public_snapshot_v2,
)
from .task_cards import ReplayTaskCardProgramV1


REPLAY_BC_SCHEMA_V2 = "replay_bc_opponent_v2"
ORDERED_TURN_SCHEMA_V3 = "replay_ordered_turn_v3"
ORDERED_STOP_SLOT_V3 = MAX_CANDIDATES_V1
STRATEGIC_SLOTS_V2 = tuple(range(21)) + (BUILD_COOP_SLOT, BUILD_PASTURE_SLOT)
_PRODUCT_ID = {name: index for index, name in enumerate(PRODUCTS)}
_CROP_ID = {name: index for index, name in enumerate(CROPS)}
_ANIMAL_ID = {name: index for index, name in enumerate(ANIMALS)}
_SHED_ID = {name: index for index, name in enumerate(SHED_ITEMS)}
_SHOP_ID = {name: index for index, name in enumerate(SHOP_NAMES)}


@dataclass(frozen=True)
class ReplayOrderedTurnV3:
    unit_op: np.ndarray
    unit_item: np.ndarray
    unit_quantity: np.ndarray
    unit_count: int
    market_candidate_slot: np.ndarray
    market_quantity: np.ndarray
    market_valid: np.ndarray
    market_count: int


@dataclass(frozen=True)
class ReplayBCSampleV2:
    episode_id: int
    source_step: int
    original_seat: int
    state: State
    positive_candidate_mask: np.ndarray
    market_quantity: np.ndarray
    ordered_turn: ReplayOrderedTurnV3
    opponent_task_type: int


@dataclass(frozen=True)
class ReplayBCFeatureBatchV2:
    """Target-free deployable features and the exact runtime eligibility mask."""

    global_features: jax.Array
    candidate_features: jax.Array
    candidate_task_type: jax.Array
    candidate_present: jax.Array
    candidate_legal: jax.Array
    candidate_bankable: jax.Array
    candidate_eligible: jax.Array


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _mapping_count(mapping: Any, name: str) -> int:
    return _integer(mapping.get(name, 0), 0) if isinstance(mapping, dict) else 0


def _canonical_farms(observation: dict[str, Any], expert_seat: int) -> list[dict[str, Any]]:
    farms = observation.get("farms", [])
    if not isinstance(farms, list) or len(farms) != NUM_PLAYERS:
        raise ValueError("observation must contain two public farms")
    return [farms[expert_seat], farms[1 - expert_seat]]


def observation_to_canonical_state_v2(
    observation: dict[str, Any], expert_seat: int
) -> State:
    """Return an unbatched State with the expert canonicalized as player 0.

    Only the expert private section is filled.  Canonical player 1 private
    arrays remain zero, exactly matching online visibility.
    """

    farms = _canonical_farms(observation, expert_seat)
    step = _integer(
        observation.get(
            "step",
            _integer(observation.get("day"), 0) * 24
            + _integer(observation.get("hour"), 0),
        )
    )
    money = np.zeros((NUM_PLAYERS,), dtype=np.int32)
    tile_kind = np.full(
        (NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE), TileKind.EMPTY, dtype=np.int8
    )
    tile_crop = np.full_like(tile_kind, -1, dtype=np.int8)
    tile_animal = np.full_like(tile_kind, -1, dtype=np.int8)
    tile_origin_day = np.full_like(tile_kind, -1, dtype=np.int8)
    tile_yield = np.zeros_like(tile_kind, dtype=np.int16)
    tile_neglect = np.zeros_like(tile_kind, dtype=np.int8)
    tile_max_lifespan = np.full_like(tile_kind, -1, dtype=np.int16)
    tile_fertilized_until = np.full_like(tile_kind, -1, dtype=np.int8)
    tile_pending_care = np.zeros_like(tile_kind, dtype=np.int8)
    tile_flags = np.zeros_like(tile_kind, dtype=np.uint8)
    unit_pos = np.zeros((NUM_PLAYERS, MAX_UNITS, 2), dtype=np.int8)
    unit_active = np.zeros((NUM_PLAYERS, MAX_UNITS), dtype=np.bool_)
    hires_today = np.zeros((NUM_PLAYERS,), dtype=np.int8)
    unlocked_count = np.ones((NUM_PLAYERS,), dtype=np.int8)

    for player, farm in enumerate(farms):
        if not isinstance(farm, dict):
            raise ValueError("farm must be a mapping")
        money[player] = _integer(farm.get("money"), 0)
        positions = [farm.get("farmer", [4, 4]), *(farm.get("hands", []) or [])]
        if len(positions) > MAX_UNITS:
            raise ValueError("replay exceeds frozen MAX_UNITS")
        for unit, position in enumerate(positions):
            if not isinstance(position, list) or len(position) < 2:
                raise ValueError("invalid public unit position")
            unit_pos[player, unit] = (_integer(position[0]), _integer(position[1]))
            unit_active[player, unit] = True
        hires_today[player] = _integer(farm.get("hires_today"), 0)
        quadrants = farm.get("unlocked_quadrants", ["NW"])
        unlocked_count[player] = len(quadrants) if isinstance(quadrants, list) else 1
        tiles = farm.get("tiles", [])
        if not isinstance(tiles, list) or len(tiles) != BOARD_SIZE:
            raise ValueError("farm tiles must be a 10x10 list")
        for y, row in enumerate(tiles):
            if not isinstance(row, list) or len(row) != BOARD_SIZE:
                raise ValueError("farm tiles must be a 10x10 list")
            for x, tile in enumerate(row):
                if tile is None:
                    tile_kind[player, y, x] = TileKind.EMPTY
                    continue
                if tile == "LOCKED":
                    tile_kind[player, y, x] = TileKind.LOCKED
                    continue
                if not isinstance(tile, dict):
                    raise ValueError(f"unsupported tile value at {(player, y, x)}")
                kind = str(tile.get("kind", ""))
                if kind == "WEED":
                    tile_kind[player, y, x] = TileKind.WEED
                    continue
                if kind == "PLANT":
                    tile_kind[player, y, x] = TileKind.PLANT
                    crop_name = str(tile.get("crop", ""))
                    if crop_name not in _CROP_ID:
                        raise ValueError(f"unknown crop {crop_name!r}")
                    tile_crop[player, y, x] = _CROP_ID[crop_name]
                    tile_origin_day[player, y, x] = _integer(tile.get("planted_day"), -1)
                    tile_yield[player, y, x] = _integer(tile.get("yield_units"), 0)
                    tile_neglect[player, y, x] = _integer(
                        tile.get("consecutive_unwatered"), 0
                    )
                    tile_max_lifespan[player, y, x] = _integer(
                        tile.get("max_lifespan_step"), -1
                    )
                    tile_fertilized_until[player, y, x] = _integer(
                        tile.get("fertilized_until_day"), -1
                    )
                    if bool(tile.get("watered_today", False)):
                        tile_flags[player, y, x] |= FLAG_WATERED
                    continue
                if kind not in {"COOP", "PASTURE"}:
                    raise ValueError(f"unknown tile kind {kind!r}")
                tile_kind[player, y, x] = (
                    TileKind.COOP if kind == "COOP" else TileKind.PASTURE
                )
                animal_name = tile.get("animal")
                if animal_name is None:
                    continue
                animal_name = str(animal_name)
                if animal_name not in _ANIMAL_ID:
                    raise ValueError(f"unknown animal {animal_name!r}")
                tile_animal[player, y, x] = _ANIMAL_ID[animal_name]
                tile_origin_day[player, y, x] = _integer(tile.get("placed_day"), -1)
                tile_yield[player, y, x] = _integer(tile.get("yield_units"), 0)
                tile_neglect[player, y, x] = _integer(
                    tile.get("consecutive_unfed"), 0
                )
                tile_pending_care[player, y, x] = _integer(
                    tile.get("pending_care_bonus"), 0
                )
                if bool(tile.get("fed_today", False)):
                    tile_flags[player, y, x] |= FLAG_FED
                if bool(tile.get("cared_today", False)):
                    tile_flags[player, y, x] |= FLAG_CARED
                if bool(tile.get("fertilizer_available", False)):
                    tile_flags[player, y, x] |= FLAG_FERTILIZER_AVAILABLE

    shed = np.zeros((NUM_PLAYERS, NUM_SHED_ITEMS), dtype=np.int16)
    seeds = np.zeros((NUM_PLAYERS, NUM_CROPS), dtype=np.int16)
    unit_inventory = np.zeros(
        (NUM_PLAYERS, MAX_UNITS, NUM_SHED_ITEMS), dtype=np.int16
    )
    unit_inventory_order = np.full_like(unit_inventory, -1, dtype=np.int8)
    unit_inventory_next_order = np.zeros((NUM_PLAYERS, MAX_UNITS), dtype=np.int8)
    private = observation.get("private", {})
    private = private if isinstance(private, dict) else {}
    for name, item in _SHED_ID.items():
        shed[0, item] = _mapping_count(private.get("shed", {}), name)
    for name, crop in _CROP_ID.items():
        seeds[0, crop] = _mapping_count(private.get("seeds", {}), name)
    inventories = private.get("inventories", [])
    inventories = inventories if isinstance(inventories, list) else []
    for unit, inventory in enumerate(inventories[:MAX_UNITS]):
        next_order = 0
        for name, item in _SHED_ID.items():
            count = _mapping_count(inventory, name)
            unit_inventory[0, unit, item] = count
            if count > 0:
                unit_inventory_order[0, unit, item] = next_order
                next_order += 1
        unit_inventory_next_order[0, unit] = next_order

    market = observation.get("market", {})
    market = market if isinstance(market, dict) else {}
    market_inventory = np.asarray(
        [_mapping_count(market.get("inventory", {}), name) for name in PRODUCTS],
        dtype=np.int32,
    )
    market_price = np.asarray(
        [_mapping_count(market.get("prices", {}), name) for name in PRODUCTS],
        dtype=np.int32,
    )
    town = observation.get("town", {})
    unlocked_shops = town.get("unlocked_shops", []) if isinstance(town, dict) else []
    unlocked_shops = unlocked_shops if isinstance(unlocked_shops, list) else []
    town_shops = np.full((len(SHOP_NAMES),), -1, dtype=np.int8)
    for index, name in enumerate(unlocked_shops[: len(SHOP_NAMES)]):
        if str(name) not in _SHOP_ID:
            raise ValueError(f"unknown town shop {name!r}")
        town_shops[index] = _SHOP_ID[str(name)]

    values = dict(
        step=np.asarray(step, dtype=np.int16),
        episode_seed=np.asarray(0, dtype=np.int32),
        money=money,
        tile_kind=tile_kind,
        tile_crop=tile_crop,
        tile_animal=tile_animal,
        tile_origin_day=tile_origin_day,
        tile_yield=tile_yield,
        tile_neglect=tile_neglect,
        tile_max_lifespan=tile_max_lifespan,
        tile_fertilized_until=tile_fertilized_until,
        tile_pending_care=tile_pending_care,
        tile_flags=tile_flags,
        unit_pos=unit_pos,
        unit_active=unit_active,
        unit_inventory=unit_inventory,
        unit_inventory_order=unit_inventory_order,
        unit_inventory_next_order=unit_inventory_next_order,
        shed=shed,
        seeds=seeds,
        hires_today=hires_today,
        unlocked_count=unlocked_count,
        market_inventory=market_inventory,
        market_price=market_price,
        town_shops=town_shops,
        town_count=np.asarray(len(unlocked_shops), dtype=np.int8),
        reward=np.zeros((NUM_PLAYERS,), dtype=np.int32),
        done=np.asarray(step >= EPISODE_STEPS - 1, dtype=np.bool_),
        hand_cap_hits=np.zeros((NUM_PLAYERS,), dtype=np.int32),
        market_loop_cap_hits=np.asarray(0, dtype=np.int32),
        price_lut_oob=np.asarray(0, dtype=np.int32),
    )
    return State(*(jnp.asarray(values[name]) for name in State._fields))


def strategic_candidate_labels_v2(action: Any) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros((MAX_CANDIDATES_V1,), dtype=np.bool_)
    quantity = np.zeros((21,), dtype=np.int16)
    for card in extract_bundle(action)["cards"]:
        slot = _runtime_slot(card)
        if slot is not None:
            positive[slot] = True
            quantity[slot] = np.int16(
                int(quantity[slot]) + max(_integer(card.get("quantity"), 1), 0)
            )
            continue
        if card.get("scope") == "unit" and card.get("op") == "BUILD_COOP":
            positive[BUILD_COOP_SLOT] = True
        if card.get("scope") == "unit" and card.get("op") == "BUILD_PASTURE":
            positive[BUILD_PASTURE_SLOT] = True
    return positive, quantity


_UNIT_OP_V3 = {
    "PASS": UnitOp.PASS,
    "NORTH": UnitOp.NORTH,
    "SOUTH": UnitOp.SOUTH,
    "EAST": UnitOp.EAST,
    "WEST": UnitOp.WEST,
    "DROP": UnitOp.DROP,
    "PICKUP": UnitOp.PICKUP,
    "PLACE": UnitOp.PLACE,
    "PLANT": UnitOp.PLANT,
    "WATER": UnitOp.WATER,
    "HARVEST": UnitOp.HARVEST,
    "FERTILIZE": UnitOp.FERTILIZE,
    "DIG": UnitOp.DIG,
    "BUILD_COOP": UnitOp.BUILD_COOP,
    "BUILD_PASTURE": UnitOp.BUILD_PASTURE,
    "FEED": UnitOp.FEED,
    "COLLECT_FERTILIZER": UnitOp.COLLECT_FERTILIZER,
    "CARE": UnitOp.CARE,
}


def _unit_item_v3(op: str, item: Any) -> int:
    if item is None:
        return -1
    name = str(item)
    if op == "PLANT":
        return _CROP_ID.get(name, -1)
    return _SHED_ID.get(name, -1)


def strategic_ordered_labels_v3(action: Any) -> ReplayOrderedTurnV3:
    """Keep official unit-stage and market-order sequence as labels only."""

    normalized = normalize_action(action)
    units = [normalized["farmer"], *normalized["hands"]][:MAX_UNITS]
    unit_op = np.full((MAX_UNITS,), UnitOp.PASS, dtype=np.int8)
    unit_item = np.full((MAX_UNITS,), -1, dtype=np.int8)
    unit_quantity = np.ones((MAX_UNITS,), dtype=np.int16)
    for actor, raw in enumerate(units):
        op = str(raw[0]) if raw else "PASS"
        unit_op[actor] = np.int8(_UNIT_OP_V3.get(op, UnitOp.PASS))
        unit_item[actor] = np.int8(
            _unit_item_v3(op, raw[1] if len(raw) >= 2 else None)
        )
        unit_quantity[actor] = np.int16(
            max(_integer(raw[2], 1), 0) if len(raw) >= 3 else 1
        )

    slots = np.full((MAX_MARKET_ORDERS + 1,), -1, dtype=np.int16)
    quantities = np.zeros((MAX_MARKET_ORDERS + 1,), dtype=np.int16)
    valid = np.zeros((MAX_MARKET_ORDERS + 1,), dtype=np.bool_)
    orders = normalized["market"][:MAX_MARKET_ORDERS]
    count = 0
    for ordinal, order in enumerate(orders):
        op = str(order[0])
        item = order[1] if len(order) >= 2 else None
        quantity = 1 if op in {"HIRE", "BUY_LAND"} else max(_integer(order[2], 1), 0)
        slot = _runtime_slot(
            {
                "scope": "market",
                "op": op,
                "item": item,
                "quantity": quantity,
                "ordinal": ordinal,
            }
        )
        if slot is None:
            continue
        slots[count] = np.int16(slot)
        quantities[count] = np.int16(quantity)
        valid[count] = True
        count += 1
    slots[count] = np.int16(ORDERED_STOP_SLOT_V3)
    valid[count] = True
    return ReplayOrderedTurnV3(
        unit_op=unit_op,
        unit_item=unit_item,
        unit_quantity=unit_quantity,
        unit_count=len(units),
        market_candidate_slot=slots,
        market_quantity=quantities,
        market_valid=valid,
        market_count=count,
    )


def opponent_primary_task_type_v2(
    action: Any,
    observation: dict[str, Any],
    opponent_original_seat: int,
) -> int:
    farms = observation.get("farms", [])
    farm = farms[opponent_original_seat] if isinstance(farms, list) and len(farms) == 2 else {}
    positions = [farm.get("farmer", [4, 4]), *(farm.get("hands", []) or [])]
    tiles = farm.get("tiles", []) if isinstance(farm, dict) else []
    for card in extract_bundle(action)["cards"]:
        op = str(card.get("op", ""))
        direct = {
            "PLANT": TaskTypeV1.CROP_PRODUCTION,
            "WATER": TaskTypeV1.WATER_CROP,
            "DIG": TaskTypeV1.CLEAR_OR_REMOVE_TILE,
            "BUILD_COOP": TaskTypeV1.BUILD_ANIMAL_STRUCTURE,
            "BUILD_PASTURE": TaskTypeV1.BUILD_ANIMAL_STRUCTURE,
            "FEED": TaskTypeV1.ANIMAL_FEED,
            "CARE": TaskTypeV1.ANIMAL_CARE,
            "COLLECT_FERTILIZER": TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
            "FERTILIZE": TaskTypeV1.APPLY_FERTILIZER,
            "PICKUP": TaskTypeV1.SHED_PICKUP,
            "DROP": TaskTypeV1.SHED_DEPOSIT,
            "BUY_LAND": TaskTypeV1.BUY_LAND,
            "BUY_PRODUCT": TaskTypeV1.BUY_PRODUCT,
            "BUY_ANIMAL": TaskTypeV1.ANIMAL_PURCHASE,
            "BUY_SEED": TaskTypeV1.CROP_PRODUCTION,
            "HIRE": TaskTypeV1.HIRE_WORKER,
            "SELL": TaskTypeV1.SELL_INVENTORY,
        }
        if op in direct:
            return int(direct[op])
        if op not in {"HARVEST", "PLACE"}:
            continue
        actor = _integer(card.get("actor_index"), 0)
        if actor >= len(positions):
            continue
        x, y = (_integer(positions[actor][0]), _integer(positions[actor][1]))
        tile = (
            tiles[y][x]
            if isinstance(tiles, list)
            and 0 <= y < len(tiles)
            and isinstance(tiles[y], list)
            and 0 <= x < len(tiles[y])
            else None
        )
        if op == "HARVEST":
            if isinstance(tile, dict) and tile.get("kind") in {"COOP", "PASTURE"}:
                return int(TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
            return int(TaskTypeV1.CROP_PRODUCTION)
        if isinstance(tile, dict) and tile.get("kind") in {"COOP", "PASTURE"}:
            return int(TaskTypeV1.ANIMAL_PLACE)
        return int(TaskTypeV1.SHED_DEPOSIT)
    return -1


def load_replay_bc_episode_v2(
    replay_path: str | os.PathLike[str], player_names: Iterable[str]
) -> tuple[list[ReplayBCSampleV2], dict[str, Any]]:
    path = Path(replay_path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    names = [str(value) for value in document.get("info", {}).get("TeamNames", [])]
    wanted = {_name_key(value) for value in player_names}
    seats = [index for index, name in enumerate(names) if _name_key(name) in wanted]
    if len(seats) != 1:
        raise ValueError(f"expected exactly one expert seat in {path.name}: {names}")
    seat = seats[0]
    opponent = 1 - seat
    episode_id = _integer(document.get("info", {}).get("EpisodeId", path.stem), 0)
    steps = document.get("steps", [])
    samples: list[ReplayBCSampleV2] = []
    for action_index in range(1, len(steps)):
        observation = steps[action_index - 1][seat].get("observation", {})
        if not isinstance(observation, dict):
            continue
        action = steps[action_index][seat].get("action")
        opponent_action = steps[action_index][opponent].get("action")
        state = observation_to_canonical_state_v2(observation, seat)
        positive, quantity = strategic_candidate_labels_v2(action)
        ordered = strategic_ordered_labels_v3(action)
        samples.append(
            ReplayBCSampleV2(
                episode_id=episode_id,
                source_step=int(state.step),
                original_seat=seat,
                state=state,
                positive_candidate_mask=positive,
                market_quantity=quantity,
                ordered_turn=ordered,
                opponent_task_type=opponent_primary_task_type_v2(
                    opponent_action, observation, opponent
                ),
            )
        )
    source = {
        "episode_id": episode_id,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expert_name": names[seat],
        "opponent_name": names[opponent],
        "original_seat": seat,
        "samples": len(samples),
        "strategic_samples": sum(
            bool(np.any(sample.positive_candidate_mask)) for sample in samples
        ),
        "strategic_cards": sum(
            int(np.sum(sample.positive_candidate_mask)) for sample in samples
        ),
    }
    return samples, source


def _stack_states(states: list[State]) -> State:
    return jax.tree.map(lambda *values: jnp.stack(values), *states)


def _controller_batch(batch_size: int):
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )


def broad_replay_bc_candidate_program_v2() -> ReplayTaskCardProgramV1:
    """Action-agnostic market enumeration for BC and its deployed policy.

    The program is derived only from the fixed action schema.  It exposes one
    unit of every buy/hire family on every non-terminal step; sell quantities
    remain state-derived.  Every expert prior is zero.
    """

    quantity = jnp.zeros((EPISODE_STEPS, 21), dtype=jnp.int16)
    quantity = quantity.at[:, :12].set(jnp.int16(1))
    return ReplayTaskCardProgramV1(
        enabled=jnp.ones((EPISODE_STEPS,), dtype=jnp.bool_),
        support=jnp.zeros((EPISODE_STEPS,), dtype=jnp.int16),
        consensus=jnp.zeros((EPISODE_STEPS,), dtype=jnp.float32),
        market_quantity=quantity,
        market_priority=jnp.zeros((EPISODE_STEPS, 21), dtype=jnp.float32),
        build_animal_id=jnp.full((EPISODE_STEPS,), -1, dtype=jnp.int8),
        build_priority=jnp.zeros((EPISODE_STEPS,), dtype=jnp.float32),
    )


def _episode_histories(states: State) -> OpponentHistoryV2:
    snapshot = jnp.stack(
        (build_public_snapshot_v2(states, 0), build_public_snapshot_v2(states, 1)),
        axis=1,
    )
    host = np.asarray(jax.device_get(snapshot), dtype=np.float32)
    recent = np.zeros_like(host)
    recent[1:] = host[1:] - host[:-1]
    ema = np.zeros_like(host)
    for index in range(1, len(host)):
        ema[index] = 0.75 * ema[index - 1] + 0.25 * recent[index]
    return OpponentHistoryV2(
        previous_snapshot=snapshot,
        recent_delta=jnp.asarray(recent),
        ema_delta=jnp.asarray(ema),
    )


def build_replay_bc_feature_batch_v2(
    states: State,
    history: OpponentHistoryV2,
    tables: StaticTables,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    *,
    allow_nonpositive_econ: bool = True,
) -> ReplayBCFeatureBatchV2:
    """Build BC inputs without accepting any expert action or label argument.

    The separation is deliberate: replay quantities, chosen cards and the
    opponent's same-turn action cannot influence actor inputs by construction.
    """

    batch_size = states.step.shape[0]
    controller = _controller_batch(batch_size)
    program = (
        broad_replay_bc_candidate_program_v2()
        if task_card_program is None
        else task_card_program
    )
    candidates = build_full_core_candidates_v1(
        states,
        controller,
        tables,
        0,
        program,
    )
    # A replay profile is a teacher label source, never an actor-visible prior.
    candidates = candidates._replace(
        replay_priority=jnp.zeros_like(candidates.replay_priority)
    )
    feasibility = evaluate_full_core_feasibility_v1(
        states, candidates, tables, 0
    )
    econ = build_full_econ_features_v1(
        states, candidates, feasibility, tables, 0
    )
    score = full_econ_score_v1(candidates, feasibility, econ)
    present = candidates.present
    legal = present & feasibility.legal_now
    bankable = legal & econ.bankable_before_terminal
    positive_or_required = (
        (score > 0.0)
        | candidates.mandatory
        | (candidates.task_type == TaskTypeV1.TERMINAL_LIQUIDATION)
        | (candidates.task_type == TaskTypeV1.SAFE_RECOVERY)
    )
    eligible = bankable & (positive_or_required | allow_nonpositive_econ)
    return ReplayBCFeatureBatchV2(
        # Offline data keeps every ablation branch.  Deployment defaults may
        # disable history, but must never silently erase the experimental arm.
        global_features=build_global_features_v2(
            states,
            0,
            history,
            include_opponent=True,
            include_history=True,
        ),
        candidate_features=build_candidate_features_v2(
            states, candidates, feasibility, econ, 0, tables
        ),
        candidate_task_type=candidates.task_type,
        candidate_present=present,
        candidate_legal=legal,
        candidate_bankable=bankable,
        candidate_eligible=eligible,
    )


def compile_replay_bc_dataset_v2(
    replay_files: Iterable[str | os.PathLike[str]],
    player_names: Iterable[str],
    tables: StaticTables,
    *,
    batch_size: int = 256,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    episodes = []
    sources = []
    for replay in replay_files:
        samples, source = load_replay_bc_episode_v2(replay, player_names)
        episodes.append(samples)
        sources.append(source)

    selected: list[ReplayBCSampleV2] = []
    histories: list[OpponentHistoryV2] = []
    for samples in episodes:
        if not samples:
            continue
        states = _stack_states([sample.state for sample in samples])
        history = _episode_histories(states)
        for index, sample in enumerate(samples):
            if not np.any(sample.positive_candidate_mask):
                continue
            selected.append(sample)
            histories.append(
                OpponentHistoryV2(
                    previous_snapshot=history.previous_snapshot[index],
                    recent_delta=history.recent_delta[index],
                    ema_delta=history.ema_delta[index],
                )
            )
    if not selected:
        raise ValueError("no replay samples contain supported strategic task cards")

    states = _stack_states([sample.state for sample in selected])
    history = jax.tree.map(lambda *values: jnp.stack(values), *histories)
    positive = np.stack(
        [sample.positive_candidate_mask for sample in selected]
    ).astype(np.bool_)
    market_quantity = np.stack(
        [sample.market_quantity for sample in selected]
    ).astype(np.int16)
    outputs: dict[str, list[np.ndarray]] = {
        "global_features": [],
        "candidate_features": [],
        "candidate_task_type": [],
        "positive_candidate_mask": [],
        "supervised_candidate_mask": [],
        "valid_sample": [],
    }
    strategic_mask = jnp.zeros((MAX_CANDIDATES_V1,), dtype=jnp.bool_).at[
        jnp.asarray(STRATEGIC_SLOTS_V2)
    ].set(True)
    eligibility_stages: dict[str, list[np.ndarray]] = {
        "present": [],
        "legal": [],
        "bankable": [],
        "eligible": [],
    }
    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        one_state = jax.tree.map(lambda value: value[start:stop], states)
        one_history = jax.tree.map(lambda value: value[start:stop], history)
        one_positive = jnp.asarray(positive[start:stop])
        features = build_replay_bc_feature_batch_v2(
            one_state,
            one_history,
            tables,
            task_card_program,
            allow_nonpositive_econ=allow_nonpositive_econ,
        )
        supervised = strategic_mask[None, :] & features.candidate_eligible
        represented_positive = one_positive & features.candidate_eligible
        valid = jnp.any(represented_positive, axis=-1)
        outputs["global_features"].append(
            np.asarray(
                jax.device_get(features.global_features),
                dtype=np.float32,
            )
        )
        outputs["candidate_features"].append(
            np.asarray(
                jax.device_get(features.candidate_features),
                dtype=np.float16,
            )
        )
        outputs["candidate_task_type"].append(
            np.asarray(jax.device_get(features.candidate_task_type), dtype=np.int8)
        )
        outputs["positive_candidate_mask"].append(
            np.asarray(jax.device_get(represented_positive), dtype=np.bool_)
        )
        outputs["supervised_candidate_mask"].append(
            np.asarray(jax.device_get(supervised), dtype=np.bool_)
        )
        outputs["valid_sample"].append(
            np.asarray(jax.device_get(valid), dtype=np.bool_)
        )
        for name, mask in (
            ("present", features.candidate_present),
            ("legal", features.candidate_legal),
            ("bankable", features.candidate_bankable),
            ("eligible", features.candidate_eligible),
        ):
            eligibility_stages[name].append(
                np.asarray(jax.device_get(mask), dtype=np.bool_)
            )

    arrays = {name: np.concatenate(values, axis=0) for name, values in outputs.items()}
    eligibility = {
        name: np.concatenate(values, axis=0)
        for name, values in eligibility_stages.items()
    }
    valid = arrays.pop("valid_sample")
    original_positive = positive
    represented_positive = arrays["positive_candidate_mask"]
    original_positive_count = np.sum(original_positive, axis=-1)
    represented_positive_count = np.sum(represented_positive, axis=-1)
    representation_fraction = represented_positive_count / np.maximum(
        original_positive_count, 1
    )
    fully_represented = represented_positive_count == original_positive_count
    rejected_illegal = int(np.sum(~valid))
    rejection_reasons = []
    rejection_counts = {
        "candidate_not_generated": 0,
        "full_core_illegal": 0,
        "full_econ_unbankable": 0,
        "full_econ_nonpositive": 0,
    }
    unsupported_card_counts = dict(rejection_counts)
    for index, sample in enumerate(selected):
        unsupported = original_positive[index] & (~eligibility["eligible"][index])
        reasons_for_sample: dict[str, list[int]] = {}
        for slot in np.flatnonzero(unsupported):
            if not eligibility["present"][index, slot]:
                reason = "candidate_not_generated"
            elif not eligibility["legal"][index, slot]:
                reason = "full_core_illegal"
            elif not eligibility["bankable"][index, slot]:
                reason = "full_econ_unbankable"
            else:
                reason = "full_econ_nonpositive"
            unsupported_card_counts[reason] += 1
            reasons_for_sample.setdefault(reason, []).append(int(slot))
        if not valid[index]:
            if not reasons_for_sample:
                raise RuntimeError("zero-label replay sample has no rejection reason")
            primary_reason = next(iter(reasons_for_sample))
            rejection_counts[primary_reason] += 1
        if reasons_for_sample and len(rejection_reasons) < 50:
            rejection_reasons.append(
                {
                    "episode_id": sample.episode_id,
                    "source_step": sample.source_step,
                    "accepted_sample": bool(valid[index]),
                    "represented_slots": np.flatnonzero(
                        represented_positive[index]
                    ).astype(int).tolist(),
                    "unsupported_slots_by_reason": reasons_for_sample,
                }
            )
    arrays = {name: value[valid] for name, value in arrays.items()}
    selected_valid = [sample for sample, keep in zip(selected, valid, strict=True) if keep]
    arrays.update(
        {
            "opponent_task_type": np.asarray(
                [sample.opponent_task_type for sample in selected_valid], dtype=np.int8
            ),
            "sample_weight": representation_fraction[valid].astype(np.float32),
            "episode_id": np.asarray(
                [sample.episode_id for sample in selected_valid], dtype=np.int64
            ),
            "source_step": np.asarray(
                [sample.source_step for sample in selected_valid], dtype=np.int16
            ),
            "original_seat": np.asarray(
                [sample.original_seat for sample in selected_valid], dtype=np.int8
            ),
            "market_quantity": np.stack(
                [sample.market_quantity for sample in selected_valid]
            ).astype(np.int16),
            "ordered_unit_op": np.stack(
                [sample.ordered_turn.unit_op for sample in selected_valid]
            ).astype(np.int8),
            "ordered_unit_item": np.stack(
                [sample.ordered_turn.unit_item for sample in selected_valid]
            ).astype(np.int8),
            "ordered_unit_quantity": np.stack(
                [sample.ordered_turn.unit_quantity for sample in selected_valid]
            ).astype(np.int16),
            "ordered_unit_count": np.asarray(
                [sample.ordered_turn.unit_count for sample in selected_valid],
                dtype=np.int8,
            ),
            "ordered_market_candidate_slot": np.stack(
                [sample.ordered_turn.market_candidate_slot for sample in selected_valid]
            ).astype(np.int16),
            "ordered_market_quantity": np.stack(
                [sample.ordered_turn.market_quantity for sample in selected_valid]
            ).astype(np.int16),
            "ordered_market_valid": np.stack(
                [sample.ordered_turn.market_valid for sample in selected_valid]
            ).astype(np.bool_),
            "ordered_market_count": np.asarray(
                [sample.ordered_turn.market_count for sample in selected_valid],
                dtype=np.int8,
            ),
        }
    )
    episode_ids = sorted({sample.episode_id for sample in selected_valid})
    validation_ids = set(episode_ids[::5])
    arrays["split"] = np.asarray(
        [1 if episode_id in validation_ids else 0 for episode_id in arrays["episode_id"]],
        dtype=np.int8,
    )
    manifest = {
        "schema_version": REPLAY_BC_SCHEMA_V2,
        "alignment": "observation[t-1] -> expert_action[t]",
        "canonical_player": 0,
        "opponent_private_policy": "zero_and_forbidden",
        "feature_dimensions": {
            "global": int(arrays["global_features"].shape[-1]),
            "candidate": int(arrays["candidate_features"].shape[-1]),
            "candidates": MAX_CANDIDATES_V1,
        },
        "coverage": {
            "episodes": len(episodes),
            "aligned_samples": sum(len(samples) for samples in episodes),
            "strategic_samples_before_legality": len(selected),
            "zero_representable_positive_samples_rejected": rejected_illegal,
            "fully_represented_samples": int(np.sum(fully_represented)),
            "partially_represented_samples": int(
                np.sum(valid & (~fully_represented))
            ),
            "rejected_sample_reason_counts": rejection_counts,
            "unsupported_positive_card_reason_counts": unsupported_card_counts,
            "accepted_samples": len(selected_valid),
            "expert_positive_cards_before_runtime_filter": int(
                np.sum(original_positive)
            ),
            "accepted_positive_cards": int(
                np.sum(arrays["positive_candidate_mask"])
            ),
            "train_samples": int(np.sum(arrays["split"] == 0)),
            "validation_samples": int(np.sum(arrays["split"] == 1)),
            "opponent_aux_labels": int(np.sum(arrays["opponent_task_type"] >= 0)),
        },
        "supervision": {
            "ordered_turn_schema": ORDERED_TURN_SCHEMA_V3,
            "ordered_stop_slot": ORDERED_STOP_SLOT_V3,
            "ordered_market_sequence_is_label_only": True,
            "strategic_slots": list(STRATEGIC_SLOTS_V2),
            "low_level_movement_and_maintenance": "retained in replay but excluded from strategic STOP labels",
            "replay_priority_feature": "forced_zero_to_prevent_label_leakage",
            "candidate_program": (
                "action_agnostic_broad_v2"
                if task_card_program is None
                else "explicit_frozen_program"
            ),
            "expert_quantity_as_actor_input": False,
            "expert_positive_card_as_candidate_presence_input": False,
            "allow_nonpositive_econ": allow_nonpositive_econ,
            "runtime_eligibility": (
                "present & full_core_legal & full_econ_bankable"
                + ("" if allow_nonpositive_econ else " & full_econ_policy_gate")
            ),
            "split_unit": "episode",
            "validation_episode_ids": sorted(validation_ids),
            "rejection_examples": rejection_reasons,
        },
        "sources": sources,
    }
    return arrays, manifest


def write_replay_bc_dataset_v2(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    dataset_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
) -> tuple[Path, Path]:
    destination = Path(dataset_path).resolve()
    manifest_destination = Path(manifest_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=destination.parent, delete=False
    ) as stream:
        temporary_dataset = Path(stream.name)
    np.savez_compressed(temporary_dataset, **arrays)
    os.replace(temporary_dataset, destination)
    dataset_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = dict(manifest)
    manifest["artifact"] = {
        "path": str(destination),
        "sha256": dataset_sha,
        "bytes": destination.stat().st_size,
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(arrays.items())
        },
    }
    content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=manifest_destination.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary_manifest = Path(stream.name)
    os.replace(temporary_manifest, manifest_destination)
    return destination, manifest_destination

"""Pre-freeze M3.5 alignment against a current gold Replay opening.

This is an audit, not a claim that a six-phase approximation reproduces the
expert policy.  It separates three questions:

1. What does the frozen conservative M3.5 default do on the Replay seed?
2. Can M3.5 execute the exact strategic commitments in the gold opening?
3. How close can the existing six-phase schema get when given an approximate
   calendar extracted from the same Replay?

The opponent is an open-loop action trace copied from the Replay.  This keeps a
realistic market disturbance but is not a causal reconstruction after our
candidate diverges from the recorded game.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for source_dir in (
    PROJECT_DIR / "src",
    REPO_ROOT / "gpu_sim" / "src",
    REPO_ROOT / "experiments" / "strategic_v5" / "src",
):
    sys.path.insert(0, str(source_dir))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kaggriculture_jax.constants import (  # noqa: E402
    ANIMALS,
    CROPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_DAYS,
    PRODUCTS,
    SHOP_NAMES,
    TileKind,
)
from kaggriculture_jax.policy import combine_player_actions  # noqa: E402
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from project_route_search_v2.lifecycle import (  # noqa: E402
    snapshot_project_controller_v2,
)
from project_route_search_v2.m35_controller import (  # noqa: E402
    m35_player_action_dict_v2,
    m35_policy_step_v2,
    update_m35_controller_from_effects_v2,
)
from project_route_search_v2.m35_genome import (  # noqa: E402
    default_m35_farm_genome_v2,
    validate_m35_farm_genome_v2,
)
from project_route_search_v2.m35_rollout import (  # noqa: E402
    initialize_m35_rollout_carry_v2,
)


UNIT_OPS = {
    name: index
    for index, name in enumerate(
        (
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
    )
}
MARKET_OPS = {
    name: index
    for index, name in enumerate(
        ("NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL")
    )
}
SHED_ITEMS = PRODUCTS + ANIMALS
PRODUCT_IDS = {name: index for index, name in enumerate(PRODUCTS)}
CROP_IDS = {name: index for index, name in enumerate(CROPS)}
ANIMAL_IDS = {name: len(PRODUCTS) + index for index, name in enumerate(ANIMALS)}
SHED_ITEM_IDS = {name: index for index, name in enumerate(SHED_ITEMS)}
EVENT_DRAWS = 100
WEED_CHANCE = 0.005


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _unit(raw) -> tuple[int, int, int]:
    if not isinstance(raw, list) or not raw:
        return UNIT_OPS["PASS"], -1, 1
    op = UNIT_OPS.get(str(raw[0]), UNIT_OPS["PASS"])
    item = -1
    amount = 1
    if op == UNIT_OPS["PLANT"] and len(raw) >= 2:
        item = CROP_IDS.get(str(raw[1]), -1)
    elif op in (UNIT_OPS["PICKUP"], UNIT_OPS["PLACE"]) and len(raw) >= 2:
        item = SHED_ITEM_IDS.get(str(raw[1]), -1)
        if len(raw) >= 3:
            amount = int(raw[2])
    return op, item, amount


def _market(raw) -> tuple[int, int, int]:
    if not isinstance(raw, list) or not raw:
        return MARKET_OPS["NONE"], -1, 0
    op = MARKET_OPS.get(str(raw[0]), MARKET_OPS["NONE"])
    if op in (MARKET_OPS["HIRE"], MARKET_OPS["BUY_LAND"]):
        return op, -1, 0
    if len(raw) < 3:
        return MARKET_OPS["NONE"], -1, 0
    if op == MARKET_OPS["BUY_SEED"]:
        item = CROP_IDS.get(str(raw[1]), -1)
    elif op == MARKET_OPS["BUY_ANIMAL"]:
        item = ANIMAL_IDS.get(str(raw[1]), -1)
    elif op in (MARKET_OPS["BUY_PRODUCT"], MARKET_OPS["SELL"]):
        item = PRODUCT_IDS.get(str(raw[1]), -1)
    else:
        return MARKET_OPS["NONE"], -1, 0
    return op, item, int(raw[2])


def _encode_actions(stream: list[dict], batch_size: int) -> Action:
    if len(stream) != 719:
        raise ValueError(f"expected 719 Replay actions, got {len(stream)}")
    unit_op = np.full((719, MAX_UNITS), UNIT_OPS["PASS"], dtype=np.int8)
    unit_item = np.full((719, MAX_UNITS), -1, dtype=np.int8)
    unit_amount = np.ones((719, MAX_UNITS), dtype=np.int32)
    unit_count = np.ones((719,), dtype=np.int8)
    market_op = np.full((719, MAX_MARKET_ORDERS), MARKET_OPS["NONE"], dtype=np.int8)
    market_item = np.full((719, MAX_MARKET_ORDERS), -1, dtype=np.int8)
    market_amount = np.zeros((719, MAX_MARKET_ORDERS), dtype=np.int32)
    market_count = np.zeros((719,), dtype=np.int8)
    for step, action in enumerate(stream):
        raw = action if isinstance(action, dict) else {}
        units = [raw.get("farmer", ["PASS"]), *(raw.get("hands", []) or [])]
        unit_count[step] = min(len(units), MAX_UNITS)
        for index, value in enumerate(units[:MAX_UNITS]):
            unit_op[step, index], unit_item[step, index], unit_amount[step, index] = _unit(value)
        orders = raw.get("market", []) or []
        market_count[step] = min(len(orders), MAX_MARKET_ORDERS)
        for index, value in enumerate(orders[:MAX_MARKET_ORDERS]):
            market_op[step, index], market_item[step, index], market_amount[step, index] = _market(value)

    def repeated(value):
        return jnp.asarray(np.repeat(value[:, None], batch_size, axis=1))

    return Action(
        repeated(unit_op),
        repeated(unit_item),
        repeated(unit_amount),
        repeated(unit_count),
        repeated(market_op),
        repeated(market_item),
        repeated(market_amount),
        repeated(market_count),
    )


def _build_events(seed: int, batch_size: int) -> Events:
    weed = np.zeros((NUM_DAYS, EVENT_DRAWS), dtype=np.bool_)
    choice = np.zeros((NUM_DAYS, EVENT_DRAWS + 1), dtype=np.int8)
    for day in range(NUM_DAYS):
        rng = random.Random((seed * 1_000_003) ^ day)
        for consumed in range(EVENT_DRAWS + 1):
            clone = random.Random()
            clone.setstate(rng.getstate())
            choice[day, consumed] = SHOP_NAMES.index(clone.choice(SHOP_NAMES))
            if consumed < EVENT_DRAWS:
                weed[day, consumed] = rng.random() < WEED_CHANCE
    return Events(
        jnp.asarray(np.repeat(weed[None], batch_size, axis=0)),
        jnp.asarray(np.repeat(choice[None], batch_size, axis=0)),
    )


def _set_shared_calendar(genome, phase_count, phase_start, land, hands):
    crop = genome.crop._replace(
        phase_count=jnp.asarray((phase_count,), dtype=jnp.int8),
        phase_start_step=jnp.asarray((phase_start,), dtype=jnp.int16),
        land_target=jnp.asarray((land,), dtype=jnp.int8),
        land_start_step=jnp.asarray((phase_start,), dtype=jnp.int16),
        hand_target=jnp.asarray((hands,), dtype=jnp.int8),
    )
    animal = genome.animal._replace(
        phase_count=jnp.asarray((phase_count,), dtype=jnp.int8),
        phase_start_step=jnp.asarray((phase_start,), dtype=jnp.int16),
        land_target=jnp.asarray((land,), dtype=jnp.int8),
        hand_target=jnp.asarray((hands,), dtype=jnp.int8),
    )
    return genome._replace(crop=crop, animal=animal)


def _gold_opening_only(candidate_id: int):
    genome = default_m35_farm_genome_v2(1)
    phase_start = (0, 720, 720, 720, 720, 720)
    genome = _set_shared_calendar(genome, 1, phase_start, (1, 1, 1, 1, 1, 1), (5, 5, 5, 5, 5, 5))
    crop_target = jnp.broadcast_to(
        jnp.asarray((7, 0, 0, 0, 12), dtype=jnp.int16), (1, 6, NUM_CROPS)
    )
    animal_target = jnp.broadcast_to(
        jnp.asarray((0, 2, 2), dtype=jnp.int16), (1, 6, NUM_ANIMALS)
    )
    crop = genome.crop._replace(
        candidate_id=jnp.asarray((candidate_id,), dtype=jnp.int32),
        crop_target=crop_target,
        seed_buy_batch=jnp.asarray(((7, 8, 8, 8, 12),), dtype=jnp.int16),
        crop_cash_cap=jnp.full_like(genome.crop.crop_cash_cap, 100_000),
        cash_floor=jnp.zeros((1,), dtype=jnp.int32),
        liquidation_start_step=jnp.asarray((27 * 24,), dtype=jnp.int16),
    )
    animal = genome.animal._replace(
        candidate_id=jnp.asarray((candidate_id,), dtype=jnp.int32),
        animal_target=animal_target,
        animal_project_cash_cap=jnp.full_like(genome.animal.animal_project_cash_cap, 100_000),
        care_policy=jnp.asarray(((2, 2, 2),), dtype=jnp.int8),
        first_cycle_care_bonus_target=jnp.asarray(((3, 5, 5),), dtype=jnp.int8),
        steady_cycle_care_bonus_target=jnp.asarray(((1, 2, 3),), dtype=jnp.int8),
        cash_floor=jnp.zeros((1,), dtype=jnp.int32),
        liquidation_start_step=jnp.asarray((27 * 24,), dtype=jnp.int16),
    )
    return genome._replace(
        candidate_id=jnp.asarray((candidate_id,), dtype=jnp.int32),
        crop=crop,
        animal=animal,
        crop_unit_share=jnp.asarray((0.45,), dtype=jnp.float32),
    )


def _gold_six_phase(candidate_id: int):
    genome = default_m35_farm_genome_v2(1)
    phase_start = (0, 6 * 24, 11 * 24, 12 * 24, 20 * 24, 27 * 24)
    # The real daily targets go both up and down.  M3.5 currently forbids a
    # decreasing hand_target, so this uses the closest legal upper envelope.
    genome = _set_shared_calendar(
        genome,
        6,
        phase_start,
        (1, 2, 3, 4, 4, 4),
        (5, 5, 12, 12, 12, 12),
    )
    crop_target = jnp.asarray(
        ((
            (7, 0, 0, 0, 12),
            (3, 0, 0, 4, 12),
            (10, 0, 0, 19, 0),
            (15, 0, 0, 42, 0),
            (28, 0, 0, 42, 0),
            (29, 3, 0, 17, 0),
        ),),
        dtype=jnp.int16,
    )
    animal_target = jnp.asarray(
        ((
            (0, 2, 2),
            (0, 4, 2),
            (0, 6, 6),
            (0, 6, 8),
            (0, 6, 12),
            (0, 6, 12),
        ),),
        dtype=jnp.int16,
    )
    crop = genome.crop._replace(
        candidate_id=jnp.asarray((candidate_id,), dtype=jnp.int32),
        crop_target=crop_target,
        seed_buy_batch=jnp.asarray(((8, 8, 8, 24, 12),), dtype=jnp.int16),
        plant_wave_size=jnp.asarray(((7, 3, 4, 8, 12),), dtype=jnp.int8),
        crop_last_plant_step=jnp.asarray(((27 * 24, 25 * 24, 19 * 24, 20 * 24, 2 * 24),), dtype=jnp.int16),
        crop_cash_cap=jnp.full_like(genome.crop.crop_cash_cap, 100_000),
        cash_floor=jnp.zeros((1,), dtype=jnp.int32),
        liquidation_start_step=jnp.asarray((27 * 24,), dtype=jnp.int16),
        sell_interval=jnp.full_like(genome.crop.sell_interval, 24),
        sell_fraction=jnp.ones_like(genome.crop.sell_fraction),
    )
    animal = genome.animal._replace(
        candidate_id=jnp.asarray((candidate_id,), dtype=jnp.int32),
        animal_target=animal_target,
        animal_investment_stop_step=jnp.asarray(((0, 8 * 24, 20 * 24),), dtype=jnp.int16),
        animal_place_wave_size=jnp.asarray(((2, 2, 2),), dtype=jnp.int8),
        animal_project_cash_cap=jnp.full_like(genome.animal.animal_project_cash_cap, 100_000),
        care_policy=jnp.asarray(((2, 2, 2),), dtype=jnp.int8),
        first_cycle_care_bonus_target=jnp.asarray(((3, 5, 5),), dtype=jnp.int8),
        steady_cycle_care_bonus_target=jnp.asarray(((1, 2, 3),), dtype=jnp.int8),
        animal_harvest_trigger_units=jnp.asarray(((1, 3, 3),), dtype=jnp.int8),
        cash_floor=jnp.zeros((1,), dtype=jnp.int32),
        liquidation_start_step=jnp.asarray((27 * 24,), dtype=jnp.int16),
        sell_interval=jnp.full_like(genome.animal.sell_interval, 24),
        sell_fraction=jnp.ones_like(genome.animal.sell_fraction),
    )
    return genome._replace(
        candidate_id=jnp.asarray((candidate_id,), dtype=jnp.int32),
        crop=crop,
        animal=animal,
        crop_unit_share=jnp.asarray((0.35,), dtype=jnp.float32),
    )


def _with_candidate_id(genome, candidate_id: int):
    ids = jnp.asarray((candidate_id,), dtype=jnp.int32)
    return genome._replace(
        candidate_id=ids,
        crop=genome.crop._replace(candidate_id=ids),
        animal=genome.animal._replace(candidate_id=ids),
    )


def _candidate_panel():
    candidates = [
        _with_candidate_id(default_m35_farm_genome_v2(1), 0),
        _gold_opening_only(1),
        _gold_six_phase(2),
    ]
    return jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *candidates)


def _state_metrics(state, player: int):
    tile_kind = state.tile_kind[:, player]
    crop = state.tile_crop[:, player]
    animal = state.tile_animal[:, player]
    return (
        state.money[:, player],
        state.unlocked_count[:, player],
        jnp.sum(state.unit_active[:, player], axis=-1, dtype=jnp.int32),
        jnp.stack(
            [jnp.sum((tile_kind == TileKind.PLANT) & (crop == item), axis=(1, 2), dtype=jnp.int32) for item in range(NUM_CROPS)],
            axis=-1,
        ),
        jnp.stack(
            [jnp.sum(animal == item, axis=(1, 2), dtype=jnp.int32) for item in range(NUM_ANIMALS)],
            axis=-1,
        ),
        state.shed[:, player],
        state.seeds[:, player],
    )


def _make_rollout(player: int):
    opponent = 1 - player

    def rollout(initial_carry, opponent_actions: Action, events: Events, tables, genome):
        zeros = jnp.zeros((initial_carry.environment_state.step.shape[0],), dtype=jnp.int32)

        def body(carry, opponent_action):
            state, controller, compiler_errors, effect_errors = carry
            player_action, controller = m35_policy_step_v2(state, controller, genome, player, tables)
            player_dict = m35_player_action_dict_v2(player_action)
            opponent_dict = opponent_action._asdict()
            joint = (
                combine_player_actions(player_dict, opponent_dict)
                if player == 0
                else combine_player_actions(opponent_dict, player_dict)
            )
            next_state = batched_step_sync(state, joint, events, tables)
            controller, effects = update_m35_controller_from_effects_v2(
                state, next_state, controller, player_action, genome, player
            )
            controller = snapshot_project_controller_v2(next_state, controller, player)
            compiler_errors = compiler_errors + (
                player_action.diagnostics.invalid_raw_action_count
                + player_action.diagnostics.unit_compiler_overlap_count
                + player_action.diagnostics.market_compiler_overlap_count
            )
            effect_errors = effect_errors + (
                effects.effect_mismatch_count
                + effects.owner_inactive_count
                + effects.deadline_missed_count
                + effects.resource_unavailable_count
            )
            return (
                next_state,
                controller,
                compiler_errors,
                effect_errors,
            ), (Action(*(joint_field[:, player] for joint_field in joint)), _state_metrics(next_state, player))

        return jax.lax.scan(
            body,
            (initial_carry.environment_state, initial_carry.controller, zeros, zeros),
            opponent_actions,
        )

    return rollout


def _make_replay_control_rollout(player: int):
    """Replay both recorded seats verbatim as a simulator/event control."""

    def rollout(initial_state, player0_actions: Action, player1_actions: Action, events: Events, tables):
        def body(state, actions):
            player0, player1 = actions
            joint = combine_player_actions(player0._asdict(), player1._asdict())
            next_state = batched_step_sync(state, joint, events, tables)
            return next_state, _state_metrics(next_state, player)

        return jax.lax.scan(body, initial_state, (player0_actions, player1_actions))

    return rollout


def _replay_metrics(data: dict, player: int, state_step: int) -> dict:
    observation = data["steps"][state_step][player]["observation"]
    farm = observation["farms"][player]
    private = observation["private"]
    crop = Counter()
    animal = Counter()
    pasture = 0
    coop = 0
    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crop[str(tile.get("crop"))] += 1
            elif kind == "PASTURE":
                pasture += 1
                if tile.get("animal"):
                    animal[str(tile["animal"])] += 1
            elif kind == "COOP":
                coop += 1
                if tile.get("animal"):
                    animal[str(tile["animal"])] += 1
    return {
        "state_step": state_step,
        "money": int(farm["money"]),
        "land": len(farm.get("unlocked_quadrants", [])),
        "units": 1 + len(farm.get("hands", [])),
        "crop_tiles": [int(crop[name]) for name in CROPS],
        "animals": [int(animal[name]) for name in ANIMALS],
        "pasture_tiles": pasture,
        "coop_tiles": coop,
        "shed": [int(private["shed"].get(name, 0)) for name in SHED_ITEMS],
        "seeds": [int(private["seeds"].get(name, 0)) for name in CROPS],
    }


def _decode_action(action: Action, lane: int, step: int = 0) -> dict:
    reverse_unit = {value: key for key, value in UNIT_OPS.items()}
    reverse_market = {value: key for key, value in MARKET_OPS.items()}
    units = []
    for index in range(int(action.unit_count[step, lane])):
        op = int(action.unit_op[step, lane, index])
        units.append([reverse_unit.get(op, str(op)), int(action.unit_item[step, lane, index]), int(action.unit_amount[step, lane, index])])
    market = []
    for index in range(int(action.market_count[step, lane])):
        op = int(action.market_op[step, lane, index])
        market.append([reverse_market.get(op, str(op)), int(action.market_item[step, lane, index]), int(action.market_amount[step, lane, index])])
    return {"units": units, "market": market}


def _replay_opening_encoded(raw: dict) -> dict:
    units = [_unit(raw.get("farmer", ["PASS"])), *[_unit(value) for value in raw.get("hands", []) or []]]
    market = [_market(value) for value in raw.get("market", []) or []]
    reverse_unit = {value: key for key, value in UNIT_OPS.items()}
    reverse_market = {value: key for key, value in MARKET_OPS.items()}
    return {
        "units": [[reverse_unit[op], item, amount] for op, item, amount in units],
        "market": [[reverse_market[op], item, amount] for op, item, amount in market],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay",
        type=Path,
        default=REPO_ROOT / "replay" / "gold_top3_recent12_2026-08-18" / "kawashigi_current_55540317" / "episodes" / "94051618.json",
    )
    parser.add_argument("--gold-seat", type=int, default=1)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "receipts" / "m35_gold_replay_alignment_v1.json",
    )
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    data = json.loads(args.replay.read_text(encoding="utf-8"))
    if data.get("module_version") != "1.32.7":
        raise ValueError(f"expected official 1.32.7 Replay, got {data.get('module_version')}")
    seed = int(data["info"]["seed"])
    player = int(args.gold_seat)
    opponent = 1 - player
    opponent_stream = [row[opponent].get("action") or {} for row in data["steps"][1:720]]
    player0_stream = [row[0].get("action") or {} for row in data["steps"][1:720]]
    player1_stream = [row[1].get("action") or {} for row in data["steps"][1:720]]
    genome = _candidate_panel()
    validation_errors = validate_m35_farm_genome_v2(genome)
    names = ("M35_DEFAULT", "GOLD_OPENING_ONLY", "GOLD_6PHASE_APPROX")
    batch_size = len(names)
    opponent_actions = _encode_actions(opponent_stream, batch_size)
    events = _build_events(seed, batch_size)
    carry = initialize_m35_rollout_carry_v2(
        jnp.full((batch_size,), np.int32(seed), dtype=jnp.int32), genome, player=player
    )
    rollout = jax.jit(_make_rollout(player))
    started = time.perf_counter()
    final, (actions, metrics) = rollout(carry, opponent_actions, events, load_tables(), genome)
    jax.block_until_ready(final)
    elapsed = time.perf_counter() - started
    final, actions, metrics = jax.device_get((final, actions, metrics))
    state, controller, compiler_errors, effect_errors = final
    money, land, units, crop_tiles, animals, shed, seeds = metrics

    milestone_steps = (1, 6 * 24, 11 * 24, 12 * 24, 20 * 24, 25 * 24, 29 * 24, 719)
    replay_milestones = [_replay_metrics(data, player, step) for step in milestone_steps]

    # Exact-action control: if this reproduces the official Replay, any route
    # gap below belongs to the controller/schema rather than event generation.
    control_genome = default_m35_farm_genome_v2(1)
    control_carry = initialize_m35_rollout_carry_v2(
        jnp.asarray((np.int32(seed),), dtype=jnp.int32), control_genome, player=player
    )
    control_events = _build_events(seed, 1)
    control_rollout = jax.jit(_make_replay_control_rollout(player))
    control_started = time.perf_counter()
    control_final, control_metrics = control_rollout(
        control_carry.environment_state,
        _encode_actions(player0_stream, 1),
        _encode_actions(player1_stream, 1),
        control_events,
        load_tables(),
    )
    jax.block_until_ready(control_final)
    control_elapsed = time.perf_counter() - control_started
    control_final, control_metrics = jax.device_get((control_final, control_metrics))
    (
        control_money,
        control_land,
        control_units,
        control_crop_tiles,
        control_animals,
        control_shed,
        control_seeds,
    ) = control_metrics
    control_milestones = []
    for state_step in milestone_steps:
        index = state_step - 1
        control_milestones.append(
            {
                "state_step": state_step,
                "money": int(control_money[index, 0]),
                "land": int(control_land[index, 0]),
                "units": int(control_units[index, 0]),
                "crop_tiles": np.asarray(control_crop_tiles[index, 0]).astype(int).tolist(),
                "animals": np.asarray(control_animals[index, 0]).astype(int).tolist(),
                "shed": np.asarray(control_shed[index, 0]).astype(int).tolist(),
                "seeds": np.asarray(control_seeds[index, 0]).astype(int).tolist(),
            }
        )
    rows = []
    for lane, name in enumerate(names):
        milestones = []
        for state_step in milestone_steps:
            index = state_step - 1
            milestones.append(
                {
                    "state_step": state_step,
                    "money": int(money[index, lane]),
                    "land": int(land[index, lane]),
                    "units": int(units[index, lane]),
                    "crop_tiles": np.asarray(crop_tiles[index, lane]).astype(int).tolist(),
                    "animals": np.asarray(animals[index, lane]).astype(int).tolist(),
                    "shed": np.asarray(shed[index, lane]).astype(int).tolist(),
                    "seeds": np.asarray(seeds[index, lane]).astype(int).tolist(),
                }
            )
        rows.append(
            {
                "candidate": name,
                "first_action": _decode_action(actions, lane),
                "milestones": milestones,
                "final_bank": int(state.money[lane, player]),
                "done": bool(state.done[lane]),
                "compiler_error_count": int(compiler_errors[lane]),
                "effect_error_count": int(effect_errors[lane]),
                "controller_unexplained_effect_failures": int(controller.unexplained_effect_failures[lane]),
            }
        )

    replay_opening = _replay_opening_encoded(data["steps"][1][player]["action"])
    opening_candidate = rows[1]["first_action"]
    receipt = {
        "receipt_id": "M35_GOLD_REPLAY_ALIGNMENT_V1",
        "status": "PASS" if not validation_errors and all(row["done"] for row in rows) else "FAIL",
        "alignment_verdict": "FAIL",
        "freeze_recommendation": "DO_NOT_FREEZE_M35_SEARCH_SCHEMA_OR_CONTROLLER",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "compile_and_execute_seconds": elapsed,
        "replay_control_compile_and_execute_seconds": control_elapsed,
        "official_version": data["module_version"],
        "replay": str(args.replay.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/"),
        "replay_sha256": _sha256(args.replay),
        "episode_id": int(data["info"]["EpisodeId"]),
        "seed": seed,
        "gold_player": data["info"]["Agents"][player]["Name"],
        "gold_seat": player,
        "opponent": data["info"]["Agents"][opponent]["Name"],
        "replay_final_bank": int(data["rewards"][player]),
        "replay_opening": replay_opening,
        "gold_opening_literal_match": opening_candidate == replay_opening,
        "gold_opening_commitment_multiset_match": (
            Counter(map(tuple, opening_candidate["units"])) == Counter(map(tuple, replay_opening["units"]))
            and Counter(map(tuple, opening_candidate["market"])) == Counter(map(tuple, replay_opening["market"]))
        ),
        "milestone_schema": {
            "crop_tiles": list(CROPS),
            "animals": list(ANIMALS),
            "shed": list(SHED_ITEMS),
            "seeds": list(CROPS),
        },
        "replay_milestones": replay_milestones,
        "jax_replay_action_control": {
            "final_bank": int(control_final.money[0, player]),
            "done": bool(control_final.done[0]),
            "exact_final_bank_match": int(control_final.money[0, player]) == int(data["rewards"][player]),
            "milestones": control_milestones,
            "exact_milestone_match_on_recorded_fields": control_milestones == [
                {key: value for key, value in row.items() if key not in ("pasture_tiles", "coop_tiles")}
                for row in replay_milestones
            ],
        },
        "candidate_rows": rows,
        "genome_validation_errors": validation_errors,
        "known_schema_gap": {
            "real_daily_hire_targets": [5, 0, 4, 5, 5, 4, 4, 8, 11, 12, 11, 12, 10, 11, 8, 12, 9, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 10, 8],
            "m35_requires_monotonic_hand_target": True,
            "approximation_used": [5, 5, 12, 12, 12, 12],
            "phase_count_limit": 6,
        },
        "truth_boundary": (
            "Replay milestones are official 1.32.7 observations. Candidate trajectories use the accepted JAX 1.32.7 simulator with the same seed and an open-loop opponent Replay action trace. After divergence this is a controlled disturbance, not an exact causal recreation of the original opponent."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

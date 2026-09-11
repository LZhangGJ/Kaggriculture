"""Observable-prefix Hasegawa V3 router with atomic V2 recovery semantics."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hasegawa_jax_v2.agent import (
    _apply_weed_repair,
    _guard_and_recover_units,
    _pair,
    _sanitize_market,
    _terminal_liquidation,
)
from kaggriculture_jax.constants import (
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import Action, Events, State, StaticTables

from .trace_bank import HasegawaTraceBankV3


ROUTER_PREFIX_REWARD = 0
ROUTER_PREFIX_PUBLIC_STATE = 1
ROUTER_FIRST_SHOP_MAP = 2
ROUTER_PREFIX_COMPATIBLE = 3
ROUTER_PREFIX_LOCK = 4
ROUTER_TWO_SHOP_COMPATIBLE_MAP = 5
ROUTER_STEP_PUBLIC_STATE = 6
ROUTER_DAY_PUBLIC_STATE = 7
_PREFIX_WEIGHTS = jnp.asarray((64, 32, 16, 8, 4, 2, 1, 1), dtype=jnp.int32)


class HasegawaCarryV3(NamedTuple):
    branch_id: jax.Array
    branch_locked: jax.Array
    branch_lock_step: jax.Array
    selected_town_count: jax.Array
    route_switch_total: jax.Array
    weed_active: jax.Array
    weed_start: jax.Array
    weed_intended_op: jax.Array
    weed_intended_item: jax.Array
    weed_intended_amount: jax.Array
    invalid_intent_total: jax.Array
    resync_total: jax.Array
    market_trim_total: jax.Array
    hard_counter_total: jax.Array


class HasegawaDiagnosticsV3(NamedTuple):
    route_id: jax.Array
    source_episode_id: jax.Array
    route_switched: jax.Array
    visible_shop_count: jax.Array
    invalid_unit_intent_count: jax.Array
    resync_unit_count: jax.Array
    market_trim_count: jax.Array
    hard_counter_delta: jax.Array
    money_deviation_from_source: jax.Array


def initialize_hasegawa_carry_v3(batch_size: int, bootstrap_route_id: int | jax.Array) -> HasegawaCarryV3:
    units = (batch_size, MAX_UNITS)
    zeros = jnp.zeros((batch_size,), dtype=jnp.int32)
    route = jnp.broadcast_to(jnp.asarray(bootstrap_route_id, dtype=jnp.int16), (batch_size,))
    return HasegawaCarryV3(
        branch_id=route,
        branch_locked=jnp.zeros((batch_size,), dtype=jnp.bool_),
        branch_lock_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        selected_town_count=jnp.zeros((batch_size,), dtype=jnp.int8),
        route_switch_total=zeros,
        weed_active=jnp.zeros(units, dtype=jnp.bool_),
        weed_start=jnp.full(units, -1, dtype=jnp.int16),
        weed_intended_op=jnp.full(units, UnitOp.PASS, dtype=jnp.int8),
        weed_intended_item=jnp.full(units, -1, dtype=jnp.int8),
        weed_intended_amount=jnp.ones(units, dtype=jnp.int32),
        invalid_intent_total=zeros,
        resync_total=zeros,
        market_trim_total=zeros,
        hard_counter_total=zeros,
    )


def _farm_summary(states: State, player: int) -> jax.Array:
    batch = states.step.shape[0]
    animal = states.tile_animal[:, player].reshape(batch, -1)
    crop = states.tile_crop[:, player].reshape(batch, -1)
    kind = states.tile_kind[:, player].reshape(batch, -1)
    animal_counts = jnp.stack(
        [jnp.sum(animal == index, axis=1, dtype=jnp.int32) for index in range(NUM_ANIMALS)], axis=1
    )
    crop_counts = jnp.stack(
        [jnp.sum(crop == index, axis=1, dtype=jnp.int32) for index in range(NUM_CROPS)], axis=1
    )
    unlocked = jnp.sum(kind != TileKind.LOCKED, axis=1, dtype=jnp.int32)[:, None]
    hands = jnp.sum(states.unit_active[:, player, 1:], axis=1, dtype=jnp.int32)[:, None]
    money = states.money[:, player : player + 1].astype(jnp.int32)
    return jnp.concatenate((animal_counts, crop_counts, unlocked, hands, money), axis=1)


def _public_state_cost(states: State, bank: HasegawaTraceBankV3, step: jax.Array, player: int):
    batch = states.step.shape[0]
    routes = bank.source_reward.shape[0]
    route_index = jnp.arange(routes, dtype=jnp.int32)[None, :]
    step_index = jnp.broadcast_to(step[:, None], (batch, routes))
    own_expected = bank.expected_self_summary[route_index, step_index]
    opponent_expected = bank.expected_opponent_summary[route_index, step_index]
    own = _farm_summary(states, player)[:, None, :]
    opponent = _farm_summary(states, 1 - player)[:, None, :]
    # Counts and staffing describe task compatibility. Money is deliberately
    # coarser because shared-market fills can move it without invalidating the
    # underlying production program.
    own_count = jnp.sum(jnp.abs(own[..., :8] - own_expected[..., :8]), axis=-1) * 2_000
    own_capacity = jnp.abs(own[..., 8] - own_expected[..., 8]) * 3_000
    own_staff = jnp.abs(own[..., 9] - own_expected[..., 9]) * 5_000
    opponent_count = jnp.sum(jnp.abs(opponent[..., :8] - opponent_expected[..., :8]), axis=-1) * 300
    opponent_capacity = jnp.abs(opponent[..., 8] - opponent_expected[..., 8]) * 300
    opponent_staff = jnp.abs(opponent[..., 9] - opponent_expected[..., 9]) * 500
    own_money = jnp.abs(own[..., 10] - own_expected[..., 10]) // 10
    opponent_money = jnp.abs(opponent[..., 10] - opponent_expected[..., 10]) // 50
    shed_cost = jnp.sum(
        jnp.abs(states.shed[:, player, None, :] - bank.expected_shed[route_index, step_index]), axis=-1
    ) * 200
    seed_cost = jnp.sum(
        jnp.abs(states.seeds[:, player, None, :] - bank.expected_seeds[route_index, step_index]), axis=-1
    ) * 200
    carried = jnp.sum(states.unit_inventory[:, player].astype(jnp.int32), axis=1)[:, None, :]
    carried_cost = jnp.sum(
        jnp.abs(carried - bank.expected_carried[route_index, step_index]), axis=-1
    ) * 200
    actual_pos = states.unit_pos[:, player, None, :, :].astype(jnp.int32)
    expected_pos = bank.expected_unit_pos[route_index, step_index].astype(jnp.int32)
    expected_active = bank.expected_unit_active[route_index, step_index]
    position_cost = jnp.sum(
        jnp.where(expected_active[..., None], jnp.abs(actual_pos - expected_pos), 0), axis=(-1, -2)
    ) * 100
    price_expected = bank.expected_market_price[route_index, step_index]
    price_cost = jnp.sum(jnp.abs(states.market_price[:, None, :] - price_expected), axis=-1) * 2
    return (
        own_count + own_capacity + own_staff + opponent_count + opponent_capacity
        + opponent_staff + own_money + opponent_money + shed_cost + seed_cost
        + carried_cost + position_cost + price_cost
    )


def select_hasegawa_route_v3(
    states: State,
    carry: HasegawaCarryV3,
    bank: HasegawaTraceBankV3,
    player: int,
    router_mode: int = ROUTER_PREFIX_PUBLIC_STATE,
    first_shop_route_map: jax.Array | None = None,
    compatibility_threshold: int = 15_000,
    compatibility_slack: int = 5_000,
    lock_shop_count: int = 2,
    second_shop_route_map: jax.Array | None = None,
    prefix_score_scale: int | jax.Array = 10_000_000,
    source_reward_scale: int | jax.Array = 1,
):
    """Reroute only when a newly unlocked shop becomes publicly visible."""

    visible_count = jnp.clip(states.town_count.astype(jnp.int32), 0, 8)
    if router_mode in (ROUTER_FIRST_SHOP_MAP, ROUTER_TWO_SHOP_COMPATIBLE_MAP):
        # This mode depends only on the first public shop and then locks.  The
        # original implementation still built the full route-prefix and
        # public-state distance tensors for every source route before throwing
        # them away.  With 100-200 Replay routes that made the combined
        # candidate-vs-FC24B graph take minutes to compile.  Return before any
        # route-bank-wide calculation; the selected actions and carry updates
        # are exactly unchanged.
        should_select = (visible_count > 0) & (carry.selected_town_count < 1)
        if first_shop_route_map is None:
            raise ValueError("first_shop_route_map is required")
        visible = states.town_shops.astype(jnp.int32)
        first_shop = jnp.clip(visible[:, 0], 0, 7)
        if first_shop_route_map.ndim == 1:
            selected = first_shop_route_map[first_shop].astype(jnp.int16)
        elif first_shop_route_map.ndim == 2 and first_shop_route_map.shape == (2, 8):
            # Seat order changes simultaneous market fills.  A route may be
            # robust as first player but fragile as second player, so permit a
            # separately trained public seat map without exposing any hidden
            # information.  Existing one-dimensional maps remain bit-exact.
            selected = first_shop_route_map[player, first_shop].astype(jnp.int16)
        else:
            raise ValueError(
                f"first_shop_route_map must have shape (8,) or (2, 8), got "
                f"{first_shop_route_map.shape}"
            )
        if router_mode == ROUTER_TWO_SHOP_COMPATIBLE_MAP:
            if second_shop_route_map is None or second_shop_route_map.shape[1:] != (8,):
                raise ValueError(
                    "second_shop_route_map must have shape (routes, 8) or (8, 8)"
                )
            should_select_second = (visible_count > 1) & (carry.selected_town_count < 2)
            second_shop = jnp.clip(visible[:, 1], 0, 7)
            if second_shop_route_map.shape == (8, 8):
                # Multiple first-shop conditions may deliberately share the
                # same base route.  Indexing only by current route would merge
                # those conditions and silently broaden a switch.  The compact
                # 8x8 form is therefore indexed by the two public shop IDs.
                selected_second = second_shop_route_map[
                    first_shop, second_shop
                ].astype(jnp.int16)
            else:
                selected_second = second_shop_route_map[
                    carry.branch_id.astype(jnp.int32), second_shop
                ].astype(jnp.int16)
            selected = jnp.where(should_select_second, selected_second, selected)
            should_select = should_select | should_select_second
        route = jnp.where(should_select, selected, carry.branch_id).astype(jnp.int16)
        switched = should_select & (route != carry.branch_id)
        first_lock = should_select & (~carry.branch_locked)
        return route, switched, visible_count.astype(jnp.int8), first_lock
    elif router_mode == ROUTER_PREFIX_LOCK:
        # A coherent atomic program cannot safely be replaced every time a new
        # town shop appears.  Wait for a short observable prefix, choose once,
        # then keep the selected program and let only local action recovery
        # handle later deviations.
        should_select = (visible_count >= lock_shop_count) & (~carry.branch_locked)
    elif router_mode == ROUTER_STEP_PUBLIC_STATE:
        # Nearest-neighbour imitation at the current step.  This is a
        # diagnostic for Replay banks whose source agent re-plans frequently:
        # it chooses only from same-step source states and uses no future event
        # or Replay identity.  The action guard below remains the safety owner.
        should_select = jnp.ones_like(carry.branch_locked)
    elif router_mode == ROUTER_DAY_PUBLIC_STATE:
        # Preserve one coherent day of choreography, then re-select from the
        # currently observable state.  This is intentionally coarser than
        # per-step nearest-neighbour routing: moving, servicing, harvesting and
        # returning often form a multi-step daily chain that must not be split
        # merely because another source trace is marginally closer next step.
        should_select = (states.step.astype(jnp.int32) % 24) == 0
    else:
        should_select = visible_count > carry.selected_town_count.astype(jnp.int32)
    visible = states.town_shops.astype(jnp.int32)
    source = bank.source_shop_sequence.astype(jnp.int32)
    active = jnp.arange(8, dtype=jnp.int32)[None, None, :] < visible_count[:, None, None]
    positional_match = (source[None, :, :] == visible[:, None, :]) & active
    prefix_score = jnp.sum(positional_match * _PREFIX_WEIGHTS[None, None, :], axis=-1, dtype=jnp.int32)
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    reward_scale = jnp.asarray(source_reward_scale, dtype=jnp.int32)
    if reward_scale.ndim == 0:
        reward_scale = reward_scale[None]
    quality = (
        jnp.broadcast_to(bank.source_reward.astype(jnp.int32)[None, :], prefix_score.shape)
        * reward_scale[:, None]
    )
    state_cost = _public_state_cost(states, bank, step, player)
    if router_mode in (
        ROUTER_PREFIX_PUBLIC_STATE,
        ROUTER_PREFIX_LOCK,
        ROUTER_STEP_PUBLIC_STATE,
        ROUTER_DAY_PUBLIC_STATE,
    ):
        quality = quality - state_cost
    elif router_mode == ROUTER_PREFIX_COMPATIBLE:
        quality = bank.source_reward.astype(jnp.int32)[None, :] // 100 - state_cost
    prefix_scale = jnp.asarray(prefix_score_scale, dtype=jnp.int32)
    if prefix_scale.ndim == 0:
        prefix_scale = prefix_scale[None]
    score = prefix_score * prefix_scale[:, None] + quality
    selected = jnp.argmax(score, axis=1).astype(jnp.int16)
    if router_mode == ROUTER_PREFIX_COMPATIBLE:
        batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
        selected_cost = state_cost[batch, selected.astype(jnp.int32)]
        current_cost = state_cost[batch, carry.branch_id.astype(jnp.int32)]
        compatible = (selected_cost <= compatibility_threshold) & (
            selected_cost <= current_cost + compatibility_slack
        )
        selected = jnp.where(compatible, selected, carry.branch_id).astype(jnp.int16)
    route = jnp.where(should_select, selected, carry.branch_id).astype(jnp.int16)
    switched = should_select & (route != carry.branch_id)
    first_lock = should_select & (~carry.branch_locked)
    return route, switched, visible_count.astype(jnp.int8), first_lock


def hasegawa_step_with_external_v3(
    states: State,
    carry: HasegawaCarryV3,
    bank: HasegawaTraceBankV3,
    external_action: Action,
    player: int,
    events: Events,
    tables: StaticTables,
    router_mode: int = ROUTER_PREFIX_PUBLIC_STATE,
    first_shop_route_map: jax.Array | None = None,
    compatibility_threshold: int = 15_000,
    compatibility_slack: int = 5_000,
    lock_shop_count: int = 2,
    forced_route: jax.Array | None = None,
    forced_lock: jax.Array | None = None,
    second_shop_route_map: jax.Array | None = None,
    controlled_action_override: Action | None = None,
    controlled_action_override_mask: jax.Array | None = None,
    prefix_score_scale: int | jax.Array = 10_000_000,
    source_reward_scale: int | jax.Array = 1,
):
    if forced_route is None:
        route, switched, visible_count, first_lock = select_hasegawa_route_v3(
            states, carry, bank, player, router_mode, first_shop_route_map,
            compatibility_threshold, compatibility_slack, lock_shop_count,
            second_shop_route_map, prefix_score_scale, source_reward_scale
        )
    else:
        # Screening and counterfactual tools already know the route being
        # evaluated.  Skipping the full route-distance scan here is both
        # semantically exact and substantially cheaper for large route banks.
        route = forced_route.astype(jnp.int16)
        switched = route != carry.branch_id
        visible_count = jnp.clip(states.town_count, 0, 8).astype(jnp.int8)
        lock_mask = (
            jnp.ones_like(carry.branch_locked)
            if forced_lock is None
            else forced_lock.astype(jnp.bool_)
        )
        first_lock = lock_mask & (~carry.branch_locked)
    carry = carry._replace(
        branch_id=route,
        branch_locked=carry.branch_locked | first_lock,
        branch_lock_step=jnp.where(first_lock, states.step, carry.branch_lock_step).astype(jnp.int16),
        selected_town_count=jnp.maximum(carry.selected_town_count, visible_count).astype(jnp.int8),
        route_switch_total=carry.route_switch_total + switched.astype(jnp.int32),
        # A repair generated by the old program must never leak into the newly
        # selected program.
        weed_active=jnp.where(switched[:, None], False, carry.weed_active),
        weed_start=jnp.where(switched[:, None], -1, carry.weed_start).astype(jnp.int16),
    )
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    present = jnp.arange(MAX_UNITS)[None, :] < count[:, None]
    op = jnp.where(present, bank.unit_op[route, step], UnitOp.PASS).astype(jnp.int8)
    item = jnp.where(present, bank.unit_item[route, step], -1).astype(jnp.int8)
    amount = jnp.where(present, bank.unit_amount[route, step], 1).astype(jnp.int32)
    op, item, amount, carry = _apply_weed_repair(
        states, bank, route, carry, op, item, amount, present, player
    )
    op, item, amount, invalid, resync = _guard_and_recover_units(
        states, bank, route, op, item, amount, present, player
    )
    action = Action(
        unit_op=op,
        unit_item=item,
        unit_amount=amount,
        unit_count=count,
        market_op=bank.market_op[route, step],
        market_item=bank.market_item[route, step],
        market_amount=bank.market_amount[route, step],
        market_count=bank.market_count[route, step],
    )
    action, trimmed = _sanitize_market(states, action, player)
    action = _terminal_liquidation(states, action, player)
    if controlled_action_override is not None:
        if controlled_action_override_mask is None:
            raise ValueError("controlled_action_override_mask is required")
        mask = controlled_action_override_mask.astype(jnp.bool_)
        selected_fields = []
        for source, override in zip(action, controlled_action_override, strict=True):
            shaped = mask.reshape((mask.shape[0],) + (1,) * (source.ndim - 1))
            selected_fields.append(jnp.where(shaped, override, source))
        action = Action(*selected_fields)
    joint = _pair(action, external_action, player)
    next_state = batched_step_sync(states, joint, events, tables)
    hard_before = states.hand_cap_hits[:, player] + states.market_loop_cap_hits + states.price_lut_oob
    hard_after = next_state.hand_cap_hits[:, player] + next_state.market_loop_cap_hits + next_state.price_lut_oob
    hard_delta = jnp.maximum(hard_after - hard_before, 0).astype(jnp.int32)
    expected_money = bank.expected_money[route, step].astype(jnp.int32)
    carry = carry._replace(
        invalid_intent_total=carry.invalid_intent_total + invalid,
        resync_total=carry.resync_total + resync,
        market_trim_total=carry.market_trim_total + trimmed,
        hard_counter_total=carry.hard_counter_total + hard_delta,
    )
    diagnostics = HasegawaDiagnosticsV3(
        route_id=route,
        source_episode_id=bank.source_episode_id[route],
        route_switched=switched,
        visible_shop_count=visible_count,
        invalid_unit_intent_count=invalid,
        resync_unit_count=resync,
        market_trim_count=trimmed,
        hard_counter_delta=hard_delta,
        money_deviation_from_source=states.money[:, player].astype(jnp.int32) - expected_money,
    )
    return next_state, carry, diagnostics, joint


__all__ = [
    "HasegawaCarryV3",
    "HasegawaDiagnosticsV3",
    "ROUTER_PREFIX_PUBLIC_STATE",
    "ROUTER_PREFIX_REWARD",
    "ROUTER_FIRST_SHOP_MAP",
    "ROUTER_PREFIX_COMPATIBLE",
    "ROUTER_PREFIX_LOCK",
    "ROUTER_TWO_SHOP_COMPATIBLE_MAP",
    "ROUTER_STEP_PUBLIC_STATE",
    "ROUTER_DAY_PUBLIC_STATE",
    "hasegawa_step_with_external_v3",
    "initialize_hasegawa_carry_v3",
    "select_hasegawa_route_v3",
]

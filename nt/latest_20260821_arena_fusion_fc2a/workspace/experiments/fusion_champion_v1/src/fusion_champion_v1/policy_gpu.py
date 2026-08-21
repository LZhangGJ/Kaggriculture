"""GPU policy candidates for Fusion Champion V1.

FC1A deliberately changes one interpretable decision only.  It keeps the
strict-parity Ray K320 execution stack, but when the opponent's *public board*
shows the already-recognized low-cash 1-cow/4-sheep expansion, it selects the
6-cow/8-sheep legacy production route instead of allowing the random shop mix
to lock a cow-heavy route.  No opponent name or hidden state is consulted.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import PRODUCTS, MarketOp
from strategic_v5 import high_potential_v20_gpu as hp
from strategic_v5 import latest_public8_gpu as lp
from strategic_v5.latest_public8_gpu import (
    KaitoV36CarryV1,
    X562CarryV1,
    initialize_kaito_v36_carry_v1,
    initialize_x562_carry_v1,
    x562_player_action_v1,
)


class FusionChampionCarryV1(NamedTuple):
    base: hp.HighPotentialV20CarryV1
    sheep_pressure: jax.Array


def initialize_fusion_champion_carry_v1(batch_size: int) -> FusionChampionCarryV1:
    return FusionChampionCarryV1(
        base=hp.initialize_high_potential_v20_carry_v1(batch_size),
        sheep_pressure=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


class FusionChampionCarryV2(NamedTuple):
    k320: hp.HighPotentialV20CarryV1
    x562: X562CarryV1
    sheep_pressure: jax.Array


def initialize_fusion_champion_carry_v2(batch_size: int) -> FusionChampionCarryV2:
    return FusionChampionCarryV2(
        k320=hp.initialize_high_potential_v20_carry_v1(batch_size),
        x562=initialize_x562_carry_v1(batch_size),
        sheep_pressure=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


class FusionChampionCarryV3(NamedTuple):
    k320: hp.HighPotentialV20CarryV1
    x562_suffix: X562OldSuffixCarryV1
    sheep_pressure: jax.Array


def initialize_fusion_champion_carry_v3(batch_size: int) -> FusionChampionCarryV3:
    return FusionChampionCarryV3(
        k320=hp.initialize_high_potential_v20_carry_v1(batch_size),
        x562_suffix=initialize_x562_old_suffix_carry_v1(batch_size),
        sheep_pressure=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


class KaitoX562CarryV1(NamedTuple):
    kaito: KaitoV36CarryV1
    counter: hp.HighPotentialV20CarryV1


def initialize_kaito_x562_carry_v1(batch_size: int) -> KaitoX562CarryV1:
    return KaitoX562CarryV1(
        kaito=initialize_kaito_v36_carry_v1(batch_size),
        counter=hp.initialize_high_potential_v20_carry_v1(batch_size),
    )


class X562RouteRuleCarryV1(NamedTuple):
    base: hp.HighPotentialV20CarryV1
    decided: jax.Array
    route_id: jax.Array


def initialize_x562_route_rule_carry_v1(batch_size: int) -> X562RouteRuleCarryV1:
    return X562RouteRuleCarryV1(
        base=hp.initialize_high_potential_v20_carry_v1(batch_size),
        decided=jnp.zeros((batch_size,), dtype=jnp.bool_),
        route_id=jnp.full((batch_size,), 12, dtype=jnp.int8),
    )


class OpponentFamilyMarketScheduleV1(NamedTuple):
    """Two public-board-distinguishable sell schedules for one route family."""

    cow_heavy_op: jax.Array
    cow_heavy_item: jax.Array
    cow_heavy_amount: jax.Array
    cow_heavy_count: jax.Array
    balanced_op: jax.Array
    balanced_item: jax.Array
    balanced_amount: jax.Array
    balanced_count: jax.Array


class X562FamilyCounterCarryV1(NamedTuple):
    route: X562RouteRuleCarryV1
    pressure_target: jax.Array
    # -1 unresolved, 0 cow-heavy, 1 balanced cow/sheep.
    opponent_family: jax.Array


def initialize_x562_family_counter_carry_v1(
    batch_size: int,
) -> X562FamilyCounterCarryV1:
    return X562FamilyCounterCarryV1(
        route=initialize_x562_route_rule_carry_v1(batch_size),
        pressure_target=jnp.zeros((batch_size,), dtype=jnp.bool_),
        opponent_family=jnp.full((batch_size,), -1, dtype=jnp.int8),
    )


class K320HeavySheepCounterCarryV1(NamedTuple):
    base: hp.HighPotentialV20CarryV1
    pressure_target: jax.Array
    # -1 unresolved, 0 high-volume 6C/12S, 1 compact 4C/9S.
    opponent_family: jax.Array


def initialize_k320_heavy_sheep_counter_carry_v1(
    batch_size: int,
) -> K320HeavySheepCounterCarryV1:
    return K320HeavySheepCounterCarryV1(
        base=hp.initialize_high_potential_v20_carry_v1(batch_size),
        pressure_target=jnp.zeros((batch_size,), dtype=jnp.bool_),
        opponent_family=jnp.full((batch_size,), -1, dtype=jnp.int8),
    )


class X562KaitoSwitchCarryV1(NamedTuple):
    x562: X562RouteRuleCarryV1
    kaito: KaitoV36CarryV1
    switched: jax.Array


def initialize_x562_kaito_switch_carry_v1(
    batch_size: int,
) -> X562KaitoSwitchCarryV1:
    return X562KaitoSwitchCarryV1(
        x562=initialize_x562_route_rule_carry_v1(batch_size),
        kaito=initialize_kaito_v36_carry_v1(batch_size),
        switched=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


class X562KaitoMarketCarryV1(NamedTuple):
    x562: X562RouteRuleCarryV1
    kaito: KaitoV36CarryV1


def initialize_x562_kaito_market_carry_v1(
    batch_size: int,
) -> X562KaitoMarketCarryV1:
    return X562KaitoMarketCarryV1(
        x562=initialize_x562_route_rule_carry_v1(batch_size),
        kaito=initialize_kaito_v36_carry_v1(batch_size),
    )


class X562OldSuffixCarryV1(NamedTuple):
    x562: X562RouteRuleCarryV1
    suffix: hp.HighPotentialV20CarryV1
    switched: jax.Array


def initialize_x562_old_suffix_carry_v1(batch_size: int) -> X562OldSuffixCarryV1:
    return X562OldSuffixCarryV1(
        x562=initialize_x562_route_rule_carry_v1(batch_size),
        suffix=hp.initialize_high_potential_v20_carry_v1(batch_size),
        switched=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


class K320OldSuffixCarryV1(NamedTuple):
    k320: hp.HighPotentialV20CarryV1
    suffix: hp.HighPotentialV20CarryV1
    switched: jax.Array


def initialize_k320_old_suffix_carry_v1(batch_size: int) -> K320OldSuffixCarryV1:
    return K320OldSuffixCarryV1(
        k320=hp.initialize_high_potential_v20_carry_v1(batch_size),
        suffix=hp.initialize_high_potential_v20_carry_v1(batch_size),
        switched=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


def _visible_opening_sheep_pressure(states, player: int) -> jax.Array:
    step = states.step.astype(jnp.int32)
    rival = 1 - player
    animals = states.tile_animal[:, rival]
    cows = jnp.sum(animals == 1, axis=(1, 2))
    sheep = jnp.sum(animals == 2, axis=(1, 2))
    return (
        (step >= 24)
        & (step < 96)
        & (sheep >= 4)
        & (cows <= 2)
        & (states.money[:, rival] <= 1000)
    )


def _select_action(mask: jax.Array, selected, fallback):
    shape = (mask.shape[0],) + (1,) * (selected.unit_op.ndim - 1)
    unit_mask = mask.reshape(shape)
    market_shape = (mask.shape[0],) + (1,) * (selected.market_op.ndim - 1)
    market_mask = mask.reshape(market_shape)
    return selected._replace(
        unit_op=jnp.where(unit_mask, selected.unit_op, fallback.unit_op),
        unit_item=jnp.where(unit_mask, selected.unit_item, fallback.unit_item),
        unit_amount=jnp.where(unit_mask, selected.unit_amount, fallback.unit_amount),
        unit_count=jnp.where(mask, selected.unit_count, fallback.unit_count),
        market_op=jnp.where(market_mask, selected.market_op, fallback.market_op),
        market_item=jnp.where(market_mask, selected.market_item, fallback.market_item),
        market_amount=jnp.where(market_mask, selected.market_amount, fallback.market_amount),
        market_count=jnp.where(mask, selected.market_count, fallback.market_count),
    )


def _select_tree(mask: jax.Array, selected, fallback):
    return jax.tree_util.tree_map(
        lambda chosen, old: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (chosen.ndim - 1)),
            chosen,
            old,
        ),
        selected,
        fallback,
    )


def fc1a_sheep_pressure_player_action_v1(
    states,
    tables,
    bank,
    runtime,
    carry: FusionChampionCarryV1,
    player: int,
):
    """Ray K320 plus one public-state route correction."""

    route, base = hp._select_route(states, carry.base, player, hp.MODE_RAY_K320)

    # Detect the economic shape, not a named opponent: during the opening the
    # rival commits at least four sheep while having no more than two cows.
    # Persist the observation because that opening commitment remains relevant
    # after the rival later buys more cows or changes visible crop counts.
    detect = _visible_opening_sheep_pressure(states, player)
    sheep_pressure = carry.sheep_pressure | detect
    # Route 2 is the current 6C/8S tape; route 7 is its legacy-layout variant.
    pressure_route = jnp.where(base.legacy_layout, 7, 2)
    route = jnp.where(sheep_pressure, pressure_route, route).astype(jnp.int32)

    action, base = hp._raw_with_weed(states, bank, route, base, player)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(
        action, base, states.step.astype(jnp.int32), hp.MODE_RAY_K320
    )
    action = hp._rank_sell_slots_exact(states, runtime, action)
    action, base = hp._preempt(
        states, bank, route, action, base, player, hp.MODE_RAY_K320
    )
    action, base = hp._market_counters(
        states, action, base, runtime, player
    )
    action = hp._room_guard(states, action, player)
    action = hp._terminal_liquidation(states, action, player)
    action = hp._trim_ray_seeds(states, runtime, route, action, player)
    return action, FusionChampionCarryV1(
        base=base,
        sheep_pressure=sheep_pressure,
    )


def fc1b_x562_pressure_player_action_v1(
    states,
    tables,
    bank,
    runtime,
    carry: FusionChampionCarryV2,
    player: int,
):
    """K320 normally; use the complete X562 capability under sheep pressure."""

    k320_action, k320 = hp.high_potential_v20_player_action_v1(
        states,
        tables,
        bank,
        runtime,
        carry.k320,
        player,
        hp.MODE_RAY_K320,
    )
    x562_action, x562 = x562_player_action_v1(
        states, runtime, bank, carry.x562, player
    )
    sheep_pressure = carry.sheep_pressure | _visible_opening_sheep_pressure(
        states, player
    )
    action = _select_action(sheep_pressure, x562_action, k320_action)
    return action, FusionChampionCarryV2(
        k320=k320,
        x562=x562,
        sheep_pressure=sheep_pressure,
    )


def fc1s_prt_suffix_pressure_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    player: int,
):
    """K320 normally; stable X562 plus learned PRT suffix under sheep pressure."""

    k320_action, k320 = hp.high_potential_v20_player_action_v1(
        states,
        tables,
        latest_bank,
        runtime,
        carry.k320,
        player,
        hp.MODE_RAY_K320,
    )
    batch = states.step.shape[0]
    x562_action, x562_suffix = x562_prt_suffix_rule_player_action_v1(
        states,
        runtime,
        latest_bank,
        old_bank,
        carry.x562_suffix,
        jnp.ones((batch,), dtype=jnp.bool_),
        player,
    )
    sheep_pressure = carry.sheep_pressure | _visible_opening_sheep_pressure(
        states, player
    )
    action = _select_action(sheep_pressure, x562_action, k320_action)
    return action, FusionChampionCarryV3(
        k320=k320,
        x562_suffix=x562_suffix,
        sheep_pressure=sheep_pressure,
    )


def fc2a_rank14_plus_anti_mirror_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    player: int,
):
    """FC1S Rank14 branch plus the frozen FC2 K320 anti-mirror capability."""

    batch = states.step.shape[0]
    k320_action, k320 = k320_preempt_parameter_player_action_v1(
        states,
        tables,
        runtime,
        latest_bank,
        carry.k320,
        jnp.zeros((batch,), dtype=jnp.bool_),
        jnp.full((batch,), 4, dtype=jnp.int16),
        jnp.full((batch,), 6, dtype=jnp.int16),
        jnp.full((batch,), 32, dtype=jnp.int16),
        jnp.full((batch,), 4, dtype=jnp.int16),
        jnp.zeros((batch,), dtype=jnp.int16),
        jnp.full((batch,), 120, dtype=jnp.int16),
        jnp.full((batch,), 200, dtype=jnp.int16),
        player,
    )
    x562_action, x562_suffix = x562_prt_suffix_rule_player_action_v1(
        states,
        runtime,
        latest_bank,
        old_bank,
        carry.x562_suffix,
        jnp.ones((batch,), dtype=jnp.bool_),
        player,
    )
    sheep_pressure = carry.sheep_pressure | _visible_opening_sheep_pressure(
        states, player
    )
    action = _select_action(sheep_pressure, x562_action, k320_action)
    return action, FusionChampionCarryV3(
        k320=k320,
        x562_suffix=x562_suffix,
        sheep_pressure=sheep_pressure,
    )


def fc2b_rank14_plus_clone_aware_preempt_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    player: int,
):
    """Rank14 pressure branch plus public-state K320 preempt routing.

    This candidate changes only two independently ablatable decisions:

    * a visible heavy-sheep opening keeps the previously validated
      X562/legacy plus PRT-suffix pressure branch;
    * all other games use the exact-clone-aware K320 preempt parameters.

    Both decisions are made from the live public board.  Opponent identity and
    future event-bank information are deliberately unavailable here.
    """

    k320_action, k320 = k320_clone_aware_preempt_player_action_v1(
        states,
        tables,
        runtime,
        latest_bank,
        carry.k320,
        player,
    )
    batch = states.step.shape[0]
    x562_action, x562_suffix = x562_prt_suffix_rule_player_action_v1(
        states,
        runtime,
        latest_bank,
        old_bank,
        carry.x562_suffix,
        jnp.ones((batch,), dtype=jnp.bool_),
        player,
    )
    sheep_pressure = carry.sheep_pressure | _visible_opening_sheep_pressure(
        states, player
    )
    action = _select_action(sheep_pressure, x562_action, k320_action)
    return action, FusionChampionCarryV3(
        k320=k320,
        x562_suffix=x562_suffix,
        sheep_pressure=sheep_pressure,
    )


def kaito_x562_market_player_action_v1(
    states,
    bank,
    runtime,
    carry: KaitoX562CarryV1,
    player: int,
):
    """Balanced Kaito route plus X562's deployable market/recovery layers.

    This is a capability fusion, not a hindsight route switch.  It keeps the
    complete Kaito V36 production plan from step zero, then adds only layers
    whose decisions can be computed from the current public state.
    """

    route = jnp.full(states.step.shape, 11, dtype=jnp.int32)
    action, kaito = lp.kaito_v36_player_action_v1(
        states, runtime, bank, route, carry.kaito, player
    )
    action, counter = lp._x562_market_counters(
        states, action, carry.counter, runtime, player
    )
    action = lp._room_guard(states, action, player)
    action = lp._x562_seed_budget_guard(states, action, player)
    action = lp._terminal_liquidation(states, action, player)
    action = lp._x562_idle_fertilizer_sale(states, action, player)
    action = lp._trim_ray_seeds(states, runtime, route, action, player)
    return action, KaitoX562CarryV1(kaito=kaito, counter=counter)


def x562_forced_route_player_action_v1(
    states,
    runtime,
    bank,
    carry: X562CarryV1,
    route: jax.Array,
    player: int,
):
    """Run X562's feedback stack on a supplied prefix-compatible route."""

    route = route.astype(jnp.int32)
    action, base = hp._raw_with_weed(states, bank, route, carry.base, player)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(
        action, base, states.step.astype(jnp.int32), hp.MODE_BOATLEE
    )
    action = hp._rank_sell_slots_exact(states, runtime, action)
    action, base = lp._x562_preempt(states, bank, route, action, base, player)
    action, base = lp._x562_market_counters(
        states, action, base, runtime, player
    )
    action = lp._room_guard(states, action, player)
    action = lp._x562_seed_budget_guard(states, action, player)
    action = lp._terminal_liquidation(states, action, player)
    action = lp._x562_idle_fertilizer_sale(states, action, player)
    action = lp._trim_ray_seeds(states, runtime, route, action, player)
    return action, carry._replace(base=base)


def x562_layer_ablation_player_action_v1(
    states,
    runtime,
    bank,
    carry: X562CarryV1,
    enable_preempt: jax.Array,
    enable_counters: jax.Array,
    enable_fertilizer_sale: jax.Array,
    player: int,
):
    """Exact X562 with per-environment masks for three market layers."""

    route, carry = lp._x562_route(states, carry)
    action, base = hp._raw_with_weed(states, bank, route, carry.base, player)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(
        action, base, states.step.astype(jnp.int32), hp.MODE_BOATLEE
    )
    action = hp._rank_sell_slots_exact(states, runtime, action)

    preempt_action, preempt_base = lp._x562_preempt(
        states, bank, route, action, base, player
    )
    action = _select_action(enable_preempt, preempt_action, action)
    base = _select_tree(enable_preempt, preempt_base, base)

    counter_action, counter_base = lp._x562_market_counters(
        states, action, base, runtime, player
    )
    action = _select_action(enable_counters, counter_action, action)
    base = _select_tree(enable_counters, counter_base, base)

    action = lp._room_guard(states, action, player)
    action = lp._x562_seed_budget_guard(states, action, player)
    action = lp._terminal_liquidation(states, action, player)
    fertilizer_action = lp._x562_idle_fertilizer_sale(states, action, player)
    action = _select_action(enable_fertilizer_sale, fertilizer_action, action)
    action = lp._trim_ray_seeds(states, runtime, route, action, player)
    return action, carry._replace(base=base)


def x562_route_rule_player_action_v1(
    states,
    runtime,
    bank,
    carry: X562RouteRuleCarryV1,
    rule_id: jax.Array,
    enable_counters: jax.Array,
    player: int,
):
    """Choose a prefix-compatible suffix at step 168 from public shops."""

    step = states.step.astype(jnp.int32)
    active = jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    has_yarn = jnp.any(active & (states.town_shops == 7), axis=1)
    first = states.town_shops[:, 0].astype(jnp.int32)
    second = states.town_shops[:, 1].astype(jnp.int32)
    dominated = (states.town_count >= 2) & (first == 3) & (second == 7)

    current = jnp.where(has_yarn & (~dominated), 13, 12)
    any_four = jnp.where(has_yarn, 4, 12)
    any_seven = jnp.where(has_yarn, 7, 12)
    custom_seven = has_yarn & (
        ((first == 7) & ((second == 3) | (second == 4)))
        | ((second == 7) & (first == 6))
    )
    custom = jnp.where(custom_seven, 7, jnp.where(has_yarn, 4, 12))
    first_four_second_thirteen = jnp.where(
        first == 7, 4, jnp.where(second == 7, 13, 12)
    )
    first_seven_second_four = jnp.where(
        first == 7, 7, jnp.where(second == 7, 4, 12)
    )
    proposed = jnp.select(
        [
            rule_id == 0,
            rule_id == 1,
            rule_id == 2,
            rule_id == 3,
            rule_id == 4,
            rule_id == 5,
        ],
        [
            current,
            any_four,
            any_seven,
            custom,
            first_four_second_thirteen,
            first_seven_second_four,
        ],
        default=current,
    ).astype(jnp.int8)
    decide = (~carry.decided) & (step >= 168)
    route_id = jnp.where(decide, proposed, carry.route_id).astype(jnp.int8)
    decided = carry.decided | decide
    route = route_id.astype(jnp.int32)

    action, base = hp._raw_with_weed(states, bank, route, carry.base, player)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(
        action, base, step, hp.MODE_BOATLEE
    )
    action = hp._rank_sell_slots_exact(states, runtime, action)
    action, base = lp._x562_preempt(states, bank, route, action, base, player)
    counter_action, counter_base = lp._x562_market_counters(
        states, action, base, runtime, player
    )
    action = _select_action(enable_counters, counter_action, action)
    base = _select_tree(enable_counters, counter_base, base)
    action = lp._room_guard(states, action, player)
    action = lp._x562_seed_budget_guard(states, action, player)
    action = lp._terminal_liquidation(states, action, player)
    action = lp._x562_idle_fertilizer_sale(states, action, player)
    action = lp._trim_ray_seeds(states, runtime, route, action, player)
    return action, X562RouteRuleCarryV1(
        base=base,
        decided=decided,
        route_id=route_id,
    )


def x562_kaito_switch_player_action_v1(
    states,
    runtime,
    bank,
    carry: X562KaitoSwitchCarryV1,
    enable_switch: jax.Array,
    switch_step: jax.Array,
    player: int,
):
    """Capability probe: switch from stable X562 execution to Kaito V36.

    Kaito's state is held at its initializer until the first selected action,
    avoiding fictitious debt or recovery state from actions that never ran.
    This is a probe of suffix recoverability, not a promoted policy.
    """

    batch = states.step.shape[0]
    x562_action, x562_next = x562_route_rule_player_action_v1(
        states,
        runtime,
        bank,
        carry.x562,
        jnp.full((batch,), 2, dtype=jnp.int8),
        jnp.zeros((batch,), dtype=jnp.bool_),
        player,
    )
    kaito_route = jnp.full((batch,), 11, dtype=jnp.int32)
    kaito_action, kaito_next = lp.kaito_v36_player_action_v1(
        states, runtime, bank, kaito_route, carry.kaito, player
    )
    switched = carry.switched | (
        enable_switch & (states.step.astype(jnp.int32) >= switch_step.astype(jnp.int32))
    )
    action = _select_action(switched, kaito_action, x562_action)
    x562_carry = _select_tree(~switched, x562_next, carry.x562)
    kaito_carry = _select_tree(switched, kaito_next, carry.kaito)
    return action, X562KaitoSwitchCarryV1(
        x562=x562_carry,
        kaito=kaito_carry,
        switched=switched,
    )


def x562_kaito_market_player_action_v1(
    states,
    runtime,
    bank,
    carry: X562KaitoMarketCarryV1,
    enable_preempt: jax.Array,
    enable_market_maker: jax.Array,
    player: int,
):
    """X562 production with independently ablatable Kaito market skills."""

    step = states.step.astype(jnp.int32)
    route_carry = carry.x562
    active = jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    has_yarn = jnp.any(active & (states.town_shops == 7), axis=1)
    decide = (~route_carry.decided) & (step >= 168)
    route_id = jnp.where(
        decide, jnp.where(has_yarn, 7, 12), route_carry.route_id
    ).astype(jnp.int8)
    decided = route_carry.decided | decide
    route = route_id.astype(jnp.int32)

    action, base = hp._raw_with_weed(states, bank, route, route_carry.base, player)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(action, base, step, hp.MODE_BOATLEE)
    action = hp._rank_sell_slots_exact(states, runtime, action)

    # Kaito's market carry has a different schema from X562's production
    # carry.  Keep the ledgers independent; only the current legal action is
    # shared with the market skill.
    kaito_seed = carry.kaito
    preempt_action, preempt_carry = lp._kaito_preempt(
        states, runtime, bank, route, action, kaito_seed, player
    )
    action = _select_action(enable_preempt, preempt_action, action)
    kaito = _select_tree(enable_preempt, preempt_carry, kaito_seed)

    maker_action, maker_carry = lp._kaito_market_maker(
        states, runtime, bank, route, action, kaito, player
    )
    action = _select_action(enable_market_maker, maker_action, action)
    kaito = _select_tree(enable_market_maker, maker_carry, kaito)

    action = lp._room_guard(states, action, player)
    action = lp._x562_seed_budget_guard(states, action, player)
    action = lp._terminal_liquidation(states, action, player)
    action = lp._x562_idle_fertilizer_sale(states, action, player)
    action = lp._trim_ray_seeds(states, runtime, route, action, player)
    return action, X562KaitoMarketCarryV1(
        x562=X562RouteRuleCarryV1(base=base, decided=decided, route_id=route_id),
        kaito=kaito,
    )


def x562_old_suffix_player_action_v1(
    states,
    runtime,
    latest_bank,
    old_bank,
    carry: X562OldSuffixCarryV1,
    enable_switch: jax.Array,
    switch_step: jax.Array,
    suffix_route: jax.Array,
    player: int,
):
    """Discovery probe for legally executable task suffixes in the old bank."""

    batch = states.step.shape[0]
    prefix_action, prefix_next = x562_route_rule_player_action_v1(
        states,
        runtime,
        latest_bank,
        carry.x562,
        jnp.full((batch,), 2, dtype=jnp.int8),
        jnp.zeros((batch,), dtype=jnp.bool_),
        player,
    )
    route = suffix_route.astype(jnp.int32)
    suffix_action, suffix_next = hp._raw_with_weed(
        states, old_bank, route, carry.suffix, player
    )
    suffix_action, suffix_next = hp._room_evac(
        states, suffix_action, suffix_next, player
    )
    suffix_action, suffix_next = hp._repay(
        suffix_action,
        suffix_next,
        states.step.astype(jnp.int32),
        hp.MODE_BOATLEE,
    )
    suffix_action = hp._rank_sell_slots_exact(states, runtime, suffix_action)
    suffix_action = lp._room_guard(states, suffix_action, player)
    suffix_action = lp._x562_seed_budget_guard(states, suffix_action, player)
    suffix_action = lp._terminal_liquidation(states, suffix_action, player)
    suffix_action = lp._x562_idle_fertilizer_sale(states, suffix_action, player)
    switched = carry.switched | (
        enable_switch
        & (states.step.astype(jnp.int32) >= switch_step.astype(jnp.int32))
    )
    action = _select_action(switched, suffix_action, prefix_action)
    return action, X562OldSuffixCarryV1(
        x562=_select_tree(~switched, prefix_next, carry.x562),
        suffix=_select_tree(switched, suffix_next, carry.suffix),
        switched=switched,
    )


def k320_old_suffix_player_action_v1(
    states,
    tables,
    runtime,
    latest_bank,
    old_bank,
    carry: K320OldSuffixCarryV1,
    enable_switch: jax.Array,
    switch_step: jax.Array,
    suffix_route: jax.Array,
    player: int,
):
    """Discovery probe for old task suffixes executable from a K320 prefix."""

    prefix_action, prefix_next = hp.high_potential_v20_player_action_v1(
        states,
        tables,
        latest_bank,
        runtime,
        carry.k320,
        player,
        hp.MODE_RAY_K320,
    )
    route = suffix_route.astype(jnp.int32)
    suffix_action, suffix_next = hp._raw_with_weed(
        states, old_bank, route, carry.suffix, player
    )
    suffix_action, suffix_next = hp._room_evac(
        states, suffix_action, suffix_next, player
    )
    suffix_action, suffix_next = hp._repay(
        suffix_action,
        suffix_next,
        states.step.astype(jnp.int32),
        hp.MODE_BOATLEE,
    )
    suffix_action = hp._rank_sell_slots_exact(states, runtime, suffix_action)
    suffix_action = lp._room_guard(states, suffix_action, player)
    suffix_action = lp._x562_seed_budget_guard(states, suffix_action, player)
    suffix_action = lp._terminal_liquidation(states, suffix_action, player)
    suffix_action = lp._x562_idle_fertilizer_sale(states, suffix_action, player)
    switched = carry.switched | (
        enable_switch
        & (states.step.astype(jnp.int32) >= switch_step.astype(jnp.int32))
    )
    action = _select_action(switched, suffix_action, prefix_action)
    return action, K320OldSuffixCarryV1(
        k320=_select_tree(~switched, prefix_next, carry.k320),
        suffix=_select_tree(switched, suffix_next, carry.suffix),
        switched=switched,
    )


def x562_prt_suffix_rule_player_action_v1(
    states,
    runtime,
    latest_bank,
    old_bank,
    carry: X562OldSuffixCarryV1,
    enable_selector: jax.Array,
    player: int,
):
    """Deploy the frozen step-288 baseline-vs-PRT suffix decision tree.

    The selector was chosen by grouped-by-seed OOF score, then passed two
    untouched event banks.  It uses only own-private and mutually public state
    available at action step 288.  Route 81 remains a discovery-bank suffix;
    this function is therefore an integration candidate, not yet a packaged
    public submission policy.
    """

    batch = states.step.shape[0]
    step = states.step.astype(jnp.int32)
    active_shops = (
        jnp.arange(states.town_shops.shape[1])[None, :]
        < states.town_count[:, None]
    )
    shop_7_count = jnp.sum(
        active_shops & (states.town_shops == 7), axis=1
    )
    # Exact deployment rule for the selected tree at probability threshold
    # 0.8.  Only leaf 3 exceeds that threshold; the fitted right branch is
    # therefore absent from the deployed rule.
    choose_suffix = (
        (states.money[:, player] <= 13376)
        & (shop_7_count <= 1)
        & (states.market_inventory[:, 3] <= 9979)
    )
    decide_and_switch = enable_selector & (step == 288) & choose_suffix
    switched = carry.switched | decide_and_switch

    prefix_action, prefix_next = x562_route_rule_player_action_v1(
        states,
        runtime,
        latest_bank,
        carry.x562,
        jnp.full((batch,), 2, dtype=jnp.int8),
        jnp.zeros((batch,), dtype=jnp.bool_),
        player,
    )
    route = jnp.full((batch,), 81, dtype=jnp.int32)
    suffix_action, suffix_next = hp._raw_with_weed(
        states, old_bank, route, carry.suffix, player
    )
    suffix_action, suffix_next = hp._room_evac(
        states, suffix_action, suffix_next, player
    )
    suffix_action, suffix_next = hp._repay(
        suffix_action,
        suffix_next,
        step,
        hp.MODE_BOATLEE,
    )
    suffix_action = hp._rank_sell_slots_exact(states, runtime, suffix_action)
    suffix_action = lp._room_guard(states, suffix_action, player)
    suffix_action = lp._x562_seed_budget_guard(states, suffix_action, player)
    suffix_action = lp._terminal_liquidation(states, suffix_action, player)
    suffix_action = lp._x562_idle_fertilizer_sale(states, suffix_action, player)
    action = _select_action(switched, suffix_action, prefix_action)
    return action, X562OldSuffixCarryV1(
        x562=_select_tree(~switched, prefix_next, carry.x562),
        suffix=_select_tree(switched, suffix_next, carry.suffix),
        switched=switched,
    )


def _family_schedule_targets(
    schedule: OpponentFamilyMarketScheduleV1,
    future_step: jax.Array,
    family: jax.Array,
) -> jax.Array:
    cow_heavy = hp._schedule_targets(
        schedule.cow_heavy_op,
        schedule.cow_heavy_item,
        schedule.cow_heavy_amount,
        schedule.cow_heavy_count,
        future_step,
    )
    balanced = hp._schedule_targets(
        schedule.balanced_op,
        schedule.balanced_item,
        schedule.balanced_amount,
        schedule.balanced_count,
        future_step,
    )
    # Before the public board distinguishes the suffix, protect against both
    # possible sale schedules.  Once it does, use only the observed family.
    unresolved = jnp.maximum(cow_heavy, balanced)
    return jnp.where(
        (family == 0)[:, None],
        cow_heavy,
        jnp.where((family == 1)[:, None], balanced, unresolved),
    )


def _rank14_family_market_counter(
    states,
    action,
    schedule: OpponentFamilyMarketScheduleV1,
    pressure_target: jax.Array,
    family: jax.Array,
    lead_steps: jax.Array,
    quantity_multiplier: jax.Array,
    quantity_cap: jax.Array,
    product_group: jax.Array,
    minimum_price_percent: jax.Array,
    require_clean_town: jax.Array,
    player: int,
):
    """Front-run a visible heavy-sheep family without opponent identity.

    The schedule is only a morphology-conditioned forecast.  Every actual
    sale is still gated by current public price, current own inventory, pickup
    reservations, market slots, and the observed opponent farm shape.
    """

    step = states.step.astype(jnp.int32)
    future_step = jnp.clip(step + lead_steps.astype(jnp.int32), 0, 718)
    targets = _family_schedule_targets(schedule, future_step, family)
    planned = hp._planned_sales(action)
    reserve = hp._pickup_reserve(action)
    shed = states.shed[:, player, : hp.NUM_PRODUCTS].astype(jnp.int32)
    available = jnp.maximum(shed - planned - reserve, 0)

    strawberry = PRODUCTS.index("STRAWBERRY")
    melon = PRODUCTS.index("MELON")
    milk = PRODUCTS.index("MILK")
    wool = PRODUCTS.index("WOOL")
    for product in hp._COUNTER_ITEMS.tolist():
        in_group = jnp.select(
            [
                product_group == 0,
                product_group == 1,
                product_group == 2,
                product_group == 3,
                product_group == 4,
            ],
            [
                jnp.asarray(product == milk),
                jnp.asarray(product == wool),
                jnp.asarray((product == milk) or (product == wool)),
                jnp.asarray(
                    (product == strawberry)
                    or (product == milk)
                    or (product == wool)
                ),
                jnp.asarray(
                    product in (melon, milk, strawberry, wool)
                ),
            ],
            default=jnp.asarray(False),
        )
        target = targets[:, product]
        requested = jnp.minimum(
            target * quantity_multiplier.astype(jnp.int32),
            quantity_cap.astype(jnp.int32),
        )
        quantity = jnp.minimum(available[:, product], requested)
        clean = (
            (hp._town_demand_at(states, product, step) == 0)
            & (hp._town_demand_at(states, product, future_step) == 0)
        )
        price_ok = states.market_price[:, product] >= (
            lp._BASE_PRICE[product] * minimum_price_percent + 99
        ) // 100
        enabled = (
            pressure_target
            & in_group
            & (step >= 120)
            & (step + lead_steps.astype(jnp.int32) < 719)
            & (target > 0)
            & (quantity > 0)
            & price_ok
            & ((~require_clean_town) | clean)
        )
        action, changed = hp._append_or_merge_sale(
            action, enabled, product, quantity
        )
        used = jnp.where(changed, quantity, 0)
        available = available.at[:, product].add(-used)
        planned = planned.at[:, product].add(used)
    return action


def x562_family_counter_player_action_v1(
    states,
    runtime,
    bank,
    schedule: OpponentFamilyMarketScheduleV1,
    carry: X562FamilyCounterCarryV1,
    enabled: jax.Array,
    lead_steps: jax.Array,
    quantity_multiplier: jax.Array,
    quantity_cap: jax.Array,
    product_group: jax.Array,
    minimum_price_percent: jax.Array,
    require_clean_town: jax.Array,
    player: int,
):
    """Stable X562 12/7 route plus a morphology-conditioned market counter."""

    step = states.step.astype(jnp.int32)
    route_carry = carry.route
    active = jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    has_yarn = jnp.any(active & (states.town_shops == 7), axis=1)
    decide = (~route_carry.decided) & (step >= 168)
    route_id = jnp.where(
        decide, jnp.where(has_yarn, 7, 12), route_carry.route_id
    ).astype(jnp.int8)
    decided = route_carry.decided | decide
    route = route_id.astype(jnp.int32)

    pressure_target = carry.pressure_target | _visible_opening_sheep_pressure(
        states, player
    )
    _, _, rival_cows, rival_sheep, _ = hp._opponent_counts(states, player)
    observed_family = jnp.where(
        rival_sheep >= 6,
        1,
        jnp.where(rival_cows >= 8, 0, carry.opponent_family),
    ).astype(jnp.int8)

    action, base = hp._raw_with_weed(states, bank, route, route_carry.base, player)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(action, base, step, hp.MODE_BOATLEE)
    action = hp._rank_sell_slots_exact(states, runtime, action)
    counter_action = _rank14_family_market_counter(
        states,
        action,
        schedule,
        pressure_target,
        observed_family,
        lead_steps,
        quantity_multiplier,
        quantity_cap,
        product_group,
        minimum_price_percent,
        require_clean_town,
        player,
    )
    action = _select_action(enabled, counter_action, action)
    action = lp._room_guard(states, action, player)
    action = lp._x562_seed_budget_guard(states, action, player)
    action = lp._terminal_liquidation(states, action, player)
    action = lp._x562_idle_fertilizer_sale(states, action, player)
    action = lp._trim_ray_seeds(states, runtime, route, action, player)
    return action, X562FamilyCounterCarryV1(
        route=X562RouteRuleCarryV1(
            base=base,
            decided=decided,
            route_id=route_id,
        ),
        pressure_target=pressure_target,
        opponent_family=observed_family,
    )


def k320_heavy_sheep_counter_player_action_v1(
    states,
    tables,
    runtime,
    bank,
    schedule: OpponentFamilyMarketScheduleV1,
    carry: K320HeavySheepCounterCarryV1,
    enabled: jax.Array,
    lead_steps: jax.Array,
    quantity_multiplier: jax.Array,
    quantity_cap: jax.Array,
    product_group: jax.Array,
    minimum_price_percent: jax.Array,
    require_clean_town: jax.Array,
    player: int,
):
    """Keep K320 production; front-run a visible heavy-sheep sell schedule.

    The detector is morphology based and becomes sticky only after the rival
    has publicly committed at least seven sheep with no more than six cows.
    It never receives an opponent ID.  The two frozen schedules are merely
    offline prototypes for high-volume and compact heavy-sheep businesses.
    """

    step = states.step.astype(jnp.int32)
    _, _, rival_cows, rival_sheep, _ = hp._opponent_counts(states, player)
    detect = (
        (step >= 144)
        & (rival_sheep >= 7)
        & (rival_cows <= 6)
    )
    pressure_target = carry.pressure_target | detect
    observed_family = jnp.where(
        pressure_target & ((rival_sheep >= 10) | (rival_cows >= 5)),
        0,
        jnp.where(pressure_target, 1, carry.opponent_family),
    ).astype(jnp.int8)

    action, base = k320_clone_aware_preempt_player_action_v1(
        states, tables, runtime, bank, carry.base, player
    )
    counter_action = _rank14_family_market_counter(
        states,
        action,
        schedule,
        pressure_target,
        observed_family,
        lead_steps,
        quantity_multiplier,
        quantity_cap,
        product_group,
        minimum_price_percent,
        require_clean_town,
        player,
    )
    action = _select_action(enabled, counter_action, action)
    return action, K320HeavySheepCounterCarryV1(
        base=base,
        pressure_target=pressure_target,
        opponent_family=observed_family,
    )


def k320_heavy_sheep_animal_substitution_player_action_v1(
    states,
    tables,
    runtime,
    bank,
    carry: hp.HighPotentialV20CarryV1,
    mode: jax.Array,
    start_step: jax.Array,
    player: int,
):
    """Modify only future sheep purchases under visible route collision.

    Mode 0 is source.  Mode 1 drops matching sheep orders; mode 2 halves them;
    mode 3 converts them to the same number of cows; mode 4 converts to half
    as many cows.  The rule activates only for K320's two heavy-sheep shop
    programs after the rival has publicly retained at least three sheep and
    no more than six cows.  Existing animals, unit actions and layout are not
    rewritten.
    """

    action, base = k320_clone_aware_preempt_player_action_v1(
        states, tables, runtime, bank, carry, player
    )
    step = states.step.astype(jnp.int32)
    _, _, rival_cows, rival_sheep, _ = hp._opponent_counts(states, player)
    route = base.ray_route_id.astype(jnp.int32)
    collision = (
        (mode > 0)
        & (step >= start_step.astype(jnp.int32))
        & base.ray_route_locked
        & ((route == 3) | (route == 4))
        & (rival_sheep >= 3)
        & (rival_cows <= 6)
    )
    slots = jnp.arange(hp.MAX_MARKET_ORDERS)[None, :]
    active = slots < action.market_count[:, None]
    sheep_item = hp.NUM_PRODUCTS + 2
    cow_item = hp.NUM_PRODUCTS + 1
    matching = (
        collision[:, None]
        & active
        & (action.market_op == MarketOp.BUY_ANIMAL)
        & (action.market_item == sheep_item)
    )
    amount = action.market_amount.astype(jnp.int32)
    half = (jnp.maximum(amount, 0) + 1) // 2
    replacement_amount = jnp.where(
        (mode == 1)[:, None],
        0,
        jnp.where(
            ((mode == 2) | (mode == 4))[:, None],
            half,
            amount,
        ),
    )
    replacement_item = jnp.where(
        ((mode == 3) | (mode == 4))[:, None],
        cow_item,
        action.market_item,
    )
    action = action._replace(
        market_item=jnp.where(matching, replacement_item, action.market_item).astype(
            jnp.int8
        ),
        market_amount=jnp.where(matching, replacement_amount, amount).astype(
            jnp.int32
        ),
    )
    keep = active & ~(
        matching & (action.market_amount <= 0)
    )
    return hp._compact_market(action, keep), base


def _append_excess_product_sale(
    states,
    action,
    product: int,
    active: jax.Array,
    quantity_cap: jax.Array,
    minimum_price_percent: jax.Array,
    player: int,
):
    planned = hp._planned_sales(action)
    reserve = hp._pickup_reserve(action)
    projected = hp._projected_shed_without_pickups(states, action, player)
    available = jnp.maximum(
        projected[:, product] - planned[:, product] - reserve[:, product], 0
    )
    quantity = jnp.minimum(available, quantity_cap.astype(jnp.int32))
    price_ok = states.market_price[:, product] >= (
        lp._BASE_PRICE[product] * minimum_price_percent + 99
    ) // 100
    enabled = active & price_ok & (quantity > 0)
    action, _ = hp._append_or_merge_sale(action, enabled, product, quantity)
    return action


def x562_excess_sale_player_action_v1(
    states,
    runtime,
    bank,
    carry: X562FamilyCounterCarryV1,
    enabled: jax.Array,
    start_step: jax.Array,
    end_step: jax.Array,
    quantity_cap: jax.Array,
    minimum_price_percent: jax.Array,
    product_group: jax.Array,
    player: int,
):
    """Sell uncommitted excess output before a visible supply glut."""

    step = states.step.astype(jnp.int32)
    route_carry = carry.route
    active_shops = jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    has_yarn = jnp.any(active_shops & (states.town_shops == 7), axis=1)
    decide = (~route_carry.decided) & (step >= 168)
    route_id = jnp.where(
        decide, jnp.where(has_yarn, 7, 12), route_carry.route_id
    ).astype(jnp.int8)
    decided = route_carry.decided | decide
    route = route_id.astype(jnp.int32)
    pressure_target = carry.pressure_target | _visible_opening_sheep_pressure(
        states, player
    )
    _, _, rival_cows, rival_sheep, _ = hp._opponent_counts(states, player)
    observed_family = jnp.where(
        rival_sheep >= 6,
        1,
        jnp.where(rival_cows >= 8, 0, carry.opponent_family),
    ).astype(jnp.int8)

    action, base = hp._raw_with_weed(states, bank, route, route_carry.base, player)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(action, base, step, hp.MODE_BOATLEE)
    action = hp._rank_sell_slots_exact(states, runtime, action)
    sale_window = (
        enabled
        & pressure_target
        & (route_id == 12)
        & (step >= start_step.astype(jnp.int32))
        & (step <= end_step.astype(jnp.int32))
    )
    milk = PRODUCTS.index("MILK")
    wool = PRODUCTS.index("WOOL")
    action = _append_excess_product_sale(
        states,
        action,
        milk,
        sale_window & ((product_group == 0) | (product_group == 1)),
        quantity_cap,
        minimum_price_percent,
        player,
    )
    action = _append_excess_product_sale(
        states,
        action,
        wool,
        sale_window & (product_group == 1),
        quantity_cap,
        minimum_price_percent,
        player,
    )
    action = lp._room_guard(states, action, player)
    action = lp._x562_seed_budget_guard(states, action, player)
    action = lp._terminal_liquidation(states, action, player)
    action = lp._x562_idle_fertilizer_sale(states, action, player)
    action = lp._trim_ray_seeds(states, runtime, route, action, player)
    return action, X562FamilyCounterCarryV1(
        route=X562RouteRuleCarryV1(base=base, decided=decided, route_id=route_id),
        pressure_target=pressure_target,
        opponent_family=observed_family,
    )


def _parameterized_one_step_preempt_player(
    states,
    bank,
    route,
    action,
    carry: hp.HighPotentialV20CarryV1,
    distance_limit: jax.Array,
    quantity_cap: jax.Array,
    minimum_price_percent: jax.Array,
    player: int,
):
    step = states.step.astype(jnp.int32)
    eligible = (
        (step >= 120)
        & (step < 680)
        & (jnp.sum(carry.due.astype(jnp.int32), axis=1) == 0)
        & (lp._kaito_clone_distance(states) <= distance_limit)
        & (action.market_count < hp.MAX_MARKET_ORDERS)
        & (quantity_cap > 0)
    )
    future_step = jnp.clip(step + 1, 0, 718)
    future_active = jnp.arange(hp.MAX_MARKET_ORDERS)[None, :] < bank.market_count[
        route, future_step
    ][:, None]
    remaining = jnp.maximum(
        lp._projected_shed_without_pickups(states, action, player)[:, : hp.NUM_PRODUCTS]
        - lp._planned_sales(action),
        0,
    )
    shifted = jnp.zeros_like(carry.due, dtype=jnp.int32)
    for product in lp._PREMIUM.tolist():
        future_quantity = jnp.sum(
            jnp.where(
                future_active
                & (bank.market_op[route, future_step] == hp.MarketOp.SELL)
                & (bank.market_item[route, future_step] == product),
                jnp.maximum(bank.market_amount[route, future_step], 0),
                0,
            ),
            axis=1,
        )
        quantity = jnp.minimum(
            jnp.minimum(remaining[:, product], future_quantity), quantity_cap
        )
        append = (
            eligible
            & (future_quantity >= 4)
            & (
                states.market_price[:, product]
                >= (
                    lp._BASE_PRICE[product] * minimum_price_percent + 99
                )
                // 100
            )
            & (quantity > 0)
            & (action.market_count < hp.MAX_MARKET_ORDERS)
        )
        action = lp._append_market(
            action, append, hp.MarketOp.SELL, product, quantity
        )
        used = jnp.where(append, quantity, 0)
        remaining = remaining.at[:, product].add(-used)
        shifted = shifted.at[:, product].add(used)
    changed = jnp.sum(shifted, axis=1) > 0
    return action, carry._replace(
        due_step=jnp.where(changed, step + 1, carry.due_step).astype(jnp.int16),
        due=jnp.where(changed[:, None], shifted, carry.due).astype(jnp.int16),
    )


def _parameterized_cumulative_preempt_player(
    states,
    bank,
    route,
    action,
    carry: hp.HighPotentialV20CarryV1,
    horizon: jax.Array,
    distance_limit: jax.Array,
    quantity_cap: jax.Array,
    minimum_future_quantity: jax.Array,
    minimum_price_percent: jax.Array,
    start_step: jax.Array,
    product_mask: jax.Array,
    player: int,
):
    """Parameter search version of K320's cumulative premium preemption."""

    step = states.step.astype(jnp.int32)
    eligible = (
        (step >= start_step.astype(jnp.int32))
        & (step < 680)
        & (horizon > 0)
        & (jnp.sum(carry.due.astype(jnp.int32), axis=1) == 0)
        & (hp._clone_distance(states) <= distance_limit)
        & (action.market_count < hp.MAX_MARKET_ORDERS)
        & (quantity_cap > 0)
    )
    future = jnp.zeros((step.shape[0], hp.NUM_PRODUCTS), dtype=jnp.int32)
    for ahead in range(1, 5):
        future_step = jnp.clip(step + ahead, 0, 718)
        active = jnp.arange(hp.MAX_MARKET_ORDERS)[None, :] < bank.market_count[
            route, future_step
        ][:, None]
        use = (ahead <= horizon) & (step + ahead < 719)
        for product in lp._PREMIUM.tolist():
            quantity = jnp.sum(
                jnp.where(
                    active
                    & (bank.market_op[route, future_step] == hp.MarketOp.SELL)
                    & (bank.market_item[route, future_step] == product),
                    jnp.maximum(bank.market_amount[route, future_step], 0),
                    0,
                ),
                axis=1,
            )
            future = future.at[:, product].add(jnp.where(use, quantity, 0))

    remaining = jnp.maximum(
        lp._projected_shed_without_pickups(states, action, player)[:, : hp.NUM_PRODUCTS]
        - lp._planned_sales(action),
        0,
    )
    shifted = jnp.zeros_like(carry.due, dtype=jnp.int32)
    for product in lp._PREMIUM.tolist():
        future_quantity = future[:, product]
        quantity = jnp.minimum(
            jnp.minimum(remaining[:, product], future_quantity), quantity_cap
        )
        append = (
            eligible
            & ((product_mask & (1 << product)) != 0)
            & (future_quantity >= minimum_future_quantity)
            & (
                states.market_price[:, product]
                >= (lp._BASE_PRICE[product] * minimum_price_percent + 99) // 100
            )
            & (quantity > 0)
            & (action.market_count < hp.MAX_MARKET_ORDERS)
        )
        action = lp._append_market(
            action, append, hp.MarketOp.SELL, product, quantity
        )
        used = jnp.where(append, quantity, 0)
        remaining = remaining.at[:, product].add(-used)
        shifted = shifted.at[:, product].add(used)
    changed = jnp.sum(shifted, axis=1) > 0
    return action, carry._replace(
        due_step=jnp.where(changed, step + horizon, carry.due_step).astype(jnp.int16),
        due=jnp.where(changed[:, None], shifted, carry.due).astype(jnp.int16),
    )


def k320_preempt_parameter_player_action_v1(
    states,
    tables,
    runtime,
    bank,
    carry: hp.HighPotentialV20CarryV1,
    source_preempt: jax.Array,
    horizon: jax.Array,
    distance_limit: jax.Array,
    quantity_cap: jax.Array,
    minimum_future_quantity: jax.Array,
    minimum_price_percent: jax.Array,
    start_step: jax.Array,
    product_mask: jax.Array,
    player: int,
):
    """Exact K320 except for one ablatable premium-sale timing layer."""

    route, base = hp._select_route(states, carry, player, hp.MODE_RAY_K320)
    action, base = hp._raw_with_weed(states, bank, route, base, player)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(
        action, base, states.step.astype(jnp.int32), hp.MODE_RAY_K320
    )
    action = hp._rank_sell_slots_exact(states, runtime, action)
    source_action, source_base = hp._preempt(
        states, bank, route, action, base, player, hp.MODE_RAY_K320
    )
    custom_action, custom_base = _parameterized_cumulative_preempt_player(
        states,
        bank,
        route,
        action,
        base,
        horizon,
        distance_limit,
        quantity_cap,
        minimum_future_quantity,
        minimum_price_percent,
        start_step,
        product_mask,
        player,
    )
    action = _select_action(source_preempt, source_action, custom_action)
    base = _select_tree(source_preempt, source_base, custom_base)
    action, base = hp._market_counters(states, action, base, runtime, player)
    action = hp._room_guard(states, action, player)
    action = hp._terminal_liquidation(states, action, player)
    action = hp._trim_ray_seeds(states, runtime, route, action, player)
    return action, base


def k320_clone_aware_preempt_player_action_v1(
    states,
    tables,
    runtime,
    bank,
    carry: hp.HighPotentialV20CarryV1,
    player: int,
):
    """Use aggressive mirror timing only while the public farms remain close.

    The exact-clone branch preserves the previously validated K320 anti-mirror
    parameters.  Once visible crops/animals/layout diverge, the smaller broad
    four-step hedge is enabled instead.  No opponent identity or future random
    event is used.
    """

    batch = states.step.shape[0]
    # A distance of six still groups Rank12's already-diverging production
    # path with a true mirror and suppresses the useful broad hedge.  Preserve
    # the mirror arm only for an exactly matching public farm layout.
    close_clone = hp._clone_distance(states) == 0
    return k320_preempt_parameter_player_action_v1(
        states,
        tables,
        runtime,
        bank,
        carry,
        jnp.zeros((batch,), dtype=jnp.bool_),
        jnp.full((batch,), 4, dtype=jnp.int16),
        jnp.where(close_clone, 6, 100).astype(jnp.int16),
        jnp.where(close_clone, 32, 12).astype(jnp.int16),
        jnp.full((batch,), 4, dtype=jnp.int16),
        jnp.zeros((batch,), dtype=jnp.int16),
        jnp.full((batch,), 120, dtype=jnp.int16),
        jnp.where(close_clone, 200, 216).astype(jnp.int16),
        player,
    )


def _raw_with_weed_idle_repay(
    states,
    bank,
    route: jax.Array,
    carry: hp.HighPotentialV20CarryV1,
    player: int,
):
    """Repair a blocked unit tape without discarding a later meaningful action.

    The frozen RC5 overlay shifts only eight subsequent unit actions and then
    resumes the absolute tape, which necessarily drops one scheduled action.
    This experimental overlay instead keeps a one-action debt until the raw
    tape exposes a PASS slot for that actor.  The PASS absorbs the delay, so
    every non-PASS action between the weed and the repayment point is retained.
    Market orders remain on their original absolute steps.
    """

    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    previous = jnp.clip(step - 1, 0, 718)
    count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    present = jnp.arange(hp.MAX_UNITS)[None, :] < count[:, None]
    source_op = jnp.where(present, bank.unit_op[route, step], hp.UnitOp.PASS).astype(
        jnp.int8
    )
    source_item = jnp.where(present, bank.unit_item[route, step], -1).astype(jnp.int8)
    source_amount = jnp.where(present, bank.unit_amount[route, step], 1).astype(
        jnp.int32
    )
    age = step[:, None] - carry.weed_start.astype(jnp.int32)
    existing = carry.weed_active & present
    use_intended = existing & (age == 1)
    use_replay = existing & (age >= 2)
    previous_present = (
        jnp.arange(hp.MAX_UNITS)[None, :]
        < bank.unit_count[route, previous][:, None].astype(jnp.int32)
    )
    replay_op = jnp.where(
        previous_present, bank.unit_op[route, previous], hp.UnitOp.PASS
    ).astype(jnp.int8)
    replay_item = jnp.where(previous_present, bank.unit_item[route, previous], -1).astype(
        jnp.int8
    )
    replay_amount = jnp.where(
        previous_present, bank.unit_amount[route, previous], 1
    ).astype(jnp.int32)
    op = jnp.where(
        use_intended,
        carry.weed_intended_op,
        jnp.where(use_replay, replay_op, source_op),
    ).astype(jnp.int8)
    item = jnp.where(
        use_intended,
        carry.weed_intended_item,
        jnp.where(use_replay, replay_item, source_item),
    ).astype(jnp.int8)
    amount = jnp.where(
        use_intended,
        carry.weed_intended_amount,
        jnp.where(use_replay, replay_amount, source_amount),
    ).astype(jnp.int32)

    # A raw PASS is the first lossless opportunity to absorb the one-step lag.
    repay = existing & (age >= 1) & (source_op == hp.UnitOp.PASS)
    active_after_repay = existing & (~repay)
    trigger = (
        (~existing)
        & present
        & ((op == hp.UnitOp.BUILD_PASTURE) | (op == hp.UnitOp.PLANT))
        & (hp._tile_under_units(states, player) == hp.TileKind.WEED)
    )
    intended_op, intended_item, intended_amount = op, item, amount
    op = jnp.where(trigger, hp.UnitOp.DIG, op).astype(jnp.int8)
    item = jnp.where(trigger, -1, item).astype(jnp.int8)
    amount = jnp.where(trigger, 1, amount).astype(jnp.int32)
    carry = carry._replace(
        weed_active=active_after_repay | trigger,
        weed_start=jnp.where(trigger, step[:, None], carry.weed_start).astype(jnp.int16),
        weed_intended_op=jnp.where(
            trigger, intended_op, carry.weed_intended_op
        ).astype(jnp.int8),
        weed_intended_item=jnp.where(
            trigger, intended_item, carry.weed_intended_item
        ).astype(jnp.int8),
        weed_intended_amount=jnp.where(
            trigger, intended_amount, carry.weed_intended_amount
        ).astype(jnp.int32),
    )
    return hp.Action(
        unit_op=op,
        unit_item=item,
        unit_amount=amount,
        unit_count=count,
        market_op=bank.market_op[route, step],
        market_item=bank.market_item[route, step],
        market_amount=bank.market_amount[route, step],
        market_count=bank.market_count[route, step],
    ), carry


def k320_clone_aware_weed_recovery_player_action_v1(
    states,
    tables,
    runtime,
    bank,
    carry: hp.HighPotentialV20CarryV1,
    use_idle_repay: jax.Array,
    player: int,
):
    """Clone-aware K320 with an independently ablatable weed-recovery layer."""

    batch = states.step.shape[0]
    route, base = hp._select_route(states, carry, player, hp.MODE_RAY_K320)
    source_action, source_base = hp._raw_with_weed(states, bank, route, base, player)
    idle_action, idle_base = _raw_with_weed_idle_repay(
        states, bank, route, base, player
    )
    action = _select_action(use_idle_repay, idle_action, source_action)
    base = _select_tree(use_idle_repay, idle_base, source_base)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(
        action, base, states.step.astype(jnp.int32), hp.MODE_RAY_K320
    )
    action = hp._rank_sell_slots_exact(states, runtime, action)
    close_clone = hp._clone_distance(states) == 0
    action, base = _parameterized_cumulative_preempt_player(
        states,
        bank,
        route,
        action,
        base,
        jnp.full((batch,), 4, dtype=jnp.int16),
        jnp.where(close_clone, 6, 100).astype(jnp.int16),
        jnp.where(close_clone, 32, 12).astype(jnp.int16),
        jnp.full((batch,), 4, dtype=jnp.int16),
        jnp.zeros((batch,), dtype=jnp.int16),
        jnp.full((batch,), 120, dtype=jnp.int16),
        jnp.where(close_clone, 200, 216).astype(jnp.int16),
        player,
    )
    action, base = hp._market_counters(states, action, base, runtime, player)
    action = hp._room_guard(states, action, player)
    action = hp._terminal_liquidation(states, action, player)
    action = hp._trim_ray_seeds(states, runtime, route, action, player)
    return action, base


def _public_opponent_product_exposure(states, player: int) -> jax.Array:
    """Cheap sell-side pressure estimate using only the rival public farm."""

    opponent = 1 - player
    crops = states.tile_crop[:, opponent]
    animals = states.tile_animal[:, opponent]
    tile_yield = jnp.maximum(states.tile_yield[:, opponent], 0).astype(jnp.float32)
    exposure = jnp.zeros(
        (states.step.shape[0], hp.NUM_PRODUCTS), dtype=jnp.float32
    )
    for crop in range(5):
        exposure = exposure.at[:, crop].set(
            jnp.sum(
                jnp.where(
                    crops == crop,
                    1.0 + tile_yield,
                    0.0,
                ),
                axis=(1, 2),
            )
        )
    for animal, product in ((0, 5), (1, 6), (2, 7)):
        exposure = exposure.at[:, product].set(
            jnp.sum(
                jnp.where(
                    animals == animal,
                    1.0 + tile_yield,
                    0.0,
                ),
                axis=(1, 2),
            )
        )
    fertilizer = (
        states.tile_flags[:, opponent]
        & jnp.uint8(hp.FLAG_FERTILIZER_AVAILABLE)
    ) != 0
    return exposure.at[:, 8].set(
        jnp.sum(fertilizer, axis=(1, 2), dtype=jnp.float32)
    )


def _terminal_priority_liquidation(
    states,
    action,
    mode: jax.Array,
    player: int,
):
    """Replace only step-718 sales with an ablatable public-state ordering.

    Modes are deliberately small and interpretable:
    0 keeps the source order; 1 ranks current liquidation value; 2 adds rival
    public supply pressure; 3 prioritizes pressure itself; 4 prioritizes the
    current quote.  All modes sell the same projected shed quantities.
    """

    projected = hp._projected_shed_without_pickups(states, action, player)
    quantities = projected[:, : hp.NUM_PRODUCTS].astype(jnp.int32)
    prices = states.market_price[:, : hp.NUM_PRODUCTS].astype(jnp.float32)
    exposure = _public_opponent_product_exposure(states, player)
    value_score = prices * quantities.astype(jnp.float32)
    pressure_value_score = (
        (1.0 + exposure) * prices * jnp.log1p(quantities.astype(jnp.float32))
    )
    pressure_score = (1.0 + exposure) * prices
    quote_score = prices
    score = jnp.where(
        (mode == 1)[:, None],
        value_score,
        jnp.where(
            (mode == 2)[:, None],
            pressure_value_score,
            jnp.where((mode == 3)[:, None], pressure_score, quote_score),
        ),
    )
    valid = quantities > 0
    order = jnp.argsort(jnp.where(valid, -score, jnp.inf), axis=1, stable=True)
    sorted_valid = jnp.take_along_axis(valid, order, axis=1)
    sorted_item = order.astype(jnp.int8)
    sorted_quantity = jnp.take_along_axis(quantities, order, axis=1)
    pad = hp.MAX_MARKET_ORDERS - hp.NUM_PRODUCTS
    market_item = jnp.pad(sorted_item, ((0, 0), (0, pad)), constant_values=-1)
    market_amount = jnp.pad(sorted_quantity, ((0, 0), (0, pad)))
    count = jnp.sum(sorted_valid, axis=1).astype(jnp.int8)
    present = jnp.arange(hp.MAX_MARKET_ORDERS)[None, :] < count[:, None]
    replacement = action._replace(
        market_op=jnp.where(present, MarketOp.SELL, MarketOp.NONE).astype(jnp.int8),
        market_item=jnp.where(present, market_item, -1).astype(jnp.int8),
        market_amount=jnp.where(present, market_amount, 0).astype(jnp.int32),
        market_count=count,
    )
    use = (states.step.astype(jnp.int32) == 718) & (mode > 0)
    return _select_action(use, replacement, action)


def k320_terminal_priority_player_action_v1(
    states,
    tables,
    runtime,
    bank,
    carry: hp.HighPotentialV20CarryV1,
    terminal_mode: jax.Array,
    player: int,
):
    """Exact-clone-aware K320 plus one terminal-ordering experiment."""

    action, carry = k320_clone_aware_preempt_player_action_v1(
        states, tables, runtime, bank, carry, player
    )
    return _terminal_priority_liquidation(
        states, action, terminal_mode, player
    ), carry


def x562_stable_route_preempt_player_action_v1(
    states,
    runtime,
    bank,
    carry: X562RouteRuleCarryV1,
    distance_limit: jax.Array,
    quantity_cap: jax.Array,
    minimum_price_percent: jax.Array,
    player: int,
):
    """Stable 12/7 shop route, no counters, parameterized one-step preempt."""

    step = states.step.astype(jnp.int32)
    active = jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    has_yarn = jnp.any(active & (states.town_shops == 7), axis=1)
    decide = (~carry.decided) & (step >= 168)
    route_id = jnp.where(decide, jnp.where(has_yarn, 7, 12), carry.route_id).astype(jnp.int8)
    decided = carry.decided | decide
    route = route_id.astype(jnp.int32)

    action, base = hp._raw_with_weed(states, bank, route, carry.base, player)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(action, base, step, hp.MODE_BOATLEE)
    action = hp._rank_sell_slots_exact(states, runtime, action)
    action, base = _parameterized_one_step_preempt_player(
        states,
        bank,
        route,
        action,
        base,
        distance_limit,
        quantity_cap,
        minimum_price_percent,
        player,
    )
    action = lp._room_guard(states, action, player)
    action = lp._x562_seed_budget_guard(states, action, player)
    action = lp._terminal_liquidation(states, action, player)
    action = lp._x562_idle_fertilizer_sale(states, action, player)
    action = lp._trim_ray_seeds(states, runtime, route, action, player)
    return action, X562RouteRuleCarryV1(base=base, decided=decided, route_id=route_id)


def _cap_animal_purchases(
    states,
    action,
    cow_cap: jax.Array,
    sheep_cap: jax.Array,
    enabled: jax.Array,
    player: int,
):
    """Legally trim future animal buys while preserving order-slot semantics."""

    cows = jnp.sum(states.tile_animal[:, player] == 1, axis=(1, 2)).astype(jnp.int32)
    sheep = jnp.sum(states.tile_animal[:, player] == 2, axis=(1, 2)).astype(jnp.int32)
    op, item, amount = action.market_op, action.market_item, action.market_amount
    active = jnp.arange(hp.MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    for slot in range(hp.MAX_MARKET_ORDERS):
        is_cow = active[:, slot] & (op[:, slot] == hp.MarketOp.BUY_ANIMAL) & (
            item[:, slot] == hp.NUM_PRODUCTS + 1
        )
        is_sheep = active[:, slot] & (op[:, slot] == hp.MarketOp.BUY_ANIMAL) & (
            item[:, slot] == hp.NUM_PRODUCTS + 2
        )
        requested = jnp.maximum(amount[:, slot], 0).astype(jnp.int32)
        allowed_cow = jnp.minimum(requested, jnp.maximum(cow_cap - cows, 0))
        allowed_sheep = jnp.minimum(requested, jnp.maximum(sheep_cap - sheep, 0))
        trim_cow = enabled & is_cow
        trim_sheep = enabled & is_sheep
        allowed = jnp.where(trim_cow, allowed_cow, jnp.where(trim_sheep, allowed_sheep, requested))
        cancel = (trim_cow | trim_sheep) & (allowed <= 0)
        op = op.at[:, slot].set(jnp.where(cancel, hp.MarketOp.NONE, op[:, slot]).astype(jnp.int8))
        item = item.at[:, slot].set(jnp.where(cancel, -1, item[:, slot]).astype(jnp.int8))
        amount = amount.at[:, slot].set(
            jnp.where(trim_cow | trim_sheep, allowed, amount[:, slot]).astype(jnp.int32)
        )
        cows = cows + jnp.where(trim_cow, allowed, 0)
        sheep = sheep + jnp.where(trim_sheep, allowed, 0)
    return action._replace(market_op=op, market_item=item, market_amount=amount)


def x562_stable_route_animal_cap_player_action_v1(
    states,
    runtime,
    bank,
    carry: X562RouteRuleCarryV1,
    cow_cap: jax.Array,
    sheep_cap: jax.Array,
    player: int,
):
    """Stable 12/7 shop route with no-yarn animal investment caps."""

    step = states.step.astype(jnp.int32)
    active = jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    has_yarn = jnp.any(active & (states.town_shops == 7), axis=1)
    decide = (~carry.decided) & (step >= 168)
    route_id = jnp.where(decide, jnp.where(has_yarn, 7, 12), carry.route_id).astype(jnp.int8)
    decided = carry.decided | decide
    route = route_id.astype(jnp.int32)

    action, base = hp._raw_with_weed(states, bank, route, carry.base, player)
    action = _cap_animal_purchases(
        states,
        action,
        cow_cap,
        sheep_cap,
        decided & (route_id == 12),
        player,
    )
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(action, base, step, hp.MODE_BOATLEE)
    action = hp._rank_sell_slots_exact(states, runtime, action)
    action = lp._room_guard(states, action, player)
    action = lp._x562_seed_budget_guard(states, action, player)
    action = lp._terminal_liquidation(states, action, player)
    action = lp._x562_idle_fertilizer_sale(states, action, player)
    action = lp._trim_ray_seeds(states, runtime, route, action, player)
    return action, X562RouteRuleCarryV1(base=base, decided=decided, route_id=route_id)

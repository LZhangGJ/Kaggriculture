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

from kaggriculture_jax.constants import (
    CROP_FIRST_YIELD_DAY,
    PRODUCTS,
    SHOP_NAMES,
    TURNS_PER_DAY,
    MarketOp,
    UnitOp,
)
from route_playbook_v1.trace_core import (
    initialize_trace_player_carry_v1,
    skeleton_player_action_v1,
)
from strategic_v5 import high_potential_v20_gpu as hp
from strategic_v5 import latest_public6_20260822_gpu as latest6
from strategic_v5 import latest_public8_gpu as lp
from strategic_v5.boatlee_v16_gpu import BoatleePlayerCarryV1
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


class FusionChampionMoonMarketCarryV1(NamedTuple):
    """FC15 execution plus Moon's public market-race observer and debt ring."""

    base: FusionChampionCarryV3
    market: latest6.MoonV92CarryV1


def initialize_fusion_champion_moon_market_carry_v1(
    batch_size: int,
) -> FusionChampionMoonMarketCarryV1:
    return FusionChampionMoonMarketCarryV1(
        base=initialize_fusion_champion_carry_v3(batch_size),
        market=latest6.initialize_moon_v92_carry_v1(batch_size),
    )


class FusionChampionFeedValueCarryV1(NamedTuple):
    """FC21 plus an explicit ledger for intentionally skipped animal feed."""

    base: FusionChampionMoonMarketCarryV1
    wheat_credit: jax.Array


def initialize_fusion_champion_feed_value_carry_v1(
    batch_size: int,
) -> FusionChampionFeedValueCarryV1:
    return FusionChampionFeedValueCarryV1(
        base=initialize_fusion_champion_moon_market_carry_v1(batch_size),
        wheat_credit=jnp.zeros((batch_size,), dtype=jnp.int16),
    )


class FusionChampionTerminalSalvageCarryV1(NamedTuple):
    """FC22 plus one last-feasible crop-to-cash obligation."""

    base: FusionChampionFeedValueCarryV1
    active: jax.Array
    actor: jax.Array
    target: jax.Array
    product: jax.Array
    quantity: jax.Array


def initialize_fusion_champion_terminal_salvage_carry_v1(
    batch_size: int,
) -> FusionChampionTerminalSalvageCarryV1:
    return FusionChampionTerminalSalvageCarryV1(
        base=initialize_fusion_champion_feed_value_carry_v1(batch_size),
        active=jnp.zeros((batch_size,), dtype=jnp.bool_),
        actor=jnp.full((batch_size,), -1, dtype=jnp.int8),
        target=jnp.zeros((batch_size, 2), dtype=jnp.int16),
        product=jnp.full((batch_size,), -1, dtype=jnp.int8),
        quantity=jnp.zeros((batch_size,), dtype=jnp.int16),
    )


class FusionChampionMoonSuffixCarryV1(NamedTuple):
    """FC16 plus one continuously shadowed old-route task suffix."""

    base: FusionChampionMoonMarketCarryV1
    suffix: hp.HighPotentialV20CarryV1
    switched: jax.Array
    opening_match: jax.Array


def initialize_fusion_champion_moon_suffix_carry_v1(
    batch_size: int,
) -> FusionChampionMoonSuffixCarryV1:
    return FusionChampionMoonSuffixCarryV1(
        base=initialize_fusion_champion_moon_market_carry_v1(batch_size),
        suffix=hp.initialize_high_potential_v20_carry_v1(batch_size),
        switched=jnp.zeros((batch_size,), dtype=jnp.bool_),
        opening_match=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


class FusionChampionBaseSuffixCarryV1(NamedTuple):
    """FC15 plus one continuously shadowed old-route task suffix."""

    base: FusionChampionCarryV3
    suffix: hp.HighPotentialV20CarryV1
    switched: jax.Array
    opening_match: jax.Array


def initialize_fusion_champion_base_suffix_carry_v1(
    batch_size: int,
) -> FusionChampionBaseSuffixCarryV1:
    return FusionChampionBaseSuffixCarryV1(
        base=initialize_fusion_champion_carry_v3(batch_size),
        suffix=hp.initialize_high_potential_v20_carry_v1(batch_size),
        switched=jnp.zeros((batch_size,), dtype=jnp.bool_),
        opening_match=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


class FusionChampionKaitoMarketCarryV1(NamedTuple):
    """FC15 production plus an independently ablatable Kaito market ledger."""

    base: FusionChampionCarryV3
    market: KaitoV36CarryV1


def initialize_fusion_champion_kaito_market_carry_v1(
    batch_size: int,
) -> FusionChampionKaitoMarketCarryV1:
    return FusionChampionKaitoMarketCarryV1(
        base=initialize_fusion_champion_carry_v3(batch_size),
        market=initialize_kaito_v36_carry_v1(batch_size),
    )


class FusionChampionCarryV4(NamedTuple):
    """FC2B state plus one audited step-120 route decision."""

    k320: hp.HighPotentialV20CarryV1
    x562_suffix: X562OldSuffixCarryV1
    sheep_pressure: jax.Array
    rank12_route_checked: jax.Array
    rank12_route_rescue: jax.Array


def initialize_fusion_champion_carry_v4(batch_size: int) -> FusionChampionCarryV4:
    return FusionChampionCarryV4(
        k320=hp.initialize_high_potential_v20_carry_v1(batch_size),
        x562_suffix=initialize_x562_old_suffix_carry_v1(batch_size),
        sheep_pressure=jnp.zeros((batch_size,), dtype=jnp.bool_),
        rank12_route_checked=jnp.zeros((batch_size,), dtype=jnp.bool_),
        rank12_route_rescue=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


class FusionChampionCarryV5(NamedTuple):
    """FC2B plus one persistent, public-state Kobe project commitment."""

    base: FusionChampionCarryV3
    goose: BoatleePlayerCarryV1
    goose_checked: jax.Array
    goose_active: jax.Array


def initialize_fusion_champion_carry_v5(batch_size: int) -> FusionChampionCarryV5:
    return FusionChampionCarryV5(
        base=initialize_fusion_champion_carry_v3(batch_size),
        goose=initialize_trace_player_carry_v1(batch_size),
        goose_checked=jnp.zeros((batch_size,), dtype=jnp.bool_),
        goose_active=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


class FusionChampionCarryV6(NamedTuple):
    """FC2B plus a continuously updated old-route shadow executor."""

    base: FusionChampionCarryV3
    suffix: BoatleePlayerCarryV1
    switched: jax.Array


def initialize_fusion_champion_carry_v6(batch_size: int) -> FusionChampionCarryV6:
    return FusionChampionCarryV6(
        base=initialize_fusion_champion_carry_v3(batch_size),
        suffix=hp.initialize_high_potential_v20_carry_v1(batch_size),
        switched=jnp.zeros((batch_size,), dtype=jnp.bool_),
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


def fc12g_fc2b_mass_hire_weed_guard_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    player: int,
):
    """FC2B with a farmer-only weed debt barrier before mass HIRE.

    The production route, market policy and pressure branch are unchanged.
    Only the K320 farmer's weed-repair debt is cancelled when the route plans
    at least five HIRE orders within two steps.  This prevents an inserted DIG
    from changing deterministic hand spawn/index semantics.
    """

    batch = states.step.shape[0]
    k320_action, k320 = k320_clone_aware_weed_mass_hire_guard_player_action_v1(
        states,
        tables,
        runtime,
        latest_bank,
        carry.k320,
        jnp.ones((batch,), dtype=jnp.bool_),
        jnp.full((batch,), 2, dtype=jnp.int16),
        jnp.full((batch,), 5, dtype=jnp.int16),
        jnp.ones((batch,), dtype=jnp.bool_),
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


def fc14_fc12g_residual_premium_sale_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    player: int,
):
    """FC12G plus one audited near-mirror premium residual-sale layer."""

    batch = states.step.shape[0]
    premium_mask = sum(
        1 << PRODUCTS.index(item) for item in ("STRAWBERRY", "MILK", "WOOL")
    )
    k320_action, k320 = k320_mass_hire_guard_residual_sale_player_action_v1(
        states,
        tables,
        runtime,
        latest_bank,
        carry.k320,
        jnp.ones((batch,), dtype=jnp.bool_),
        jnp.full((batch,), 120, dtype=jnp.int16),
        jnp.full((batch,), 6, dtype=jnp.int16),
        jnp.ones((batch,), dtype=jnp.int16),
        jnp.full((batch,), 100, dtype=jnp.int16),
        jnp.full((batch,), premium_mask, dtype=jnp.int16),
        jnp.zeros((batch,), dtype=jnp.bool_),
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


def fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    player: int,
):
    """FC14 plus actor-specific X562/PRT weed-HIRE synchronization."""

    batch = states.step.shape[0]
    premium_mask = sum(
        1 << PRODUCTS.index(item) for item in ("STRAWBERRY", "MILK", "WOOL")
    )
    k320_action, k320 = k320_mass_hire_guard_residual_sale_player_action_v1(
        states,
        tables,
        runtime,
        latest_bank,
        carry.k320,
        jnp.ones((batch,), dtype=jnp.bool_),
        jnp.full((batch,), 120, dtype=jnp.int16),
        jnp.full((batch,), 6, dtype=jnp.int16),
        jnp.ones((batch,), dtype=jnp.int16),
        jnp.full((batch,), 100, dtype=jnp.int16),
        jnp.full((batch,), premium_mask, dtype=jnp.int16),
        jnp.zeros((batch,), dtype=jnp.bool_),
        player,
    )
    x562_action, x562_suffix = x562_prt_suffix_weed_hire_guard_player_action_v1(
        states,
        runtime,
        latest_bank,
        old_bank,
        carry.x562_suffix,
        jnp.ones((batch,), dtype=jnp.bool_),
        jnp.full((batch,), 4, dtype=jnp.int16),
        jnp.full((batch,), 2, dtype=jnp.int16),
        jnp.full((batch,), 5, dtype=jnp.int16),
        jnp.ones((batch,), dtype=jnp.int16),
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


def fc15_opening_wheat_quantity_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    opening_wheat_quantity: jax.Array,
    player: int,
):
    """FC15 with only its existing step-zero wheat order quantity changed.

    This is an ablation hook for a public-state route interaction.  It neither
    adds a market slot nor branches on opponent identity: the ordinary FC15
    action is generated first, then the already-present BUY_PRODUCT/WHEAT
    order is assigned the requested quantity on state step zero.
    """

    return fc15_opening_transaction_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry,
        jnp.full(states.step.shape, 7, dtype=jnp.int16),
        jnp.full(states.step.shape, 12, dtype=jnp.int16),
        opening_wheat_quantity,
        player,
    )


def fc15_opening_transaction_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    opening_wheat_seed_quantity: jax.Array,
    opening_melon_seed_quantity: jax.Array,
    opening_wheat_product_quantity: jax.Array,
    player: int,
):
    """FC15 with three existing step-zero order amounts parameterized."""

    action, carry = fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry,
        player,
    )
    active = (
        jnp.arange(action.market_op.shape[1])[None, :]
        < action.market_count[:, None]
    )
    opening = (states.step.astype(jnp.int32) == 0)[:, None] & active
    wheat_seed = (
        opening
        & (action.market_op == MarketOp.BUY_SEED)
        & (action.market_item == PRODUCTS.index("WHEAT"))
    )
    melon_seed = (
        opening
        & (action.market_op == MarketOp.BUY_SEED)
        & (action.market_item == PRODUCTS.index("MELON"))
    )
    wheat_product = (
        opening
        & (action.market_op == MarketOp.BUY_PRODUCT)
        & (action.market_item == PRODUCTS.index("WHEAT"))
    )
    amount = jnp.where(
        wheat_seed,
        opening_wheat_seed_quantity.astype(action.market_amount.dtype)[:, None],
        action.market_amount,
    )
    amount = jnp.where(
        melon_seed,
        opening_melon_seed_quantity.astype(action.market_amount.dtype)[:, None],
        amount,
    )
    amount = jnp.where(
        wheat_product,
        opening_wheat_product_quantity.astype(action.market_amount.dtype)[:, None],
        amount,
    )
    return action._replace(market_amount=amount), carry


def fc15_moon_market_observer_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionMoonMarketCarryV1,
    player: int,
):
    """Keep FC15 production and add only Moon's learned sale-horizon overlay.

    The observer reconstructs unexpected public market supply after removing
    town demand and this agent's own prior sales.  It never sees the rival's
    private shed, identity, action, or future events.  A ring debt removes each
    advanced quantity from the original FC15 sale step, so this layer changes
    timing rather than manufacturing inventory.
    """

    action, base = fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    route, _ = hp._select_route(
        states, base.k320, player, hp.MODE_RAY_K320
    )
    market = latest6._moon_observe(
        states,
        latest_bank,
        route,
        carry.market,
    )
    repaid_action, repaid_market = latest6._moon_repay(
        states, action, market
    )
    preempted_action, preempted_market = latest6._moon_preempt(
        states,
        latest_bank,
        route,
        repaid_action,
        repaid_market,
        player,
    )
    use_overlay = ~base.sheep_pressure
    action = _select_action(use_overlay, preempted_action, repaid_action)
    market = _select_tree(use_overlay, preempted_market, repaid_market)
    market = latest6._moon_record_own_sells(states, action, market, player)
    return action, FusionChampionMoonMarketCarryV1(base=base, market=market)


def fc15_moon_embedded_market_observer_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionMoonMarketCarryV1,
    player: int,
):
    """FC15 with Moon's more responsive public clone-race calibration.

    This is an experimental, opponent-identity-free variant of
    :func:`fc15_moon_market_observer_player_action_v1`.  It uses the stricter
    public-board similarity gate from Boatlee V21, but lowers the evidence
    requirement from two observations to one-and-a-half and protects a
    learned clone race with a minimum three-step sale horizon.  Production,
    route choice, task repair and the repayment ledger are unchanged.
    """

    action, base = fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    route, _ = hp._select_route(
        states, base.k320, player, hp.MODE_RAY_K320
    )
    market = latest6._moon_observe(
        states,
        latest_bank,
        route,
        carry.market,
        embedded_v21=True,
    )
    repaid_action, repaid_market = latest6._moon_repay(
        states, action, market
    )
    preempted_action, preempted_market = latest6._moon_preempt(
        states,
        latest_bank,
        route,
        repaid_action,
        repaid_market,
        player,
        embedded_v21=True,
    )
    use_overlay = ~base.sheep_pressure
    action = _select_action(use_overlay, preempted_action, repaid_action)
    market = _select_tree(use_overlay, preempted_market, repaid_market)
    market = latest6._moon_record_own_sells(states, action, market, player)
    return action, FusionChampionMoonMarketCarryV1(base=base, market=market)


def fc15_moon_parameter_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionMoonMarketCarryV1,
    minimum_horizon: jax.Array,
    horizon_floor_start: jax.Array,
    horizon_floor_distance: jax.Array,
    action_distance: jax.Array,
    player: int,
):
    """Parameterized, public-state-only refinement of the embedded Moon layer.

    The frozen Moon implementation remains untouched.  This wrapper screens
    only two interpretable controls: how close the public farms must remain
    before changing a sale, and the minimum learned sale horizon after a given
    step.  The original FC15 production tape and Moon repayment ledger remain
    the sole action owners.
    """

    action, base = fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    route, _ = hp._select_route(states, base.k320, player, hp.MODE_RAY_K320)
    market = latest6._moon_observe(
        states,
        latest_bank,
        route,
        carry.market,
        embedded_v21=True,
    )
    repaid_action, repaid_market = latest6._moon_repay(states, action, market)
    step = states.step.astype(jnp.int32)
    distance = lp._kaito_clone_distance(states)
    floor = (
        (step >= horizon_floor_start.astype(jnp.int32))
        & (distance <= horizon_floor_distance.astype(jnp.int32))
    )
    tuned_market = repaid_market._replace(
        horizons=jnp.where(
            floor[:, None],
            jnp.maximum(
                repaid_market.horizons.astype(jnp.int32),
                minimum_horizon.astype(jnp.int32)[:, None],
            ),
            repaid_market.horizons.astype(jnp.int32),
        ).astype(jnp.int8)
    )
    preempted_action, preempted_market = latest6._moon_preempt(
        states,
        latest_bank,
        route,
        repaid_action,
        tuned_market,
        player,
        embedded_v21=True,
    )
    use_overlay = (
        (~base.sheep_pressure)
        & (distance <= action_distance.astype(jnp.int32))
    )
    action = _select_action(use_overlay, preempted_action, repaid_action)
    market = _select_tree(use_overlay, preempted_market, repaid_market)
    market = latest6._moon_record_own_sells(states, action, market, player)
    return action, FusionChampionMoonMarketCarryV1(base=base, market=market)


def fc16_moon_h4_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionMoonMarketCarryV1,
    player: int,
):
    """FC16 candidate: FC15 plus the two-bank-validated Moon h4 hedge."""

    shape = states.step.shape
    return fc15_moon_parameter_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry,
        jnp.full(shape, 4, dtype=jnp.int16),
        jnp.full(shape, 120, dtype=jnp.int16),
        jnp.full(shape, 2, dtype=jnp.int16),
        jnp.full(shape, 6, dtype=jnp.int16),
        player,
    )


def fc19_moon_h4_wheat8_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionMoonMarketCarryV1,
    player: int,
):
    """FC16 Moon hedge plus the independently replicated wheat-8 opening."""

    action, carry = fc16_moon_h4_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry,
        player,
    )
    active = (
        jnp.arange(action.market_op.shape[1])[None, :]
        < action.market_count[:, None]
    )
    opening_wheat_seed = (
        (states.step.astype(jnp.int32) == 0)[:, None]
        & active
        & (action.market_op == MarketOp.BUY_SEED)
        & (action.market_item == PRODUCTS.index("WHEAT"))
    )
    amount = jnp.where(
        opening_wheat_seed,
        jnp.asarray(8, dtype=action.market_amount.dtype),
        action.market_amount,
    )
    return action._replace(market_amount=amount), carry


def fc21_wheat8_feed_cash_guard_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionMoonMarketCarryV1,
    player: int,
):
    """Wheat-8 opening with one early bulk-carrot purchase deferred.

    The wheat-8 opening moves Boatlee V21 away from its strongest mirror
    route, but the ten-coin opening debit can cross a later feed-cash
    threshold.  On the first observed route this manifests as a bulk carrot
    request of six instead of five, followed by an unaffordable wheat-product
    order and a silent failed feed.  This guard changes only an *actual*
    early bulk carrot order (amount >= 5), never keys on opponent identity,
    and leaves all unit actions and all other transactions untouched.

    This remains an experimental ablation until it passes independent B21,
    Moon, Rank14 and full-roster panels.
    """

    action, carry = fc19_moon_h4_wheat8_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry,
        player,
    )
    active = (
        jnp.arange(action.market_op.shape[1])[None, :]
        < action.market_count[:, None]
    )
    bulk_carrot = (
        active
        & (states.step.astype(jnp.int32) > 72)[:, None]
        & (states.step.astype(jnp.int32) < 192)[:, None]
        & (action.market_op == MarketOp.BUY_SEED)
        & (action.market_item == PRODUCTS.index("CARROT"))
        & (action.market_amount >= 5)
    )
    amount = jnp.where(
        bulk_carrot,
        jnp.maximum(action.market_amount - 1, 0),
        action.market_amount,
    )
    return action._replace(market_amount=amount), carry


def fc22_feed_value_guard_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionFeedValueCarryV1,
    player: int,
):
    """FC21 plus Steven's generic late feed-value decision.

    From day 10 onward, the first missed feeding day is tolerated only when
    the currently observable animal-product value (including pending CARE)
    is below the observable wheat replacement cost.  Every skipped feed is
    recorded as wheat credit and deducted from later wheat purchases.  The
    rule uses no opponent identity, replay id, event seed, or future shop.

    This is an ablation candidate; FC21 remains the accepted baseline until
    the latest-six and old-roster panels prove that the value guard is safe.
    """

    action, base = fc21_wheat8_feed_cash_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    action, credit = latest6._x631_skip_feed_and_credit(
        states, action, carry.wheat_credit, player
    )
    action, credit = latest6._x631_trim_wheat_buys(action, credit)
    return action, FusionChampionFeedValueCarryV1(
        base=base,
        wheat_credit=credit,
    )


def _parameterized_feed_value_skip_and_credit(
    states,
    action,
    credit: jax.Array,
    day_start: jax.Array,
    value_numerator: jax.Array,
    value_denominator: jax.Array,
    player: int,
):
    """Ablation form of the public-value feed guard used by FC22.

    A feed is skipped when ``product_value * denominator`` is strictly below
    ``wheat_price * numerator``.  Positive integer ratios keep the comparison
    exact and avoid floating-point threshold drift between Python and JAX.
    """

    batch = jnp.arange(states.step.shape[0])
    present = (
        jnp.arange(latest6.MAX_UNITS)[None, :]
        < action.unit_count[:, None]
    )
    position = states.unit_pos[:, player].astype(jnp.int32)
    x = jnp.clip(position[..., 0], 0, 9)
    y = jnp.clip(position[..., 1], 0, 9)
    animal = states.tile_animal[
        batch[:, None], player, y, x
    ].astype(jnp.int32)
    valid_animal = (
        (animal >= 0) & (animal < latest6._ANIMAL_PRODUCT.shape[0])
    )
    safe_animal = jnp.clip(
        animal, 0, latest6._ANIMAL_PRODUCT.shape[0] - 1
    )
    product = latest6._ANIMAL_PRODUCT[safe_animal]
    fed_today = (
        states.tile_flags[batch[:, None], player, y, x]
        & latest6.FLAG_FED
    ) != 0
    first_unfed_day = (
        states.tile_neglect[batch[:, None], player, y, x].astype(jnp.int32)
        == 0
    )
    pending = jnp.maximum(
        states.tile_pending_care[
            batch[:, None], player, y, x
        ].astype(jnp.int32),
        0,
    )
    product_price = jnp.take_along_axis(
        states.market_price, product, axis=1
    ).astype(jnp.int32)
    product_value = product_price * (1 + pending)
    skip = (
        ((states.step.astype(jnp.int32) // 24) >= day_start.astype(jnp.int32))[:, None]
        & present
        & (action.unit_op == latest6.UnitOp.FEED)
        & valid_animal
        & first_unfed_day
        & (~fed_today)
        & (
            product_value * value_denominator.astype(jnp.int32)[:, None]
            < states.market_price[:, latest6._WHEAT, None]
            * value_numerator.astype(jnp.int32)[:, None]
        )
    )
    action = action._replace(
        unit_op=jnp.where(
            skip, latest6.UnitOp.PASS, action.unit_op
        ).astype(jnp.int8),
        unit_item=jnp.where(skip, -1, action.unit_item).astype(jnp.int8),
        unit_amount=jnp.where(skip, 1, action.unit_amount).astype(jnp.int32),
    )
    return action, (
        credit.astype(jnp.int32) + jnp.sum(skip, axis=1)
    ).astype(jnp.int16)


def fc22_feed_value_parameter_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionFeedValueCarryV1,
    day_start: jax.Array,
    value_numerator: jax.Array,
    value_denominator: jax.Array,
    player: int,
):
    """Vectorized parameter screen for FC22's one economic decision."""

    action, base = fc21_wheat8_feed_cash_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    action, credit = _parameterized_feed_value_skip_and_credit(
        states,
        action,
        carry.wheat_credit,
        day_start,
        value_numerator,
        value_denominator,
        player,
    )
    action, credit = latest6._x631_trim_wheat_buys(action, credit)
    return action, FusionChampionFeedValueCarryV1(
        base=base,
        wheat_credit=credit,
    )


def fc23_route3_rebalance_parameter_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionFeedValueCarryV1,
    enabled: jax.Array,
    switch_step: jax.Array,
    target_route: jax.Array,
    player: int,
):
    """FC22 counterfactual for the public first-YARN heavy-sheep route.

    The route selected from public town shops remains the sole trigger.  When
    that source route is the first-YARN program (route 3), a screen arm may
    replace it at a public step with another already-validated legal route.
    All other source routes and all pre-switch actions are byte-identical to
    FC22.  This hook is experimental until paired train/holdout panels prove a
    deployable public-state rule; it never keys on opponent identity or seed.
    """

    k320 = carry.base.base.k320
    do_switch = (
        enabled
        & k320.ray_route_locked
        & (k320.ray_route_id.astype(jnp.int32) == 3)
        & (states.step.astype(jnp.int32) >= switch_step.astype(jnp.int32))
    )
    k320 = k320._replace(
        ray_route_locked=k320.ray_route_locked | do_switch,
        ray_route_id=jnp.where(
            do_switch,
            target_route.astype(jnp.int8),
            k320.ray_route_id,
        ).astype(jnp.int8),
    )
    base_inner = carry.base.base._replace(k320=k320)
    carry = carry._replace(base=carry.base._replace(base=base_inner))
    return fc22_feed_value_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry,
        player,
    )


def fc23_terminal_priority_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionFeedValueCarryV1,
    terminal_mode: jax.Array,
    player: int,
):
    """FC22 plus the existing public-state step-718 sale-order ablation."""

    action, carry = fc22_feed_value_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry,
        player,
    )
    return _terminal_priority_liquidation(
        states, action, terminal_mode, player
    ), carry


def fc24_terminal_crop_salvage_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionTerminalSalvageCarryV1,
    player: int,
):
    """FC22 plus one generic last-feasible terminal crop recovery.

    During the final day, if an active unit is standing on a mature crop with
    positive yield, the source action would move away, and *this exact step*
    is the last time HARVEST -> shortest return -> DROP can still finish by
    step 718, the highest current-value candidate becomes a sticky obligation.
    The wrapper then owns that unit until deposit and adds only the newly
    deposited product to the existing terminal sale order.  No coordinates,
    route IDs, opponent identity, seed, or future event are used.
    """

    action, base = fc22_feed_value_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    step = states.step.astype(jnp.int32)
    batch_size = step.shape[0]
    batch = jnp.arange(batch_size)
    positions = states.unit_pos[:, player].astype(jnp.int16)
    x = jnp.clip(positions[..., 0].astype(jnp.int32), 0, 9)
    y = jnp.clip(positions[..., 1].astype(jnp.int32), 0, 9)
    crop = states.tile_crop[batch[:, None], player, y, x].astype(jnp.int32)
    safe_crop = jnp.clip(crop, 0, len(CROP_FIRST_YIELD_DAY) - 1)
    tile_yield = states.tile_yield[batch[:, None], player, y, x].astype(jnp.int32)
    origin = states.tile_origin_day[batch[:, None], player, y, x].astype(jnp.int32)
    first_yield = jnp.asarray(CROP_FIRST_YIELD_DAY, dtype=jnp.int32)[safe_crop]
    mature = (step[:, None] // TURNS_PER_DAY - origin) >= first_yield
    distances = jnp.sum(
        jnp.abs(positions[:, :, None, :] - hp._EVAC_ACCESS[None, None, :, :]),
        axis=3,
    ).astype(jnp.int32)
    access_index = jnp.argmin(distances, axis=2)
    distance = jnp.min(distances, axis=2)
    unit_target = hp._EVAC_ACCESS[access_index]
    present = states.unit_active[:, player] & (
        jnp.arange(hp.MAX_UNITS)[None, :] < action.unit_count[:, None]
    )
    moving_away = (action.unit_op >= UnitOp.NORTH) & (
        action.unit_op <= UnitOp.WEST
    )
    remaining_actions = 719 - step
    last_feasible = distance + 2 == remaining_actions[:, None]
    crop_value = tile_yield * jnp.take_along_axis(
        states.market_price[:, : len(CROP_FIRST_YIELD_DAY)],
        safe_crop,
        axis=1,
    )
    carried_value = jnp.sum(
        states.unit_inventory[:, player, :, : hp.NUM_PRODUCTS].astype(jnp.int32)
        * states.market_price[:, None, : hp.NUM_PRODUCTS].astype(jnp.int32),
        axis=2,
    )
    # Delaying the source return by one action can push its existing cargo into
    # the crowded final market step.  Only take that risk when the crop on the
    # current tile is worth at least twice the already-secured cargo.  This is
    # a route-independent opportunity-cost guard; an empty actor passes it.
    value_dominates_cargo = crop_value >= (2 * carried_value)
    valid = (
        (~carry.active)[:, None]
        & (step >= 696)[:, None]
        & present
        & moving_away
        & (crop >= 0)
        & (crop < len(CROP_FIRST_YIELD_DAY))
        & mature
        & (tile_yield > 0)
        & value_dominates_cargo
        & last_feasible
    )
    # Lexicographic objective: current liquidation value, then lower actor ID.
    score = jnp.where(
        valid,
        crop_value * 100 - jnp.arange(hp.MAX_UNITS)[None, :],
        -1,
    )
    best_actor = jnp.argmax(score, axis=1).astype(jnp.int8)
    selected = jnp.max(score, axis=1) >= 0
    safe_best = jnp.clip(best_actor.astype(jnp.int32), 0, hp.MAX_UNITS - 1)
    selected_target = unit_target[batch, safe_best]
    selected_product = crop[batch, safe_best].astype(jnp.int8)
    selected_quantity = tile_yield[batch, safe_best].astype(jnp.int16)

    actor = jnp.where(selected, best_actor, carry.actor).astype(jnp.int8)
    target = jnp.where(selected[:, None], selected_target, carry.target).astype(jnp.int16)
    product = jnp.where(selected, selected_product, carry.product).astype(jnp.int8)
    quantity = jnp.where(selected, selected_quantity, carry.quantity).astype(jnp.int16)
    active = carry.active | selected
    safe_actor = jnp.clip(actor.astype(jnp.int32), 0, hp.MAX_UNITS - 1)
    position = positions[batch, safe_actor]
    at_target = jnp.all(position == target, axis=1)
    continuation = active & (~selected)
    drop = continuation & at_target
    continuation_op = jnp.where(
        drop,
        UnitOp.DROP,
        hp._move_toward(position, target),
    ).astype(jnp.int8)
    replacement_op = jnp.where(
        selected,
        UnitOp.HARVEST,
        continuation_op,
    ).astype(jnp.int8)
    old_op = action.unit_op[batch, safe_actor]
    action = action._replace(
        unit_op=action.unit_op.at[batch, safe_actor].set(
            jnp.where(active, replacement_op, old_op)
        ),
        unit_item=action.unit_item.at[batch, safe_actor].set(
            jnp.where(active, -1, action.unit_item[batch, safe_actor])
        ),
        unit_amount=action.unit_amount.at[batch, safe_actor].set(
            jnp.where(active, 1, action.unit_amount[batch, safe_actor])
        ),
    )

    # Unit actions execute before market orders.  Preserve FC22's order and
    # merge only the incremental product that the sticky salvage actor drops.
    projected = hp._projected_shed_without_pickups(states, action, player)
    planned = hp._planned_sales(action)
    for item in range(hp.NUM_PRODUCTS):
        incremental = jnp.minimum(
            quantity.astype(jnp.int32),
            jnp.maximum(projected[:, item] - planned[:, item], 0),
        )
        action, _ = hp._append_or_merge_sale(
            action,
            drop
            & (step == 718)
            & (product.astype(jnp.int32) == item)
            & (incremental > 0),
            item,
            incremental,
        )

    return action, FusionChampionTerminalSalvageCarryV1(
        base=base,
        active=active,
        actor=actor,
        target=target,
        product=product,
        quantity=quantity,
    )


def fc20_wheat8_virtual_baseline_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionMoonMarketCarryV1,
    player: int,
):
    """Expose the real wheat-8 opening while preserving FC16's plan ledger.

    After the opening transaction, the controller evaluates its own private
    cash/seed budget in the equivalent pre-hedge coordinates (cash +10 and one
    fewer wheat seed).  The simulator and the opponent always receive the real
    state and real action; this is only a planned-state normalization for the
    controller that paid for the hedge.
    """

    after_opening = states.step.astype(jnp.int32) > 0
    money_offset = jnp.where(after_opening, 10, 0).astype(states.money.dtype)
    virtual_money = states.money.at[:, player].add(money_offset)
    wheat = PRODUCTS.index("WHEAT")
    seed_offset = jnp.where(
        after_opening & (states.seeds[:, player, wheat] > 0),
        -1,
        0,
    ).astype(states.seeds.dtype)
    virtual_seeds = states.seeds.at[:, player, wheat].add(seed_offset)
    virtual_states = states._replace(money=virtual_money, seeds=virtual_seeds)
    action, carry = fc16_moon_h4_player_action_v1(
        virtual_states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry,
        player,
    )
    active = (
        jnp.arange(action.market_op.shape[1])[None, :]
        < action.market_count[:, None]
    )
    opening_wheat_seed = (
        (states.step.astype(jnp.int32) == 0)[:, None]
        & active
        & (action.market_op == MarketOp.BUY_SEED)
        & (action.market_item == wheat)
    )
    amount = jnp.where(
        opening_wheat_seed,
        jnp.asarray(8, dtype=action.market_amount.dtype),
        action.market_amount,
    )
    return action._replace(market_amount=amount), carry


def fc16_shadow_old_suffix_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionMoonSuffixCarryV1,
    enable_switch: jax.Array,
    switch_step: jax.Array,
    suffix_route: jax.Array,
    player: int,
):
    """Run FC16 and an old task route in parallel, then safely hand off.

    The suffix ledger is updated from step zero on the same realized states.
    This prevents the cold-start corruption seen when a fixed route is first
    initialized only at the takeover step.
    """

    base_action, base = fc16_moon_h4_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    route = suffix_route.astype(jnp.int32)
    suffix_action, suffix = hp._raw_with_weed(
        states, old_bank, route, carry.suffix, player
    )
    suffix_action, suffix = hp._room_evac(
        states, suffix_action, suffix, player
    )
    suffix_action, suffix = hp._repay(
        suffix_action,
        suffix,
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
    action = _select_action(switched, suffix_action, base_action)
    return action, FusionChampionMoonSuffixCarryV1(
        base=base,
        suffix=suffix,
        switched=switched,
        opening_match=carry.opening_match,
    )


def fc17_ueddy_s192_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionMoonSuffixCarryV1,
    player: int,
):
    """FC17 screen arm: FC16 then rank02/ueddy route 30 at step 192."""

    shape = states.step.shape
    return fc16_shadow_old_suffix_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry,
        jnp.ones(shape, dtype=jnp.bool_),
        jnp.full(shape, 192, dtype=jnp.int16),
        jnp.full(shape, 30, dtype=jnp.int16),
        player,
    )


def fc17_opening_hedge_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionMoonSuffixCarryV1,
    player: int,
):
    """FC16 with a narrow public-state hedge for the 2C/2S opening family.

    The flag is learned once from the post-opening public state and retained in
    carry.  No opponent identity, Replay id, seed, or hidden inventory is used.
    Route 30 is shadowed from step zero and only takes over at step 192.
    """

    rival = 1 - player
    active_own = jnp.sum(states.unit_active[:, player], axis=-1)
    active_rival = jnp.sum(states.unit_active[:, rival], axis=-1)
    opening_signature = (
        (states.step.astype(jnp.int32) == 1)
        & (states.money[:, player].astype(jnp.int32) == 22)
        & (states.money[:, rival].astype(jnp.int32) == 140)
        & (active_own == 6)
        & (active_rival == 6)
    )
    opening_match = carry.opening_match | opening_signature
    updated = FusionChampionMoonSuffixCarryV1(
        base=carry.base,
        suffix=carry.suffix,
        switched=carry.switched,
        opening_match=opening_match,
    )
    shape = states.step.shape
    action, next_carry = fc16_shadow_old_suffix_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        updated,
        opening_match,
        jnp.full(shape, 192, dtype=jnp.int16),
        jnp.full(shape, 30, dtype=jnp.int16),
        player,
    )
    return action, FusionChampionMoonSuffixCarryV1(
        base=next_carry.base,
        suffix=next_carry.suffix,
        switched=next_carry.switched,
        opening_match=opening_match,
    )


def fc15_opening_hedge_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionBaseSuffixCarryV1,
    player: int,
):
    """Deployable FC15 repair with the same narrow opening-family hedge."""

    rival = 1 - player
    active_own = jnp.sum(states.unit_active[:, player], axis=-1)
    active_rival = jnp.sum(states.unit_active[:, rival], axis=-1)
    opening_signature = (
        (states.step.astype(jnp.int32) == 1)
        & (states.money[:, player].astype(jnp.int32) == 22)
        & (states.money[:, rival].astype(jnp.int32) == 140)
        & (active_own == 6)
        & (active_rival == 6)
    )
    opening_match = carry.opening_match | opening_signature
    base_action, base = fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    suffix_action, suffix = hp._raw_with_weed(
        states, old_bank, jnp.full(states.step.shape, 30, dtype=jnp.int32), carry.suffix, player
    )
    suffix_action, suffix = hp._room_evac(states, suffix_action, suffix, player)
    suffix_action, suffix = hp._repay(
        suffix_action,
        suffix,
        states.step.astype(jnp.int32),
        hp.MODE_BOATLEE,
    )
    suffix_action = hp._rank_sell_slots_exact(states, runtime, suffix_action)
    suffix_action = lp._room_guard(states, suffix_action, player)
    suffix_action = lp._x562_seed_budget_guard(states, suffix_action, player)
    suffix_action = lp._terminal_liquidation(states, suffix_action, player)
    suffix_action = lp._x562_idle_fertilizer_sale(states, suffix_action, player)
    switched = carry.switched | (
        opening_match & (states.step.astype(jnp.int32) >= 192)
    )
    action = _select_action(switched, suffix_action, base_action)
    return action, FusionChampionBaseSuffixCarryV1(
        base=base,
        suffix=suffix,
        switched=switched,
        opening_match=opening_match,
    )


def fc15_kaito_market_skill_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionKaitoMarketCarryV1,
    enable_preempt: jax.Array,
    enable_market_maker: jax.Array,
    player: int,
):
    """Keep FC15 production and test Kaito's public-state market skills.

    This candidate never switches to Kaito's production tape.  It applies the
    Kaito preemption and one-turn wheat market-maker only to FC15's current
    legal action, and only while FC15 remains on its K320 lane.  The pressure
    lane can hand off to an old-bank PRT suffix, whose future commitments are
    not represented by ``latest_bank``; disabling the overlay there avoids a
    fictitious investment reserve.
    """

    action, base = fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    route, _ = hp._select_route(states, base.k320, player, hp.MODE_RAY_K320)
    eligible = ~base.sheep_pressure

    preempt_action, preempt_market = lp._kaito_preempt(
        states,
        runtime,
        latest_bank,
        route,
        action,
        carry.market,
        player,
    )
    use_preempt = eligible & enable_preempt
    action = _select_action(use_preempt, preempt_action, action)
    market = _select_tree(use_preempt, preempt_market, carry.market)

    maker_action, maker_market = lp._kaito_market_maker(
        states,
        runtime,
        latest_bank,
        route,
        action,
        market,
        player,
    )
    use_maker = eligible & enable_market_maker
    action = _select_action(use_maker, maker_action, action)
    market = _select_tree(use_maker, maker_market, market)
    return action, FusionChampionKaitoMarketCarryV1(base=base, market=market)


def fc16_public_residual_sale_screen_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    sale_enabled: jax.Array,
    sale_start_step: jax.Array,
    distance_limit: jax.Array,
    quantity_cap: jax.Array,
    minimum_price_percent: jax.Array,
    product_mask: jax.Array,
    pressure_branch_enabled: jax.Array,
    player: int,
):
    """FC15 with one lane-parameterized, public-state residual-sale layer.

    The production tapes, public route selector, PRT handoff and task repair
    remain FC15.  The experiment changes only whether genuinely uncommitted
    shed stock is sold earlier.  ``distance_limit`` is the public-board clone
    distance already used by FC15; no opponent identity, event seed, hidden
    inventory or future shop is visible to the policy.
    """

    batch = states.step.shape[0]
    k320_action, k320 = k320_mass_hire_guard_residual_sale_player_action_v1(
        states,
        tables,
        runtime,
        latest_bank,
        carry.k320,
        sale_enabled,
        sale_start_step,
        distance_limit,
        quantity_cap,
        minimum_price_percent,
        product_mask,
        jnp.zeros((batch,), dtype=jnp.bool_),
        player,
    )
    x562_action, x562_suffix = (
        x562_prt_suffix_weed_hire_guard_residual_sale_player_action_v1(
            states,
            runtime,
            latest_bank,
            old_bank,
            carry.x562_suffix,
            jnp.ones((batch,), dtype=jnp.bool_),
            jnp.full((batch,), 4, dtype=jnp.int16),
            jnp.full((batch,), 2, dtype=jnp.int16),
            jnp.full((batch,), 5, dtype=jnp.int16),
            jnp.ones((batch,), dtype=jnp.int16),
            sale_enabled & pressure_branch_enabled,
            sale_start_step,
            distance_limit,
            quantity_cap,
            minimum_price_percent,
            product_mask,
            jnp.zeros((batch,), dtype=jnp.bool_),
            player,
        )
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


def fc16_b21_forecast_preempt_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    enabled: jax.Array,
    start_step: jax.Array,
    distance_limit: jax.Array,
    quantity_cap: jax.Array,
    minimum_price_percent: jax.Array,
    player: int,
):
    """Thin B21-style foreign-sale forecast layered after one FC15 action.

    Route 16 is B21's public Munib production schedule.  The layer never plays
    that route.  It only asks whether the schedule predicts a premium sale on
    the next step and, if the live public boards are sufficiently similar,
    advances an available quantity by one step.  The existing repayment ledger
    removes the same quantity from the next planned sale.
    """

    action, next_carry = fc15_fc14_x562_split_weed_hire_guard_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry,
        player,
    )
    step = states.step.astype(jnp.int32)
    active_cap = jnp.where(
        enabled & (step >= start_step.astype(jnp.int32)),
        quantity_cap,
        0,
    ).astype(jnp.int16)
    foreign_route = jnp.full(step.shape, 16, dtype=jnp.int32)
    k320_action, k320 = _parameterized_one_step_preempt_player(
        states,
        latest_bank,
        foreign_route,
        action,
        next_carry.k320,
        distance_limit,
        active_cap,
        minimum_price_percent,
        player,
    )
    prefix_action, prefix_base = _parameterized_one_step_preempt_player(
        states,
        latest_bank,
        foreign_route,
        action,
        next_carry.x562_suffix.x562.base,
        distance_limit,
        active_cap,
        minimum_price_percent,
        player,
    )
    suffix_action, suffix_base = _parameterized_one_step_preempt_player(
        states,
        latest_bank,
        foreign_route,
        action,
        next_carry.x562_suffix.suffix,
        distance_limit,
        active_cap,
        minimum_price_percent,
        player,
    )
    pressure = next_carry.sheep_pressure
    switched = next_carry.x562_suffix.switched
    use_prefix = pressure & ~switched
    use_suffix = pressure & switched
    action = _select_action(use_prefix, prefix_action, k320_action)
    action = _select_action(use_suffix, suffix_action, action)
    x562_route = next_carry.x562_suffix.x562._replace(
        base=_select_tree(
            use_prefix,
            prefix_base,
            next_carry.x562_suffix.x562.base,
        )
    )
    x562_suffix = next_carry.x562_suffix._replace(
        x562=x562_route,
        suffix=_select_tree(
            use_suffix,
            suffix_base,
            next_carry.x562_suffix.suffix,
        ),
    )
    return action, FusionChampionCarryV3(
        k320=_select_tree(~pressure, k320, next_carry.k320),
        x562_suffix=x562_suffix,
        sheep_pressure=pressure,
    )


def fc11e_fc2b_late_sheep_pressure_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    enable_late: jax.Array,
    start_step: jax.Array,
    minimum_rival_sheep: jax.Array,
    maximum_rival_cows: jax.Array,
    player: int,
):
    """Extend FC2B's existing public heavy-sheep recovery window.

    FC2B already advances both its K320 policy and the X562/PRT recovery policy
    on every real state, but it only selects the recovery policy when a very
    early 24--72 step morphology is visible.  PRT-like businesses often do not
    expose their final sheep commitment until later.  This ablation keeps both
    planners and all execution guards unchanged; it changes only the sticky,
    public-state trigger window.

    The function is parameterized for counterfactual screening.  It receives
    no opponent identity, hidden inventory, event seed or future shop data.
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
    step = states.step.astype(jnp.int32)
    _, _, rival_cows, rival_sheep, _ = hp._opponent_counts(states, player)
    late_pressure = (
        enable_late
        & (step >= start_step.astype(jnp.int32))
        & (rival_sheep >= minimum_rival_sheep.astype(jnp.int32))
        & (rival_cows <= maximum_rival_cows.astype(jnp.int32))
    )
    sheep_pressure = (
        carry.sheep_pressure
        | _visible_opening_sheep_pressure(states, player)
        | late_pressure
    )
    action = _select_action(sheep_pressure, x562_action, k320_action)
    return action, FusionChampionCarryV3(
        k320=k320,
        x562_suffix=x562_suffix,
        sheep_pressure=sheep_pressure,
    )


def fc10h_fc2b_kobe_goose_commitment_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    goose_bank,
    carry: FusionChampionCarryV5,
    player: int,
    opponent_melon_min: int = 18,
):
    """FC2B plus Kobe's public-shop goose project commitment.

    The decision is evaluated once, no earlier than step 192, when the first
    two public town shops are available.  The goose route is committed only
    when those shops contain neither YARN demand nor MILK-support demand and
    the opponent already exposes at least ``opponent_melon_min`` melon tiles.

    Once committed, the project remains active for the season.  This is
    intentional: the holdout ablation showed that returning to an FC2B shadow
    policy after building coops destroys the already-paid project state.  The
    policy reads no opponent identity, hidden inventory, or future shop/event
    information.
    """

    base_action, base = fc2b_rank14_plus_clone_aware_preempt_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    batch = states.step.shape[0]
    route = jnp.zeros((batch,), dtype=jnp.int32)
    goose_action, goose = skeleton_player_action_v1(
        states,
        tables,
        goose_bank,
        route,
        carry.goose,
        player,
    )

    step = states.step.astype(jnp.int32)
    evaluate_now = (
        (~carry.goose_checked)
        & (step >= 192)
        & (states.town_count >= 2)
    )
    first_two = states.town_shops[:, :2]
    yarn = SHOP_NAMES.index("YARN_STORE")
    milk_support = jnp.asarray(
        [
            SHOP_NAMES.index("PIZZA_SHOP"),
            SHOP_NAMES.index("ICE_CREAM_SHOP"),
            SHOP_NAMES.index("SMOOTHIE_SHOP"),
        ],
        dtype=jnp.int8,
    )
    no_yarn = jnp.all(first_two != yarn, axis=1)
    no_milk_support = ~jnp.any(
        first_two[..., None] == milk_support[None, None, :],
        axis=(1, 2),
    )
    rival = 1 - player
    opponent_melon = jnp.sum(
        states.tile_crop[:, rival] == PRODUCTS.index("MELON"),
        axis=(1, 2),
    )
    activate = (
        evaluate_now
        & no_yarn
        & no_milk_support
        & (opponent_melon >= opponent_melon_min)
    )
    goose_active = carry.goose_active | activate
    goose_checked = carry.goose_checked | evaluate_now
    action = _select_action(goose_active, goose_action, base_action)
    return action, FusionChampionCarryV5(
        base=base,
        goose=goose,
        goose_checked=goose_checked,
        goose_active=goose_active,
    )


def fc2b_shadow_old_suffix_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV6,
    enable_switch: jax.Array,
    switch_step: jax.Array,
    suffix_route: jax.Array,
    player: int,
):
    """Probe an old route that shadows the live FC2B state before takeover.

    Both executors are advanced on every real state.  Only their emitted
    action is selected.  This avoids the historical cold-start failure where
    a suffix first initialized its ledgers at the takeover step.  It remains
    a capability probe: a route-specific switch must still pass causal
    selection and holdout gates before deployment.
    """

    base_action, base = fc2b_rank14_plus_clone_aware_preempt_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    route = suffix_route.astype(jnp.int32)
    suffix_action, suffix = hp._raw_with_weed(
        states, old_bank, route, carry.suffix, player
    )
    suffix_action, suffix = hp._room_evac(
        states, suffix_action, suffix, player
    )
    suffix_action, suffix = hp._repay(
        suffix_action,
        suffix,
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
    action = _select_action(switched, suffix_action, base_action)
    return action, FusionChampionCarryV6(
        base=base,
        suffix=suffix,
        switched=switched,
    )


def fc11i_fc2b_shadow_suffix_component_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV6,
    switch_step: jax.Array,
    suffix_route: jax.Array,
    blend_mode: jax.Array,
    player: int,
):
    """Ablate market versus unit components of a shadow project suffix.

    ``blend_mode`` is 0 for the untouched FC2B source, 1 for suffix market
    transactions with FC2B unit scheduling, 2 for suffix unit scheduling with
    FC2B market transactions, and 3 for the complete suffix.  Both carries are
    advanced from the same real state on every step.  This is a diagnostic
    capability probe, not a deployable identity-aware policy.
    """

    base_action, base = fc2b_rank14_plus_clone_aware_preempt_player_action_v1(
        states,
        tables,
        latest_bank,
        old_bank,
        runtime,
        carry.base,
        player,
    )
    route = suffix_route.astype(jnp.int32)
    suffix_action, suffix = hp._raw_with_weed(
        states, old_bank, route, carry.suffix, player
    )
    suffix_action, suffix = hp._room_evac(
        states, suffix_action, suffix, player
    )
    suffix_action, suffix = hp._repay(
        suffix_action,
        suffix,
        states.step.astype(jnp.int32),
        hp.MODE_BOATLEE,
    )
    suffix_action = hp._rank_sell_slots_exact(states, runtime, suffix_action)
    suffix_action = lp._room_guard(states, suffix_action, player)
    suffix_action = lp._x562_seed_budget_guard(states, suffix_action, player)
    suffix_action = lp._terminal_liquidation(states, suffix_action, player)
    suffix_action = lp._x562_idle_fertilizer_sale(states, suffix_action, player)

    active = (
        (blend_mode > 0)
        & (states.step.astype(jnp.int32) >= switch_step.astype(jnp.int32))
    )
    use_suffix_market = active & ((blend_mode == 1) | (blend_mode == 3))
    use_suffix_units = active & ((blend_mode == 2) | (blend_mode == 3))
    unit_mask = use_suffix_units[:, None]
    market_mask = use_suffix_market[:, None]
    action = base_action._replace(
        unit_op=jnp.where(unit_mask, suffix_action.unit_op, base_action.unit_op),
        unit_item=jnp.where(unit_mask, suffix_action.unit_item, base_action.unit_item),
        unit_amount=jnp.where(unit_mask, suffix_action.unit_amount, base_action.unit_amount),
        unit_count=jnp.where(use_suffix_units, suffix_action.unit_count, base_action.unit_count),
        market_op=jnp.where(market_mask, suffix_action.market_op, base_action.market_op),
        market_item=jnp.where(market_mask, suffix_action.market_item, base_action.market_item),
        market_amount=jnp.where(market_mask, suffix_action.market_amount, base_action.market_amount),
        market_count=jnp.where(
            use_suffix_market, suffix_action.market_count, base_action.market_count
        ),
    )
    switched = carry.switched | active
    return action, FusionChampionCarryV6(
        base=base,
        suffix=suffix,
        switched=switched,
    )


def fc4f_rank14_plus_rank12_threshold_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV4,
    player: int,
):
    """FC2B plus one train-only-selected public wool-supply rule.

    At state step 120 exactly, a wool market inventory no greater than 9983
    locks K320 to route 4 (the two-YARN route).  The threshold, step and route
    were selected by grouped training-bank OOF; two independent event banks
    were kept as untouched holdouts.  The rule sees neither opponent identity
    nor future events.  The one-shot flags prevent a later market move from
    silently changing the original decision semantics.
    """

    step = states.step.astype(jnp.int32)
    evaluate_now = (~carry.rank12_route_checked) & (step == 120)
    rescue_now = evaluate_now & (states.market_inventory[:, 7] <= 9983)
    checked = carry.rank12_route_checked | (step >= 120)
    rescue = carry.rank12_route_rescue | rescue_now
    k320_seed = carry.k320._replace(
        ray_route_locked=carry.k320.ray_route_locked | rescue_now,
        ray_route_id=jnp.where(
            rescue_now,
            jnp.full_like(carry.k320.ray_route_id, 4),
            carry.k320.ray_route_id,
        ).astype(jnp.int8),
    )
    k320_action, k320 = k320_clone_aware_preempt_player_action_v1(
        states,
        tables,
        runtime,
        latest_bank,
        k320_seed,
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
    return action, FusionChampionCarryV4(
        k320=k320,
        x562_suffix=x562_suffix,
        sheep_pressure=sheep_pressure,
        rank12_route_checked=checked,
        rank12_route_rescue=rescue,
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


def boatlee_v21_horizon_counter_player_action_v1(
    states,
    runtime,
    bank,
    carry: latest6.BoatleeV21CarryV1,
    enabled: jax.Array,
    horizon: jax.Array,
    distance_limit: jax.Array,
    quantity_cap: jax.Array,
    minimum_future_quantity: jax.Array,
    minimum_price_percent: jax.Array,
    start_step: jax.Array,
    product_mask: jax.Array,
    player: int,
):
    """Ablatable B21 counter that advances only its active Munib sale lane.

    The public B21 route tree remains the sole owner of production, movement,
    recovery, and route selection.  This candidate changes one decision: while
    the selected route is Munib, compare the next ``horizon`` public tape
    sales with current own inventory and optionally sell a bounded quantity
    earlier.  The normal repayment ledger removes the advanced quantity at the
    original due step, so the candidate cannot sell the same stock twice.
    """

    action, next_carry = latest6.boatlee_v21_player_action_v1(
        states, runtime, bank, carry, player
    )
    route = jnp.full(states.step.shape, 16, dtype=jnp.int32)
    active_horizon = jnp.where(enabled, horizon, 0).astype(jnp.int16)
    counter_action, counter_base = _parameterized_cumulative_preempt_player(
        states,
        bank,
        route,
        action,
        next_carry.munib_front,
        active_horizon,
        distance_limit,
        quantity_cap,
        minimum_future_quantity,
        minimum_price_percent,
        start_step,
        product_mask,
        player,
    )
    use_counter = enabled & (next_carry.route == 2)
    return _select_action(use_counter, counter_action, action), next_carry._replace(
        munib_front=_select_tree(
            use_counter, counter_base, next_carry.munib_front
        )
    )


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


def _raw_with_weed_hire_barrier(
    states,
    bank,
    route: jax.Array,
    carry: hp.HighPotentialV20CarryV1,
    hire_guard_horizon: jax.Array,
    minimum_hires: jax.Array,
    farmer_only: jax.Array,
    player: int,
):
    """Keep unit-index semantics stable across a planned HIRE transaction.

    A weed repair inserts one extra unit action.  If that debt is still active
    when HIRE is processed, the shifted unit position can change deterministic
    hand spawn order.  All later per-index tape actions then target the wrong
    tiles.  This guard uses only the agent's own future transaction tape: when
    a HIRE lies within ``hire_guard_horizon`` steps, it cancels an existing
    shift and declines to start a new one.  The blocked BUILD/PLANT may be lost,
    but the rest of the farm remains synchronized.
    """

    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    previous = jnp.clip(step - 1, 0, 718)
    count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    present = jnp.arange(hp.MAX_UNITS)[None, :] < count[:, None]
    op = jnp.where(present, bank.unit_op[route, step], hp.UnitOp.PASS).astype(jnp.int8)
    item = jnp.where(present, bank.unit_item[route, step], -1).astype(jnp.int8)
    amount = jnp.where(present, bank.unit_amount[route, step], 1).astype(jnp.int32)

    future_hire_count = jnp.zeros_like(step, dtype=jnp.int16)
    for offset in range(13):
        future_step = jnp.clip(step + offset, 0, 718)
        within = offset <= hire_guard_horizon.astype(jnp.int32)
        future_hire_count = future_hire_count + jnp.sum(
            within[:, None] & (bank.market_op[route, future_step] == MarketOp.HIRE),
            axis=1,
        ).astype(jnp.int16)
    future_hire = future_hire_count >= minimum_hires.astype(jnp.int16)
    actor_index = jnp.arange(hp.MAX_UNITS)[None, :]
    guarded_actor = (~farmer_only[:, None]) | (actor_index == 0)
    barrier = future_hire[:, None] & guarded_actor

    age = step[:, None] - carry.weed_start.astype(jnp.int32)
    existing = carry.weed_active & present & (age <= 9) & (~barrier)
    use_intended = existing & (age == 1)
    use_replay = existing & (age >= 2) & (age <= 9)
    op = jnp.where(use_intended, carry.weed_intended_op, op)
    item = jnp.where(use_intended, carry.weed_intended_item, item)
    amount = jnp.where(use_intended, carry.weed_intended_amount, amount)
    op = jnp.where(use_replay, bank.unit_op[route, previous], op)
    item = jnp.where(use_replay, bank.unit_item[route, previous], item)
    amount = jnp.where(use_replay, bank.unit_amount[route, previous], amount)
    trigger = (
        (~existing)
        & (~barrier)
        & present
        & ((op == hp.UnitOp.BUILD_PASTURE) | (op == hp.UnitOp.PLANT))
        & (hp._tile_under_units(states, player) == hp.TileKind.WEED)
    )
    intended_op, intended_item, intended_amount = op, item, amount
    op = jnp.where(trigger, hp.UnitOp.DIG, op).astype(jnp.int8)
    item = jnp.where(trigger, -1, item).astype(jnp.int8)
    amount = jnp.where(trigger, 1, amount).astype(jnp.int32)
    carry = carry._replace(
        weed_active=existing | trigger,
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


def _x562_route_rule_weed_hire_guard_player_action_v1(
    states,
    runtime,
    bank,
    carry: X562RouteRuleCarryV1,
    use_guard: jax.Array,
    farmer_horizon: jax.Array,
    hand_horizon: jax.Array,
    farmer_minimum_hires: jax.Array,
    hand_minimum_hires: jax.Array,
    player: int,
):
    """Frozen X562 12/7 route with one independently ablatable weed barrier."""

    step = states.step.astype(jnp.int32)
    active = (
        jnp.arange(states.town_shops.shape[1])[None, :]
        < states.town_count[:, None]
    )
    has_yarn = jnp.any(active & (states.town_shops == 7), axis=1)
    proposed = jnp.where(has_yarn, 7, 12).astype(jnp.int8)
    decide = (~carry.decided) & (step >= 168)
    route_id = jnp.where(decide, proposed, carry.route_id).astype(jnp.int8)
    decided = carry.decided | decide
    route = route_id.astype(jnp.int32)

    source_action, source_base = hp._raw_with_weed(
        states, bank, route, carry.base, player
    )
    guarded_action, guarded_base = _raw_with_weed_split_hire_barrier(
        states,
        bank,
        route,
        carry.base,
        farmer_horizon,
        hand_horizon,
        farmer_minimum_hires,
        hand_minimum_hires,
        player,
    )
    action = _select_action(use_guard, guarded_action, source_action)
    base = _select_tree(use_guard, guarded_base, source_base)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(action, base, step, hp.MODE_BOATLEE)
    action = hp._rank_sell_slots_exact(states, runtime, action)
    action, base = lp._x562_preempt(states, bank, route, action, base, player)
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


def x562_prt_suffix_weed_hire_guard_player_action_v1(
    states,
    runtime,
    latest_bank,
    old_bank,
    carry: X562OldSuffixCarryV1,
    use_guard: jax.Array,
    farmer_horizon: jax.Array,
    hand_horizon: jax.Array,
    farmer_minimum_hires: jax.Array,
    hand_minimum_hires: jax.Array,
    player: int,
):
    """Apply the same ablatable HIRE barrier to X562 prefix and PRT suffix.

    With ``use_guard=False`` this is a semantic control for the deployed X562
    12/7 route plus the frozen public-state step-288 PRT selector.  The guard
    reads only the candidate's own deterministic future transaction tape.
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
    choose_suffix = (
        (states.money[:, player] <= 13376)
        & (shop_7_count <= 1)
        & (states.market_inventory[:, 3] <= 9979)
    )
    switched = carry.switched | (
        (step == 288) & choose_suffix
    )

    prefix_action, prefix_next = _x562_route_rule_weed_hire_guard_player_action_v1(
        states,
        runtime,
        latest_bank,
        carry.x562,
        use_guard,
        farmer_horizon,
        hand_horizon,
        farmer_minimum_hires,
        hand_minimum_hires,
        player,
    )

    route = jnp.full((batch,), 81, dtype=jnp.int32)
    source_action, source_next = hp._raw_with_weed(
        states, old_bank, route, carry.suffix, player
    )
    guarded_action, guarded_next = _raw_with_weed_split_hire_barrier(
        states,
        old_bank,
        route,
        carry.suffix,
        farmer_horizon,
        hand_horizon,
        farmer_minimum_hires,
        hand_minimum_hires,
        player,
    )
    suffix_action = _select_action(use_guard, guarded_action, source_action)
    suffix_next = _select_tree(use_guard, guarded_next, source_next)
    suffix_action, suffix_next = hp._room_evac(
        states, suffix_action, suffix_next, player
    )
    suffix_action, suffix_next = hp._repay(
        suffix_action, suffix_next, step, hp.MODE_BOATLEE
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


def x562_prt_suffix_weed_hire_guard_residual_sale_player_action_v1(
    states,
    runtime,
    latest_bank,
    old_bank,
    carry: X562OldSuffixCarryV1,
    use_guard: jax.Array,
    farmer_horizon: jax.Array,
    hand_horizon: jax.Array,
    farmer_minimum_hires: jax.Array,
    hand_minimum_hires: jax.Array,
    sale_enabled: jax.Array,
    sale_start_step: jax.Array,
    clone_distance_limit: jax.Array,
    sale_quantity_cap: jax.Array,
    minimum_price_percent: jax.Array,
    product_mask: jax.Array,
    seat1_only: jax.Array,
    player: int,
):
    """FC15 pressure branch plus an independently ablatable residual sale.

    The production route, PRT handoff and weed/HIRE synchronization are kept
    byte-for-byte in the existing action owner above.  This extra layer may
    only sell stock left after the current action's planned sales and PICKUP
    reservations.  Activation uses the live public-board clone distance; it
    has no opponent identity, event seed or future-shop access.
    """

    action, next_carry = x562_prt_suffix_weed_hire_guard_player_action_v1(
        states,
        runtime,
        latest_bank,
        old_bank,
        carry,
        use_guard,
        farmer_horizon,
        hand_horizon,
        farmer_minimum_hires,
        hand_minimum_hires,
        player,
    )
    active = (
        sale_enabled
        & (states.step.astype(jnp.int32) >= sale_start_step.astype(jnp.int32))
        & (hp._clone_distance(states) <= clone_distance_limit.astype(jnp.int32))
        & ((~seat1_only) | (player == 1))
    )
    for product in range(hp.NUM_PRODUCTS):
        action = _append_excess_product_sale(
            states,
            action,
            product,
            active & ((product_mask & (1 << product)) != 0),
            sale_quantity_cap,
            minimum_price_percent,
            player,
        )
    return action, next_carry


def _raw_with_weed_split_hire_barrier(
    states,
    bank,
    route: jax.Array,
    carry: hp.HighPotentialV20CarryV1,
    farmer_horizon: jax.Array,
    hand_horizon: jax.Array,
    farmer_minimum_hires: jax.Array,
    hand_minimum_hires: jax.Array,
    player: int,
):
    """Actor-specific weed barrier for deterministic per-index route tapes.

    The farmer can change the shed-adjacent occupancy that determines new hand
    spawn positions, so it may need a longer look-ahead than existing hands.
    A negative hand horizon disables the barrier for hands.
    """

    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    previous = jnp.clip(step - 1, 0, 718)
    count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    present = jnp.arange(hp.MAX_UNITS)[None, :] < count[:, None]
    op = jnp.where(present, bank.unit_op[route, step], hp.UnitOp.PASS).astype(jnp.int8)
    item = jnp.where(present, bank.unit_item[route, step], -1).astype(jnp.int8)
    amount = jnp.where(present, bank.unit_amount[route, step], 1).astype(jnp.int32)

    farmer_hires = jnp.zeros_like(step, dtype=jnp.int16)
    hand_hires = jnp.zeros_like(step, dtype=jnp.int16)
    for offset in range(13):
        future_step = jnp.clip(step + offset, 0, 718)
        hires = jnp.sum(
            bank.market_op[route, future_step] == MarketOp.HIRE, axis=1
        ).astype(jnp.int16)
        farmer_hires += jnp.where(
            offset <= farmer_horizon.astype(jnp.int32), hires, 0
        ).astype(jnp.int16)
        hand_hires += jnp.where(
            (hand_horizon >= 0)
            & (offset <= hand_horizon.astype(jnp.int32)),
            hires,
            0,
        ).astype(jnp.int16)
    farmer_barrier = farmer_hires >= farmer_minimum_hires.astype(jnp.int16)
    hand_barrier = (hand_horizon >= 0) & (
        hand_hires >= hand_minimum_hires.astype(jnp.int16)
    )
    actor_index = jnp.arange(hp.MAX_UNITS)[None, :]
    barrier = jnp.where(actor_index == 0, farmer_barrier[:, None], hand_barrier[:, None])

    age = step[:, None] - carry.weed_start.astype(jnp.int32)
    existing = carry.weed_active & present & (age <= 9) & (~barrier)
    use_intended = existing & (age == 1)
    use_replay = existing & (age >= 2) & (age <= 9)
    op = jnp.where(use_intended, carry.weed_intended_op, op)
    item = jnp.where(use_intended, carry.weed_intended_item, item)
    amount = jnp.where(use_intended, carry.weed_intended_amount, amount)
    op = jnp.where(use_replay, bank.unit_op[route, previous], op)
    item = jnp.where(use_replay, bank.unit_item[route, previous], item)
    amount = jnp.where(use_replay, bank.unit_amount[route, previous], amount)
    trigger = (
        (~existing)
        & (~barrier)
        & present
        & ((op == hp.UnitOp.BUILD_PASTURE) | (op == hp.UnitOp.PLANT))
        & (hp._tile_under_units(states, player) == hp.TileKind.WEED)
    )
    intended_op, intended_item, intended_amount = op, item, amount
    op = jnp.where(trigger, hp.UnitOp.DIG, op).astype(jnp.int8)
    item = jnp.where(trigger, -1, item).astype(jnp.int8)
    amount = jnp.where(trigger, 1, amount).astype(jnp.int32)
    carry = carry._replace(
        weed_active=existing | trigger,
        weed_start=jnp.where(trigger, step[:, None], carry.weed_start).astype(jnp.int16),
        weed_intended_op=jnp.where(trigger, intended_op, carry.weed_intended_op).astype(jnp.int8),
        weed_intended_item=jnp.where(trigger, intended_item, carry.weed_intended_item).astype(jnp.int8),
        weed_intended_amount=jnp.where(trigger, intended_amount, carry.weed_intended_amount).astype(jnp.int32),
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


def k320_clone_aware_weed_hire_barrier_player_action_v1(
    states,
    tables,
    runtime,
    bank,
    carry: hp.HighPotentialV20CarryV1,
    use_hire_barrier: jax.Array,
    hire_guard_horizon: jax.Array,
    player: int,
):
    """K320 with one independently ablatable weed/HIRE synchronization guard."""

    batch = states.step.shape[0]
    route, base = hp._select_route(states, carry, player, hp.MODE_RAY_K320)
    source_action, source_base = hp._raw_with_weed(states, bank, route, base, player)
    guarded_action, guarded_base = _raw_with_weed_hire_barrier(
        states,
        bank,
        route,
        base,
        hire_guard_horizon,
        jnp.ones((batch,), dtype=jnp.int16),
        jnp.zeros((batch,), dtype=jnp.bool_),
        player,
    )
    action = _select_action(use_hire_barrier, guarded_action, source_action)
    base = _select_tree(use_hire_barrier, guarded_base, source_base)
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


def k320_clone_aware_weed_mass_hire_guard_player_action_v1(
    states,
    tables,
    runtime,
    bank,
    carry: hp.HighPotentialV20CarryV1,
    use_guard: jax.Array,
    hire_guard_horizon: jax.Array,
    minimum_hires: jax.Array,
    farmer_only: jax.Array,
    player: int,
    route_map: jax.Array | None = None,
):
    """Refined guard: protect only the actor that can perturb mass-HIRE spawn order."""

    batch = states.step.shape[0]
    route, base = hp._select_route(states, carry, player, hp.MODE_RAY_K320)
    if route_map is not None:
        route = jnp.take_along_axis(
            route_map,
            jnp.clip(route, 0, route_map.shape[1] - 1)[:, None],
            axis=1,
        )[:, 0].astype(jnp.int32)
    source_action, source_base = hp._raw_with_weed(states, bank, route, base, player)
    guarded_action, guarded_base = _raw_with_weed_hire_barrier(
        states,
        bank,
        route,
        base,
        hire_guard_horizon,
        minimum_hires,
        farmer_only,
        player,
    )
    action = _select_action(use_guard, guarded_action, source_action)
    base = _select_tree(use_guard, guarded_base, source_base)
    action, base = hp._room_evac(states, action, base, player)
    action, base = hp._repay(action, base, states.step.astype(jnp.int32), hp.MODE_RAY_K320)
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


def k320_mass_hire_guard_residual_sale_player_action_v1(
    states,
    tables,
    runtime,
    bank,
    carry: hp.HighPotentialV20CarryV1,
    enabled: jax.Array,
    start_step: jax.Array,
    distance_limit: jax.Array,
    quantity_cap: jax.Array,
    minimum_price_percent: jax.Array,
    product_mask: jax.Array,
    seat1_only: jax.Array,
    player: int,
    route_map: jax.Array | None = None,
):
    """Ablatable sale of uncommitted premium leftovers in near mirrors.

    The source production tape, weed/HIRE barrier, preemption and market
    counters are unchanged.  This layer can only sell stock that remains after
    the current step's planned sales and PICKUP reservations.  It is intended
    to test the repeated one-unit residual sales observed in official FC2B
    close-mirror losses; it is not a promoted policy by itself.
    """

    batch = states.step.shape[0]
    action, base = k320_clone_aware_weed_mass_hire_guard_player_action_v1(
        states,
        tables,
        runtime,
        bank,
        carry,
        jnp.ones((batch,), dtype=jnp.bool_),
        jnp.full((batch,), 2, dtype=jnp.int16),
        jnp.full((batch,), 5, dtype=jnp.int16),
        jnp.ones((batch,), dtype=jnp.bool_),
        player,
        route_map,
    )
    active = (
        enabled
        & (states.step.astype(jnp.int32) >= start_step.astype(jnp.int32))
        & (hp._clone_distance(states) <= distance_limit.astype(jnp.int32))
        & ((~seat1_only) | (player == 1))
    )
    for product in (
        PRODUCTS.index("STRAWBERRY"),
        PRODUCTS.index("MILK"),
        PRODUCTS.index("WOOL"),
    ):
        action = _append_excess_product_sale(
            states,
            action,
            product,
            active & ((product_mask & (1 << product)) != 0),
            quantity_cap,
            minimum_price_percent,
            player,
        )
    return action, base


def k320_route_map_candidate_player_action_v1(
    states,
    tables,
    runtime,
    bank,
    carry: hp.HighPotentialV20CarryV1,
    route_map: jax.Array,
    player: int,
):
    """FC15's K320 lane with only its selected route remapped.

    ``route_map`` is a per-environment lookup table over the frozen route bank.
    Every legality, weed/HIRE recovery, preemption, residual sale, market
    counter, and terminal guard stays byte-for-byte on the FC15 path.
    """

    batch = states.step.shape[0]
    premium_mask = sum(
        1 << PRODUCTS.index(item) for item in ("STRAWBERRY", "MILK", "WOOL")
    )
    return k320_mass_hire_guard_residual_sale_player_action_v1(
        states,
        tables,
        runtime,
        bank,
        carry,
        jnp.ones((batch,), dtype=jnp.bool_),
        jnp.full((batch,), 120, dtype=jnp.int16),
        jnp.full((batch,), 6, dtype=jnp.int16),
        jnp.ones((batch,), dtype=jnp.int16),
        jnp.full((batch,), 100, dtype=jnp.int16),
        jnp.full((batch,), premium_mask, dtype=jnp.int16),
        jnp.zeros((batch,), dtype=jnp.bool_),
        player,
        route_map,
    )


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


_PET_CAFE_ID = SHOP_NAMES.index("PET_CAFE")
_WHEAT_ID = PRODUCTS.index("WHEAT")
_CARROT_ID = PRODUCTS.index("CARROT")


def _pet_cafe_project_shift_v1(
    states,
    action,
    seed_switch_step: jax.Array,
    plant_switch_step: jax.Array,
    sale_switch_step: jax.Array,
    sale_mode: jax.Array,
    animal_mode: jax.Array,
    animal_switch_step: jax.Array,
    goose_cap: jax.Array,
    feed_mode: jax.Array,
    feed_switch_step: jax.Array,
    release_step: jax.Array,
    player: int,
):
    """Ablatable demand-driven wheat-to-carrot project edit.

    The rule reads only the shared shop state.  It activates when the first
    three revealed shops are all PET_CAFE, whose official demand is CARROT.
    Dates and optional animal-service edits are parameters of the experiment,
    not values embedded from one Replay.

    animal_mode: 0 keeps source purchases; 1 converts future cow purchases to
    geese up to goose_cap.
    feed_mode: 0 keeps source service; 1 skips FEED and wheat-product buying
    on alternating days; 2 stops them from release_step onward; 3 combines
    alternating service before release with a full stop afterward.
    sale_mode: 0 keeps source sales; 1 reuses an existing wheat-sale slot for
    available carrot at the same step.
    """

    step = states.step.astype(jnp.int32)
    triple_pet = (states.town_count >= 3) & jnp.all(
        states.town_shops[:, :3] == _PET_CAFE_ID,
        axis=1,
    )
    unit_slots = jnp.arange(action.unit_op.shape[1])[None, :]
    unit_active = unit_slots < action.unit_count[:, None]
    convert_plant = (
        triple_pet[:, None]
        & (step >= plant_switch_step.astype(jnp.int32))[:, None]
        & unit_active
        & (action.unit_op == hp.UnitOp.PLANT)
        & (action.unit_item == _WHEAT_ID)
    )

    day = step // TURNS_PER_DAY
    alternating_skip = (
        ((feed_mode == 1) | (feed_mode == 3))
        & (step >= feed_switch_step.astype(jnp.int32))
        & ((day & 1) == 1)
    )
    terminal_release = ((feed_mode == 2) | (feed_mode == 3)) & (
        step >= release_step.astype(jnp.int32)
    )
    skip_feed = triple_pet & (alternating_skip | terminal_release)
    remove_feed = (
        skip_feed[:, None]
        & unit_active
        & (action.unit_op == hp.UnitOp.FEED)
    )
    unit_op = jnp.where(remove_feed, hp.UnitOp.PASS, action.unit_op).astype(jnp.int8)
    unit_item = jnp.where(
        convert_plant,
        _CARROT_ID,
        jnp.where(remove_feed, -1, action.unit_item),
    ).astype(jnp.int8)
    unit_amount = jnp.where(remove_feed, 0, action.unit_amount).astype(jnp.int32)
    action = action._replace(
        unit_op=unit_op,
        unit_item=unit_item,
        unit_amount=unit_amount,
    )

    market_slots = jnp.arange(hp.MAX_MARKET_ORDERS)[None, :]
    market_active = market_slots < action.market_count[:, None]
    market_op = action.market_op
    market_item = action.market_item
    market_amount = action.market_amount.astype(jnp.int32)
    convert_seed = (
        triple_pet[:, None]
        & (step >= seed_switch_step.astype(jnp.int32))[:, None]
        & market_active
        & (market_op == MarketOp.BUY_SEED)
        & (market_item == _WHEAT_ID)
    )
    market_item = jnp.where(convert_seed, _CARROT_ID, market_item).astype(jnp.int8)

    goose_item = hp.NUM_PRODUCTS
    cow_item = hp.NUM_PRODUCTS + 1
    geese = jnp.sum(states.tile_animal[:, player] == 0, axis=(1, 2)).astype(jnp.int32)
    for slot in range(hp.MAX_MARKET_ORDERS):
        convert_cow = (
            triple_pet
            & (animal_mode > 0)
            & (step >= animal_switch_step.astype(jnp.int32))
            & market_active[:, slot]
            & (market_op[:, slot] == MarketOp.BUY_ANIMAL)
            & (market_item[:, slot] == cow_item)
        )
        requested = jnp.maximum(market_amount[:, slot], 0)
        allowed = jnp.minimum(requested, jnp.maximum(goose_cap - geese, 0))
        cancel = convert_cow & (allowed <= 0)
        market_op = market_op.at[:, slot].set(
            jnp.where(cancel, MarketOp.NONE, market_op[:, slot]).astype(jnp.int8)
        )
        market_item = market_item.at[:, slot].set(
            jnp.where(convert_cow & ~cancel, goose_item, jnp.where(cancel, -1, market_item[:, slot])).astype(
                jnp.int8
            )
        )
        market_amount = market_amount.at[:, slot].set(
            jnp.where(convert_cow, allowed, market_amount[:, slot]).astype(jnp.int32)
        )
        geese = geese + jnp.where(convert_cow, allowed, 0)

    projected = hp._projected_shed_without_pickups(states, action, player)
    carrot_available = projected[:, _CARROT_ID].astype(jnp.int32)
    for slot in range(hp.MAX_MARKET_ORDERS):
        active_slot = market_slots[:, slot] < action.market_count
        existing_carrot_sale = (
            active_slot
            & (market_op[:, slot] == MarketOp.SELL)
            & (market_item[:, slot] == _CARROT_ID)
        )
        carrot_available = jnp.maximum(
            carrot_available
            - jnp.where(existing_carrot_sale, jnp.maximum(market_amount[:, slot], 0), 0),
            0,
        )
        wheat_sale = (
            triple_pet
            & (sale_mode > 0)
            & (step >= sale_switch_step.astype(jnp.int32))
            & active_slot
            & (market_op[:, slot] == MarketOp.SELL)
            & (market_item[:, slot] == _WHEAT_ID)
        )
        carrot_quantity = jnp.minimum(
            jnp.maximum(market_amount[:, slot], 0), carrot_available
        )
        replace_sale = wheat_sale & (carrot_quantity > 0)
        market_item = market_item.at[:, slot].set(
            jnp.where(replace_sale, _CARROT_ID, market_item[:, slot]).astype(jnp.int8)
        )
        market_amount = market_amount.at[:, slot].set(
            jnp.where(replace_sale, carrot_quantity, market_amount[:, slot]).astype(jnp.int32)
        )
        carrot_available = jnp.maximum(
            carrot_available - jnp.where(replace_sale, carrot_quantity, 0), 0
        )

    suppress_wheat_buy = (
        skip_feed[:, None]
        & market_active
        & (market_op == MarketOp.BUY_PRODUCT)
        & (market_item == _WHEAT_ID)
    )
    market_op = jnp.where(suppress_wheat_buy, MarketOp.NONE, market_op).astype(jnp.int8)
    market_item = jnp.where(suppress_wheat_buy, -1, market_item).astype(jnp.int8)
    market_amount = jnp.where(suppress_wheat_buy, 0, market_amount).astype(jnp.int32)
    action = action._replace(
        market_op=market_op,
        market_item=market_item,
        market_amount=market_amount,
    )
    keep = market_active & (action.market_op != MarketOp.NONE)
    return hp._compact_market(action, keep)


def fc7_pet_cafe_project_shift_player_action_v1(
    states,
    tables,
    latest_bank,
    old_bank,
    runtime,
    carry: FusionChampionCarryV3,
    seed_switch_step: jax.Array,
    plant_switch_step: jax.Array,
    sale_switch_step: jax.Array,
    sale_mode: jax.Array,
    animal_mode: jax.Array,
    animal_switch_step: jax.Array,
    goose_cap: jax.Array,
    feed_mode: jax.Array,
    feed_switch_step: jax.Array,
    release_step: jax.Array,
    player: int,
):
    """FC2B plus one public-demand, parameterized project shift."""

    action, carry = fc2b_rank14_plus_clone_aware_preempt_player_action_v1(
        states, tables, latest_bank, old_bank, runtime, carry, player
    )
    action = _pet_cafe_project_shift_v1(
        states,
        action,
        seed_switch_step,
        plant_switch_step,
        sale_switch_step,
        sale_mode,
        animal_mode,
        animal_switch_step,
        goose_cap,
        feed_mode,
        feed_switch_step,
        release_step,
        player,
    )
    return action, carry


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

"""Exact JAX controllers for the 2026-08-22 latest-public-six freeze.

Raw production tapes are stored in ``latest_public6_20260822_route_bank_v1.npz``.
This module owns only observation-conditioned runtime behavior.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    ANIMAL_PRODUCT,
    FLAG_FED,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    MARKET_MAX_INVENTORY,
    MARKET_MIN_INVENTORY,
    NUM_PRODUCTS,
    PRODUCTS,
    SHOP_NAMES,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State
from strategic_v5.high_potential_v20_gpu import (
    MODE_BOATLEE,
    HighPotentialRuntimeTablesV1,
    HighPotentialV20CarryV1,
    _market_counters,
    _append_or_merge_sale,
    _compact_market,
    _preempt,
    _rank_sell_slots_exact,
    _raw_with_weed,
    _repay as _hp_repay,
    _room_evac,
    _room_guard,
    _terminal_liquidation,
    _select_route,
    initialize_high_potential_v20_carry_v1,
)
from strategic_v5.latest_public8_gpu import (
    KaitoV36CarryV1,
    X562CarryV1,
    _append_market,
    _kaito_clone_distance,
    _planned_sales,
    _projected_shed_without_pickups,
    _x562_market_counters,
    _x562_preempt,
    _x562_route,
    initialize_kaito_v36_carry_v1,
    initialize_x562_carry_v1,
    kaito_v36_player_action_v1,
)
_WHEAT = PRODUCTS.index("WHEAT")
_FERTILIZER = PRODUCTS.index("FERTILIZER")
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int32)
_PREMIUM_IDS = jnp.asarray(
    [PRODUCTS.index(name) for name in ("STRAWBERRY", "MELON", "MILK", "WOOL")],
    dtype=jnp.int32,
)
_PREMIUM_LEXICAL = jnp.asarray((2, 0, 1, 3), dtype=jnp.int32)
_MOON_HORIZONS = 6
_MOON_RING = 7
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_SHOP_PREMIUM_DRAIN = jnp.asarray(
    [
        [
            (2 if len(_SHOP_PRODUCTS[name]) == 1 else 1)
            if PRODUCTS[int(product)] in _SHOP_PRODUCTS[name]
            else 0
            for product in _PREMIUM_IDS.tolist()
        ]
        for name in SHOP_NAMES
    ],
    dtype=jnp.int32,
)
_SHOP_DEMAND = jnp.asarray(
    [
        [
            (2 if len(_SHOP_PRODUCTS[name]) == 1 else 1)
            if product in _SHOP_PRODUCTS[name]
            else 0
            for product in PRODUCTS
        ]
        for name in SHOP_NAMES
    ],
    dtype=jnp.float32,
)


def _build_moon_changed_price_lut() -> jax.Array:
    """Freeze V92's three patched price curves, not the older BL curve."""

    inventories = np.arange(
        MARKET_MIN_INVENTORY, MARKET_MAX_INVENTORY + 1, dtype=np.float64
    )

    def shape(name: str, value: np.ndarray, scale: float) -> np.ndarray:
        value = np.maximum(value, 0.0)
        if name == "hinge":
            unit = value / scale
            return unit + 8.0 * np.maximum(unit - 1.0, 0.0) ** 2
        if name == "linear":
            return value
        if name == "sqrt":
            return np.sqrt(value)
        if name == "log":
            return np.log1p(value)
        raise ValueError(name)

    rows = []
    for base, scale, below_name, below_target, above_name, above_target in (
        (35.0, 450.0, "hinge", 1.0, "sqrt", 0.7),
        (60.0, 200.0, "hinge", 0.4, "sqrt", 0.6),
        (50.0, 332.0, "hinge", 0.4, "log", 0.2),
    ):
        below_delta = 10000.0 - inventories
        above_delta = inventories - 10000.0
        below_amplitude = below_target * base / shape(
            below_name, np.asarray(scale), scale
        )
        above_amplitude = above_target * base / shape(
            above_name, np.asarray(scale), scale
        )
        price = np.where(
            inventories < 10000.0,
            base + below_amplitude * shape(below_name, below_delta, scale),
            base - above_amplitude * shape(above_name, above_delta, scale),
        )
        rows.append(np.maximum(np.rint(price), 1.0).astype(np.int32))
    return jnp.asarray(np.stack(rows, axis=0))


_MOON_CHANGED_PRICE_LUT = _build_moon_changed_price_lut()
_MOON_CHANGED_PRODUCTS = (PRODUCTS.index("CARROT"), PRODUCTS.index("TOMATO"), PRODUCTS.index("EGG"))
_MOON_TERMINAL_SHEDS = jnp.asarray(
    ((4, 4), (5, 4), (5, 5), (4, 5)), dtype=jnp.int16
)
_MOON_SELLABLE = jnp.arange(NUM_PRODUCTS, dtype=jnp.int32)
_MOON_LEXICAL_RANK = jnp.asarray((7, 0, 6, 5, 3, 1, 4, 8, 2), dtype=jnp.int32)


class MoonV92CarryV1(NamedTuple):
    base: HighPotentialV20CarryV1
    race_initialized: jax.Array
    previous_inventory: jax.Array
    previous_price: jax.Array
    previous_shops: jax.Array
    previous_shop_count: jax.Array
    own_sells: jax.Array
    scores: jax.Array
    evidence: jax.Array
    horizons: jax.Array
    policy_scores: jax.Array
    policy_evidence: jax.Array
    policy_horizon: jax.Array
    debts: jax.Array


def initialize_moon_v92_carry_v1(batch_size: int) -> MoonV92CarryV1:
    return MoonV92CarryV1(
        base=initialize_high_potential_v20_carry_v1(batch_size),
        race_initialized=jnp.zeros((batch_size,), dtype=jnp.bool_),
        previous_inventory=jnp.zeros((batch_size, 4), dtype=jnp.int32),
        previous_price=jnp.zeros((batch_size, 4), dtype=jnp.int32),
        previous_shops=jnp.full((batch_size, 8), -1, dtype=jnp.int8),
        previous_shop_count=jnp.zeros((batch_size,), dtype=jnp.int8),
        own_sells=jnp.zeros((batch_size, 4), dtype=jnp.int16),
        scores=jnp.zeros((batch_size, 4, _MOON_HORIZONS), dtype=jnp.float32),
        evidence=jnp.zeros((batch_size, 4), dtype=jnp.float32),
        horizons=jnp.ones((batch_size, 4), dtype=jnp.int8),
        policy_scores=jnp.zeros((batch_size, _MOON_HORIZONS), dtype=jnp.float32),
        policy_evidence=jnp.zeros((batch_size,), dtype=jnp.float32),
        policy_horizon=jnp.ones((batch_size,), dtype=jnp.int8),
        debts=jnp.zeros((batch_size, _MOON_RING, 4), dtype=jnp.int16),
    )


def _moon_raw_sales(bank, route: jax.Array, step: jax.Array, product: int) -> jax.Array:
    step = jnp.clip(step.astype(jnp.int32), 0, 718)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < bank.market_count[
        route, step
    ][:, None]
    return jnp.sum(
        jnp.where(
            active
            & (bank.market_op[route, step] == MarketOp.SELL)
            & (bank.market_item[route, step] == product),
            jnp.maximum(bank.market_amount[route, step], 0),
            0,
        ),
        axis=1,
    ).astype(jnp.int32)


def _moon_town_drain(carry: MoonV92CarryV1, previous_step: jax.Array, ordinal: int):
    active = jnp.arange(carry.previous_shops.shape[1])[None, :] < carry.previous_shop_count[:, None]
    safe = jnp.clip(carry.previous_shops.astype(jnp.int32), 0, len(SHOP_NAMES) - 1)
    shop_drain = jnp.sum(
        jnp.where(active, _SHOP_PREMIUM_DRAIN[safe, ordinal], 0), axis=1
    )
    return jnp.where(previous_step % 4 == 0, shop_drain, 0) + (
        previous_step % 24 == 0
    ).astype(jnp.int32)


def _moon_observe(
    states: State,
    bank,
    route: jax.Array,
    carry: MoonV92CarryV1,
    *,
    embedded_v21: bool = False,
    soil_v26: bool = False,
) -> MoonV92CarryV1:
    step = states.step.astype(jnp.int32)
    current_inventory = states.market_inventory[:, _PREMIUM_IDS].astype(jnp.int32)
    current_price = states.market_price[:, _PREMIUM_IDS].astype(jnp.int32)
    scores = carry.scores * 0.999
    evidence = carry.evidence * 0.999
    policy_scores = carry.policy_scores * 0.999
    policy_evidence = carry.policy_evidence * 0.999
    horizons = carry.horizons
    evidence_threshold = 1.5 if (embedded_v21 or soil_v26) else 2.0
    previous_step = step - 1
    sequential = carry.race_initialized & (step > 0)
    clone_like = _kaito_clone_distance(states) <= 6

    for ordinal, product in enumerate(_PREMIUM_IDS.tolist()):
        delta = current_inventory[:, ordinal] - carry.previous_inventory[:, ordinal]
        opponent_supply = (
            delta
            + _moon_town_drain(carry, previous_step, ordinal)
            - carry.own_sells[:, ordinal].astype(jnp.int32)
        )
        extra = opponent_supply - _moon_raw_sales(bank, route, previous_step, product)
        usable = (
            sequential
            & (carry.previous_price[:, ordinal] > 1)
            & (current_price[:, ordinal] > 1)
            & (extra >= 4)
        )
        if embedded_v21 or soil_v26:
            usable = usable & clone_like
        evidence = evidence.at[:, ordinal].add(usable.astype(jnp.float32))
        policy_evidence = policy_evidence + usable.astype(jnp.float32)
        for horizon_index in range(_MOON_HORIZONS):
            horizon = horizon_index + 1
            expected = _moon_raw_sales(
                bank, route, previous_step + horizon, product
            )
            similarity = jnp.minimum(extra, expected).astype(jnp.float32) / jnp.maximum(
                jnp.maximum(extra, expected), 1
            ).astype(jnp.float32)
            increment = jnp.where(expected > 0, 1.0 + similarity, -0.15)
            scores = scores.at[:, ordinal, horizon_index].add(
                jnp.where(usable, increment, 0.0)
            )
            policy_scores = policy_scores.at[:, horizon_index].add(
                jnp.where(usable, increment, 0.0)
            )

        order = jnp.argsort(-scores[:, ordinal], axis=1, stable=True)
        best_index = order[:, 0]
        runner_index = order[:, 1]
        best_score = jnp.take_along_axis(
            scores[:, ordinal], best_index[:, None], axis=1
        )[:, 0]
        runner_score = jnp.take_along_axis(
            scores[:, ordinal], runner_index[:, None], axis=1
        )[:, 0]
        update = (evidence[:, ordinal] >= evidence_threshold) & (
            best_score >= runner_score + 0.25
        )
        horizons = horizons.at[:, ordinal].set(
            jnp.where(
                update,
                jnp.minimum(6, best_index + 2),
                horizons[:, ordinal],
            ).astype(jnp.int8)
        )

    policy_order = jnp.argsort(-policy_scores, axis=1, stable=True)
    policy_best = policy_order[:, 0]
    policy_runner = policy_order[:, 1]
    policy_best_score = jnp.take_along_axis(
        policy_scores, policy_best[:, None], axis=1
    )[:, 0]
    policy_runner_score = jnp.take_along_axis(
        policy_scores, policy_runner[:, None], axis=1
    )[:, 0]
    policy_update = (policy_evidence >= evidence_threshold) & (
        policy_best_score >= policy_runner_score + 0.25
    )
    policy_horizon = jnp.where(
        policy_update, jnp.minimum(6, policy_best + 2), carry.policy_horizon
    ).astype(jnp.int8)
    horizons = jnp.where(
        policy_update[:, None] & (horizons == 1),
        policy_horizon[:, None],
        horizons,
    ).astype(jnp.int8)
    shop_width = carry.previous_shops.shape[1]
    return carry._replace(
        race_initialized=jnp.ones_like(carry.race_initialized),
        previous_inventory=current_inventory,
        previous_price=current_price,
        previous_shops=states.town_shops[:, :shop_width].astype(jnp.int8),
        previous_shop_count=jnp.minimum(states.town_count, shop_width).astype(jnp.int8),
        own_sells=jnp.zeros_like(carry.own_sells),
        scores=scores,
        evidence=evidence,
        horizons=horizons,
        policy_scores=policy_scores,
        policy_evidence=policy_evidence,
        policy_horizon=policy_horizon,
    )


def _moon_compact_market(action: Action, keep: jax.Array, amount: jax.Array) -> Action:
    rank = jnp.cumsum(keep.astype(jnp.int32), axis=1) - 1
    op = jnp.zeros_like(action.market_op)
    item = jnp.full_like(action.market_item, -1)
    out_amount = jnp.zeros_like(action.market_amount)
    rows = jnp.arange(action.market_count.shape[0])
    for slot in range(MAX_MARKET_ORDERS):
        target = jnp.clip(rank[:, slot], 0, MAX_MARKET_ORDERS - 1)
        use = keep[:, slot]
        op = op.at[rows, target].set(
            jnp.where(use, action.market_op[:, slot], op[rows, target])
        )
        item = item.at[rows, target].set(
            jnp.where(use, action.market_item[:, slot], item[rows, target])
        )
        out_amount = out_amount.at[rows, target].set(
            jnp.where(use, amount[:, slot], out_amount[rows, target])
        )
    return action._replace(
        market_op=op,
        market_item=item,
        market_amount=out_amount,
        market_count=jnp.sum(keep, axis=1).astype(jnp.int8),
    )


def _moon_rank_sell_slots(
    states: State, runtime: HighPotentialRuntimeTablesV1, action: Action
) -> Action:
    """V92 ranking with its patched CARROT/TOMATO/EGG price curves."""

    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slot < action.market_count[:, None]
    sell = (
        active
        & (action.market_op == MarketOp.SELL)
        & (action.market_item >= 0)
        & (action.market_item < NUM_PRODUCTS)
    )
    safe_item = jnp.clip(action.market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    quantity = jnp.maximum(action.market_amount, 0).astype(jnp.int32)
    inventory = jnp.take_along_axis(states.market_inventory, safe_item, axis=1)
    current_quote = jnp.take_along_axis(states.market_price, safe_item, axis=1)
    later_index = jnp.clip(
        inventory + quantity - MARKET_MIN_INVENTORY,
        0,
        runtime.agent_market_price_lut.shape[1] - 1,
    )
    later_quote = runtime.agent_market_price_lut[safe_item, later_index].astype(
        jnp.int32
    )
    for changed_ordinal, product in enumerate(_MOON_CHANGED_PRODUCTS):
        later_quote = jnp.where(
            safe_item == product,
            _MOON_CHANGED_PRICE_LUT[changed_ordinal, later_index],
            later_quote,
        )
    impact = quantity.astype(jnp.float32) * jnp.maximum(
        current_quote - later_quote, 0
    ).astype(jnp.float32)
    town_active = (
        jnp.arange(states.town_shops.shape[1])[None, :]
        < states.town_count[:, None]
    )
    safe_shops = jnp.clip(
        states.town_shops.astype(jnp.int32), 0, len(SHOP_NAMES) - 1
    )
    demand_by_item = jnp.sum(
        _SHOP_DEMAND[safe_shops] * town_active[..., None], axis=1
    ) * 6.0
    demand_by_item = demand_by_item + jnp.asarray(
        [1.0] * (NUM_PRODUCTS - 1) + [0.0], dtype=jnp.float32
    )[None, :]
    demand = jnp.maximum(
        jnp.take_along_axis(demand_by_item, safe_item, axis=1), 0.25
    )
    excess = jnp.maximum(inventory + quantity - 10000, 0).astype(jnp.float32)
    urgency = jnp.minimum(1.0, (excess / demand) / 10.0)
    score = jnp.where(sell, impact * (1.0 + 0.25 * urgency), -jnp.inf)
    ranked_index = jnp.argsort(-score, axis=1, stable=True)
    ranked_op = jnp.take_along_axis(action.market_op, ranked_index, axis=1)
    ranked_item = jnp.take_along_axis(action.market_item, ranked_index, axis=1)
    ranked_amount = jnp.take_along_axis(action.market_amount, ranked_index, axis=1)
    ordinal = jnp.maximum(jnp.cumsum(sell, axis=1) - 1, 0)
    return action._replace(
        market_op=jnp.where(
            sell, jnp.take_along_axis(ranked_op, ordinal, axis=1), action.market_op
        ),
        market_item=jnp.where(
            sell,
            jnp.take_along_axis(ranked_item, ordinal, axis=1),
            action.market_item,
        ),
        market_amount=jnp.where(
            sell,
            jnp.take_along_axis(ranked_amount, ordinal, axis=1),
            action.market_amount,
        ),
    )


def _moon_repay(
    states: State, action: Action, carry: MoonV92CarryV1
) -> tuple[Action, MoonV92CarryV1]:
    step = states.step.astype(jnp.int32)
    ring = step % _MOON_RING
    due = carry.debts[jnp.arange(step.shape[0]), ring].astype(jnp.int32)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    keep = active.copy()
    amount = action.market_amount.astype(jnp.int32)
    for slot in range(MAX_MARKET_ORDERS):
        is_sell = active[:, slot] & (action.market_op[:, slot] == MarketOp.SELL)
        adjusted = jnp.zeros_like(is_sell)
        for ordinal, product in enumerate(_PREMIUM_IDS.tolist()):
            matching = (
                is_sell
                & (action.market_item[:, slot] == product)
                & (due[:, ordinal] > 0)
            )
            adjusted = adjusted | matching
            reduction = jnp.where(
                matching,
                jnp.minimum(jnp.maximum(amount[:, slot], 0), due[:, ordinal]),
                0,
            )
            amount = amount.at[:, slot].add(-reduction)
            due = due.at[:, ordinal].add(-reduction)
        # Only a SELL order consumed entirely by a due preemption debt is
        # removed.  HIRE/BUY_LAND and other zero-amount atomic orders remain.
        keep = keep.at[:, slot].set(
            active[:, slot] & ~(adjusted & (amount[:, slot] <= 0))
        )
    debts = carry.debts.at[jnp.arange(step.shape[0]), ring].set(0)
    return _moon_compact_market(action, keep, amount), carry._replace(debts=debts)


def _moon_preempt(
    states: State,
    bank,
    route: jax.Array,
    action: Action,
    carry: MoonV92CarryV1,
    player: int,
    *,
    embedded_v21: bool = False,
    soil_v26: bool = False,
) -> tuple[Action, MoonV92CarryV1]:
    step = states.step.astype(jnp.int32)
    clone_like = _kaito_clone_distance(states) <= 6
    remaining = jnp.maximum(
        _projected_shed_without_pickups(states, action, player)[:, :NUM_PRODUCTS]
        - _planned_sales(action),
        0,
    )
    valid = []
    choice_h = []
    targets = []
    values = []
    for ordinal, product in enumerate(_PREMIUM_IDS.tolist()):
        chosen_h = jnp.zeros(step.shape, dtype=jnp.int32)
        future_quantity = jnp.zeros(step.shape, dtype=jnp.int32)
        preferred = carry.horizons[:, ordinal].astype(jnp.int32)
        if embedded_v21:
            preferred = jnp.where(
                (step >= 120) & (_kaito_clone_distance(states) <= 2),
                jnp.maximum(preferred, 3),
                preferred,
            )
        for horizon in range(_MOON_HORIZONS, 0, -1):
            quantity = _moon_raw_sales(bank, route, step + horizon, product)
            choose = (chosen_h == 0) & (horizon <= preferred) & (quantity >= 4)
            chosen_h = jnp.where(choose, horizon, chosen_h)
            future_quantity = jnp.where(choose, quantity, future_quantity)
        target = jnp.minimum(
            jnp.minimum(remaining[:, product], future_quantity), 12
        )
        allowed = clone_like
        if not (embedded_v21 or soil_v26):
            allowed = allowed | (
                jnp.maximum(
                    carry.evidence[:, ordinal], carry.policy_evidence
                )
                >= 2.0
            )
        row_valid = (
            (step >= 120)
            & (step < 680)
            & (action.market_count < MAX_MARKET_ORDERS)
            & (target > 0)
            & allowed
        )
        valid.append(row_valid)
        choice_h.append(chosen_h)
        targets.append(target)
        values.append(states.market_price[:, product].astype(jnp.int32) * target)

    any_adapted = jnp.zeros(step.shape, dtype=jnp.bool_)
    best_value = jnp.full(step.shape, -1, dtype=jnp.int32)
    best_lex = jnp.full(step.shape, -1, dtype=jnp.int32)
    best_target = jnp.full(step.shape, -1, dtype=jnp.int32)
    best_h = jnp.full(step.shape, -1, dtype=jnp.int32)
    best_ordinal = jnp.full(step.shape, -1, dtype=jnp.int32)
    for ordinal in range(4):
        adapted = valid[ordinal] & (choice_h[ordinal] > 1)
        improve = adapted & (
            (values[ordinal] > best_value)
            | (
                (values[ordinal] == best_value)
                & (
                    (_PREMIUM_LEXICAL[ordinal] > best_lex)
                    | (
                        (_PREMIUM_LEXICAL[ordinal] == best_lex)
                        & (
                            (targets[ordinal] > best_target)
                            | (
                                (targets[ordinal] == best_target)
                                & (choice_h[ordinal] > best_h)
                            )
                        )
                    )
                )
            )
        )
        best_value = jnp.where(improve, values[ordinal], best_value)
        best_lex = jnp.where(improve, _PREMIUM_LEXICAL[ordinal], best_lex)
        best_target = jnp.where(improve, targets[ordinal], best_target)
        best_h = jnp.where(improve, choice_h[ordinal], best_h)
        best_ordinal = jnp.where(improve, ordinal, best_ordinal)
        any_adapted = any_adapted | adapted

    debts = carry.debts
    for ordinal, product in enumerate(_PREMIUM_IDS.tolist()):
        select = valid[ordinal] & jnp.where(
            any_adapted,
            best_ordinal == ordinal,
            clone_like
            if (embedded_v21 or soil_v26)
            else jnp.ones_like(clone_like),
        )
        append = select & (action.market_count < MAX_MARKET_ORDERS)
        action = _append_market(
            action, append, MarketOp.SELL, product, targets[ordinal]
        )
        due_ring = (step + choice_h[ordinal]) % _MOON_RING
        rows = jnp.arange(step.shape[0])
        current = debts[rows, due_ring, ordinal].astype(jnp.int32)
        debts = debts.at[rows, due_ring, ordinal].set(
            (current + jnp.where(append, targets[ordinal], 0)).astype(jnp.int16)
        )
    return action, carry._replace(debts=debts)


def _moon_record_own_sells(
    states: State, action: Action, carry: MoonV92CarryV1, player: int
) -> MoonV92CarryV1:
    remaining = _projected_shed_without_pickups(states, action, player).astype(jnp.int32)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    sold = jnp.zeros_like(carry.own_sells, dtype=jnp.int32)
    rows = jnp.arange(states.step.shape[0])
    for slot in range(MAX_MARKET_ORDERS):
        for ordinal, product in enumerate(_PREMIUM_IDS.tolist()):
            matching = (
                active[:, slot]
                & (action.market_op[:, slot] == MarketOp.SELL)
                & (action.market_item[:, slot] == product)
            )
            quantity = jnp.where(
                matching,
                jnp.minimum(
                    jnp.maximum(action.market_amount[:, slot], 0),
                    remaining[:, product],
                ),
                0,
            )
            sold = sold.at[:, ordinal].add(quantity)
            remaining = remaining.at[rows, product].add(-quantity)
    return carry._replace(own_sells=sold.astype(jnp.int16))


def _moon_terminal_move(
    states: State, player: int, actor: int, target: jax.Array
) -> jax.Array:
    """V20 movement: try horizontal first, then vertical if blocked."""

    rows = jnp.arange(states.step.shape[0])
    position = states.unit_pos[:, player, actor].astype(jnp.int32)
    x, y = position[:, 0], position[:, 1]
    tx, ty = target[:, 0].astype(jnp.int32), target[:, 1].astype(jnp.int32)
    horizontal = jnp.where(
        tx < x, UnitOp.WEST, jnp.where(tx > x, UnitOp.EAST, UnitOp.PASS)
    ).astype(jnp.int8)
    hx = (
        x
        + (horizontal == UnitOp.EAST).astype(jnp.int32)
        - (horizontal == UnitOp.WEST).astype(jnp.int32)
    )
    horizontal_open = (
        (horizontal != UnitOp.PASS)
        & (states.tile_kind[rows, player, jnp.clip(y, 0, 9), jnp.clip(hx, 0, 9)] != TileKind.LOCKED)
    )
    vertical = jnp.where(
        ty < y, UnitOp.NORTH, jnp.where(ty > y, UnitOp.SOUTH, UnitOp.PASS)
    ).astype(jnp.int8)
    vy = (
        y
        + (vertical == UnitOp.SOUTH).astype(jnp.int32)
        - (vertical == UnitOp.NORTH).astype(jnp.int32)
    )
    vertical_open = (
        (vertical != UnitOp.PASS)
        & (states.tile_kind[rows, player, jnp.clip(vy, 0, 9), jnp.clip(x, 0, 9)] != TileKind.LOCKED)
    )
    return jnp.where(
        horizontal_open,
        horizontal,
        jnp.where(vertical_open, vertical, UnitOp.PASS),
    ).astype(jnp.int8)


def _moon_terminal_action(states: State, player: int) -> Action:
    """Exact Moon V92 terminal harvest/bank/liquidation controller."""

    batch_size = states.step.shape[0]
    rows = jnp.arange(batch_size)
    unit_count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    unit_op = jnp.full((batch_size, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8)
    unit_item = jnp.full((batch_size, MAX_UNITS), -1, dtype=jnp.int8)
    unit_amount = jnp.ones((batch_size, MAX_UNITS), dtype=jnp.int32)
    available = (
        (states.tile_yield[:, player] > 0)
        & (states.tile_kind[:, player] != TileKind.LOCKED)
    ).reshape(batch_size, 100)
    pending = jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int32)
    tile_index = jnp.arange(100)[None, :]
    tx = tile_index % 10
    ty = tile_index // 10
    for actor in range(MAX_UNITS):
        present = actor < unit_count
        position = states.unit_pos[:, player, actor].astype(jnp.int32)
        x, y = position[:, 0], position[:, 1]
        inventory = states.unit_inventory[:, player, actor].astype(jnp.int32)
        load = jnp.sum(jnp.maximum(inventory, 0), axis=1)
        at_shed = jnp.any(
            jnp.all(position[:, None, :] == _MOON_TERMINAL_SHEDS[None, :, :], axis=2),
            axis=1,
        )
        current_yield = (
            states.tile_yield[
                rows, player, jnp.clip(y, 0, 9), jnp.clip(x, 0, 9)
            ]
            > 0
        )
        shed_distance = jnp.sum(
            jnp.abs(position[:, None, :] - _MOON_TERMINAL_SHEDS[None, :, :]),
            axis=2,
        )
        shed_target = _MOON_TERMINAL_SHEDS[jnp.argmin(shed_distance, axis=1)]
        distance = jnp.abs(tx - x[:, None]) + jnp.abs(ty - y[:, None])
        key = jnp.where(available, distance * 100 + tile_index, 1_000_000)
        target_index = jnp.argmin(key, axis=1)
        has_target = jnp.any(available, axis=1)
        target = jnp.stack((target_index % 10, target_index // 10), axis=1).astype(
            jnp.int16
        )
        move_shed = _moon_terminal_move(states, player, actor, shed_target)
        move_yield = _moon_terminal_move(states, player, actor, target)
        tile_kind = states.tile_kind[
            rows, player, jnp.clip(y, 0, 9), jnp.clip(x, 0, 9)
        ]
        tile_flags = states.tile_flags[
            rows, player, jnp.clip(y, 0, 9), jnp.clip(x, 0, 9)
        ]
        fertilizer_available = (tile_flags & 8) != 0
        chosen = jnp.where(
            (load > 0) & at_shed,
            UnitOp.DROP,
            jnp.where(
                current_yield,
                UnitOp.HARVEST,
                jnp.where(
                    load > 0,
                    move_shed,
                    jnp.where(
                        has_target,
                        move_yield,
                        jnp.where(
                            fertilizer_available & (tile_kind != TileKind.LOCKED),
                            UnitOp.COLLECT_FERTILIZER,
                            UnitOp.PASS,
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int8)
        unit_op = unit_op.at[:, actor].set(
            jnp.where(present, chosen, UnitOp.PASS)
        )
        drop = present & (chosen == UnitOp.DROP)
        pending = pending + jnp.where(
            drop[:, None], inventory[:, :NUM_PRODUCTS], 0
        )
        claim = (
            present
            & ~((load > 0) & at_shed)
            & ~current_yield
            & (load == 0)
            & has_target
        )
        available = available.at[rows, target_index].set(
            jnp.where(claim, False, available[rows, target_index])
        )
        current_index = jnp.clip(y, 0, 9) * 10 + jnp.clip(x, 0, 9)
        available = available.at[rows, current_index].set(
            jnp.where(present & current_yield, False, available[rows, current_index])
        )

    shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32) + pending
    values = shed * states.market_price.astype(jnp.int32)
    valid = shed > 0
    rank_key = jnp.where(valid, values * 32 + _MOON_LEXICAL_RANK[None, :], -1)
    ranked = jnp.argsort(-rank_key, axis=1, stable=True)
    action = Action(
        unit_op,
        unit_item,
        unit_amount,
        unit_count,
        jnp.zeros((batch_size, MAX_MARKET_ORDERS), dtype=jnp.int8),
        jnp.full((batch_size, MAX_MARKET_ORDERS), -1, dtype=jnp.int8),
        jnp.zeros((batch_size, MAX_MARKET_ORDERS), dtype=jnp.int32),
        jnp.zeros((batch_size,), dtype=jnp.int8),
    )
    for ordinal in range(NUM_PRODUCTS):
        product = ranked[:, ordinal]
        quantity = jnp.take_along_axis(shed, product[:, None], axis=1)[:, 0]
        append = (quantity > 0) & (action.market_count < MAX_MARKET_ORDERS)
        action = _append_market(
            action, append, MarketOp.SELL, product.astype(jnp.int8), quantity
        )
    return action


def prvsiyan_moon_v92_player_action_v1(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    carry: MoonV92CarryV1,
    player: int,
    *,
    embedded_v21: bool = False,
) -> tuple[Action, MoonV92CarryV1]:
    """V92 base controller; source-specific crop overlays are added below."""

    route, base = _select_route(states, carry.base, player, MODE_BOATLEE)
    carry = carry._replace(base=base)
    carry = _moon_observe(
        states, bank, route, carry, embedded_v21=embedded_v21
    )

    action, base = _raw_with_weed(states, bank, route, carry.base, player)
    action, base = _room_evac(states, action, base, player)
    carry = carry._replace(base=base)
    action, carry = _moon_repay(states, action, carry)
    action = _moon_rank_sell_slots(states, runtime, action)
    action, carry = _moon_preempt(
        states,
        bank,
        route,
        action,
        carry,
        player,
        embedded_v21=embedded_v21,
    )
    action, base = _market_counters(
        states, action, carry.base, runtime, player
    )
    carry = carry._replace(base=base)
    action = _room_guard(states, action, player)
    action = _terminal_liquidation(states, action, player)
    terminal = states.step.astype(jnp.int32) >= 708
    terminal_action = _moon_terminal_action(states, player)
    action = Action(
        *(
            jnp.where(
                terminal.reshape((terminal.shape[0],) + (1,) * (left.ndim - 1)),
                left,
                right,
            )
            for left, right in zip(terminal_action, action, strict=True)
        )
    )
    carry = _moon_record_own_sells(states, action, carry, player)
    return action, carry


def initialize_soil_v26h_carry_v1(batch_size: int) -> MoonV92CarryV1:
    """Soil inherits Moon's public-market observer and repayment ledger."""

    return initialize_moon_v92_carry_v1(batch_size)


def _soil_terminal_feed_horizon(
    states: State, action: Action, player: int
) -> Action:
    """Exact V26-H late feed admission test."""

    rows = jnp.arange(states.step.shape[0])
    day = states.step.astype(jnp.int32) // 24
    price_gate = (
        states.market_price[:, _WHEAT].astype(jnp.int32)
        >= states.market_price[:, _FERTILIZER].astype(jnp.int32) + 15
    )
    unit_op = action.unit_op
    # Source-specific schedule: goose, cow, sheep.
    first_yield = jnp.asarray((4, 8, 6), dtype=jnp.int32)
    interval = jnp.asarray((2, 2, 3), dtype=jnp.int32)
    for actor in range(MAX_UNITS):
        position = states.unit_pos[:, player, actor].astype(jnp.int32)
        x = jnp.clip(position[:, 0], 0, 9)
        y = jnp.clip(position[:, 1], 0, 9)
        animal = states.tile_animal[rows, player, y, x].astype(jnp.int32)
        safe_animal = jnp.clip(animal, 0, 2)
        placed = states.tile_origin_day[rows, player, y, x].astype(jnp.int32)
        placed = jnp.where(placed < 0, day, placed)
        usable = jnp.zeros(day.shape, dtype=jnp.bool_)
        for feed_day in range(29):
            delta = feed_day + 1 - placed - first_yield[safe_animal]
            usable = usable | (
                (feed_day >= day)
                & (delta >= 0)
                & (delta % interval[safe_animal] == 0)
            )
        skip = (
            (actor < action.unit_count)
            & (action.unit_op[:, actor] == UnitOp.FEED)
            & (day <= 28)
            & price_gate
            & (animal >= 0)
            & (states.tile_yield[rows, player, y, x] == 0)
            & (states.tile_neglect[rows, player, y, x] == 0)
            & ~usable
        )
        unit_op = unit_op.at[:, actor].set(
            jnp.where(skip, UnitOp.PASS, unit_op[:, actor])
        )
    return action._replace(unit_op=unit_op)


def prvsiyan_soil_v26h_player_action_v1(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    carry: MoonV92CarryV1,
    player: int,
) -> tuple[Action, MoonV92CarryV1]:
    """Soil V26-H fixed modal route plus its inherited market controller.

    This first exact slice deliberately leaves the optional tomato substitution
    and terminal feed-horizon overlay inert.  The strict official traces decide
    whether either branch is reachable before those state machines are added.
    """

    # Soil explicitly monkey-patches Moon's ``_kawa_actions`` to the same
    # modal route.  Both execution and premium-sale forecasting therefore use
    # route 9, independent of the public shop sequence.
    forecast_route = jnp.full(states.step.shape, 9, dtype=jnp.int32)
    carry = _moon_observe(
        states, bank, forecast_route, carry, soil_v26=True
    )
    execution_route = forecast_route
    action, base = _raw_with_weed(
        states, bank, execution_route, carry.base, player
    )
    carry = carry._replace(base=base)
    action, carry = _moon_repay(states, action, carry)
    action = _moon_rank_sell_slots(states, runtime, action)
    action, carry = _moon_preempt(
        states,
        bank,
        forecast_route,
        action,
        carry,
        player,
        soil_v26=True,
    )
    action = _soil_terminal_feed_horizon(states, action, player)
    terminal = states.step.astype(jnp.int32) >= 712
    terminal_action = _moon_terminal_action(states, player)
    action = Action(
        *(
            jnp.where(
                terminal.reshape((terminal.shape[0],) + (1,) * (left.ndim - 1)),
                left,
                right,
            )
            for left, right in zip(terminal_action, action, strict=True)
        )
    )
    carry = _moon_record_own_sells(states, action, carry, player)
    return action, carry


class KaitoV39CarryV1(NamedTuple):
    base: KaitoV36CarryV1
    route_decided: jax.Array
    route_id: jax.Array


def initialize_kaito_v39_carry_v1(batch_size: int) -> KaitoV39CarryV1:
    return KaitoV39CarryV1(
        base=initialize_kaito_v36_carry_v1(batch_size),
        route_decided=jnp.zeros((batch_size,), dtype=jnp.bool_),
        route_id=jnp.full((batch_size,), 11, dtype=jnp.int8),
    )


def kaito_v39_history_gate_player_action_v1(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    carry: KaitoV39CarryV1,
    player: int,
) -> tuple[Action, KaitoV39CarryV1]:
    """V39's accepted base branch with its step-96 V36/V37 split."""

    step = states.step.astype(jnp.int32)
    decide = (~carry.route_decided) & (step >= 96)
    first_yarn = (
        (states.town_count > 0)
        & (states.town_shops[:, 0] == SHOP_NAMES.index("YARN_STORE"))
    )
    route_id = jnp.where(
        decide, jnp.where(first_yarn, 17, 11), carry.route_id
    ).astype(jnp.int8)
    action, base = kaito_v36_player_action_v1(
        states, runtime, bank, route_id.astype(jnp.int32), carry.base, player
    )
    return action, carry._replace(
        base=base,
        route_decided=carry.route_decided | decide,
        route_id=route_id,
    )


class BoatleeV21CarryV1(NamedTuple):
    """Runtime state for V21's Moon/Mutoy/Munib public router."""

    moon: MoonV92CarryV1
    mutoy: HighPotentialV20CarryV1
    munib_base: HighPotentialV20CarryV1
    munib_front: HighPotentialV20CarryV1
    route: jax.Array
    market_overlay: jax.Array
    previous_opponent_money: jax.Array


def initialize_boatlee_v21_carry_v1(batch_size: int) -> BoatleeV21CarryV1:
    return BoatleeV21CarryV1(
        moon=initialize_moon_v92_carry_v1(batch_size),
        mutoy=initialize_high_potential_v20_carry_v1(batch_size),
        munib_base=initialize_high_potential_v20_carry_v1(batch_size),
        munib_front=initialize_high_potential_v20_carry_v1(batch_size),
        route=jnp.full((batch_size,), -1, dtype=jnp.int8),
        market_overlay=jnp.zeros((batch_size,), dtype=jnp.bool_),
        previous_opponent_money=jnp.full((batch_size,), 3000, dtype=jnp.int32),
    )


def _choose_action(mask: jax.Array, left: Action, right: Action) -> Action:
    return Action(
        *(
            jnp.where(
                mask.reshape((mask.shape[0],) + (1,) * (lhs.ndim - 1)), lhs, rhs
            )
            for lhs, rhs in zip(left, right, strict=True)
        )
    )


def _action_equal(left: Action, right: Action) -> jax.Array:
    equal = jnp.ones(left.unit_count.shape, dtype=jnp.bool_)
    for lhs, rhs in zip(left, right, strict=True):
        equal = equal & jnp.all(lhs == rhs, axis=tuple(range(1, lhs.ndim))) if lhs.ndim > 1 else equal & (lhs == rhs)
    return equal


def _boatlee_town_demand_now(
    states: State, product: int, step: jax.Array
) -> jax.Array:
    demand = ((product != _FERTILIZER) & (step % 24 == 0)).astype(jnp.int32)
    active = (
        jnp.arange(states.town_shops.shape[1])[None, :]
        < states.town_count[:, None]
    )
    safe = jnp.clip(
        states.town_shops.astype(jnp.int32), 0, len(SHOP_NAMES) - 1
    )
    shop_demand = jnp.sum(
        jnp.where(active, _SHOP_DEMAND[safe, product], 0), axis=1
    ).astype(jnp.int32)
    return demand + jnp.where(step % 4 == 0, shop_demand, 0)


def _boatlee_future_sale(
    bank, route: int, step: jax.Array, product: int
) -> jax.Array:
    future = jnp.clip(step.astype(jnp.int32) + 1, 0, 718)
    active = (
        jnp.arange(MAX_MARKET_ORDERS)[None, :]
        < bank.market_count[route, future][:, None]
    )
    return jnp.sum(
        jnp.where(
            active
            & (bank.market_op[route, future] == MarketOp.SELL)
            & (bank.market_item[route, future] == product),
            jnp.maximum(bank.market_amount[route, future], 0),
            0,
        ),
        axis=1,
    ).astype(jnp.int32)


def _boatlee_front_run(
    states: State,
    bank,
    action: Action,
    carry: HighPotentialV20CarryV1,
    player: int,
) -> tuple[Action, HighPotentialV20CarryV1]:
    """Exact one-step Munib front-run and repayment ledger."""

    step = states.step.astype(jnp.int32)
    action, carry = _hp_repay(action, carry, step, MODE_BOATLEE)
    present = jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
    moved = jnp.zeros((step.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    for product in (
        PRODUCTS.index("WOOL"),
        PRODUCTS.index("MILK"),
        PRODUCTS.index("MELON"),
        PRODUCTS.index("STRAWBERRY"),
    ):
        target = _boatlee_future_sale(bank, 16, step, product)
        pickup = jnp.sum(
            jnp.where(
                present
                & (action.unit_op == UnitOp.PICKUP)
                & (action.unit_item == product),
                jnp.maximum(action.unit_amount, 0),
                0,
            ),
            axis=1,
        ).astype(jnp.int32)
        active = (
            jnp.arange(MAX_MARKET_ORDERS)[None, :]
            < action.market_count[:, None]
        )
        existing = jnp.sum(
            jnp.where(
                active
                & (action.market_op == MarketOp.SELL)
                & (action.market_item == product),
                jnp.maximum(action.market_amount, 0),
                0,
            ),
            axis=1,
        ).astype(jnp.int32)
        stock = states.shed[:, player, product].astype(jnp.int32)
        quantity = jnp.minimum(target, jnp.maximum(stock - pickup - existing, 0))
        enabled = (
            (target > 0)
            & (_boatlee_town_demand_now(states, product, step) == 0)
            & (quantity > 0)
        )
        action, changed = _append_or_merge_sale(
            action, enabled, product, quantity
        )
        moved = moved.at[:, product].set(jnp.where(changed, quantity, 0))
    changed = jnp.sum(moved, axis=1) > 0
    carry = carry._replace(
        due_step=jnp.where(changed, step + 1, carry.due_step).astype(jnp.int16),
        due=jnp.where(changed[:, None], moved, carry.due).astype(jnp.int16),
    )
    return action, carry


def _boatlee_tape_action(
    states: State,
    bank,
    route: int,
    carry: HighPotentialV20CarryV1,
    player: int,
) -> tuple[Action, HighPotentialV20CarryV1]:
    fixed = jnp.full(states.step.shape, route, dtype=jnp.int32)
    return _raw_with_weed(states, bank, fixed, carry, player)


def _boatlee_sell_totals(action: Action) -> jax.Array:
    active = (
        jnp.arange(MAX_MARKET_ORDERS)[None, :]
        < action.market_count[:, None]
    )
    totals = jnp.zeros((action.market_count.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    for product in range(NUM_PRODUCTS):
        totals = totals.at[:, product].set(
            jnp.sum(
                jnp.where(
                    active
                    & (action.market_op == MarketOp.SELL)
                    & (action.market_item == product),
                    jnp.maximum(action.market_amount, 0),
                    0,
                ),
                axis=1,
            )
        )
    return totals


def _boatlee_market_delta(
    action: Action, base_action: Action, shifted_action: Action
) -> Action:
    """Apply Munib-front's per-product sale delta to Moon's action."""

    delta = _boatlee_sell_totals(shifted_action) - _boatlee_sell_totals(base_action)
    for product in range(NUM_PRODUCTS):
        positive = jnp.maximum(delta[:, product], 0)
        action, _ = _append_or_merge_sale(
            action, positive > 0, product, positive
        )
        remaining = jnp.maximum(-delta[:, product], 0)
        active = (
            jnp.arange(MAX_MARKET_ORDERS)[None, :]
            < action.market_count[:, None]
        )
        amount = action.market_amount.astype(jnp.int32)
        keep = active.copy()
        for slot in range(MAX_MARKET_ORDERS):
            matching = (
                active[:, slot]
                & (action.market_op[:, slot] == MarketOp.SELL)
                & (action.market_item[:, slot] == product)
            )
            reduction = jnp.where(
                matching, jnp.minimum(jnp.maximum(amount[:, slot], 0), remaining), 0
            )
            amount = amount.at[:, slot].add(-reduction)
            remaining = remaining - reduction
            keep = keep.at[:, slot].set(~(matching & (amount[:, slot] <= 0)))
        action = _compact_market(action._replace(market_amount=amount), keep)
    return action


def boatlee_v21_player_action_v1(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    carry: BoatleeV21CarryV1,
    player: int,
) -> tuple[Action, BoatleeV21CarryV1]:
    """V21's public-state route tree with all three children kept live."""

    step = states.step.astype(jnp.int32)
    opponent_money = states.money[:, 1 - player].astype(jnp.int32)
    opponent_hires = states.hires_today[:, 1 - player].astype(jnp.int32)

    moon_action, moon = prvsiyan_moon_v92_player_action_v1(
        states, runtime, bank, carry.moon, player, embedded_v21=True
    )
    mutoy_action, mutoy = _boatlee_tape_action(
        states, bank, 15, carry.mutoy, player
    )
    munib_base_action, munib_base = _boatlee_tape_action(
        states, bank, 16, carry.munib_base, player
    )
    munib_action, munib_front = _boatlee_tape_action(
        states, bank, 16, carry.munib_front, player
    )
    munib_action, munib_front = _boatlee_front_run(
        states, bank, munib_action, munib_front, player
    )

    route = carry.route
    early_mutoy = (
        (step == 1)
        & (route < 0)
        & (opponent_hires >= 4)
        & (opponent_money <= 20)
    )
    route = jnp.where(early_mutoy, 1, route).astype(jnp.int8)
    spend = carry.previous_opponent_money.astype(jnp.int32) - opponent_money
    overlay = carry.market_overlay | ((step == 217) & (spend >= 100))

    divergent = ~_action_equal(moon_action, munib_action)
    first_three_ready = states.town_count >= 3
    bakery = first_three_ready & jnp.all(
        states.town_shops[:, :3] == SHOP_NAMES.index("BAKERY"), axis=1
    )
    extra = first_three_ready & jnp.all(
        states.town_shops[:, :3]
        == jnp.asarray(
            (
                SHOP_NAMES.index("PET_CAFE"),
                SHOP_NAMES.index("ICE_CREAM_SHOP"),
                SHOP_NAMES.index("ICE_CREAM_SHOP"),
            ),
            dtype=states.town_shops.dtype,
        )[None, :],
        axis=1,
    )
    choose_now = (route < 0) & divergent
    selected = jnp.where((step < 200) | bakery | extra, 2, 0).astype(jnp.int8)
    route = jnp.where(choose_now, selected, route).astype(jnp.int8)

    moon_overlay = _boatlee_market_delta(
        moon_action, munib_base_action, munib_action
    )
    moon_result = _choose_action(overlay, moon_overlay, moon_action)
    result = _choose_action(route == 2, munib_action, moon_result)
    result = _choose_action(route == 1, mutoy_action, result)
    # Source returns Moon at step zero after advancing every child.
    result = _choose_action(step == 0, moon_action, result)
    return result, BoatleeV21CarryV1(
        moon=moon,
        mutoy=mutoy,
        munib_base=munib_base,
        munib_front=munib_front,
        route=route,
        market_overlay=overlay,
        previous_opponent_money=opponent_money,
    )


class LatestE284CarryV1(NamedTuple):
    """E279 route state plus X631's skipped-feed wheat credit."""

    e279: X562CarryV1
    wheat_credit: jax.Array


def initialize_latest_e284_carry_v1(batch_size: int) -> LatestE284CarryV1:
    return LatestE284CarryV1(
        e279=initialize_x562_carry_v1(batch_size),
        wheat_credit=jnp.zeros((batch_size,), dtype=jnp.int16),
    )


def _e283_base_action(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    carry: X562CarryV1,
    player: int,
    *,
    preempt: bool,
) -> tuple[Action, X562CarryV1]:
    """Shared E279/E283 action stack; no X562-only deployment patches."""

    route, carry = _x562_route(states, carry)
    action, base = _raw_with_weed(states, bank, route, carry.base, player)
    # The inherited V17 feed guard is present in source but frozen off.
    action, base = _room_evac(states, action, base, player)
    action, base = _hp_repay(
        action, base, states.step.astype(jnp.int32), MODE_BOATLEE
    )
    action = _rank_sell_slots_exact(states, runtime, action)
    if preempt:
        action, base = _x562_preempt(states, bank, route, action, base, player)
    action, base = _x562_market_counters(states, action, base, runtime, player)
    action = _room_guard(states, action, player)
    action = _terminal_liquidation(states, action, player)
    return action, carry._replace(base=base)


def salem_harvestforge_x_player_action_v1(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    carry: X562CarryV1,
    player: int,
) -> tuple[Action, X562CarryV1]:
    """Exact E283 public route: E279 selector with preemption disabled."""

    return _e283_base_action(
        states, runtime, bank, carry, player, preempt=False
    )


def _x631_skip_feed_and_credit(
    states: State,
    action: Action,
    credit: jax.Array,
    player: int,
) -> tuple[Action, jax.Array]:
    batch = jnp.arange(states.step.shape[0])
    present = jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
    position = states.unit_pos[:, player].astype(jnp.int32)
    x = jnp.clip(position[..., 0], 0, 9)
    y = jnp.clip(position[..., 1], 0, 9)
    animal = states.tile_animal[batch[:, None], player, y, x].astype(jnp.int32)
    valid_animal = (animal >= 0) & (animal < _ANIMAL_PRODUCT.shape[0])
    safe_animal = jnp.clip(animal, 0, _ANIMAL_PRODUCT.shape[0] - 1)
    product = _ANIMAL_PRODUCT[safe_animal]
    fed_today = (
        states.tile_flags[batch[:, None], player, y, x] & FLAG_FED
    ) != 0
    first_unfed_day = (
        states.tile_neglect[batch[:, None], player, y, x].astype(jnp.int32) == 0
    )
    pending = jnp.maximum(
        states.tile_pending_care[batch[:, None], player, y, x].astype(jnp.int32), 0
    )
    product_price = jnp.take_along_axis(
        states.market_price, product, axis=1
    ).astype(jnp.int32)
    skip = (
        (states.step.astype(jnp.int32) // 24 >= 10)[:, None]
        & present
        & (action.unit_op == UnitOp.FEED)
        & valid_animal
        & first_unfed_day
        & (~fed_today)
        & (product_price * (1 + pending) < states.market_price[:, _WHEAT, None])
    )
    action = action._replace(
        unit_op=jnp.where(skip, UnitOp.PASS, action.unit_op).astype(jnp.int8),
        unit_item=jnp.where(skip, -1, action.unit_item).astype(jnp.int8),
        unit_amount=jnp.where(skip, 1, action.unit_amount).astype(jnp.int32),
    )
    return action, (credit.astype(jnp.int32) + jnp.sum(skip, axis=1)).astype(jnp.int16)


def _x631_trim_wheat_buys(action: Action, credit: jax.Array) -> tuple[Action, jax.Array]:
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    keep = active.copy()
    amount = action.market_amount.astype(jnp.int32)
    remaining = credit.astype(jnp.int32)
    for slot in range(MAX_MARKET_ORDERS):
        wheat_buy = (
            active[:, slot]
            & (action.market_op[:, slot] == MarketOp.BUY_PRODUCT)
            & (action.market_item[:, slot] == _WHEAT)
        )
        take = jnp.where(
            wheat_buy,
            jnp.minimum(jnp.maximum(amount[:, slot], 0), remaining),
            0,
        )
        amount = amount.at[:, slot].add(-take)
        remaining = remaining - take
        keep = keep.at[:, slot].set(
            keep[:, slot] & (~(wheat_buy & (amount[:, slot] <= 0)))
        )

    # Stable compaction mirrors the Python list-comprehension removal.
    rank = jnp.cumsum(keep.astype(jnp.int32), axis=1) - 1
    new_op = jnp.zeros_like(action.market_op)
    new_item = jnp.full_like(action.market_item, -1)
    new_amount = jnp.zeros_like(action.market_amount)
    for slot in range(MAX_MARKET_ORDERS):
        target = jnp.clip(rank[:, slot], 0, MAX_MARKET_ORDERS - 1)
        use = keep[:, slot]
        rows = jnp.arange(action.market_count.shape[0])
        new_op = new_op.at[rows, target].set(
            jnp.where(use, action.market_op[:, slot], new_op[rows, target])
        )
        new_item = new_item.at[rows, target].set(
            jnp.where(use, action.market_item[:, slot], new_item[rows, target])
        )
        new_amount = new_amount.at[rows, target].set(
            jnp.where(use, amount[:, slot], new_amount[rows, target])
        )
    return (
        action._replace(
            market_op=new_op,
            market_item=new_item,
            market_amount=new_amount,
            market_count=jnp.sum(keep, axis=1).astype(jnp.int8),
        ),
        remaining.astype(jnp.int16),
    )


def _x631_liquidate_fertilizer(
    states: State, action: Action, player: int
) -> Action:
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    already = jnp.sum(
        jnp.where(
            active
            & (action.market_op == MarketOp.SELL)
            & (action.market_item == _FERTILIZER),
            jnp.maximum(action.market_amount, 0),
            0,
        ),
        axis=1,
    )
    held = states.shed[:, player, _FERTILIZER].astype(jnp.int32)
    extra = held - 14 - already
    enabled = (extra > 0) & (action.market_count < MAX_MARKET_ORDERS)
    return _append_market(action, enabled, MarketOp.SELL, _FERTILIZER, extra)


def steven_e284_hadouken_player_action_v1(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    carry: LatestE284CarryV1,
    player: int,
) -> tuple[Action, LatestE284CarryV1]:
    """Exact E284 quote-gated preemption plus X631 Hadouken rules."""

    action, e279 = _e283_base_action(
        states, runtime, bank, carry.e279, player, preempt=True
    )
    action, credit = _x631_skip_feed_and_credit(
        states, action, carry.wheat_credit, player
    )
    action, credit = _x631_trim_wheat_buys(action, credit)
    action = _x631_liquidate_fertilizer(states, action, player)
    return action, carry._replace(e279=e279, wheat_credit=credit)

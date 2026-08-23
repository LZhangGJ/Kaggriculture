"""Exact per-product sales ledger for synchronized official market batches.

The official simulator deliberately exposes only the post-step state.  That is
enough for score parity, but not for deciding whether a route really belongs
to (for example) the tomato or egg income family.  This module mirrors the
official 1.32.7 lockstep market loop and records only successful SELL units and
their quoted gross revenue.  It accepts the state *after* the unit phase, which
matters when a worker deposits inventory and sells it in the same turn.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from kaggriculture_jax.constants import (
    EPISODE_STEPS,
    NUM_PLAYERS,
    NUM_PRODUCTS,
    TURNS_PER_DAY,
    MarketOp,
)
from kaggriculture_jax.simulator import (
    _decay_plants,
    _end_of_day,
    _batch_buy_land_player,
    _batch_commit_market_player,
    _batch_hire_player,
    _batch_market_quote,
    _refresh_prices_batch,
    _town_consume,
    batched_project_unit_phase,
)
from kaggriculture_jax.types import Action, Events, State, StaticTables


class MarketSalesLedgerV1(NamedTuple):
    """Successful sales, shaped ``[batch, player, product]``."""

    units: jax.Array
    gross_revenue: jax.Array


class MarketLedgerResultV1(NamedTuple):
    final_state: State
    sales: MarketSalesLedgerV1


class StepWithSalesLedgerResultV1(NamedTuple):
    """One complete synchronized official step plus successful sales."""

    final_state: State
    sales: MarketSalesLedgerV1


def batch_market_sales_ledger_sync_v1(
    projected_states: State,
    actions: Action,
    tables: StaticTables,
) -> MarketLedgerResultV1:
    """Run the exact synchronized market phase and retain a sales ledger.

    ``projected_states`` must already include the official unit phase.  The
    returned state is therefore comparable to
    ``batched_project_action_phases_sync`` rather than a full environment step.
    Private simulator helpers are imported intentionally and are guarded by a
    state-parity regression test; this experiment is frozen to 1.32.7.
    """

    batch_size = projected_states.money.shape[0]
    max_count = jnp.max(actions.market_count).astype(jnp.int32)
    empty = jnp.zeros(
        (batch_size, NUM_PLAYERS, NUM_PRODUCTS), dtype=jnp.int32
    )

    def order_cond(carry) -> jax.Array:
        order_index, *_ = carry
        return order_index < max_count

    def order_body(carry):
        order_index, batch_states, sale_units, sale_revenue = carry
        within = order_index < actions.market_count
        ops = jnp.where(
            within, actions.market_op[:, :, order_index], MarketOp.NONE
        ).astype(jnp.int8)
        items = actions.market_item[:, :, order_index]
        remaining = actions.market_amount[:, :, order_index]

        for player in range(NUM_PLAYERS):
            batch_states = _batch_hire_player(
                batch_states, player, ops[:, player] == MarketOp.HIRE
            )
            batch_states = _batch_buy_land_player(
                batch_states, player, ops[:, player] == MarketOp.BUY_LAND
            )
        atomic = (ops == MarketOp.HIRE) | (ops == MarketOp.BUY_LAND)
        active = (~atomic) & (ops != MarketOp.NONE) & (remaining > 0)

        def unit_cond(unit_carry) -> jax.Array:
            _, rem, act, iteration, *_ = unit_carry
            return jnp.any(act & (rem > 0)) & (iteration < 99_999)

        def unit_body(unit_carry):
            current, rem, act, iteration, units, revenue = unit_carry
            act = act & (rem > 0)
            current, prices, quoted = _batch_market_quote(
                current, ops, items, act, tables
            )
            act = act & quoted
            successes = []
            for player in range(NUM_PLAYERS):
                current, ok = _batch_commit_market_player(
                    current,
                    player,
                    ops[:, player],
                    items[:, player],
                    prices[:, player],
                    act[:, player],
                )
                successes.append(ok)
            success = jnp.stack(successes, axis=1)
            sold = success & (ops == MarketOp.SELL)
            safe_item = jnp.clip(items.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
            product = jax.nn.one_hot(
                safe_item, NUM_PRODUCTS, dtype=jnp.int32
            )
            valid_product = (items >= 0) & (items < NUM_PRODUCTS)
            sold_product = sold & valid_product
            units = units + product * sold_product[..., None].astype(jnp.int32)
            revenue = revenue + product * (
                sold_product.astype(jnp.int32) * prices.astype(jnp.int32)
            )[..., None]
            rem = rem - success.astype(jnp.int32)
            act = act & success
            return current, rem, act, iteration + 1, units, revenue

        (
            batch_states,
            remaining_after,
            active_after,
            iterations,
            sale_units,
            sale_revenue,
        ) = lax.while_loop(
            unit_cond,
            unit_body,
            (
                batch_states,
                remaining,
                active,
                jnp.asarray(0, dtype=jnp.int32),
                sale_units,
                sale_revenue,
            ),
        )
        del remaining_after
        hit_cap = (iterations >= 99_999) & jnp.any(active_after, axis=1)
        batch_states = batch_states._replace(
            market_loop_cap_hits=batch_states.market_loop_cap_hits
            + hit_cap.astype(jnp.int32)
        )
        batch_states = _refresh_prices_batch(batch_states, tables)
        return order_index + 1, batch_states, sale_units, sale_revenue

    _, final_state, sale_units, sale_revenue = lax.while_loop(
        order_cond,
        order_body,
        (jnp.asarray(0, dtype=jnp.int32), projected_states, empty, empty),
    )
    return MarketLedgerResultV1(
        final_state=final_state,
        sales=MarketSalesLedgerV1(
            units=sale_units,
            gross_revenue=sale_revenue,
        ),
    )


def batched_step_with_sales_ledger_sync_v1(
    states: State,
    actions: Action,
    events: Events,
    tables: StaticTables,
) -> StepWithSalesLedgerResultV1:
    """Advance one exact synchronized step without simulating the market twice.

    This is the same unit -> lockstep market -> town -> decay -> optional
    end-of-day suffix as ``batched_step_sync``.  The only addition is the
    successful SELL ledger produced inside the already-required market phase.
    """

    step0 = states.step[0]
    done0 = states.done[0]
    empty = jnp.zeros(
        (states.step.shape[0], NUM_PLAYERS, NUM_PRODUCTS), dtype=jnp.int32
    )

    def advance(batch: State) -> tuple[State, MarketSalesLedgerV1]:
        day = (step0 // TURNS_PER_DAY).astype(jnp.int8)
        projected = batched_project_unit_phase(batch, actions)
        market = batch_market_sales_ledger_sync_v1(projected, actions, tables)
        following = jax.vmap(_town_consume, in_axes=(0, None))(
            market.final_state, tables
        )
        following = jax.vmap(_decay_plants)(following)
        following = jax.lax.cond(
            ((step0.astype(jnp.int32) + 1) % TURNS_PER_DAY) == 0,
            lambda value: jax.vmap(_end_of_day, in_axes=(0, 0, None))(
                value, events, day
            ),
            lambda value: value,
            following,
        )
        terminal = step0.astype(jnp.int32) >= EPISODE_STEPS - 2
        reward = jnp.where(terminal, following.money, following.reward)
        following = following._replace(
            step=(following.step + 1).astype(jnp.int16),
            reward=reward.astype(jnp.int32),
            done=jnp.full_like(following.done, terminal),
        )
        return following, market.sales

    final_state, sales = jax.lax.cond(
        done0,
        lambda batch: (
            batch,
            MarketSalesLedgerV1(units=empty, gross_revenue=empty),
        ),
        advance,
        states,
    )
    return StepWithSalesLedgerResultV1(final_state=final_state, sales=sales)

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import load_tables, reset
from kaggriculture_jax.simulator import _refresh_prices, _refresh_prices_batch


def test_hinge_prices_use_int32_without_overflow() -> None:
    tables = load_tables()
    assert tables.market_price.dtype == jnp.int32

    inventories = jnp.asarray(
        [10000, -32768, -32768, 10000, 10000, -32768, 10000, 10000, 10000],
        dtype=jnp.int32,
    )
    state = reset()._replace(market_inventory=inventories)
    refreshed = jax.jit(_refresh_prices)(state, tables)
    prices = np.asarray(refreshed.market_price)

    assert prices.dtype == np.int32
    assert int(prices[1]) == 2_479_547  # CARROT
    assert int(prices[2]) == 8_702_958  # TOMATO
    assert int(prices[5]) == 2_616_669  # EGG
    assert int(refreshed.price_lut_oob) == 0


def test_batched_hinge_price_refresh_matches_scalar() -> None:
    tables = load_tables()
    inventories = jnp.asarray(
        [10000, 9100, 9600, 10000, 10000, 9336, 10000, 10000, 10000],
        dtype=jnp.int32,
    )
    scalar = jax.jit(_refresh_prices)(
        reset()._replace(market_inventory=inventories), tables
    )
    states = jax.tree.map(lambda value: jnp.stack([value, value]), reset())
    states = states._replace(
        market_inventory=jnp.stack([inventories, inventories])
    )
    batched = jax.jit(_refresh_prices_batch)(states, tables)
    np.testing.assert_array_equal(
        np.asarray(batched.market_price),
        np.stack([np.asarray(scalar.market_price), np.asarray(scalar.market_price)]),
    )

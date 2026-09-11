from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import reset
from kaggriculture_jax.constants import TileKind


def test_reset_is_jittable_and_vmappable() -> None:
    single = jax.jit(reset)(jnp.int32(7))
    batch = jax.jit(jax.vmap(reset))(jnp.arange(64, dtype=jnp.int32))
    jax.block_until_ready(batch)
    np.testing.assert_array_equal(single.money, [3000, 3000])
    assert single.tile_kind.shape == (2, 10, 10)
    assert int(np.sum(np.asarray(single.tile_kind) == TileKind.EMPTY)) == 50
    assert batch.money.shape == (64, 2)


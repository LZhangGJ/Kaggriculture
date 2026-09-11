"""Full Hasegawa winning-Replay bank used by the observable-prefix router."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class HasegawaTraceBankV3(NamedTuple):
    unit_op: jax.Array
    unit_item: jax.Array
    unit_amount: jax.Array
    unit_count: jax.Array
    market_op: jax.Array
    market_item: jax.Array
    market_amount: jax.Array
    market_count: jax.Array
    expected_unit_pos: jax.Array
    expected_unit_active: jax.Array
    expected_money: jax.Array
    expected_self_summary: jax.Array
    expected_opponent_summary: jax.Array
    expected_shed: jax.Array
    expected_seeds: jax.Array
    expected_carried: jax.Array
    expected_market_price: jax.Array
    expected_market_inventory: jax.Array
    source_shop_sequence: jax.Array
    source_episode_id: jax.Array
    source_reward: jax.Array
    source_opponent_reward: jax.Array
    source_margin: jax.Array
    source_seat: jax.Array
    bootstrap_route_id: jax.Array


def load_hasegawa_trace_bank_v3(path: Path) -> HasegawaTraceBankV3:
    with np.load(path, allow_pickle=False) as data:
        return HasegawaTraceBankV3(
            *(jnp.asarray(data[field]) for field in HasegawaTraceBankV3._fields)
        )


__all__ = ["HasegawaTraceBankV3", "load_hasegawa_trace_bank_v3"]

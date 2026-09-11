"""Coherent Hasegawa atomic-action programs compiled from public Replays."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array


class HasegawaTraceBankV2(NamedTuple):
    unit_op: Array
    unit_item: Array
    unit_amount: Array
    unit_count: Array
    market_op: Array
    market_item: Array
    market_amount: Array
    market_count: Array
    expected_unit_pos: Array
    expected_unit_active: Array
    expected_money: Array
    source_episode_id: Array
    source_reward: Array
    source_seat: Array
    branch_sample_count: Array
    branch_shop_demand: Array


def load_hasegawa_trace_bank_v2(path: Path) -> HasegawaTraceBankV2:
    with np.load(path, allow_pickle=False) as data:
        return HasegawaTraceBankV2(
            *(jnp.asarray(data[field]) for field in HasegawaTraceBankV2._fields)
        )


__all__ = ["HasegawaTraceBankV2", "load_hasegawa_trace_bank_v2"]

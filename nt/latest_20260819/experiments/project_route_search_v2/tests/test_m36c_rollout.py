from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.state import load_tables
from kaggriculture_jax.types import Events

from project_route_search_v2.m35_genome import default_m35_farm_genome_v2
from project_route_search_v2.m36_calendar import kawashigi_opening_calendar_v3
from project_route_search_v2.m36_rollout import summarize_m36c_rollout_v3
from project_route_search_v2.m36_split_rollout import (
    initialize_m36c_split_v3,
    make_m36c_light_chunk_v3,
)


def _events(batch_size: int) -> Events:
    return Events(
        jnp.zeros((batch_size, 30, 100), dtype=jnp.bool_),
        jnp.zeros((batch_size, 30, 101), dtype=jnp.int8),
    )


def test_m36c_short_rollout_uses_opening_then_unified_scheduler():
    calendar = kawashigi_opening_calendar_v3(1)
    template = default_m35_farm_genome_v2(1)
    opening = jax.jit(initialize_m36c_split_v3)
    chunk = jax.jit(make_m36c_light_chunk_v3(chunk_steps=11))
    carry = opening(
        jnp.asarray((17,), dtype=jnp.int32),
        calendar,
        _events(1),
        load_tables(),
        template,
    )
    carry = chunk(carry, calendar, _events(1), load_tables(), template)
    summary = summarize_m36c_rollout_v3(carry)
    jax.block_until_ready(summary)
    assert int(carry.farm.environment_state.step[0]) == 12
    assert int(summary.transaction_hard_error_count[0]) == 0
    assert int(summary.illegal_action_count[0]) == 0
    assert int(summary.fertilizer_market_slot_overflow_count[0]) == 0
    assert (
        int(summary.scheduler_crop_selection_count[0])
        + int(summary.scheduler_animal_selection_count[0])
        > 0
    )

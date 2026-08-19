import jax
import jax.numpy as jnp

from kaggriculture_jax.state import reset
from project_route_search_v2.lifecycle import initialize_project_controller_v2
from project_route_search_v2.m25_rollout import empty_m25_metrics_v2
from project_route_search_v2.m26_rollout import empty_m26_coverage_v2
from project_route_search_v2.m3_rollout import empty_m3_coverage_v2, empty_m3_metrics_v2
from project_route_search_v2.m35_rollout import empty_m35_flow_v2
from project_route_search_v2.m35_schema import M35RolloutCarryV2
from project_route_search_v2.m36_rollout import (
    M36CRolloutCarryV3,
    _empty_aggregates,
)
from project_route_search_v2.m36_calendar import empty_route_calendar_v3
from project_route_search_v2.m37_lifecycle import (
    M37LifecycleAggregatesV1,
    empty_m37_lifecycle_aggregates_v1,
    update_m37_daily_lifecycle_v1,
)


def _carry(states):
    batch_size = states.step.shape[0]
    farm = M35RolloutCarryV2(
        states,
        initialize_project_controller_v2(states, 0),
        empty_m25_metrics_v2(batch_size),
        empty_m3_metrics_v2(batch_size),
        empty_m26_coverage_v2(batch_size),
        empty_m3_coverage_v2(batch_size),
        empty_m35_flow_v2(batch_size),
    )
    return M36CRolloutCarryV3(farm, _empty_aggregates(batch_size))


def test_m37_gpu_aggregate_has_fixed_small_shapes_only():
    aggregate = empty_m37_lifecycle_aggregates_v1(8)
    assert isinstance(aggregate, M37LifecycleAggregatesV1)
    assert aggregate.blocker_day_counts.shape == (8, 10)
    assert aggregate.latest_crop_target.shape == (8, 5)
    assert aggregate.latest_animal_active.shape == (8, 3)
    assert all(value.ndim <= 2 for value in jax.tree.leaves(aggregate))


def test_m37_daily_update_is_idempotent_within_same_day_and_ages_debt():
    states = jax.vmap(reset)(jnp.asarray((301,), dtype=jnp.int32))
    carry = _carry(states)
    calendar = empty_route_calendar_v3(1)._replace(
        crop_target_by_day=jnp.ones((1, 30, 5), dtype=jnp.int16),
        animal_purchase_additions_by_day=jnp.ones(
            (1, 30, 3), dtype=jnp.int16
        ),
        animal_service_target_by_day=jnp.ones((1, 30, 3), dtype=jnp.int16),
    )
    initial = empty_m37_lifecycle_aggregates_v1(1)
    first = update_m37_daily_lifecycle_v1(carry, initial, calendar, player=0)
    duplicate = update_m37_daily_lifecycle_v1(
        carry, first, calendar, player=0
    )
    assert int(first.crop_target_debt_days.sum()) == 5
    assert int(duplicate.crop_target_debt_days.sum()) == 5
    assert int(duplicate.animal_purchase_debt_days.sum()) == 3

    day_two_states = states._replace(step=jnp.asarray((24,), dtype=jnp.int16))
    day_two = update_m37_daily_lifecycle_v1(
        _carry(day_two_states), duplicate, calendar, player=0
    )
    assert int(day_two.crop_target_debt_days.sum()) == 10
    assert int(day_two.max_crop_debt_age_days.max()) == 2
    assert int(day_two.max_animal_debt_age_days.max()) == 2

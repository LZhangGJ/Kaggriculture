"""Human-readable smoke anchors before large parameter sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import NUM_DAYS

from .schema import (
    FertilizerPolicyV1,
    RouteFamilyV1,
    RouteScheduleV1,
    empty_route_schedule_v1,
)


def _stage(values: np.ndarray, start_day: int, target: list[int]) -> None:
    values[start_day:, :] = np.asarray(target, dtype=values.dtype)


def smoke_route_schedules_v1() -> RouteScheduleV1:
    """Return one deliberately distinct, plausible anchor per route family."""

    batch = 8
    base = empty_route_schedule_v1(batch)
    crops = np.zeros((batch, NUM_DAYS, 5), dtype=np.int8)
    animals = np.zeros((batch, NUM_DAYS, 3), dtype=np.int8)
    land = np.ones((batch, NUM_DAYS), dtype=np.int8)
    hires = np.full((batch, NUM_DAYS), 8, dtype=np.int8)

    # WHEAT/CARROT fast cash.
    _stage(crops[0], 0, [12, 8, 0, 0, 0])
    # TOMATO with support wheat.
    _stage(crops[1], 0, [6, 0, 18, 0, 0])
    land[1, 6:] = 2
    # STRAWBERRY with support wheat.
    _stage(crops[2], 0, [6, 0, 0, 16, 0])
    land[2, 6:] = 2
    # MELON burst with support wheat.
    _stage(crops[3], 0, [5, 0, 0, 0, 12])
    # Goose/egg.
    _stage(crops[4], 0, [6, 0, 0, 0, 4])
    _stage(animals[4], 0, [2, 0, 0])
    _stage(animals[4], 5, [4, 0, 0])
    _stage(animals[4], 9, [6, 0, 0])
    land[4, 6:] = 2
    # Cow/milk.
    _stage(crops[5], 0, [8, 0, 0, 0, 6])
    _stage(animals[5], 0, [0, 2, 0])
    _stage(animals[5], 6, [0, 4, 0])
    _stage(animals[5], 11, [0, 8, 0])
    land[5, 6:] = 2
    land[5, 11:] = 3
    # Sheep/wool.
    _stage(crops[6], 0, [8, 0, 0, 0, 5])
    _stage(animals[6], 0, [0, 0, 2])
    _stage(animals[6], 5, [0, 0, 4])
    _stage(animals[6], 10, [0, 0, 8])
    land[6, 6:] = 2
    land[6, 10:] = 3
    # Gold-like 1C4S -> 8C4S -> 75L calibration route.  These are public
    # state targets, not a 719-action copy; the deterministic executor must
    # independently satisfy them under new weeds and shops.
    crops[7] = np.asarray(
        [
            [5, 0, 0, 0, 5], [5, 0, 0, 0, 5], [5, 0, 0, 0, 5],
            [6, 0, 0, 3, 5], [6, 0, 0, 3, 5], [7, 0, 0, 7, 5],
            [7, 0, 0, 8, 5], [10, 0, 0, 18, 5], [13, 0, 0, 20, 5],
            [13, 0, 0, 20, 5], [13, 0, 0, 31, 12],
            [13, 0, 0, 36, 12], [13, 0, 0, 36, 12],
            [13, 0, 0, 36, 12], [13, 0, 0, 36, 12],
            [13, 0, 0, 36, 12], [13, 0, 0, 36, 12],
            [13, 0, 0, 36, 12], [13, 0, 0, 36, 12],
            [16, 0, 0, 33, 12], [26, 0, 0, 33, 0], [30, 0, 0, 29, 0],
            [33, 0, 0, 28, 0], [44, 0, 0, 18, 0], [45, 0, 0, 16, 0],
            [46, 0, 0, 16, 0], [56, 0, 0, 5, 0], [39, 0, 0, 5, 0],
            [26, 0, 0, 0, 0], [0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    animals[7, :, 1] = np.asarray(
        [1, 1, 1, 1, 1, 2, 3, 6, *([8] * 22)], dtype=np.int8
    )
    animals[7, :, 2] = 4
    land[7, 6:] = 2
    land[7, 10:] = 3
    hires[7] = np.asarray(
        [
            5, 1, 2, 3, 4, 3, 4, 7, 6, 7,
            14, 10, 10, 8, 9, 9, 13, 9, 11, 13,
            14, 12, 10, 14, 11, 12, 12, 11, 10, 10,
        ],
        dtype=np.int8,
    )

    # Stop long-horizon re-investment by driving targets to zero near terminal.
    crops[:7, 26:, :] = 0
    animals[:, 19:, :] = animals[:, 18:19, :]
    fertilizer = np.asarray(
        [
            FertilizerPolicyV1.SELL,
            FertilizerPolicyV1.APPLY,
            FertilizerPolicyV1.APPLY,
            FertilizerPolicyV1.APPLY,
            FertilizerPolicyV1.SELL,
            FertilizerPolicyV1.MIXED,
            FertilizerPolicyV1.SELL,
            FertilizerPolicyV1.MIXED,
        ],
        dtype=np.int8,
    )
    family = np.arange(8, dtype=np.int8)
    return base._replace(
        route_id=jnp.arange(batch, dtype=jnp.int32),
        family_id=jnp.asarray(family),
        enabled=jnp.ones((batch,), dtype=jnp.bool_),
        crop_target_by_day=jnp.asarray(crops),
        animal_target_by_day=jnp.asarray(animals),
        land_target_by_day=jnp.asarray(land),
        hire_target_by_day=jnp.asarray(hires),
        # Frozen by hire_batch_grid_v1 on the paired E0_SMOKE panel.  Hiring
        # is family-sensitive: aggressive same-step labor helps most routes,
        # while melon loses cash when its sparse early program over-hires.
        hire_batch_max=jnp.asarray([8, 6, 8, 1, 6, 4, 6, 6], dtype=jnp.int8),
        seed_batch=jnp.full((batch, 5), 6, dtype=jnp.int8),
        feed_reserve_days=jnp.asarray([1, 1, 1, 1, 2, 2, 2, 3], dtype=jnp.int8),
        feed_sell_reserve_days=jnp.asarray(
            [0, 0, 0, 0, 4, 4, 0, 2], dtype=jnp.int8
        ),
        fertilizer_policy=jnp.asarray(fertilizer),
        parallel_plant_lanes=jnp.full((batch,), 4, dtype=jnp.int8),
        parallel_plant_min_maintenance_code=jnp.full(
            (batch,), 3, dtype=jnp.int8
        ),
        # Confirmed on the frozen 128-seed E0_SEARCH panel.  Pure crop routes
        # lose watering/planting capacity from reserved harvest lanes, while
        # animal/hybrid routes benefit after same-tile service chaining.
        harvest_dispatch_lanes=jnp.asarray(
            [0, 0, 0, 0, 4, 9, 6, 9], dtype=jnp.int8
        ),
        harvest_dispatch_start_step=jnp.asarray(
            [0, 0, 0, 0, 0, 6 * 24, 0, 0], dtype=jnp.int16
        ),
        deposit_batch_units=jnp.ones((batch,), dtype=jnp.int16),
        # Confirmed on the frozen 128-seed E0_SEARCH panel.  Chained harvests
        # avoid repeated depot travel only for families whose paired median
        # and mean improve without violating the preregistered tail floor.
        crop_harvest_batch_units=jnp.asarray(
            [16, 8, 4, 1, 32, 1, 1, 16], dtype=jnp.int16
        ),
        chain_care_after_collection=jnp.asarray(
            [False, False, False, False, True, True, False, True],
            dtype=jnp.bool_,
        ),
        chain_animal_service_after_action=jnp.asarray(
            [False, False, False, False, True, True, True, True],
            dtype=jnp.bool_,
        ),
        chain_crop_harvest_after_action=jnp.asarray(
            [True, True, True, False, True, False, False, True],
            dtype=jnp.bool_,
        ),
        financing_cash_floor=jnp.zeros((batch,), dtype=jnp.int32),
        sell_interval=jnp.asarray(
            np.vstack(
                [
                    np.full((7, 9), 24, dtype=np.int16),
                    np.full((1, 9), 8, dtype=np.int16),
                ]
            )
        ),
        sell_phase=jnp.asarray(
            np.vstack(
                [
                    np.full((7, 9), 17, dtype=np.int16),
                    np.full((1, 9), 5, dtype=np.int16),
                ]
            )
        ),
        sell_price_floor=jnp.ones((batch, 9), dtype=jnp.int16),
        investment_stop_step=jnp.asarray(
            [650, 620, 600, 500, 500, 500, 500, 624], dtype=jnp.int16
        ),
        liquidation_start_step=jnp.asarray(
            [680, 680, 680, 680, 680, 680, 680, 648], dtype=jnp.int16
        ),
    )


def repeat_route_schedules_v1(
    schedules: RouteScheduleV1, repeats: int
) -> RouteScheduleV1:
    """Repeat each route consecutively for candidate-by-seed batching."""

    repeats = int(repeats)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return jax.tree.map(lambda value: jnp.repeat(value, repeats, axis=0), schedules)

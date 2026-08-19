"""Adapt a complete RouteCalendarV3 to the accepted M3.5 sub-executors."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import NUM_ANIMALS, NUM_CROPS, TURNS_PER_DAY
from kaggriculture_jax.types import State

from .m35_genome import default_m35_farm_genome_v2
from .m35_schema import M35FarmGenomeV2
from .m3_constants import M3AnimalCarePolicyV2, M3AnimalFertilizerPolicyV2
from .m36_schema import ROUTE_DAYS_V3, RouteCalendarV3


_FIRST_CYCLE_CARE_TARGET = jnp.asarray((3, 5, 5), dtype=jnp.int8)
_STEADY_CYCLE_CARE_TARGET = jnp.asarray((1, 2, 3), dtype=jnp.int8)


def _repeat_phase(value: jax.Array, phase_slots: int = 6) -> jax.Array:
    return jnp.repeat(value[:, None], phase_slots, axis=1)


def calendar_step_genome_v3(
    states: State,
    calendar: RouteCalendarV3,
    template: M35FarmGenomeV2 | None = None,
) -> M35FarmGenomeV2:
    """Return a one-phase executor genome for the current calendar day.

    The daily arrays remain the source of truth.  The six legacy phase slots
    are merely broadcast compatibility views and therefore do not truncate the
    route back to six events.
    """

    batch_size = states.step.shape[0]
    template = default_m35_farm_genome_v2(batch_size) if template is None else template
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    day = jnp.clip(
        states.step.astype(jnp.int32) // TURNS_PER_DAY, 0, ROUTE_DAYS_V3 - 1
    )
    hand = calendar.hand_target_by_day[batch, day].astype(jnp.int8)
    crop_target = calendar.crop_target_by_day[batch, day].astype(jnp.int16)
    care_policy = calendar.animal_care_policy_by_day[batch, day].astype(jnp.int8)
    care_enabled = care_policy != M3AnimalCarePolicyV2.OFF
    # Animal ownership is irreversible.  The M3 placement/build executor must
    # therefore see the cumulative purchase commitment, not the (possibly
    # lower) number selected for continued daily service.  M3.6's service mask
    # separately gates FEED/CARE and implements deliberate terminal release.
    cumulative_animal = jnp.sum(
        calendar.animal_purchase_additions_by_day
        * (
            jnp.arange(ROUTE_DAYS_V3, dtype=jnp.int32)[None, :, None]
            <= day[:, None, None]
        ),
        axis=1,
        dtype=jnp.int16,
    )
    cumulative_land = (
        1
        + jnp.sum(
            calendar.land_additions_by_day
            * (jnp.arange(ROUTE_DAYS_V3, dtype=jnp.int32)[None] <= day[:, None]),
            axis=-1,
            dtype=jnp.int16,
        )
    )
    cumulative_land = jnp.clip(cumulative_land, 1, 4).astype(jnp.int8)

    crop_last_day = jnp.max(
        jnp.where(
            calendar.crop_target_by_day > 0,
            jnp.arange(ROUTE_DAYS_V3, dtype=jnp.int16)[None, :, None],
            -1,
        ),
        axis=1,
    )
    crop_last_step = jnp.where(
        crop_last_day >= 0,
        jnp.minimum((crop_last_day + 1) * TURNS_PER_DAY - 1, 718),
        0,
    ).astype(jnp.int16)
    max_purchase_wave = jnp.maximum(
        jnp.max(calendar.animal_purchase_additions_by_day, axis=1), 1
    ).astype(jnp.int8)
    liquidation_start = jnp.full(
        (batch_size,), 27 * TURNS_PER_DAY, dtype=jnp.int16
    )

    phase_count = jnp.ones((batch_size,), dtype=jnp.int8)
    phase_start = jnp.zeros_like(template.crop.phase_start_step)
    land_phase = _repeat_phase(cumulative_land)
    hand_phase = _repeat_phase(hand)
    crop = template.crop._replace(
        candidate_id=calendar.candidate_id.astype(jnp.int32),
        phase_count=phase_count,
        phase_start_step=phase_start,
        crop_target=_repeat_phase(crop_target),
        land_target=land_phase,
        land_start_step=phase_start,
        hand_target=hand_phase,
        crop_last_plant_step=crop_last_step,
        # The source gold route harvests strawberry at one unit when early
        # liquidity is needed (26 observed one-unit harvests).  The frozen
        # M2.6 default of two is a conservative standalone setting and delays
        # the M3.6 cash loop, so the calendar executor uses the observed
        # minimum while leaving every other crop threshold unchanged.
        harvest_trigger_units=template.crop.harvest_trigger_units.at[:, 3].set(
            jnp.int8(1)
        ),
        cash_floor=jnp.zeros((batch_size,), dtype=jnp.int32),
        liquidation_start_step=jnp.full(
            (batch_size,), 27 * TURNS_PER_DAY, dtype=jnp.int16
        ),
    )
    fertilizer_safety = calendar.fertilizer_safety_stock_by_day[
        batch, day
    ].astype(jnp.int16)
    fertilizer_market_policy = jnp.where(
        fertilizer_safety > 0,
        M3AnimalFertilizerPolicyV2.RESERVE_FOR_CROPS,
        M3AnimalFertilizerPolicyV2.ACTIVE_COLLECT_AND_SELL,
    ).astype(jnp.int8)
    animal = template.animal._replace(
        candidate_id=calendar.candidate_id.astype(jnp.int32),
        phase_count=phase_count,
        phase_start_step=phase_start,
        animal_target=_repeat_phase(cumulative_animal),
        land_target=land_phase,
        hand_target=hand_phase,
        # A calendar addition day raises the cumulative ownership commitment;
        # it is not a hard expiry for an unfilled order.  If cash, capacity or
        # the shared market delays a purchase, keep the gap live and catch it
        # up while the M3 bankability check still proves that the animal can
        # produce before liquidation.  The previous "last observed buy day"
        # cutoff permanently discarded missed cows and changed a 6C/12S route
        # into an unintended 2C/10S route.
        animal_investment_stop_step=jnp.broadcast_to(
            liquidation_start[:, None],
            template.animal.animal_investment_stop_step.shape,
        ),
        animal_place_wave_size=max_purchase_wave,
        care_policy=care_policy,
        # The exact M3.6 fertilizer ledger remains authoritative for the final
        # sell quantity.  With zero declared safety stock, expose the same sale
        # to M3 before transaction admission so its proceeds can finance an
        # animal in the *same* ten-slot market bundle.  A one-step interval is
        # required because full replans also run at hour 20 after deposits; the
        # legacy 24-step interval hid that newly banked working capital until
        # the following day.
        animal_fertilizer_policy=fertilizer_market_policy,
        sell_interval=template.animal.sell_interval.at[:, 8].set(
            jnp.int16(1)
        ),
        first_cycle_care_bonus_target=jnp.where(
            care_enabled, _FIRST_CYCLE_CARE_TARGET[None], 0
        ).astype(jnp.int8),
        steady_cycle_care_bonus_target=jnp.where(
            care_enabled, _STEADY_CYCLE_CARE_TARGET[None], 0
        ).astype(jnp.int8),
        maintenance_utilization_cap=jnp.ones(
            (batch_size,), dtype=jnp.float32
        ),
        # The complete calendar already gives the intended hand and service
        # schedule.  A second, conservative species-split cap would silently
        # replace that proven plan (for example 6 cows + 12 sheep) with a
        # smaller route.  Feasibility is enforced by the live scheduler and
        # the hard deadline/escape gates instead.
        enforce_productive_cap=jnp.zeros((batch_size,), dtype=jnp.bool_),
        cash_floor=jnp.zeros((batch_size,), dtype=jnp.int32),
        liquidation_start_step=liquidation_start,
    )
    return M35FarmGenomeV2(
        candidate_id=calendar.candidate_id.astype(jnp.int32),
        crop=crop,
        animal=animal,
        crop_unit_share=template.crop_unit_share,
    )


__all__ = ["calendar_step_genome_v3"]

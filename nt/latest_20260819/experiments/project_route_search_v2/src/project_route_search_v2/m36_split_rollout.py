"""Split-JIT M3.6C rollout: daily planning plus lightweight step chunks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import TURNS_PER_DAY
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.state import reset
from kaggriculture_jax.types import Events, StaticTables

from .lifecycle import initialize_project_controller_v2, snapshot_project_controller_v2
from .m25_rollout import empty_m25_metrics_v2
from .m26_rollout import (
    _same_day_water_failures,
    _update_coverage as _update_crop_coverage,
    _update_metrics as _update_crop_metrics,
    empty_m26_coverage_v2,
)
from .m3_rollout import (
    _animal_admission_diagnostics,
    _duplicate_reservations,
    _end_day_losses,
    _update_coverage as _update_animal_coverage,
    _update_metrics as _update_animal_metrics,
    empty_m3_coverage_v2,
    empty_m3_metrics_v2,
)
from .m35_controller import (
    ensure_m35_projects_v2,
    m35_player_action_dict_v2,
    update_m35_controller_from_effects_v2,
)
from .m35_rollout import _update_flow, empty_m35_flow_v2
from .m35_schema import M35FarmGenomeV2, M35RolloutCarryV2
from .m36_controller import (
    continue_m36_fertilizer_batch_after_effects_v3,
    m36c_daily_plan_v3,
    m36c_light_policy_step_v3,
    update_route_cards_from_effects_v3,
)
from .m36_genome_adapter import calendar_step_genome_v3
from .m36_rollout import M36CAggregatesV3, M36CRolloutCarryV3
from .m36_scheduler import planned_service_tile_mask_v3
from .m38_route_cards import RouteCardModeV4
from .m36_schema import RouteCalendarV3
from .m36_transaction import (
    m36_player_action_dict_v3,
    m36a_policy_step_v3,
    reconcile_commitment_bundle_v3,
)
from .null_opponent import combine_with_null_opponent


# A day can require more than ten atomic market orders (for example twelve
# hires plus animal/seed/feed commitments).  Full economic planning remains a
# low-frequency operation, but one same-day retry at hour 1 is required so a
# legal ten-slot cap does not silently erase the tail of the daily calendar.
M36_MARKET_PLAN_HOURS_V3 = (0, 1)


def _empty_aggregates(batch_size: int) -> M36CAggregatesV3:
    zeros = jnp.zeros((batch_size,), dtype=jnp.int32)
    return M36CAggregatesV3(*(zeros for _ in M36CAggregatesV3._fields))


def initialize_m36c_split_v3(
    seeds: jax.Array,
    calendar: RouteCalendarV3,
    events: Events,
    tables: StaticTables,
    template: M35FarmGenomeV2,
    *,
    player: int = 0,
) -> M36CRolloutCarryV3:
    """Execute the opening commitment and initialize persistent ledgers."""

    states = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
    opening, projected = m36a_policy_step_v3(states, calendar, player)
    joint = combine_with_null_opponent(
        m36_player_action_dict_v3(opening), player_seat=player
    )
    next_states = batched_step_sync(states, joint, events, tables)
    _, transaction = reconcile_commitment_bundle_v3(
        projected, next_states, opening.bundle, player
    )
    genome = calendar_step_genome_v3(next_states, calendar, template)
    controller = initialize_project_controller_v2(next_states, player)
    controller = ensure_m35_projects_v2(next_states, controller, genome, player)
    controller = snapshot_project_controller_v2(next_states, controller, player)
    batch_size = next_states.step.shape[0]
    farm = M35RolloutCarryV2(
        next_states,
        controller,
        empty_m25_metrics_v2(batch_size),
        empty_m3_metrics_v2(batch_size),
        empty_m26_coverage_v2(batch_size),
        empty_m3_coverage_v2(batch_size),
        empty_m35_flow_v2(batch_size),
    )
    aggregate = _empty_aggregates(batch_size)._replace(
        transaction_hard_error_count=(
            transaction.hard_error_count + opening.diagnostics.hard_error_count
        )
    )
    return M36CRolloutCarryV3(farm, aggregate)


def m36c_daily_plan_carry_v3(
    carry: M36CRolloutCarryV3,
    calendar: RouteCalendarV3,
    tables: StaticTables,
    template: M35FarmGenomeV2,
    *,
    player: int = 0,
    enable_unlock_committed_capital: bool = False,
) -> M36CRolloutCarryV3:
    controller, fertilizer = m36c_daily_plan_v3(
        carry.farm.environment_state,
        carry.farm.controller,
        calendar,
        template,
        player,
        tables,
        enable_unlock_committed_capital=enable_unlock_committed_capital,
    )
    farm = carry.farm._replace(controller=controller)
    aggregate = carry.aggregates._replace(
        fertilizer_market_slot_overflow_count=(
            carry.aggregates.fertilizer_market_slot_overflow_count
            + fertilizer.market_slot_overflow_count
        )
    )
    return M36CRolloutCarryV3(farm, aggregate)


def make_m36c_light_chunk_v3(
    *,
    chunk_steps: int,
    player: int = 0,
    enable_unlock_committed_capital: bool = False,
    enable_route_cards: bool = False,
    route_card_mode: int = RouteCardModeV4.LEGACY_M38,
):
    if chunk_steps <= 0 or chunk_steps > 24:
        raise ValueError("chunk_steps must be in 1..24")

    def chunk(
        carry: M36CRolloutCarryV3,
        calendar: RouteCalendarV3,
        events: Events,
        tables: StaticTables,
        template: M35FarmGenomeV2,
        active_steps: jax.Array | int | None = None,
    ) -> M36CRolloutCarryV3:
        effective_steps = jnp.asarray(
            chunk_steps if active_steps is None else active_steps,
            dtype=jnp.int32,
        )

        def execute_body(carry: M36CRolloutCarryV3, _):
            farm = carry.farm
            states = farm.environment_state
            action, controller, genome, scheduler = m36c_light_policy_step_v3(
                states,
                farm.controller,
                calendar,
                template,
                player,
                enable_unlock_committed_capital=enable_unlock_committed_capital,
                enable_route_cards=enable_route_cards,
                route_card_mode=route_card_mode,
            )
            allowed_service, _ = planned_service_tile_mask_v3(
                states, calendar, player
            )
            joint = combine_with_null_opponent(
                m35_player_action_dict_v2(action), player_seat=player
            )
            last_turn = ((states.step[0] + 1) % TURNS_PER_DAY) == 0
            same_day_failure = jax.lax.cond(
                last_turn,
                lambda _: _same_day_water_failures(states, joint, player),
                lambda _: jnp.zeros_like(states.step, dtype=jnp.int32),
                operand=None,
            )
            hard_miss, capacity_loss, care_forfeited, care_clipped = _end_day_losses(
                states, joint, player
            )
            duplicate = _duplicate_reservations(controller)
            bought_without_plan, pasture_conflict = _animal_admission_diagnostics(
                states, action, controller, player
            )
            crop_coverage = _update_crop_coverage(
                farm.crop_coverage,
                states,
                action,
                controller,
                genome.crop,
                player,
            )
            animal_coverage = _update_animal_coverage(
                farm.animal_coverage,
                states,
                action,
                controller,
                genome.animal,
                player,
            )
            next_states = batched_step_sync(states, joint, events, tables)
            lost = (states.tile_animal[:, player] >= 0) & (
                next_states.tile_animal[:, player] < 0
            )
            planned_escape = jnp.sum(
                lost & (~allowed_service), axis=(1, 2), dtype=jnp.int32
            )
            unplanned_escape = jnp.sum(
                lost & allowed_service, axis=(1, 2), dtype=jnp.int32
            )
            planned_controller = controller
            controller, effects = update_m35_controller_from_effects_v2(
                states, next_states, controller, action, genome, player
            )
            if enable_route_cards:
                controller = update_route_cards_from_effects_v3(
                    states,
                    next_states,
                    planned_controller,
                    controller,
                    action,
                    player,
                )
            controller = continue_m36_fertilizer_batch_after_effects_v3(
                states,
                next_states,
                planned_controller,
                controller,
                action,
                player,
            )
            controller = snapshot_project_controller_v2(
                next_states, controller, player
            )
            crop_metrics = _update_crop_metrics(
                farm.crop_metrics,
                action,
                effects,
                same_day_failure,
                next_states,
                controller,
                player,
            )
            animal_metrics = _update_animal_metrics(
                farm.animal_metrics,
                states,
                next_states,
                action,
                effects,
                hard_miss,
                capacity_loss,
                care_forfeited,
                care_clipped,
                bought_without_plan,
                pasture_conflict,
                duplicate,
                player,
            )
            flow = _update_flow(farm.flow, states, action, effects, player)
            unexplained = (
                effects.effect_mismatch_count
                + effects.owner_inactive_count
                + effects.resource_unavailable_count
            )
            aggregate = carry.aggregates._replace(
                invalid_raw_action_count=carry.aggregates.invalid_raw_action_count
                + action.diagnostics.invalid_raw_action_count,
                unexplained_effect_error_count=(
                    carry.aggregates.unexplained_effect_error_count + unexplained
                ),
                unplanned_deadline_miss_count=(
                    carry.aggregates.unplanned_deadline_miss_count
                    + effects.deadline_missed_count
                    + hard_miss
                ),
                duplicate_reservation_count=(
                    carry.aggregates.duplicate_reservation_count + duplicate
                ),
                planned_release_escape_count=(
                    carry.aggregates.planned_release_escape_count + planned_escape
                ),
                unplanned_animal_escape_count=(
                    carry.aggregates.unplanned_animal_escape_count + unplanned_escape
                ),
                scheduler_crop_selection_count=(
                    carry.aggregates.scheduler_crop_selection_count
                    + scheduler.selected_crop_count
                ),
                scheduler_animal_selection_count=(
                    carry.aggregates.scheduler_animal_selection_count
                    + scheduler.selected_animal_count
                ),
                scheduler_deadline_preemption_count=(
                    carry.aggregates.scheduler_deadline_preemption_count
                    + scheduler.deadline_preemption_count
                ),
                scheduler_duplicate_prevented_count=(
                    carry.aggregates.scheduler_duplicate_prevented_count
                    + scheduler.duplicate_reservation_prevented
                ),
                scheduler_local_swap_count=(
                    carry.aggregates.scheduler_local_swap_count
                    + scheduler.local_swap_count
                ),
                max_planned_release_tiles=jnp.maximum(
                    carry.aggregates.max_planned_release_tiles,
                    scheduler.planned_release_tile_count,
                ),
            )
            next_farm = M35RolloutCarryV2(
                next_states,
                controller,
                crop_metrics,
                animal_metrics,
                crop_coverage,
                animal_coverage,
                flow,
            )
            return M36CRolloutCarryV3(next_farm, aggregate), None

        def body(carry: M36CRolloutCarryV3, index: jax.Array):
            return jax.lax.cond(
                index < effective_steps,
                lambda current: execute_body(current, None),
                lambda current: (current, None),
                carry,
            )

        carry, _ = jax.lax.scan(
            body, carry, xs=jnp.arange(chunk_steps, dtype=jnp.int32)
        )
        return carry

    return chunk


__all__ = [
    "M36_MARKET_PLAN_HOURS_V3",
    "initialize_m36c_split_v3",
    "m36c_daily_plan_carry_v3",
    "make_m36c_light_chunk_v3",
]

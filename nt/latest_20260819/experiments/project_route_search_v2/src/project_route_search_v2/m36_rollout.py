"""M3.6C full-season rollout and hard/soft acceptance diagnostics."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import EPISODE_STEPS, TURNS_PER_DAY
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
from .m35_genome import default_m35_farm_genome_v2
from .m35_rollout import _update_flow, empty_m35_flow_v2, summarize_m35_rollout_v2
from .m35_schema import M35FarmGenomeV2, M35RolloutCarryV2, M35RolloutSummaryV2
from .m36_controller import (
    continue_m36_fertilizer_batch_after_effects_v3,
    m36c_policy_step_v3,
    update_route_cards_from_effects_v3,
)
from .m36_genome_adapter import calendar_step_genome_v3
from .m36_scheduler import planned_service_tile_mask_v3
from .m36_schema import RouteCalendarV3
from .m38_route_cards import RouteCardModeV4
from .m36_transaction import (
    m36_player_action_dict_v3,
    m36a_policy_step_v3,
    reconcile_commitment_bundle_v3,
)
from .null_opponent import combine_with_null_opponent


class M36CAggregatesV3(NamedTuple):
    transaction_hard_error_count: jax.Array
    invalid_raw_action_count: jax.Array
    unexplained_effect_error_count: jax.Array
    unplanned_deadline_miss_count: jax.Array
    duplicate_reservation_count: jax.Array
    planned_release_escape_count: jax.Array
    unplanned_animal_escape_count: jax.Array
    fertilizer_market_slot_overflow_count: jax.Array
    scheduler_crop_selection_count: jax.Array
    scheduler_animal_selection_count: jax.Array
    scheduler_deadline_preemption_count: jax.Array
    scheduler_duplicate_prevented_count: jax.Array
    scheduler_local_swap_count: jax.Array
    max_planned_release_tiles: jax.Array


class M36CRolloutCarryV3(NamedTuple):
    farm: M35RolloutCarryV2
    aggregates: M36CAggregatesV3


class M36CRolloutSummaryV3(NamedTuple):
    farm: M35RolloutSummaryV2
    transaction_hard_error_count: jax.Array
    illegal_action_count: jax.Array
    unexplained_effect_error_count: jax.Array
    unplanned_deadline_miss_count: jax.Array
    duplicate_resource_reservation_count: jax.Array
    planned_release_count: jax.Array
    unplanned_animal_escape: jax.Array
    fertilizer_market_slot_overflow_count: jax.Array
    hard_error_count: jax.Array
    scheduler_crop_selection_count: jax.Array
    scheduler_animal_selection_count: jax.Array
    scheduler_deadline_preemption_count: jax.Array
    scheduler_duplicate_prevented_count: jax.Array
    scheduler_local_swap_count: jax.Array
    max_planned_release_tiles: jax.Array


def _empty_aggregates(batch_size: int) -> M36CAggregatesV3:
    zeros = jnp.zeros((batch_size,), dtype=jnp.int32)
    return M36CAggregatesV3(*(zeros for _ in M36CAggregatesV3._fields))


def _initialize_after_opening(
    next_states,
    calendar: RouteCalendarV3,
    template: M35FarmGenomeV2,
    player: int,
) -> M35RolloutCarryV2:
    genome = calendar_step_genome_v3(next_states, calendar, template)
    controller = initialize_project_controller_v2(next_states, player)
    controller = ensure_m35_projects_v2(next_states, controller, genome, player)
    controller = snapshot_project_controller_v2(next_states, controller, player)
    batch_size = next_states.step.shape[0]
    return M35RolloutCarryV2(
        environment_state=next_states,
        controller=controller,
        crop_metrics=empty_m25_metrics_v2(batch_size),
        animal_metrics=empty_m3_metrics_v2(batch_size),
        crop_coverage=empty_m26_coverage_v2(batch_size),
        animal_coverage=empty_m3_coverage_v2(batch_size),
        flow=empty_m35_flow_v2(batch_size),
    )


def make_m36c_farm_rollout_v3(
    *,
    rollout_steps: int = 719,
    player: int = 0,
    trace: bool = False,
    enable_route_cards: bool = False,
    route_card_mode: int = RouteCardModeV4.LEGACY_M38,
):
    if rollout_steps <= 0 or rollout_steps >= EPISODE_STEPS:
        raise ValueError("rollout_steps must be in 1..719")
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    def rollout(
        seeds: jax.Array,
        calendar: RouteCalendarV3,
        events: Events,
        tables: StaticTables,
        template: M35FarmGenomeV2,
    ):
        states = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
        opening, projected = m36a_policy_step_v3(states, calendar, player)
        opening_joint = combine_with_null_opponent(
            m36_player_action_dict_v3(opening), player_seat=player
        )
        next_states = batched_step_sync(states, opening_joint, events, tables)
        _, transaction = reconcile_commitment_bundle_v3(
            projected, next_states, opening.bundle, player
        )
        carry = M36CRolloutCarryV3(
            farm=_initialize_after_opening(next_states, calendar, template, player),
            aggregates=_empty_aggregates(next_states.step.shape[0])._replace(
                transaction_hard_error_count=transaction.hard_error_count
                + opening.diagnostics.hard_error_count
            ),
        )

        def body(carry: M36CRolloutCarryV3, _):
            farm = carry.farm
            states = farm.environment_state
            action, controller, genome, scheduler, fertilizer = m36c_policy_step_v3(
                states,
                farm.controller,
                calendar,
                template,
                player,
                tables,
                enable_route_cards=enable_route_cards,
                route_card_mode=route_card_mode,
            )
            planned_controller = controller
            allowed_service, _ = planned_service_tile_mask_v3(
                states, calendar, player
            )
            joint = combine_with_null_opponent(
                m35_player_action_dict_v2(action), player_seat=player
            )
            last_turn = ((states.step + 1) % TURNS_PER_DAY) == 0
            same_day_failure = jax.lax.cond(
                jnp.any(last_turn),
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
            aggregate = carry.aggregates
            unexplained = (
                effects.effect_mismatch_count
                + effects.owner_inactive_count
                + effects.resource_unavailable_count
            )
            aggregate = aggregate._replace(
                invalid_raw_action_count=aggregate.invalid_raw_action_count
                + action.diagnostics.invalid_raw_action_count,
                unexplained_effect_error_count=(
                    aggregate.unexplained_effect_error_count + unexplained
                ),
                unplanned_deadline_miss_count=(
                    aggregate.unplanned_deadline_miss_count
                    + effects.deadline_missed_count
                    + hard_miss
                ),
                duplicate_reservation_count=(
                    aggregate.duplicate_reservation_count + duplicate
                ),
                planned_release_escape_count=(
                    aggregate.planned_release_escape_count + planned_escape
                ),
                unplanned_animal_escape_count=(
                    aggregate.unplanned_animal_escape_count + unplanned_escape
                ),
                fertilizer_market_slot_overflow_count=(
                    aggregate.fertilizer_market_slot_overflow_count
                    + fertilizer.market_slot_overflow_count
                ),
                scheduler_crop_selection_count=(
                    aggregate.scheduler_crop_selection_count
                    + scheduler.selected_crop_count
                ),
                scheduler_animal_selection_count=(
                    aggregate.scheduler_animal_selection_count
                    + scheduler.selected_animal_count
                ),
                scheduler_deadline_preemption_count=(
                    aggregate.scheduler_deadline_preemption_count
                    + scheduler.deadline_preemption_count
                ),
                scheduler_duplicate_prevented_count=(
                    aggregate.scheduler_duplicate_prevented_count
                    + scheduler.duplicate_reservation_prevented
                ),
                scheduler_local_swap_count=(
                    aggregate.scheduler_local_swap_count
                    + scheduler.local_swap_count
                ),
                max_planned_release_tiles=jnp.maximum(
                    aggregate.max_planned_release_tiles,
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
            trace_value = (
                joint,
                next_states,
                planned_controller,
                scheduler,
                fertilizer,
            ) if trace else None
            return M36CRolloutCarryV3(next_farm, aggregate), trace_value

        remaining = rollout_steps - 1
        carry, scan_trace = jax.lax.scan(body, carry, xs=None, length=remaining)
        if trace:
            return carry, (opening_joint, next_states, scan_trace)
        return carry, None

    return rollout


def summarize_m36c_rollout_v3(
    carry: M36CRolloutCarryV3, *, player: int = 0
) -> M36CRolloutSummaryV3:
    farm = summarize_m35_rollout_v2(carry.farm, player=player)
    aggregate = carry.aggregates
    hard = (
        aggregate.transaction_hard_error_count
        + aggregate.invalid_raw_action_count
        + aggregate.unexplained_effect_error_count
        + aggregate.unplanned_deadline_miss_count
        + aggregate.duplicate_reservation_count
        + aggregate.unplanned_animal_escape_count
        + aggregate.fertilizer_market_slot_overflow_count
        + farm.plant_without_same_day_water
    )
    return M36CRolloutSummaryV3(
        farm=farm,
        transaction_hard_error_count=aggregate.transaction_hard_error_count,
        illegal_action_count=aggregate.invalid_raw_action_count,
        unexplained_effect_error_count=aggregate.unexplained_effect_error_count,
        unplanned_deadline_miss_count=aggregate.unplanned_deadline_miss_count,
        duplicate_resource_reservation_count=aggregate.duplicate_reservation_count,
        planned_release_count=aggregate.planned_release_escape_count,
        unplanned_animal_escape=aggregate.unplanned_animal_escape_count,
        fertilizer_market_slot_overflow_count=(
            aggregate.fertilizer_market_slot_overflow_count
        ),
        hard_error_count=hard,
        scheduler_crop_selection_count=aggregate.scheduler_crop_selection_count,
        scheduler_animal_selection_count=aggregate.scheduler_animal_selection_count,
        scheduler_deadline_preemption_count=(
            aggregate.scheduler_deadline_preemption_count
        ),
        scheduler_duplicate_prevented_count=(
            aggregate.scheduler_duplicate_prevented_count
        ),
        scheduler_local_swap_count=aggregate.scheduler_local_swap_count,
        max_planned_release_tiles=aggregate.max_planned_release_tiles,
    )


__all__ = [
    "M36CAggregatesV3",
    "M36CRolloutCarryV3",
    "M36CRolloutSummaryV3",
    "make_m36c_farm_rollout_v3",
    "summarize_m36c_rollout_v3",
]

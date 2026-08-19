"""Full-season M3.5 rollout, cross-flow metrics and joint summary."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    EPISODE_STEPS,
    MarketOp,
    NUM_PRODUCTS,
    TURNS_PER_DAY,
    TileKind,
    UnitOp,
)
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
    summarize_m26_rollout_v2,
)
from .m3_rollout import (
    _animal_admission_diagnostics,
    _duplicate_reservations,
    _end_day_losses,
    _update_coverage as _update_animal_coverage,
    _update_metrics as _update_animal_metrics,
    empty_m3_coverage_v2,
    empty_m3_metrics_v2,
    summarize_m3_rollout_v2,
)
from .m35_controller import (
    ensure_m35_projects_v2,
    m35_player_action_dict_v2,
    m35_policy_step_v2,
    update_m35_controller_from_effects_v2,
)
from .m35_genome import default_m35_farm_genome_v2
from .m35_schema import (
    M35FarmGenomeV2,
    M35FlowMetricsV2,
    M35RolloutCarryV2,
    M35RolloutSummaryV2,
)
from .m3_schema import M3RolloutCarryV2
from .null_opponent import combine_with_null_opponent
from .schema import M26RolloutCarryV2


def empty_m35_flow_v2(batch_size: int) -> M35FlowMetricsV2:
    zeros = jnp.zeros((batch_size,), dtype=jnp.int32)
    return M35FlowMetricsV2(*(zeros for _ in M35FlowMetricsV2._fields))


def initialize_m35_rollout_carry_v2(
    seeds: jax.Array,
    genome: M35FarmGenomeV2 | None = None,
    *,
    player: int = 0,
) -> M35RolloutCarryV2:
    states = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
    if genome is None:
        genome = default_m35_farm_genome_v2(states.step.shape[0])
    controller = initialize_project_controller_v2(states, player)
    controller = ensure_m35_projects_v2(states, controller, genome, player)
    return M35RolloutCarryV2(
        environment_state=states,
        controller=controller,
        crop_metrics=empty_m25_metrics_v2(states.step.shape[0]),
        animal_metrics=empty_m3_metrics_v2(states.step.shape[0]),
        crop_coverage=empty_m26_coverage_v2(states.step.shape[0]),
        animal_coverage=empty_m3_coverage_v2(states.step.shape[0]),
        flow=empty_m35_flow_v2(states.step.shape[0]),
    )


def _update_flow(flow, states, action, effects, player: int):
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    positions = states.unit_pos[:, player].astype(jnp.int32)
    x = jnp.clip(positions[..., 0], 0, 9)
    y = jnp.clip(positions[..., 1], 0, 9)
    kind = states.tile_kind[batch[:, None], player, y, x]
    crop = states.tile_crop[batch[:, None], player, y, x]
    tile_yield = states.tile_yield[batch[:, None], player, y, x].astype(jnp.int32)
    wheat_harvest = (
        (action.unit_op == UnitOp.HARVEST)
        & (kind == TileKind.PLANT)
        & (crop == 0)
    )
    market_slot = jnp.arange(action.market_op.shape[1], dtype=jnp.int8)[None]
    active_market = market_slot < action.market_count[:, None]
    wheat_buy = (
        active_market
        & (action.market_op == MarketOp.BUY_PRODUCT)
        & (action.market_item == 0)
    )
    fertilizer_buy = (
        active_market
        & (action.market_op == MarketOp.BUY_PRODUCT)
        & (action.market_item == 8)
    )
    conflict = (
        effects.resource_unavailable_count
        + action.diagnostics.unit_compiler_overlap_count
        + action.diagnostics.market_compiler_overlap_count
    )
    return M35FlowMetricsV2(
        wheat_harvest_actions=flow.wheat_harvest_actions
        + jnp.sum(jnp.where(wheat_harvest, tile_yield, 0), axis=-1, dtype=jnp.int32),
        wheat_market_buy_units=flow.wheat_market_buy_units
        + jnp.sum(
            jnp.where(wheat_buy, action.market_amount, 0), axis=-1, dtype=jnp.int32
        ),
        animal_feed_actions=flow.animal_feed_actions
        + jnp.sum(action.unit_op == UnitOp.FEED, axis=-1, dtype=jnp.int32),
        animal_fertilizer_collect_actions=flow.animal_fertilizer_collect_actions
        + jnp.sum(
            action.unit_op == UnitOp.COLLECT_FERTILIZER,
            axis=-1,
            dtype=jnp.int32,
        ),
        crop_fertilizer_apply_actions=flow.crop_fertilizer_apply_actions
        + jnp.sum(action.unit_op == UnitOp.FERTILIZE, axis=-1, dtype=jnp.int32),
        fertilizer_market_buy_units=flow.fertilizer_market_buy_units
        + jnp.sum(
            jnp.where(fertilizer_buy, action.market_amount, 0),
            axis=-1,
            dtype=jnp.int32,
        ),
        joint_resource_conflict_count=flow.joint_resource_conflict_count + conflict,
    )


def make_m35_farm_rollout_v2(
    *, rollout_steps: int = 719, player: int = 0, trace: bool | str = False
):
    if rollout_steps <= 0 or rollout_steps >= EPISODE_STEPS:
        raise ValueError("rollout_steps must be in 1..719")
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    def rollout(
        initial_carry: M35RolloutCarryV2,
        events: Events,
        tables: StaticTables,
        genome: M35FarmGenomeV2,
    ):
        def body(carry: M35RolloutCarryV2, _):
            states = carry.environment_state
            action, controller = m35_policy_step_v2(
                states, carry.controller, genome, player, tables
            )
            planned_controller = controller
            joint_action = combine_with_null_opponent(
                m35_player_action_dict_v2(action), player_seat=player
            )
            last_turn = ((states.step[0].astype(jnp.int32) + 1) % TURNS_PER_DAY) == 0
            same_day_failure = jax.lax.cond(
                last_turn,
                lambda _: _same_day_water_failures(states, joint_action, player),
                lambda _: jnp.zeros_like(states.step, dtype=jnp.int32),
                operand=None,
            )
            hard_miss, capacity_loss, care_forfeited, care_clipped = _end_day_losses(
                states, joint_action, player
            )
            duplicate = _duplicate_reservations(controller)
            bought_without_plan, pasture_conflict = _animal_admission_diagnostics(
                states, action, controller, player
            )
            crop_coverage = _update_crop_coverage(
                carry.crop_coverage, states, action, controller, genome.crop, player
            )
            animal_coverage = _update_animal_coverage(
                carry.animal_coverage,
                states,
                action,
                controller,
                genome.animal,
                player,
            )
            next_states = batched_step_sync(states, joint_action, events, tables)
            controller, effects = update_m35_controller_from_effects_v2(
                states, next_states, controller, action, genome, player
            )
            controller = snapshot_project_controller_v2(
                next_states, controller, player
            )
            crop_metrics = _update_crop_metrics(
                carry.crop_metrics,
                action,
                effects,
                same_day_failure,
                next_states,
                controller,
                player,
            )
            animal_metrics = _update_animal_metrics(
                carry.animal_metrics,
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
            flow = _update_flow(carry.flow, states, action, effects, player)
            next_carry = M35RolloutCarryV2(
                next_states,
                controller,
                crop_metrics,
                animal_metrics,
                crop_coverage,
                animal_coverage,
                flow,
            )
            if trace == "diagnostic":
                trace_value = (
                    joint_action,
                    next_states,
                    effects,
                    planned_controller,
                    (hard_miss, capacity_loss, care_forfeited, care_clipped),
                )
            elif trace:
                trace_value = (joint_action, next_states, effects)
            else:
                trace_value = None
            return next_carry, trace_value

        return jax.lax.scan(body, initial_carry, xs=None, length=rollout_steps)

    return rollout


def summarize_m35_rollout_v2(
    carry: M35RolloutCarryV2, *, player: int = 0
) -> M35RolloutSummaryV2:
    crop = summarize_m26_rollout_v2(
        M26RolloutCarryV2(
            carry.environment_state,
            carry.controller,
            carry.crop_metrics,
            carry.crop_coverage,
        ),
        player=player,
    )
    animal = summarize_m3_rollout_v2(
        M3RolloutCarryV2(
            carry.environment_state,
            carry.controller,
            carry.animal_metrics,
            carry.animal_coverage,
        ),
        player=player,
    )
    shared_terminal = (
        crop.terminal_sellable_shed_value + crop.terminal_unit_inventory_value
    )
    avoidable = (
        shared_terminal
        + crop.terminal_harvestable_map_value
        + animal.terminal_animal_product_value
    ).astype(jnp.int32)
    return M35RolloutSummaryV2(
        final_bank=crop.final_bank,
        done=crop.done,
        terminal_sellable_shed_value=crop.terminal_sellable_shed_value,
        terminal_unit_inventory_value=crop.terminal_unit_inventory_value,
        terminal_harvestable_crop_value=crop.terminal_harvestable_map_value,
        terminal_animal_product_value=animal.terminal_animal_product_value,
        avoidable_liquidation_loss=avoidable,
        plant_without_same_day_water=crop.plant_without_same_day_water,
        unexplained_failure_count=animal.unexplained_failure_count,
        unplanned_animal_escape=animal.unplanned_animal_escape,
        animal_capacity_loss=animal.animal_capacity_loss,
        feed_hard_deadline_miss=animal.feed_hard_deadline_miss,
        care_bonus_forfeited_unexplained=animal.care_bonus_forfeited_unexplained,
        care_bonus_capacity_clipped_unexplained=(
            animal.care_bonus_capacity_clipped_unexplained
        ),
        animal_bought_without_place_plan=animal.animal_bought_without_place_plan,
        cow_sheep_pasture_conflict=animal.cow_sheep_pasture_conflict,
        animals_stranded_in_shed_at_terminal=(
            animal.animals_stranded_in_shed_at_terminal
        ),
        animals_stranded_in_unit_inventory_at_terminal=(
            animal.animals_stranded_in_unit_inventory_at_terminal
        ),
        planted_tiles_by_crop=crop.planted_tiles_by_crop,
        active_animals_by_species=animal.active_animals_by_species,
        flow=carry.flow,
        crop_coverage=carry.crop_coverage,
        animal_coverage=carry.animal_coverage,
    )


__all__ = [
    "empty_m35_flow_v2",
    "initialize_m35_rollout_carry_v2",
    "make_m35_farm_rollout_v2",
    "summarize_m35_rollout_v2",
]

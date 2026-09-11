"""Full-season rollout, coverage counters and diagnostics for M2.6."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    EPISODE_STEPS,
    FLAG_WATERED,
    MarketOp,
    NUM_CROPS,
    NUM_PRODUCTS,
    SHED_ACCESS,
    TURNS_PER_DAY,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_project_unit_phase, batched_step_sync
from kaggriculture_jax.state import reset
from kaggriculture_jax.types import Events, State, StaticTables
from strategic_v5.constants import TaskTypeV1

from .lifecycle import initialize_project_controller_v2, snapshot_project_controller_v2
from .m25_rollout import empty_m25_metrics_v2
from .m26_controller import (
    ensure_m26_projects_v2,
    m26_phase_v2,
    m26_player_action_dict_v2,
    m26_policy_step_v2,
    update_m26_controller_from_effects_v2,
)
from .m26_genome import default_m26_crop_genome_v2
from .null_opponent import combine_with_null_opponent
from .schema import (
    M25RolloutMetricsV2,
    M26CoverageMetricsV2,
    M26CropGenomeV2,
    M26RolloutCarryV2,
    M26RolloutSummaryV2,
)


_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)


def empty_m26_coverage_v2(batch_size: int) -> M26CoverageMetricsV2:
    return M26CoverageMetricsV2(
        phase_step_count=jnp.zeros((batch_size, 6), dtype=jnp.int32),
        crop_target_step_count=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
        crop_plant_actions=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
        crop_harvest_actions=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
        layout_plant_actions=jnp.zeros((batch_size, 4), dtype=jnp.int32),
        fertilizer_actions=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
        weed_recovery_actions=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
        abandon_actions=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
        sell_orders_by_product=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int32),
        seed_orders_by_crop=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
        land_orders=jnp.zeros((batch_size,), dtype=jnp.int32),
        hire_orders=jnp.zeros((batch_size,), dtype=jnp.int32),
        expansion_events=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
        shrink_events=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
        stop_events=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
        restart_events=jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32),
    )


def initialize_m26_rollout_carry_v2(
    seeds: jax.Array,
    genome: M26CropGenomeV2 | None = None,
    *,
    player: int = 0,
) -> M26RolloutCarryV2:
    states = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
    if genome is None:
        genome = default_m26_crop_genome_v2(states.step.shape[0])
    controller = initialize_project_controller_v2(states, player)
    controller = ensure_m26_projects_v2(states, controller, genome, player)
    return M26RolloutCarryV2(
        environment_state=states,
        controller=controller,
        metrics=empty_m25_metrics_v2(states.step.shape[0]),
        coverage=empty_m26_coverage_v2(states.step.shape[0]),
    )


def _same_day_water_failures(states: State, joint_action, player: int) -> jax.Array:
    projected = batched_project_unit_phase(states, joint_action)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    bad = (
        (projected.tile_kind[:, player] == TileKind.PLANT)
        & (projected.tile_origin_day[:, player].astype(jnp.int16) == current_day[:, None, None])
        & ((projected.tile_flags[:, player] & jnp.uint8(FLAG_WATERED)) == 0)
    )
    return jnp.sum(bad, axis=(1, 2), dtype=jnp.int32)


def _update_metrics(
    metrics: M25RolloutMetricsV2,
    action,
    effects,
    same_day_failure: jax.Array,
    next_states: State,
    controller,
    player: int,
) -> M25RolloutMetricsV2:
    compile_diag = action.diagnostics
    crop_compile = action.crop_inventory.diagnostics
    e2_compile = action.e2.diagnostics
    crop_effect = effects.crop_inventory
    e2_effect = effects.e2
    return M25RolloutMetricsV2(
        invalid_raw_action_count=metrics.invalid_raw_action_count + compile_diag.invalid_raw_action_count,
        unexpected_pass_count=metrics.unexpected_pass_count + compile_diag.unexpected_pass_count,
        effect_mismatch_count=metrics.effect_mismatch_count + effects.effect_mismatch_count,
        owner_inactive_count=metrics.owner_inactive_count + effects.owner_inactive_count,
        deadline_missed_count=metrics.deadline_missed_count + effects.deadline_missed_count,
        resource_unavailable_count=metrics.resource_unavailable_count + effects.resource_unavailable_count,
        unit_compiler_overlap_count=metrics.unit_compiler_overlap_count + compile_diag.unit_compiler_overlap_count,
        market_compiler_overlap_count=metrics.market_compiler_overlap_count + compile_diag.market_compiler_overlap_count,
        plant_actions=metrics.plant_actions + crop_compile.plant_actions,
        water_actions=metrics.water_actions + crop_compile.water_actions,
        harvest_actions=metrics.harvest_actions + crop_compile.harvest_actions,
        deposit_actions=metrics.deposit_actions + crop_compile.deposit_actions,
        seed_buy_orders=metrics.seed_buy_orders + crop_compile.seed_buy_orders,
        hire_orders=metrics.hire_orders + crop_compile.hire_orders,
        land_orders=metrics.land_orders + e2_compile.land_orders,
        sell_orders=metrics.sell_orders + crop_compile.sell_orders + crop_compile.terminal_sell_orders,
        plant_success_count=metrics.plant_success_count + crop_effect.plant_success_count,
        water_success_count=metrics.water_success_count + crop_effect.water_success_count,
        harvest_success_count=metrics.harvest_success_count + crop_effect.harvest_success_count,
        deposit_success_count=metrics.deposit_success_count + crop_effect.deposit_success_count,
        seed_purchase_success_count=metrics.seed_purchase_success_count + crop_effect.seed_purchase_success_count,
        hire_success_count=metrics.hire_success_count + crop_effect.hire_success_count,
        land_purchase_success_count=metrics.land_purchase_success_count + e2_effect.land_purchase_success_count,
        sold_product_units=metrics.sold_product_units + crop_effect.sold_product_units,
        max_hires_observed=jnp.maximum(metrics.max_hires_observed, next_states.hires_today[:, player].astype(jnp.int32)),
        plant_without_same_day_water=metrics.plant_without_same_day_water + same_day_failure,
        project_cap_hits=jnp.maximum(metrics.project_cap_hits, controller.project_cap_hits),
        obligation_cap_hits=jnp.maximum(metrics.obligation_cap_hits, controller.obligation_cap_hits),
    )


def _scatter_counts(ids: jax.Array, mask: jax.Array, size: int) -> jax.Array:
    safe = jnp.clip(ids.astype(jnp.int32), 0, size - 1)
    return jnp.sum(
        jax.nn.one_hot(safe, size, dtype=jnp.int32) * mask[..., None], axis=1, dtype=jnp.int32
    )


def _update_coverage(
    coverage: M26CoverageMetricsV2,
    states: State,
    action,
    controller,
    genome: M26CropGenomeV2,
    player: int,
) -> M26CoverageMetricsV2:
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    phase = m26_phase_v2(states, genome).astype(jnp.int32)
    phase_steps = coverage.phase_step_count.at[batch, phase].add(1)
    target = genome.crop_target[batch, phase]
    crop_target_steps = coverage.crop_target_step_count + (target > 0).astype(jnp.int32)

    unit_op = action.unit_op
    unit_item = jnp.clip(action.unit_item.astype(jnp.int32), 0, NUM_CROPS - 1)
    plant_mask = unit_op == UnitOp.PLANT
    crop_plant = coverage.crop_plant_actions + _scatter_counts(unit_item, plant_mask, NUM_CROPS)
    layout = genome.crop_layout_policy[batch[:, None], unit_item]
    layout_plant = coverage.layout_plant_actions + _scatter_counts(layout, plant_mask, 4)

    positions = states.unit_pos[:, player].astype(jnp.int32)
    x = jnp.clip(positions[..., 0], 0, 9)
    y = jnp.clip(positions[..., 1], 0, 9)
    tile_kind_at_unit = states.tile_kind[batch[:, None], player, y, x]
    tile_crop = jnp.clip(states.tile_crop[batch[:, None], player, y, x].astype(jnp.int32), 0, NUM_CROPS - 1)
    harvest_mask = (unit_op == UnitOp.HARVEST) & (tile_kind_at_unit == TileKind.PLANT)
    crop_harvest = coverage.crop_harvest_actions + _scatter_counts(tile_crop, harvest_mask, NUM_CROPS)
    fertilizer_mask = unit_op == UnitOp.FERTILIZE
    fertilizer = coverage.fertilizer_actions + _scatter_counts(tile_crop, fertilizer_mask, NUM_CROPS)
    dig_mask = unit_op == UnitOp.DIG
    pre_kind = tile_kind_at_unit
    remembered = jnp.clip(
        controller.tile_project_id[batch[:, None], y, x].astype(jnp.int32), 0, NUM_CROPS - 1
    )
    weed = coverage.weed_recovery_actions + _scatter_counts(
        remembered, dig_mask & (pre_kind == TileKind.WEED), NUM_CROPS
    )
    abandon = coverage.abandon_actions + _scatter_counts(
        tile_crop, dig_mask & (pre_kind == TileKind.PLANT), NUM_CROPS
    )

    active_market = jnp.arange(action.market_op.shape[1], dtype=jnp.int8)[None] < action.market_count[:, None]
    market_item = jnp.clip(action.market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    sell = coverage.sell_orders_by_product + _scatter_counts(
        market_item, active_market & (action.market_op == MarketOp.SELL), NUM_PRODUCTS
    )
    seed = coverage.seed_orders_by_crop + _scatter_counts(
        jnp.clip(market_item, 0, NUM_CROPS - 1),
        active_market & (action.market_op == MarketOp.BUY_SEED),
        NUM_CROPS,
    )
    land = coverage.land_orders + jnp.sum(
        active_market & (action.market_op == MarketOp.BUY_LAND), axis=-1, dtype=jnp.int32
    )
    hire = coverage.hire_orders + jnp.sum(
        active_market & (action.market_op == MarketOp.HIRE), axis=-1, dtype=jnp.int32
    )

    previous_phase = jnp.maximum(phase - 1, 0)
    previous_target = genome.crop_target[batch, previous_phase]
    is_phase_start = (phase > 0) & (states.step == genome.phase_start_step[batch, phase])
    phase_slots = jnp.arange(genome.crop_target.shape[1], dtype=jnp.int32)[None, :, None]
    prior_positive = jnp.any(
        (genome.crop_target > 0) & (phase_slots < phase[:, None, None]),
        axis=1,
    )
    expansion_event = is_phase_start[:, None] & (target > previous_target)
    shrink_event = is_phase_start[:, None] & (target > 0) & (target < previous_target)
    stop_event = is_phase_start[:, None] & (target == 0) & (previous_target > 0)
    restart_event = is_phase_start[:, None] & (target > 0) & (previous_target == 0) & prior_positive

    return M26CoverageMetricsV2(
        phase_step_count=phase_steps,
        crop_target_step_count=crop_target_steps,
        crop_plant_actions=crop_plant,
        crop_harvest_actions=crop_harvest,
        layout_plant_actions=layout_plant,
        fertilizer_actions=fertilizer,
        weed_recovery_actions=weed,
        abandon_actions=abandon,
        sell_orders_by_product=sell,
        seed_orders_by_crop=seed,
        land_orders=land,
        hire_orders=hire,
        expansion_events=coverage.expansion_events + expansion_event.astype(jnp.int32),
        shrink_events=coverage.shrink_events + shrink_event.astype(jnp.int32),
        stop_events=coverage.stop_events + stop_event.astype(jnp.int32),
        restart_events=coverage.restart_events + restart_event.astype(jnp.int32),
    )


def make_m26_crop_rollout_v2(*, rollout_steps: int = 719, player: int = 0, trace: bool = False):
    if rollout_steps <= 0 or rollout_steps > EPISODE_STEPS - 1:
        raise ValueError("rollout_steps must be in 1..719")
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    def rollout(
        initial_carry: M26RolloutCarryV2,
        events: Events,
        tables: StaticTables,
        genome: M26CropGenomeV2,
    ):
        def body(carry: M26RolloutCarryV2, _):
            states = carry.environment_state
            action, controller = m26_policy_step_v2(
                states, carry.controller, genome, player, tables
            )
            joint_action = combine_with_null_opponent(
                m26_player_action_dict_v2(action), player_seat=player
            )
            last_turn = ((states.step[0].astype(jnp.int32) + 1) % TURNS_PER_DAY) == 0
            same_day_failure = jax.lax.cond(
                last_turn,
                lambda _: _same_day_water_failures(states, joint_action, player),
                lambda _: jnp.zeros_like(states.step, dtype=jnp.int32),
                operand=None,
            )
            coverage = _update_coverage(carry.coverage, states, action, controller, genome, player)
            next_states = batched_step_sync(states, joint_action, events, tables)
            controller, effects = update_m26_controller_from_effects_v2(
                states, next_states, controller, action, genome, player
            )
            controller = snapshot_project_controller_v2(next_states, controller, player)
            metrics = _update_metrics(
                carry.metrics, action, effects, same_day_failure, next_states, controller, player
            )
            next_carry = M26RolloutCarryV2(next_states, controller, metrics, coverage)
            return next_carry, (joint_action, next_states, effects) if trace else None

        return jax.lax.scan(body, initial_carry, xs=None, length=rollout_steps)

    return rollout


def summarize_m26_rollout_v2(
    carry: M26RolloutCarryV2, *, player: int = 0
) -> M26RolloutSummaryV2:
    states = carry.environment_state
    metrics = carry.metrics
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    product_shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)
    shed_value = jnp.sum(product_shed * states.market_price.astype(jnp.int32), axis=-1, dtype=jnp.int32)
    inventory = states.unit_inventory[:, player, :, :NUM_PRODUCTS].astype(jnp.int32)
    inventory_value = jnp.sum(
        inventory * states.market_price[:, None, :].astype(jnp.int32), axis=(1, 2), dtype=jnp.int32
    )
    kind = states.tile_kind[:, player]
    crop = states.tile_crop[:, player]
    safe_crop = jnp.clip(crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    harvestable_units = jnp.where(
        (kind == TileKind.PLANT) & (states.tile_yield[:, player] > 0),
        states.tile_yield[:, player].astype(jnp.int32),
        0,
    )
    harvestable_value = jnp.sum(
        harvestable_units * states.market_price[batch[:, None, None], safe_crop],
        axis=(1, 2),
        dtype=jnp.int32,
    )
    positions = states.unit_pos[:, player].astype(jnp.int16)
    at_shed = jnp.any(
        jnp.all(positions[:, :, None, :] == _SHED_ACCESS[None, None], axis=-1), axis=-1
    )
    immediately_bankable = jnp.sum(
        jnp.where(
            at_shed[..., None],
            inventory * states.market_price[:, None, :].astype(jnp.int32),
            0,
        ),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    planted = jnp.sum(
        jax.nn.one_hot(safe_crop, NUM_CROPS, dtype=jnp.int32)
        * (kind == TileKind.PLANT)[..., None],
        axis=(1, 2),
        dtype=jnp.int32,
    )
    unexplained = (
        metrics.invalid_raw_action_count
        + metrics.unexpected_pass_count
        + metrics.effect_mismatch_count
        + metrics.owner_inactive_count
        + metrics.deadline_missed_count
        + metrics.resource_unavailable_count
        + metrics.unit_compiler_overlap_count
        + metrics.market_compiler_overlap_count
        + metrics.project_cap_hits
        + metrics.obligation_cap_hits
    )
    return M26RolloutSummaryV2(
        final_bank=states.money[:, player].astype(jnp.int32),
        done=states.done,
        terminal_sellable_shed_value=shed_value,
        terminal_unit_inventory_value=inventory_value,
        terminal_harvestable_map_value=harvestable_value,
        avoidable_liquidation_loss=(shed_value + immediately_bankable).astype(jnp.int32),
        plant_without_same_day_water=metrics.plant_without_same_day_water,
        unexplained_failure_count=unexplained.astype(jnp.int32),
        planted_tiles_by_crop=planted,
        max_hires_observed=metrics.max_hires_observed,
        final_unlocked_count=states.unlocked_count[:, player].astype(jnp.int32),
        plant_success_count=metrics.plant_success_count,
        harvest_success_count=metrics.harvest_success_count,
        sold_product_units=metrics.sold_product_units,
        coverage=carry.coverage,
    )


__all__ = [
    "empty_m26_coverage_v2",
    "initialize_m26_rollout_carry_v2",
    "make_m26_crop_rollout_v2",
    "summarize_m26_rollout_v2",
]

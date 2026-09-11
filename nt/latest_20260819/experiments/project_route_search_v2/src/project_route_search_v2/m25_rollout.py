"""Full-season JAX rollout and diagnostics for the M2.5 crop controller."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    EPISODE_STEPS,
    FLAG_WATERED,
    NUM_CROPS,
    NUM_PRODUCTS,
    SHED_ACCESS,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.simulator import batched_project_unit_phase, batched_step_sync
from kaggriculture_jax.state import reset
from kaggriculture_jax.types import Events, State, StaticTables

from .lifecycle import initialize_project_controller_v2, snapshot_project_controller_v2
from .m25_controller import (
    default_r2_tomato_m25_config_v2,
    ensure_m25_projects_v2,
    m25_player_action_dict_v2,
    m25_policy_step_v2,
    update_m25_controller_from_effects_v2,
)
from .null_opponent import combine_with_null_opponent
from .schema import (
    M25CropExpansionConfigV2,
    M25RolloutCarryV2,
    M25RolloutMetricsV2,
    M25RolloutSummaryV2,
)


_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)


def empty_m25_metrics_v2(batch_size: int) -> M25RolloutMetricsV2:
    zero = lambda: jnp.zeros((batch_size,), dtype=jnp.int32)
    return M25RolloutMetricsV2(*(zero() for _ in M25RolloutMetricsV2._fields))


def initialize_m25_rollout_carry_v2(
    seeds: jax.Array,
    config: M25CropExpansionConfigV2 | None = None,
    *,
    player: int = 0,
) -> M25RolloutCarryV2:
    states = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
    if config is None:
        config = default_r2_tomato_m25_config_v2(states.step.shape[0])
    controller = initialize_project_controller_v2(states, player)
    controller = ensure_m25_projects_v2(states, controller, config, player)
    return M25RolloutCarryV2(
        environment_state=states,
        controller=controller,
        metrics=empty_m25_metrics_v2(states.step.shape[0]),
    )


def _same_day_water_failures(
    states: State,
    joint_action,
    player: int,
) -> jax.Array:
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
        invalid_raw_action_count=metrics.invalid_raw_action_count
        + compile_diag.invalid_raw_action_count,
        unexpected_pass_count=metrics.unexpected_pass_count
        + compile_diag.unexpected_pass_count,
        effect_mismatch_count=metrics.effect_mismatch_count
        + effects.effect_mismatch_count,
        owner_inactive_count=metrics.owner_inactive_count
        + effects.owner_inactive_count,
        deadline_missed_count=metrics.deadline_missed_count
        + effects.deadline_missed_count,
        resource_unavailable_count=metrics.resource_unavailable_count
        + effects.resource_unavailable_count,
        unit_compiler_overlap_count=metrics.unit_compiler_overlap_count
        + compile_diag.unit_compiler_overlap_count,
        market_compiler_overlap_count=metrics.market_compiler_overlap_count
        + compile_diag.market_compiler_overlap_count,
        plant_actions=metrics.plant_actions + crop_compile.plant_actions,
        water_actions=metrics.water_actions + crop_compile.water_actions,
        harvest_actions=metrics.harvest_actions + crop_compile.harvest_actions,
        deposit_actions=metrics.deposit_actions + crop_compile.deposit_actions,
        seed_buy_orders=metrics.seed_buy_orders + crop_compile.seed_buy_orders,
        hire_orders=metrics.hire_orders + crop_compile.hire_orders,
        land_orders=metrics.land_orders + e2_compile.land_orders,
        sell_orders=metrics.sell_orders
        + crop_compile.sell_orders
        + crop_compile.terminal_sell_orders,
        plant_success_count=metrics.plant_success_count
        + crop_effect.plant_success_count,
        water_success_count=metrics.water_success_count
        + crop_effect.water_success_count,
        harvest_success_count=metrics.harvest_success_count
        + crop_effect.harvest_success_count,
        deposit_success_count=metrics.deposit_success_count
        + crop_effect.deposit_success_count,
        seed_purchase_success_count=metrics.seed_purchase_success_count
        + crop_effect.seed_purchase_success_count,
        hire_success_count=metrics.hire_success_count
        + crop_effect.hire_success_count,
        land_purchase_success_count=metrics.land_purchase_success_count
        + e2_effect.land_purchase_success_count,
        sold_product_units=metrics.sold_product_units
        + crop_effect.sold_product_units,
        max_hires_observed=jnp.maximum(
            metrics.max_hires_observed,
            next_states.hires_today[:, player].astype(jnp.int32),
        ),
        plant_without_same_day_water=metrics.plant_without_same_day_water
        + same_day_failure,
        project_cap_hits=jnp.maximum(metrics.project_cap_hits, controller.project_cap_hits),
        obligation_cap_hits=jnp.maximum(
            metrics.obligation_cap_hits, controller.obligation_cap_hits
        ),
    )


def make_m25_crop_rollout_v2(*, rollout_steps: int = 719, player: int = 0):
    if rollout_steps <= 0 or rollout_steps > EPISODE_STEPS - 1:
        raise ValueError("rollout_steps must be in 1..719")
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    def rollout(
        initial_carry: M25RolloutCarryV2,
        events: Events,
        tables: StaticTables,
        config: M25CropExpansionConfigV2,
    ) -> M25RolloutCarryV2:
        def body(carry: M25RolloutCarryV2, _):
            states = carry.environment_state
            action, controller = m25_policy_step_v2(
                states, carry.controller, config, player
            )
            joint_action = combine_with_null_opponent(
                m25_player_action_dict_v2(action), player_seat=player
            )
            last_turn = ((states.step[0].astype(jnp.int32) + 1) % TURNS_PER_DAY) == 0
            same_day_failure = jax.lax.cond(
                last_turn,
                lambda _: _same_day_water_failures(states, joint_action, player),
                lambda _: jnp.zeros_like(states.step, dtype=jnp.int32),
                operand=None,
            )
            next_states = batched_step_sync(states, joint_action, events, tables)
            controller, effects = update_m25_controller_from_effects_v2(
                states, next_states, controller, action, config, player
            )
            controller = snapshot_project_controller_v2(next_states, controller, player)
            metrics = _update_metrics(
                carry.metrics,
                action,
                effects,
                same_day_failure,
                next_states,
                controller,
                player,
            )
            return M25RolloutCarryV2(next_states, controller, metrics), None

        final, _ = jax.lax.scan(body, initial_carry, xs=None, length=rollout_steps)
        return final

    return rollout


def make_m25_trace_rollout_v2(*, rollout_steps: int = 719, player: int = 0):
    """Diagnostic rollout that also returns every exact primitive joint action."""

    if rollout_steps <= 0 or rollout_steps > EPISODE_STEPS - 1:
        raise ValueError("rollout_steps must be in 1..719")
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    def rollout(
        initial_carry: M25RolloutCarryV2,
        events: Events,
        tables: StaticTables,
        config: M25CropExpansionConfigV2,
    ):
        def body(carry: M25RolloutCarryV2, _):
            states = carry.environment_state
            action, controller = m25_policy_step_v2(
                states, carry.controller, config, player
            )
            joint_action = combine_with_null_opponent(
                m25_player_action_dict_v2(action), player_seat=player
            )
            last_turn = ((states.step[0].astype(jnp.int32) + 1) % TURNS_PER_DAY) == 0
            same_day_failure = jax.lax.cond(
                last_turn,
                lambda _: _same_day_water_failures(states, joint_action, player),
                lambda _: jnp.zeros_like(states.step, dtype=jnp.int32),
                operand=None,
            )
            next_states = batched_step_sync(states, joint_action, events, tables)
            controller, effects = update_m25_controller_from_effects_v2(
                states, next_states, controller, action, config, player
            )
            controller = snapshot_project_controller_v2(next_states, controller, player)
            metrics = _update_metrics(
                carry.metrics,
                action,
                effects,
                same_day_failure,
                next_states,
                controller,
                player,
            )
            return M25RolloutCarryV2(next_states, controller, metrics), (
                joint_action,
                next_states,
            )

        return jax.lax.scan(body, initial_carry, xs=None, length=rollout_steps)

    return rollout


def summarize_m25_rollout_v2(
    carry: M25RolloutCarryV2,
    config: M25CropExpansionConfigV2,
    *,
    player: int = 0,
) -> M25RolloutSummaryV2:
    states = carry.environment_state
    metrics = carry.metrics
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    product_shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)
    shed_value = jnp.sum(
        product_shed * states.market_price.astype(jnp.int32), axis=-1, dtype=jnp.int32
    )
    inventory = states.unit_inventory[:, player, :, :NUM_PRODUCTS].astype(jnp.int32)
    inventory_value = jnp.sum(
        inventory * states.market_price[:, None, :].astype(jnp.int32),
        axis=(1, 2),
        dtype=jnp.int32,
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
    immediately_bankable_inventory = jnp.sum(
        jnp.where(
            at_shed[..., None],
            inventory * states.market_price[:, None, :].astype(jnp.int32),
            0,
        ),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    primary = config.primary_crop_id.astype(jnp.int32)
    support = config.support_crop_id.astype(jnp.int32)
    primary_count = jnp.sum(
        (kind == TileKind.PLANT) & (crop == primary[:, None, None]),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    support_count = jnp.sum(
        (kind == TileKind.PLANT) & (crop == support[:, None, None]),
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
    return M25RolloutSummaryV2(
        final_bank=states.money[:, player].astype(jnp.int32),
        done=states.done,
        terminal_sellable_shed_value=shed_value,
        terminal_unit_inventory_value=inventory_value,
        terminal_harvestable_map_value=harvestable_value,
        avoidable_liquidation_loss=(shed_value + immediately_bankable_inventory).astype(jnp.int32),
        plant_without_same_day_water=metrics.plant_without_same_day_water,
        unexplained_failure_count=unexplained.astype(jnp.int32),
        primary_planted_tiles=primary_count,
        support_planted_tiles=support_count,
        max_hires_observed=metrics.max_hires_observed,
        final_unlocked_count=states.unlocked_count[:, player].astype(jnp.int32),
        plant_success_count=metrics.plant_success_count,
        harvest_success_count=metrics.harvest_success_count,
        sold_product_units=metrics.sold_product_units,
    )


__all__ = [
    "empty_m25_metrics_v2",
    "initialize_m25_rollout_carry_v2",
    "make_m25_crop_rollout_v2",
    "make_m25_trace_rollout_v2",
    "summarize_m25_rollout_v2",
]

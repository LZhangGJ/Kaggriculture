"""Full-season JAX rollout for the minimal M2 crop business loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    EPISODE_STEPS,
    FLAG_WATERED,
    NUM_PRODUCTS,
    SHED_ACCESS,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.simulator import (
    batched_project_unit_phase,
    batched_step_sync,
)
from kaggriculture_jax.state import reset
from kaggriculture_jax.types import Events, State, StaticTables

from .crop_executor import (
    crop_player_action_dict_v2,
    crop_policy_step_v2,
    update_crop_controller_from_effects_v2,
)
from .crop_project import ensure_crop_project_v2, refresh_crop_project_v2
from .lifecycle import initialize_project_controller_v2, snapshot_project_controller_v2
from .null_opponent import combine_with_null_opponent
from .schema import (
    CropProjectConfigV2,
    CropRolloutCarryV2,
    CropRolloutMetricsV2,
    CropRolloutSummaryV2,
)


_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)


def empty_crop_rollout_metrics_v2(batch_size: int) -> CropRolloutMetricsV2:
    zero = lambda: jnp.zeros((batch_size,), dtype=jnp.int32)
    return CropRolloutMetricsV2(*(zero() for _ in CropRolloutMetricsV2._fields))


def initialize_crop_rollout_carry_v2(
    seeds: jax.Array,
    config: CropProjectConfigV2,
    player: int,
) -> CropRolloutCarryV2:
    states = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
    controller = initialize_project_controller_v2(states, player)
    controller = ensure_crop_project_v2(states, controller, config)
    controller = refresh_crop_project_v2(states, controller, config, player)
    return CropRolloutCarryV2(
        environment_state=states,
        controller=controller,
        metrics=empty_crop_rollout_metrics_v2(states.step.shape[0]),
    )


def _same_day_water_failures(
    states: State,
    joint_action,
    config: CropProjectConfigV2,
    player: int,
) -> jax.Array:
    """Count newly planted tiles still unwatered immediately before day refresh."""

    projected = batched_project_unit_phase(states, joint_action)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    kind = projected.tile_kind[:, player]
    crop = projected.tile_crop[:, player]
    origin = projected.tile_origin_day[:, player].astype(jnp.int16)
    flags = projected.tile_flags[:, player]
    bad = (
        (kind == TileKind.PLANT)
        & (crop == config.crop_id[:, None, None])
        & (origin == current_day[:, None, None])
        & ((flags & jnp.uint8(FLAG_WATERED)) == 0)
    )
    return jnp.sum(bad, axis=(1, 2), dtype=jnp.int32)


def _update_metrics(
    metrics: CropRolloutMetricsV2,
    compile_diag,
    effect_diag,
    same_day_failure: jax.Array,
    controller,
) -> CropRolloutMetricsV2:
    return CropRolloutMetricsV2(
        invalid_raw_action_count=metrics.invalid_raw_action_count
        + compile_diag.invalid_raw_action_count,
        unexpected_pass_count=metrics.unexpected_pass_count
        + compile_diag.unexpected_pass_count,
        effect_mismatch_count=metrics.effect_mismatch_count
        + effect_diag.effect_mismatch_count,
        owner_inactive_count=metrics.owner_inactive_count
        + effect_diag.owner_inactive_count,
        deadline_missed_count=metrics.deadline_missed_count
        + effect_diag.deadline_missed_count,
        resource_unavailable_count=metrics.resource_unavailable_count
        + effect_diag.resource_unavailable_count,
        plant_actions=metrics.plant_actions + compile_diag.plant_actions,
        water_actions=metrics.water_actions + compile_diag.water_actions,
        harvest_actions=metrics.harvest_actions + compile_diag.harvest_actions,
        deposit_actions=metrics.deposit_actions + compile_diag.deposit_actions,
        seed_buy_orders=metrics.seed_buy_orders + compile_diag.seed_buy_orders,
        sell_orders=metrics.sell_orders
        + compile_diag.sell_orders
        + compile_diag.terminal_sell_orders,
        plant_success_count=metrics.plant_success_count
        + effect_diag.plant_success_count,
        water_success_count=metrics.water_success_count
        + effect_diag.water_success_count,
        harvest_success_count=metrics.harvest_success_count
        + effect_diag.harvest_success_count,
        deposit_success_count=metrics.deposit_success_count
        + effect_diag.deposit_success_count,
        seed_purchase_success_count=metrics.seed_purchase_success_count
        + effect_diag.seed_purchase_success_count,
        sold_product_units=metrics.sold_product_units
        + effect_diag.sold_product_units,
        plant_without_same_day_water=metrics.plant_without_same_day_water
        + same_day_failure,
        project_cap_hits=jnp.maximum(
            metrics.project_cap_hits, controller.project_cap_hits
        ),
        obligation_cap_hits=jnp.maximum(
            metrics.obligation_cap_hits, controller.obligation_cap_hits
        ),
    )


def make_m2_crop_rollout_v2(*, rollout_steps: int = 719, player: int = 0):
    if rollout_steps <= 0 or rollout_steps > EPISODE_STEPS - 1:
        raise ValueError("rollout_steps must be in 1..719")
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    def rollout(
        initial_carry: CropRolloutCarryV2,
        events: Events,
        tables: StaticTables,
        config: CropProjectConfigV2,
    ) -> CropRolloutCarryV2:
        def body(carry: CropRolloutCarryV2, _):
            states = carry.environment_state
            player_action, controller, _ = crop_policy_step_v2(
                states, carry.controller, config, player
            )
            joint_action = combine_with_null_opponent(
                crop_player_action_dict_v2(player_action),
                player_seat=player,
            )
            last_turn = ((states.step[0].astype(jnp.int32) + 1) % TURNS_PER_DAY) == 0
            same_day_failure = jax.lax.cond(
                last_turn,
                lambda _: _same_day_water_failures(
                    states, joint_action, config, player
                ),
                lambda _: jnp.zeros_like(states.step, dtype=jnp.int32),
                operand=None,
            )
            next_states = batched_step_sync(states, joint_action, events, tables)
            controller, effect_diag = update_crop_controller_from_effects_v2(
                states,
                next_states,
                controller,
                player_action,
                config,
                player,
            )
            controller = snapshot_project_controller_v2(
                next_states, controller, player
            )
            metrics = _update_metrics(
                carry.metrics,
                player_action.diagnostics,
                effect_diag,
                same_day_failure,
                controller,
            )
            return CropRolloutCarryV2(next_states, controller, metrics), None

        final_carry, _ = jax.lax.scan(
            body,
            initial_carry,
            xs=None,
            length=rollout_steps,
        )
        return final_carry

    return rollout


def summarize_m2_crop_rollout_v2(
    carry: CropRolloutCarryV2,
    config: CropProjectConfigV2,
    player: int,
) -> CropRolloutSummaryV2:
    states = carry.environment_state
    metrics = carry.metrics
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    product_shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)
    terminal_shed_units = jnp.sum(product_shed, axis=-1, dtype=jnp.int32)
    terminal_shed_value = jnp.sum(
        product_shed * states.market_price.astype(jnp.int32),
        axis=-1,
        dtype=jnp.int32,
    )
    product_inventory = states.unit_inventory[
        :, player, :, :NUM_PRODUCTS
    ].astype(jnp.int32)
    terminal_inventory_units = jnp.sum(
        product_inventory, axis=(1, 2), dtype=jnp.int32
    )
    terminal_inventory_value = jnp.sum(
        product_inventory * states.market_price[:, None, :].astype(jnp.int32),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    crop_id = config.crop_id.astype(jnp.int32)
    kind = states.tile_kind[:, player]
    crop = states.tile_crop[:, player]
    origin = states.tile_origin_day[:, player].astype(jnp.int16)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    harvestable = (
        (kind == TileKind.PLANT)
        & (crop == crop_id[:, None, None])
        & (
            current_day[:, None, None] - origin
            >= config.harvest_age_days.astype(jnp.int16)[:, None, None]
        )
        & (states.tile_yield[:, player] > 0)
    )
    harvestable_units = jnp.sum(
        jnp.where(harvestable, states.tile_yield[:, player], 0),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    harvestable_value = harvestable_units * states.market_price[batch, crop_id]
    positions = states.unit_pos[:, player].astype(jnp.int16)
    at_shed = jnp.any(
        jnp.all(
            positions[:, :, None, :] == _SHED_ACCESS[None, None, :, :],
            axis=-1,
        ),
        axis=-1,
    )
    avoidable_inventory_value = jnp.sum(
        jnp.where(
            at_shed[..., None],
            product_inventory * states.market_price[:, None, :],
            0,
        ),
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
        + metrics.project_cap_hits
        + metrics.obligation_cap_hits
    )
    return CropRolloutSummaryV2(
        final_bank=states.money[:, player].astype(jnp.int32),
        done=states.done,
        terminal_sellable_shed_units=terminal_shed_units,
        terminal_sellable_shed_value=terminal_shed_value,
        terminal_unit_inventory_units=terminal_inventory_units,
        terminal_unit_inventory_value=terminal_inventory_value,
        terminal_harvestable_map_units=harvestable_units,
        terminal_harvestable_map_value=harvestable_value,
        avoidable_liquidation_loss=(
            terminal_shed_value + avoidable_inventory_value
        ).astype(jnp.int32),
        plant_without_same_day_water=metrics.plant_without_same_day_water,
        unexplained_failure_count=unexplained.astype(jnp.int32),
        plant_success_count=metrics.plant_success_count,
        water_success_count=metrics.water_success_count,
        harvest_success_count=metrics.harvest_success_count,
        deposit_success_count=metrics.deposit_success_count,
        sold_product_units=metrics.sold_product_units,
    )


__all__ = [
    "empty_crop_rollout_metrics_v2",
    "initialize_crop_rollout_carry_v2",
    "make_m2_crop_rollout_v2",
    "summarize_m2_crop_rollout_v2",
]


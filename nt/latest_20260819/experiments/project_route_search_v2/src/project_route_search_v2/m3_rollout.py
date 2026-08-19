"""Full-season rollout, diagnostics and coverage for M3A/M3B animals."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
    EPISODE_STEPS,
    FLAG_FED,
    MarketOp,
    NUM_ANIMALS,
    NUM_PRODUCTS,
    SHED_ACCESS,
    TURNS_PER_DAY,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_project_unit_phase, batched_step_sync
from kaggriculture_jax.state import reset
from kaggriculture_jax.types import Events, State, StaticTables
from strategic_v5.constants import TaskStatusV1, TaskTypeV1

from .lifecycle import initialize_project_controller_v2, snapshot_project_controller_v2
from .m3_controller import (
    ensure_m3_projects_v2,
    m3_phase_v2,
    m3_player_action_dict_v2,
    m3_policy_step_v2,
    update_m3_controller_from_effects_v2,
)
from .m3_genome import default_m3_animal_genome_v2
from .null_opponent import combine_with_null_opponent
from .m3_schema import (
    M3AnimalGenomeV2,
    M3CoverageMetricsV2,
    M3RolloutCarryV2,
    M3RolloutMetricsV2,
    M3RolloutSummaryV2,
)


_ANIMAL_MAX = jnp.asarray(ANIMAL_MAX_HELD, dtype=jnp.int16)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int32)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)


def empty_m3_metrics_v2(batch_size: int) -> M3RolloutMetricsV2:
    z = lambda: jnp.zeros((batch_size,), dtype=jnp.int32)
    return M3RolloutMetricsV2(
        invalid_raw_action_count=z(),
        unexpected_pass_count=z(),
        effect_mismatch_count=z(),
        owner_inactive_count=z(),
        deadline_missed_count=z(),
        resource_unavailable_count=z(),
        unit_compiler_overlap_count=z(),
        market_compiler_overlap_count=z(),
        build_success_count=z(),
        animal_purchase_success_count=z(),
        place_success_count=z(),
        feed_success_count=z(),
        care_success_count=z(),
        harvest_success_count=z(),
        collect_fertilizer_success_count=z(),
        deposit_success_count=z(),
        wheat_purchase_success_count=z(),
        sold_product_units=z(),
        unplanned_animal_escape=z(),
        animal_capacity_loss=z(),
        feed_hard_deadline_miss=z(),
        care_bonus_forfeited_unexplained=z(),
        care_bonus_capacity_clipped_unexplained=z(),
        animal_bought_without_place_plan=z(),
        cow_sheep_pasture_conflict=z(),
        duplicate_structure_reservation=z(),
        max_hires_observed=z(),
    )


def empty_m3_coverage_v2(batch_size: int) -> M3CoverageMetricsV2:
    return M3CoverageMetricsV2(
        phase_step_count=jnp.zeros((batch_size, 6), dtype=jnp.int32),
        animal_target_step_count=jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int32),
        build_actions=jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int32),
        purchase_orders=jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int32),
        place_actions=jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int32),
        feed_actions=jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int32),
        care_actions=jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int32),
        harvest_actions=jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int32),
        fertilizer_actions=jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int32),
        sell_orders_by_product=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int32),
        land_orders=jnp.zeros((batch_size,), dtype=jnp.int32),
        hire_orders=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def initialize_m3_rollout_carry_v2(
    seeds: jax.Array,
    genome: M3AnimalGenomeV2 | None = None,
    *,
    player: int = 0,
) -> M3RolloutCarryV2:
    states = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
    if genome is None:
        genome = default_m3_animal_genome_v2(states.step.shape[0])
    controller = initialize_project_controller_v2(states, player)
    controller = ensure_m3_projects_v2(states, controller, genome, player)
    return M3RolloutCarryV2(
        environment_state=states,
        controller=controller,
        metrics=empty_m3_metrics_v2(states.step.shape[0]),
        coverage=empty_m3_coverage_v2(states.step.shape[0]),
    )


def _scatter(ids: jax.Array, mask: jax.Array, size: int) -> jax.Array:
    safe = jnp.clip(ids.astype(jnp.int32), 0, size - 1)
    return jnp.sum(
        jax.nn.one_hot(safe, size, dtype=jnp.int32) * mask[..., None],
        axis=1,
        dtype=jnp.int32,
    )


def _update_coverage(
    coverage: M3CoverageMetricsV2,
    states: State,
    action,
    controller,
    genome: M3AnimalGenomeV2,
    player: int,
) -> M3CoverageMetricsV2:
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    phase = m3_phase_v2(states, genome).astype(jnp.int32)
    phase_steps = coverage.phase_step_count.at[batch, phase].add(1)
    target = genome.animal_target[batch, phase]
    positions = states.unit_pos[:, player].astype(jnp.int32)
    x = jnp.clip(positions[..., 0], 0, 9)
    y = jnp.clip(positions[..., 1], 0, 9)
    tile_animal_raw = states.tile_animal[
        batch[:, None], player, y, x
    ].astype(jnp.int32)
    tile_animal = jnp.clip(
        tile_animal_raw,
        0,
        NUM_ANIMALS - 1,
    )
    unit_item_animal = jnp.clip(
        action.unit_item.astype(jnp.int32) - NUM_PRODUCTS, 0, NUM_ANIMALS - 1
    )
    # BUILD actions do not carry an item in the raw environment action.  The
    # species lives on the still-active controller task until effects are
    # reconciled, so use that source of truth for coverage attribution.
    build_animal = jnp.clip(controller.unit_tasks.item_id.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    build = coverage.build_actions + _scatter(
        build_animal,
        (action.unit_op == UnitOp.BUILD_COOP) | (action.unit_op == UnitOp.BUILD_PASTURE),
        NUM_ANIMALS,
    )
    place = coverage.place_actions + _scatter(
        unit_item_animal, action.unit_op == UnitOp.PLACE, NUM_ANIMALS
    )
    feed = coverage.feed_actions + _scatter(
        tile_animal, action.unit_op == UnitOp.FEED, NUM_ANIMALS
    )
    care = coverage.care_actions + _scatter(
        tile_animal, action.unit_op == UnitOp.CARE, NUM_ANIMALS
    )
    harvest = coverage.harvest_actions + _scatter(
        tile_animal,
        (action.unit_op == UnitOp.HARVEST) & (tile_animal_raw >= 0),
        NUM_ANIMALS,
    )
    fertilizer = coverage.fertilizer_actions + _scatter(
        tile_animal, action.unit_op == UnitOp.COLLECT_FERTILIZER, NUM_ANIMALS
    )

    slots = jnp.arange(action.market_op.shape[1], dtype=jnp.int8)[None]
    market_active = slots < action.market_count[:, None]
    market_animal = jnp.clip(
        action.market_item.astype(jnp.int32) - NUM_PRODUCTS, 0, NUM_ANIMALS - 1
    )
    purchases = coverage.purchase_orders + _scatter(
        market_animal,
        market_active & (action.market_op == MarketOp.BUY_ANIMAL),
        NUM_ANIMALS,
    )
    market_product = jnp.clip(action.market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    sells = coverage.sell_orders_by_product + _scatter(
        market_product,
        market_active & (action.market_op == MarketOp.SELL),
        NUM_PRODUCTS,
    )
    return M3CoverageMetricsV2(
        phase_step_count=phase_steps,
        animal_target_step_count=coverage.animal_target_step_count + (target > 0).astype(jnp.int32),
        build_actions=build,
        purchase_orders=purchases,
        place_actions=place,
        feed_actions=feed,
        care_actions=care,
        harvest_actions=harvest,
        fertilizer_actions=fertilizer,
        sell_orders_by_product=sells,
        land_orders=coverage.land_orders
        + jnp.sum(market_active & (action.market_op == MarketOp.BUY_LAND), axis=-1, dtype=jnp.int32),
        hire_orders=coverage.hire_orders
        + jnp.sum(market_active & (action.market_op == MarketOp.HIRE), axis=-1, dtype=jnp.int32),
    )


def _end_day_losses(
    states: State, joint_action, player: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    projected = batched_project_unit_phase(states, joint_action)
    is_last_turn = ((states.step + 1) % TURNS_PER_DAY) == 0
    animal = projected.tile_animal[:, player]
    present = animal >= 0
    safe = jnp.clip(animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    fed = (projected.tile_flags[:, player] & jnp.uint8(FLAG_FED)) != 0
    hard_miss = present & (projected.tile_neglect[:, player] == 1) & (~fed)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)[:, None, None]
    from kaggriculture_jax.constants import ANIMAL_FIRST_YIELD_DAY, ANIMAL_INTERVAL

    first = jnp.asarray(ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int16)[safe]
    interval = jnp.asarray(ANIMAL_INTERVAL, dtype=jnp.int16)[safe]
    days_since_first = current_day + 1 - projected.tile_origin_day[:, player].astype(jnp.int16) - first
    production = present & (days_since_first >= 0) & (jnp.mod(days_since_first, interval) == 0)
    incoming = jnp.where(
        production,
        1 + jnp.where(fed, projected.tile_pending_care[:, player].astype(jnp.int16), 0),
        0,
    )
    capacity = jnp.maximum(
        projected.tile_yield[:, player].astype(jnp.int16) + incoming - _ANIMAL_MAX[safe], 0
    )
    care_forfeited = jnp.where(
        production & (projected.tile_pending_care[:, player] > 0) & (~fed),
        projected.tile_pending_care[:, player].astype(jnp.int16),
        0,
    )
    care_clipped = jnp.where(
        production & fed & (projected.tile_pending_care[:, player] > 0),
        capacity,
        0,
    )
    return (
        jnp.where(is_last_turn, jnp.sum(hard_miss, axis=(1, 2), dtype=jnp.int32), 0),
        jnp.where(is_last_turn, jnp.sum(capacity, axis=(1, 2), dtype=jnp.int32), 0),
        jnp.where(
            is_last_turn,
            jnp.sum(care_forfeited, axis=(1, 2), dtype=jnp.int32),
            0,
        ),
        jnp.where(
            is_last_turn,
            jnp.sum(care_clipped, axis=(1, 2), dtype=jnp.int32),
            0,
        ),
    )


def _animal_admission_diagnostics(states: State, action, controller, player: int):
    slots = jnp.arange(action.market_op.shape[1], dtype=jnp.int8)[None]
    active = slots < action.market_count[:, None]
    species = jnp.clip(
        action.market_item.astype(jnp.int32) - NUM_PRODUCTS, 0, NUM_ANIMALS - 1
    )
    bought = jnp.sum(
        jax.nn.one_hot(species, NUM_ANIMALS, dtype=jnp.int16)
        * (active & (action.market_op == MarketOp.BUY_ANIMAL))[..., None]
        * action.market_amount[..., None].astype(jnp.int16),
        axis=1,
        dtype=jnp.int16,
    )
    kind = states.tile_kind[:, player]
    animal = states.tile_animal[:, player]
    coop_free = jnp.sum(
        (kind == TileKind.COOP) & (animal < 0), axis=(1, 2), dtype=jnp.int16
    )
    pasture_free = jnp.sum(
        (kind == TileKind.PASTURE) & (animal < 0), axis=(1, 2), dtype=jnp.int16
    )
    unplaced = (
        states.shed[:, player, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS].astype(jnp.int16)
        + jnp.sum(
            states.unit_inventory[
                :, player, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
            ].astype(jnp.int16),
            axis=1,
            dtype=jnp.int16,
        )
    )
    excess_goose = jnp.maximum(bought[:, 0] + unplaced[:, 0] - coop_free, 0)
    excess_pasture = jnp.maximum(
        jnp.sum(bought[:, 1:] + unplaced[:, 1:], axis=-1, dtype=jnp.int16)
        - pasture_free,
        0,
    )
    bought_without_plan = (excess_goose + excess_pasture).astype(jnp.int32)

    project = controller.tile_project_id
    mismatch = (
        ((animal == 1) & (project == 7))
        | ((animal == 2) & (project == 6))
    )
    pasture_conflict = jnp.sum(mismatch, axis=(1, 2), dtype=jnp.int32)
    return bought_without_plan, pasture_conflict


def _duplicate_reservations(controller) -> jax.Array:
    tasks = controller.unit_tasks
    relevant = (
        (tasks.status == TaskStatusV1.ACTIVE)
        & (
            (tasks.task_type == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
            | (tasks.task_type == TaskTypeV1.ANIMAL_PLACE)
        )
    )
    same = tasks.target_id[:, :, None] == tasks.target_id[:, None, :]
    pairs = jnp.triu(jnp.ones((tasks.target_id.shape[1], tasks.target_id.shape[1]), dtype=jnp.bool_), 1)
    return jnp.sum(
        same & relevant[:, :, None] & relevant[:, None, :] & pairs[None],
        axis=(1, 2),
        dtype=jnp.int32,
    )


def _update_metrics(
    metrics: M3RolloutMetricsV2,
    states: State,
    next_states: State,
    action,
    effects,
    hard_miss: jax.Array,
    capacity_loss: jax.Array,
    care_forfeited: jax.Array,
    care_clipped: jax.Array,
    bought_without_plan: jax.Array,
    pasture_conflict: jax.Array,
    duplicate: jax.Array,
    player: int,
) -> M3RolloutMetricsV2:
    pre_active = jnp.sum(states.tile_animal[:, player] >= 0, axis=(1, 2), dtype=jnp.int32)
    post_active = jnp.sum(next_states.tile_animal[:, player] >= 0, axis=(1, 2), dtype=jnp.int32)
    escaped = jnp.maximum(pre_active - post_active, 0)
    e3 = effects.e3
    return M3RolloutMetricsV2(
        invalid_raw_action_count=metrics.invalid_raw_action_count + action.diagnostics.invalid_raw_action_count,
        unexpected_pass_count=metrics.unexpected_pass_count + action.diagnostics.unexpected_pass_count,
        effect_mismatch_count=metrics.effect_mismatch_count + effects.effect_mismatch_count,
        owner_inactive_count=metrics.owner_inactive_count + effects.owner_inactive_count,
        deadline_missed_count=metrics.deadline_missed_count + effects.deadline_missed_count,
        resource_unavailable_count=metrics.resource_unavailable_count + effects.resource_unavailable_count,
        unit_compiler_overlap_count=metrics.unit_compiler_overlap_count + action.diagnostics.unit_compiler_overlap_count,
        market_compiler_overlap_count=metrics.market_compiler_overlap_count + action.diagnostics.market_compiler_overlap_count,
        build_success_count=metrics.build_success_count + e3.build_success_count,
        animal_purchase_success_count=metrics.animal_purchase_success_count + e3.animal_purchase_success_count,
        place_success_count=metrics.place_success_count + e3.place_success_count,
        feed_success_count=metrics.feed_success_count + e3.feed_success_count,
        care_success_count=metrics.care_success_count + e3.care_success_count,
        harvest_success_count=metrics.harvest_success_count + e3.harvest_success_count,
        collect_fertilizer_success_count=metrics.collect_fertilizer_success_count + e3.collect_fertilizer_success_count,
        deposit_success_count=metrics.deposit_success_count + e3.deposit_success_count,
        wheat_purchase_success_count=metrics.wheat_purchase_success_count + e3.wheat_purchase_success_count,
        sold_product_units=metrics.sold_product_units + e3.sold_product_units,
        unplanned_animal_escape=metrics.unplanned_animal_escape + escaped,
        animal_capacity_loss=metrics.animal_capacity_loss + capacity_loss,
        feed_hard_deadline_miss=metrics.feed_hard_deadline_miss + hard_miss,
        care_bonus_forfeited_unexplained=(
            metrics.care_bonus_forfeited_unexplained + care_forfeited
        ),
        care_bonus_capacity_clipped_unexplained=(
            metrics.care_bonus_capacity_clipped_unexplained + care_clipped
        ),
        animal_bought_without_place_plan=(
            metrics.animal_bought_without_place_plan + bought_without_plan
        ),
        cow_sheep_pasture_conflict=(
            metrics.cow_sheep_pasture_conflict + pasture_conflict
        ),
        duplicate_structure_reservation=metrics.duplicate_structure_reservation + duplicate,
        max_hires_observed=jnp.maximum(
            metrics.max_hires_observed, next_states.hires_today[:, player].astype(jnp.int32)
        ),
    )


def make_m3_animal_rollout_v2(
    *, rollout_steps: int = 719, player: int = 0, trace: bool | str = False
):
    if rollout_steps <= 0 or rollout_steps >= EPISODE_STEPS:
        raise ValueError("rollout_steps must be in 1..719")

    def rollout(
        initial_carry: M3RolloutCarryV2,
        events: Events,
        tables: StaticTables,
        genome: M3AnimalGenomeV2,
    ):
        def body(carry: M3RolloutCarryV2, _):
            states = carry.environment_state
            action, controller = m3_policy_step_v2(
                states, carry.controller, genome, player, tables
            )
            planned_controller = controller
            joint_action = combine_with_null_opponent(
                m3_player_action_dict_v2(action), player_seat=player
            )
            hard_miss, capacity_loss, care_forfeited, care_clipped = _end_day_losses(
                states, joint_action, player
            )
            duplicate = _duplicate_reservations(controller)
            bought_without_plan, pasture_conflict = _animal_admission_diagnostics(
                states, action, controller, player
            )
            coverage = _update_coverage(
                carry.coverage, states, action, controller, genome, player
            )
            next_states = batched_step_sync(states, joint_action, events, tables)
            controller, effects = update_m3_controller_from_effects_v2(
                states, next_states, controller, action, genome, player
            )
            controller = snapshot_project_controller_v2(next_states, controller, player)
            metrics = _update_metrics(
                carry.metrics,
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
            next_carry = M3RolloutCarryV2(next_states, controller, metrics, coverage)
            if trace == "actions":
                trace_value = joint_action
            elif trace == "diagnostic":
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


def summarize_m3_rollout_v2(
    carry: M3RolloutCarryV2, *, player: int = 0
) -> M3RolloutSummaryV2:
    states = carry.environment_state
    metrics = carry.metrics
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    products = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)
    shed_value = jnp.sum(
        products * states.market_price.astype(jnp.int32), axis=-1, dtype=jnp.int32
    )
    inventory = states.unit_inventory[:, player, :, :NUM_PRODUCTS].astype(jnp.int32)
    inventory_value = jnp.sum(
        inventory * states.market_price[:, None, :].astype(jnp.int32),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    animal = states.tile_animal[:, player]
    safe = jnp.clip(animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    product = _ANIMAL_PRODUCT[safe]
    map_value = jnp.sum(
        jnp.where(
            animal >= 0,
            states.tile_yield[:, player].astype(jnp.int32)
            * states.market_price[batch[:, None, None], product],
            0,
        ),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    stranded_shed = jnp.sum(
        states.shed[:, player, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS],
        axis=-1,
        dtype=jnp.int32,
    )
    stranded_unit = jnp.sum(
        states.unit_inventory[
            :, player, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
        ],
        axis=(1, 2),
        dtype=jnp.int32,
    )
    active_by_species = jnp.sum(
        jax.nn.one_hot(safe, NUM_ANIMALS, dtype=jnp.int32)
        * (animal >= 0)[..., None],
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
        + metrics.duplicate_structure_reservation
    )
    return M3RolloutSummaryV2(
        final_bank=states.money[:, player].astype(jnp.int32),
        done=states.done,
        terminal_sellable_shed_value=shed_value,
        terminal_unit_inventory_value=inventory_value,
        terminal_animal_product_value=map_value,
        avoidable_liquidation_loss=(shed_value + inventory_value).astype(jnp.int32),
        animals_stranded_in_shed_at_terminal=stranded_shed,
        animals_stranded_in_unit_inventory_at_terminal=stranded_unit,
        active_animals_by_species=active_by_species,
        unexplained_failure_count=unexplained.astype(jnp.int32),
        unplanned_animal_escape=metrics.unplanned_animal_escape,
        animal_capacity_loss=metrics.animal_capacity_loss,
        feed_hard_deadline_miss=metrics.feed_hard_deadline_miss,
        care_bonus_forfeited_unexplained=metrics.care_bonus_forfeited_unexplained,
        care_bonus_capacity_clipped_unexplained=(
            metrics.care_bonus_capacity_clipped_unexplained
        ),
        animal_bought_without_place_plan=metrics.animal_bought_without_place_plan,
        cow_sheep_pasture_conflict=metrics.cow_sheep_pasture_conflict,
        coverage=carry.coverage,
    )


__all__ = [
    "empty_m3_coverage_v2",
    "empty_m3_metrics_v2",
    "initialize_m3_rollout_carry_v2",
    "make_m3_animal_rollout_v2",
    "summarize_m3_rollout_v2",
]

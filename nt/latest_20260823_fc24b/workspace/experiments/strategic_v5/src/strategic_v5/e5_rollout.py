"""Full-econ RULE_ONLY rollout and hard-safety diagnostics for V5 E5."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax import reset
from kaggriculture_jax.constants import (
    EPISODE_STEPS,
    MAX_UNITS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import Events, State, StaticTables

from .constants import FailureCodeV1, TaskStatusV1, TaskTypeV1
from .e4_core import (
    build_full_core_candidates_v1,
    evaluate_full_core_feasibility_v1,
    full_core_priority_v1,
    select_full_core_candidates_v1,
)
from .e4_executor import (
    FullCoreEffectDiagnosticsV1,
    compile_full_core_action_bundle_v1,
    update_full_core_controller_from_effects_v1,
)
from .e5_econ import (
    build_full_econ_features_v1,
    full_econ_score_v1,
    select_full_econ_candidates_v1,
)
from .lifecycle import (
    clear_finished_tasks_v1,
    clear_invalidated_full_core_tasks_v1,
    empty_unit_tasks_v1,
    reset_controller_state_v1,
)
from .schema import ControllerStateV1


TASK_TYPE_COUNT_V1 = max(int(value) for value in TaskTypeV1) + 1


class FullRuleDiagnosticsV1(NamedTuple):
    invalid_raw_action_count: jax.Array
    internal_resource_conflict_count: jax.Array
    unexpected_silent_noop_count: jax.Array
    effect_mismatch_count: jax.Array
    effect_mismatch_by_module: jax.Array
    effect_mismatch_by_task_type: jax.Array
    owner_inactive_count: jax.Array
    deadline_missed_count: jax.Array
    resource_unavailable_count: jax.Array
    cross_episode_task_contamination_count: jax.Array
    nan_or_inf_count: jax.Array
    terminal_unbanked_product_count: jax.Array
    terminal_unbanked_nonproduct_count: jax.Array
    terminal_unbanked_tile_count: jax.Array
    success_by_task_type: jax.Array


class FullRuleCarryV1(NamedTuple):
    environment_state: State
    player0_controller: ControllerStateV1
    player1_controller: ControllerStateV1
    current_events: Events
    diagnostics: FullRuleDiagnosticsV1


class FullRuleArenaResultV1(NamedTuple):
    final_carry: FullRuleCarryV1
    player0_outcome: jax.Array


def _controller_batch(batch_size: int) -> ControllerStateV1:
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )


def empty_full_rule_diagnostics_v1(batch_size: int) -> FullRuleDiagnosticsV1:
    pair = lambda: jnp.zeros((batch_size, 2), dtype=jnp.int32)
    return FullRuleDiagnosticsV1(
        invalid_raw_action_count=pair(),
        internal_resource_conflict_count=pair(),
        unexpected_silent_noop_count=pair(),
        effect_mismatch_count=pair(),
        effect_mismatch_by_module=jnp.zeros(
            (batch_size, 2, 3), dtype=jnp.int32
        ),
        effect_mismatch_by_task_type=jnp.zeros(
            (batch_size, 2, TASK_TYPE_COUNT_V1), dtype=jnp.int32
        ),
        owner_inactive_count=pair(),
        deadline_missed_count=pair(),
        resource_unavailable_count=pair(),
        cross_episode_task_contamination_count=pair(),
        nan_or_inf_count=pair(),
        terminal_unbanked_product_count=pair(),
        terminal_unbanked_nonproduct_count=pair(),
        terminal_unbanked_tile_count=pair(),
        success_by_task_type=jnp.zeros(
            (batch_size, 2, TASK_TYPE_COUNT_V1), dtype=jnp.int32
        ),
    )


def initialize_full_rule_carry_v1(
    seeds: jax.Array, events: Events
) -> FullRuleCarryV1:
    seeds = jnp.asarray(seeds, dtype=jnp.int32)
    batch_size = seeds.shape[0]
    return FullRuleCarryV1(
        environment_state=jax.vmap(reset)(seeds),
        player0_controller=_controller_batch(batch_size),
        player1_controller=_controller_batch(batch_size),
        current_events=events,
        diagnostics=empty_full_rule_diagnostics_v1(batch_size),
    )


def cleanup_full_controller_day_end_v1(
    controller: ControllerStateV1,
    next_unit_active: jax.Array,
    day_end: jax.Array,
) -> ControllerStateV1:
    """Clear vanished hand tasks while preserving a farmer route across days."""

    slot = jnp.arange(MAX_UNITS, dtype=jnp.int16)[None, :]
    clear = day_end[:, None] & (slot > 0) & (~next_unit_active)
    empty = empty_unit_tasks_v1()
    empty = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (day_end.shape[0],) + value.shape),
        empty,
    )
    unit_tasks = jax.tree.map(
        lambda value, replacement: jnp.where(clear, replacement, value),
        controller.unit_tasks,
        empty,
    )
    return controller._replace(unit_tasks=unit_tasks)


def _stack_players(left, right):
    return jax.tree.map(
        lambda value0, value1: jnp.stack((value0, value1), axis=1), left, right
    )


def _task_successes(
    effect: FullCoreEffectDiagnosticsV1,
    terminal_orders: jax.Array,
) -> jax.Array:
    batch_size = effect.effect_mismatch_count.shape[0]
    result = jnp.zeros((batch_size, TASK_TYPE_COUNT_V1), dtype=jnp.int32)

    def add(task: TaskTypeV1, value: jax.Array) -> None:
        nonlocal result
        result = result.at[:, int(task)].add(value.astype(jnp.int32))

    add(
        TaskTypeV1.SAFE_RECOVERY,
        effect.crop_inventory.pickup_success_count
        + effect.crop_inventory.deposit_success_count,
    )
    add(
        TaskTypeV1.CROP_PRODUCTION,
        effect.crop_inventory.seed_purchase_success_count
        + effect.crop_inventory.plant_success_count
        + effect.crop_inventory.harvest_success_count,
    )
    add(TaskTypeV1.WATER_CROP, effect.crop_inventory.water_success_count)
    add(TaskTypeV1.CLEAR_OR_REMOVE_TILE, effect.crop_inventory.clear_success_count)
    add(TaskTypeV1.BUILD_ANIMAL_STRUCTURE, effect.e3.build_success_count)
    add(TaskTypeV1.ANIMAL_PURCHASE, effect.e3.animal_purchase_success_count)
    add(TaskTypeV1.ANIMAL_PLACE, effect.e3.place_success_count)
    add(TaskTypeV1.ANIMAL_FEED, effect.e3.feed_success_count)
    add(TaskTypeV1.ANIMAL_CARE, effect.e3.care_success_count)
    add(TaskTypeV1.ANIMAL_COLLECT_PRODUCT, effect.e3.harvest_success_count)
    add(
        TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
        effect.e3.collect_fertilizer_success_count,
    )
    add(TaskTypeV1.BUY_LAND, effect.e2.land_purchase_success_count)
    add(
        TaskTypeV1.BUY_PRODUCT,
        effect.e2.fertilizer_buy_success_count
        + effect.e3.wheat_purchase_success_count,
    )
    add(TaskTypeV1.APPLY_FERTILIZER, effect.e2.fertilizer_apply_success_count)
    add(TaskTypeV1.HIRE_WORKER, effect.crop_inventory.hire_success_count)
    add(
        TaskTypeV1.SHED_PICKUP,
        effect.e2.fertilizer_pickup_success_count + effect.e3.pickup_success_count,
    )
    add(
        TaskTypeV1.SHED_DEPOSIT,
        effect.e3.deposit_success_count
        + effect.crop_inventory.deposit_success_count,
    )
    add(
        TaskTypeV1.SELL_INVENTORY,
        effect.e3.sold_product_units + effect.crop_inventory.sold_product_units,
    )
    add(TaskTypeV1.TERMINAL_LIQUIDATION, terminal_orders)
    return result


def _terminal_assets(next_states: State) -> tuple[jax.Array, jax.Array, jax.Array]:
    done = next_states.done[:, None]
    product = (
        jnp.sum(next_states.shed[:, :, :NUM_PRODUCTS].astype(jnp.int32), axis=-1)
        + jnp.sum(
            next_states.unit_inventory[:, :, :, :NUM_PRODUCTS].astype(jnp.int32),
            axis=(2, 3),
        )
    )
    nonproduct = (
        jnp.sum(next_states.shed[:, :, NUM_PRODUCTS:].astype(jnp.int32), axis=-1)
        + jnp.sum(
            next_states.unit_inventory[:, :, :, NUM_PRODUCTS:].astype(jnp.int32),
            axis=(2, 3),
        )
    )
    occupied = (next_states.tile_kind != TileKind.EMPTY) & (
        next_states.tile_kind != TileKind.LOCKED
    )
    tile = jnp.sum(occupied.astype(jnp.int32), axis=(2, 3))
    return (
        jnp.where(done, product, 0),
        jnp.where(done, nonproduct, 0),
        jnp.where(done, tile, 0),
    )


def _active_task_count(controller: ControllerStateV1) -> jax.Array:
    return (
        jnp.sum(
            controller.unit_tasks.status == TaskStatusV1.ACTIVE,
            axis=-1,
            dtype=jnp.int32,
        )
        + jnp.sum(
            controller.market_tasks.status == TaskStatusV1.ACTIVE,
            axis=-1,
            dtype=jnp.int32,
        )
    )


def _effect_mismatch_tasks(controller: ControllerStateV1) -> jax.Array:
    task_ids = jnp.arange(TASK_TYPE_COUNT_V1, dtype=jnp.int8)
    unit = jnp.sum(
        (
            controller.unit_tasks.failure_code[..., None]
            == FailureCodeV1.EFFECT_MISMATCH
        )
        & (
            controller.unit_tasks.task_type[..., None] == task_ids
        ),
        axis=1,
        dtype=jnp.int32,
    )
    market = jnp.sum(
        (
            controller.market_tasks.failure_code[..., None]
            == FailureCodeV1.EFFECT_MISMATCH
        )
        & (
            controller.market_tasks.task_type[..., None] == task_ids
        ),
        axis=1,
        dtype=jnp.int32,
    )
    return unit + market


def _add_diagnostics(
    prior: FullRuleDiagnosticsV1,
    states: State,
    next_states: State,
    bundle,
    selection0,
    selection1,
    effect0: FullCoreEffectDiagnosticsV1,
    effect1: FullCoreEffectDiagnosticsV1,
    score0: jax.Array,
    score1: jax.Array,
    controller0: ControllerStateV1,
    controller1: ControllerStateV1,
) -> FullRuleDiagnosticsV1:
    effects = _stack_players(effect0, effect1)
    product, nonproduct, tile = _terminal_assets(next_states)
    active = jnp.stack(
        (_active_task_count(controller0), _active_task_count(controller1)), axis=1
    )
    contamination = jnp.where(next_states.done[:, None], active, 0)
    score_nonfinite = jnp.stack(
        (
            jnp.sum(~jnp.isfinite(score0), axis=-1, dtype=jnp.int32),
            jnp.sum(~jnp.isfinite(score1), axis=-1, dtype=jnp.int32),
        ),
        axis=1,
    )
    state_nonfinite = jnp.zeros_like(score_nonfinite)
    del states
    success = jnp.stack(
        (
            _task_successes(
                effect0, bundle.player0.crop_inventory.diagnostics.terminal_sell_orders
            ),
            _task_successes(
                effect1, bundle.player1.crop_inventory.diagnostics.terminal_sell_orders
            ),
        ),
        axis=1,
    )
    increment = FullRuleDiagnosticsV1(
        invalid_raw_action_count=bundle.diagnostics.invalid_raw_action_count,
        internal_resource_conflict_count=jnp.stack(
            (
                selection0.internal_resource_conflict,
                selection1.internal_resource_conflict,
            ),
            axis=1,
        ),
        unexpected_silent_noop_count=bundle.diagnostics.unexpected_pass_count,
        effect_mismatch_count=effects.effect_mismatch_count,
        effect_mismatch_by_module=jnp.stack(
            (
                jnp.stack(
                    (
                        effect0.e2.effect_mismatch_count,
                        effect0.e3.effect_mismatch_count,
                        effect0.crop_inventory.effect_mismatch_count,
                    ),
                    axis=-1,
                ),
                jnp.stack(
                    (
                        effect1.e2.effect_mismatch_count,
                        effect1.e3.effect_mismatch_count,
                        effect1.crop_inventory.effect_mismatch_count,
                    ),
                    axis=-1,
                ),
            ),
            axis=1,
        ),
        effect_mismatch_by_task_type=jnp.stack(
            (
                _effect_mismatch_tasks(controller0),
                _effect_mismatch_tasks(controller1),
            ),
            axis=1,
        ),
        owner_inactive_count=effects.owner_inactive_count,
        deadline_missed_count=effects.deadline_missed_count,
        resource_unavailable_count=effects.resource_unavailable_count,
        cross_episode_task_contamination_count=contamination,
        nan_or_inf_count=score_nonfinite + state_nonfinite,
        terminal_unbanked_product_count=product,
        terminal_unbanked_nonproduct_count=nonproduct,
        terminal_unbanked_tile_count=tile,
        success_by_task_type=success,
    )
    return jax.tree.map(lambda left, right: left + right, prior, increment)


def full_rule_only_step_v1(
    carry: FullRuleCarryV1,
    tables: StaticTables,
    *,
    use_econ: bool = True,
    include_expected_risk: bool = True,
    include_scenario_risk: bool = True,
    audit: bool = True,
) -> tuple[FullRuleCarryV1, None]:
    states = carry.environment_state
    controller0 = clear_invalidated_full_core_tasks_v1(
        states, carry.player0_controller, 0
    )
    controller1 = clear_invalidated_full_core_tasks_v1(
        states, carry.player1_controller, 1
    )

    def decide(controller: ControllerStateV1, player: int):
        candidates = build_full_core_candidates_v1(states, controller, tables, player)
        feasibility = evaluate_full_core_feasibility_v1(
            states, candidates, tables, player
        )
        if use_econ:
            econ = build_full_econ_features_v1(
                states,
                candidates,
                feasibility,
                tables,
                player,
                include_expected_risk=include_expected_risk,
                include_scenario_risk=include_scenario_risk,
            )
            selection = select_full_econ_candidates_v1(
                states, candidates, feasibility, econ, controller, player
            )
            score = full_econ_score_v1(candidates, feasibility, econ)
        else:
            selection = select_full_core_candidates_v1(
                states, candidates, feasibility, controller, player
            )
            score = full_core_priority_v1(candidates, feasibility)
        return selection, score

    selection0, score0 = decide(controller0, 0)
    selection1, score1 = decide(controller1, 1)
    bundle = compile_full_core_action_bundle_v1(
        states, selection0.controller, selection1.controller
    )
    next_states = batched_step_sync(
        states, bundle.action, carry.current_events, tables
    )
    updated0, effect0 = update_full_core_controller_from_effects_v1(
        states, next_states, selection0.controller, bundle.player0, 0
    )
    updated1, effect1 = update_full_core_controller_from_effects_v1(
        states, next_states, selection1.controller, bundle.player1, 1
    )
    diagnostics = (
        _add_diagnostics(
            carry.diagnostics,
            states,
            next_states,
            bundle,
            selection0,
            selection1,
            effect0,
            effect1,
            score0,
            score1,
            updated0,
            updated1,
        )
        if audit
        else carry.diagnostics
    )
    day_end = ((next_states.step % TURNS_PER_DAY) == 0) & (~next_states.done)
    updated0 = cleanup_full_controller_day_end_v1(
        updated0, next_states.unit_active[:, 0], day_end
    )
    updated1 = cleanup_full_controller_day_end_v1(
        updated1, next_states.unit_active[:, 1], day_end
    )
    return FullRuleCarryV1(
        environment_state=next_states,
        player0_controller=updated0,
        player1_controller=updated1,
        current_events=carry.current_events,
        diagnostics=diagnostics,
    ), None


def make_full_rule_only_arena_rollout_v1(
    *,
    rollout_steps: int = EPISODE_STEPS - 1,
    use_econ: bool = True,
    include_expected_risk: bool = True,
    include_scenario_risk: bool = True,
    audit: bool = True,
):
    def rollout(
        initial_carry: FullRuleCarryV1, tables: StaticTables
    ) -> FullRuleArenaResultV1:
        def body(carry, _):
            return full_rule_only_step_v1(
                carry,
                tables,
                use_econ=use_econ,
                include_expected_risk=include_expected_risk,
                include_scenario_risk=include_scenario_risk,
                audit=audit,
            )

        final_carry, _ = jax.lax.scan(
            body, initial_carry, xs=None, length=rollout_steps
        )
        money = final_carry.environment_state.money
        return FullRuleArenaResultV1(
            final_carry=final_carry,
            player0_outcome=jnp.sign(money[:, 0] - money[:, 1]).astype(jnp.int8),
        )

    return rollout

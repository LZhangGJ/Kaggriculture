"""Exact-GPU rollout for two route schedules on the accepted V5 stack."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_PRODUCT,
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    NUM_CROPS,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import Action, StaticTables
from strategic_v5.e4_core import (
    build_full_core_candidates_v1,
    evaluate_full_core_feasibility_v1,
    full_core_priority_v1,
    select_full_core_candidates_v1,
)
from strategic_v5.e4_executor import (
    compile_full_core_action_bundle_v1,
    update_full_core_controller_from_effects_v1,
)
from strategic_v5.constants import TaskPhaseV1, TaskStatusV1, TaskTypeV1
from strategic_v5.e5_rollout import (
    FullRuleArenaResultV1,
    FullRuleCarryV1,
    _add_diagnostics,
    cleanup_full_controller_day_end_v1,
)
from strategic_v5.lifecycle import (
    clear_invalidated_full_core_tasks_v1,
    reset_controller_state_v1,
)
from strategic_v5.task_cards import empty_replay_task_card_program_v1

from .route_core import apply_route_schedule_v1
from .schema import RouteScheduleV1


EMPTY_ROUTE_TASK_CARD_V1 = empty_replay_task_card_program_v1()
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)
_CROP_FIRST_YIELD_DAY = jnp.asarray(CROP_FIRST_YIELD_DAY, dtype=jnp.int16)


def _clear_terminal_controller_v1(controller, done: jax.Array):
    """Discard the plan ledger after DONE; environment assets remain audited."""

    reset = reset_controller_state_v1()
    reset = jax.tree.map(
        lambda value: jnp.broadcast_to(
            value, (done.shape[0],) + value.shape
        ),
        reset,
    )
    return jax.tree.map(
        lambda value, replacement: jnp.where(
            done.reshape((done.shape[0],) + (1,) * (value.ndim - 1)),
            replacement,
            value,
        ),
        controller,
        reset,
    )


def chain_animal_care_after_collection_v1(
    controller,
    schedule: RouteScheduleV1,
    next_states,
    player: int,
    step: jax.Array,
):
    """Care at distance zero after collecting, before the depot return."""

    tasks = controller.unit_tasks
    collected = (
        (tasks.task_type == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (tasks.task_type == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    batch_size = next_states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    animal = next_states.tile_animal[batch, player, y, x]
    flags = next_states.tile_flags[batch, player, y, x]
    target = jnp.stack((x, y), axis=-1).astype(jnp.int8)
    at_target = jnp.all(next_states.unit_pos[:, player] == target, axis=-1)
    chain = (
        schedule.enabled[:, None]
        & schedule.chain_care_after_collection[:, None]
        & (~schedule.chain_animal_service_after_action[:, None])
        & (step[:, None] < schedule.liquidation_start_step[:, None])
        & (tasks.status == TaskStatusV1.ACTIVE)
        & (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT)
        & collected
        & at_target
        & (animal >= 0)
        & ((flags & jnp.uint8(FLAG_CARED)) == 0)
    )
    return controller._replace(
        unit_tasks=tasks._replace(
            task_type=jnp.where(
                chain, TaskTypeV1.ANIMAL_CARE, tasks.task_type
            ).astype(jnp.int8),
            item_id=jnp.where(chain, animal, tasks.item_id).astype(jnp.int8),
            phase=jnp.where(
                chain, TaskPhaseV1.OPERATE, tasks.phase
            ).astype(jnp.int8),
        )
    )


def chain_animal_service_after_action_v1(
    controller,
    previous_controller,
    schedule: RouteScheduleV1,
    next_states,
    player: int,
    step: jax.Array,
):
    """Finish other due services at the same animal tile before returning."""

    tasks = controller.unit_tasks
    previous = previous_controller.unit_tasks
    previous_service = (
        (previous.task_type == TaskTypeV1.ANIMAL_FEED)
        | (previous.task_type == TaskTypeV1.ANIMAL_CARE)
        | (previous.task_type == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (previous.task_type == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    batch_size = next_states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    unit = jnp.arange(next_states.unit_active.shape[-1], dtype=jnp.int32)[None, :]
    x = jnp.clip(previous.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(previous.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    target = jnp.stack((x, y), axis=-1).astype(jnp.int8)
    at_target = jnp.all(next_states.unit_pos[:, player] == target, axis=-1)
    animal = next_states.tile_animal[batch, player, y, x]
    safe_animal = jnp.clip(animal.astype(jnp.int32), 0, len(_ANIMAL_PRODUCT) - 1)
    flags = next_states.tile_flags[batch, player, y, x]
    yield_units = next_states.tile_yield[batch, player, y, x]
    carried_wheat = next_states.unit_inventory[batch, player, unit, 0]
    need_feed = (
        (flags & jnp.uint8(FLAG_FED)) == 0
    ) & (carried_wheat > 0)
    need_product = yield_units > 0
    need_fertilizer = (
        flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)
    ) != 0
    need_care = (flags & jnp.uint8(FLAG_CARED)) == 0
    has_next = need_feed | need_product | need_fertilizer | need_care
    recent_effect = (
        (previous.status == TaskStatusV1.ACTIVE)
        & previous_service
        & (
            (tasks.status == TaskStatusV1.DONE)
            | (
                (tasks.status == TaskStatusV1.ACTIVE)
                & (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT)
            )
        )
    )
    chain = (
        schedule.enabled[:, None]
        & schedule.chain_animal_service_after_action[:, None]
        & (step[:, None] < schedule.liquidation_start_step[:, None])
        & recent_effect
        & at_target
        & (animal >= 0)
        & has_next
    )
    next_task = jnp.where(
        need_feed,
        TaskTypeV1.ANIMAL_FEED,
        jnp.where(
            need_product,
            TaskTypeV1.ANIMAL_COLLECT_PRODUCT,
            jnp.where(
                need_fertilizer,
                TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
                TaskTypeV1.ANIMAL_CARE,
            ),
        ),
    ).astype(jnp.int8)
    next_item = jnp.where(
        need_feed,
        0,
        jnp.where(
            need_product,
            _ANIMAL_PRODUCT[safe_animal],
            jnp.where(need_fertilizer, 8, animal),
        ),
    ).astype(jnp.int8)
    return controller._replace(
        unit_tasks=tasks._replace(
            task_type=jnp.where(chain, next_task, tasks.task_type).astype(jnp.int8),
            item_id=jnp.where(chain, next_item, tasks.item_id).astype(jnp.int8),
            target_id=jnp.where(chain, previous.target_id, tasks.target_id).astype(
                jnp.int16
            ),
            target_x=jnp.where(chain, previous.target_x, tasks.target_x).astype(
                jnp.int8
            ),
            target_y=jnp.where(chain, previous.target_y, tasks.target_y).astype(
                jnp.int8
            ),
            phase=jnp.where(chain, TaskPhaseV1.OPERATE, tasks.phase).astype(jnp.int8),
            status=jnp.where(chain, TaskStatusV1.ACTIVE, tasks.status).astype(jnp.int8),
        )
    )
def batch_route_collection_tasks_v1(
    controller,
    schedule: RouteScheduleV1,
    next_states,
    player: int,
    step: jax.Array,
):
    """Let a collector take another job until its carried batch is large enough."""

    tasks = controller.unit_tasks
    collected = (
        (tasks.task_type == TaskTypeV1.CROP_PRODUCTION)
        | (tasks.task_type == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (tasks.task_type == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    carried = jnp.sum(
        next_states.unit_inventory[:, player].astype(jnp.int32), axis=-1
    )
    release = (
        schedule.enabled[:, None]
        & (step[:, None] < schedule.liquidation_start_step[:, None])
        & (tasks.status == TaskStatusV1.ACTIVE)
        & (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT)
        & collected
        & (
            carried
            < jnp.maximum(schedule.deposit_batch_units, 1)[:, None]
        )
    )
    return controller._replace(
        unit_tasks=tasks._replace(
            status=jnp.where(
                release, TaskStatusV1.DONE, tasks.status
            ).astype(jnp.int8)
        )
    )


def chain_crop_harvest_after_action_v1(
    controller,
    previous_controller,
    schedule: RouteScheduleV1,
    states,
    next_states,
    player: int,
    step: jax.Array,
):
    """Send a successful harvester to a distinct ready crop before depositing."""

    tasks = controller.unit_tasks
    previous = previous_controller.unit_tasks
    carried_before = jnp.sum(
        states.unit_inventory[:, player].astype(jnp.int32), axis=-1
    )
    carried_after = jnp.sum(
        next_states.unit_inventory[:, player].astype(jnp.int32), axis=-1
    )
    just_harvested = (
        (previous.status == TaskStatusV1.ACTIVE)
        & (previous.task_type == TaskTypeV1.CROP_PRODUCTION)
        & (previous.phase != TaskPhaseV1.MOVE_TO_DEPOT)
        & (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT)
        & (carried_after > carried_before)
        & (
            carried_after
            < jnp.maximum(schedule.crop_harvest_batch_units, 1)[:, None]
        )
    )
    batch_size = next_states.step.shape[0]
    flat_kind = next_states.tile_kind[:, player].reshape(batch_size, -1)
    flat_crop = next_states.tile_crop[:, player].reshape(batch_size, -1)
    flat_yield = next_states.tile_yield[:, player].reshape(batch_size, -1)
    flat_origin = next_states.tile_origin_day[:, player].reshape(batch_size, -1)
    current_day = (next_states.step // TURNS_PER_DAY).astype(jnp.int16)
    safe_crop = jnp.clip(flat_crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    ready = (
        (flat_kind == TileKind.PLANT)
        & (flat_yield > 0)
        & (
            current_day[:, None] - flat_origin.astype(jnp.int16)
            >= _CROP_FIRST_YIELD_DAY[safe_crop]
        )
    )
    harvester_count = jnp.maximum(
        jnp.sum(just_harvested, axis=-1, dtype=jnp.int16), 1
    )
    harvester_rank = (
        jnp.cumsum(just_harvested.astype(jnp.int16), axis=-1) - 1
    ).astype(jnp.int16)
    crop_rank = (jnp.cumsum(ready.astype(jnp.int16), axis=-1) - 1).astype(
        jnp.int16
    )
    crop_owner_rank = jnp.mod(crop_rank, harvester_count[:, None])
    tile_id = jnp.arange(BOARD_SIZE * BOARD_SIZE, dtype=jnp.int16)
    tile_x = (tile_id % BOARD_SIZE).astype(jnp.int8)
    tile_y = (tile_id // BOARD_SIZE).astype(jnp.int8)
    tile_xy = jnp.stack((tile_x, tile_y), axis=-1)
    position = next_states.unit_pos[:, player].astype(jnp.int16)
    distance = jnp.sum(
        jnp.abs(position[:, :, None, :] - tile_xy[None, None].astype(jnp.int16)),
        axis=-1,
    )
    assigned = (
        ready[:, None, :]
        & just_harvested[:, :, None]
        & (crop_owner_rank[:, None, :] == harvester_rank[:, :, None])
    )
    target_key = jnp.where(
        assigned, distance * 100 + tile_id[None, None, :], 32767
    )
    target_id = jnp.argmin(target_key, axis=-1).astype(jnp.int16)
    target_valid = jnp.min(target_key, axis=-1) < 32767
    chain = (
        schedule.enabled[:, None]
        & schedule.chain_crop_harvest_after_action[:, None]
        & (step[:, None] < schedule.liquidation_start_step[:, None])
        & just_harvested
        & target_valid
    )
    target_x = tile_x[target_id]
    target_y = tile_y[target_id]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    target_crop = next_states.tile_crop[
        batch, player, target_y.astype(jnp.int32), target_x.astype(jnp.int32)
    ]
    return controller._replace(
        unit_tasks=tasks._replace(
            task_type=jnp.where(
                chain, TaskTypeV1.CROP_PRODUCTION, tasks.task_type
            ).astype(jnp.int8),
            item_id=jnp.where(chain, target_crop, tasks.item_id).astype(jnp.int8),
            target_id=jnp.where(chain, target_id, tasks.target_id).astype(jnp.int16),
            target_x=jnp.where(chain, target_x, tasks.target_x).astype(jnp.int8),
            target_y=jnp.where(chain, target_y, tasks.target_y).astype(jnp.int8),
            phase=jnp.where(
                chain, TaskPhaseV1.MOVE_TO_TARGET, tasks.phase
            ).astype(jnp.int8),
            status=jnp.where(
                chain, TaskStatusV1.ACTIVE, tasks.status
            ).astype(jnp.int8),
        )
    )


def route_rule_step_with_action_v1(
    carry: FullRuleCarryV1,
    tables: StaticTables,
    schedule0: RouteScheduleV1,
    schedule1: RouteScheduleV1,
    *,
    include_expected_risk: bool = True,
    include_scenario_risk: bool = True,
    audit: bool = True,
) -> tuple[FullRuleCarryV1, Action]:
    """Advance one exact step and expose the executed joint action."""

    states = carry.environment_state
    controller0 = clear_invalidated_full_core_tasks_v1(
        states, carry.player0_controller, 0
    )
    controller1 = clear_invalidated_full_core_tasks_v1(
        states, carry.player1_controller, 1
    )

    def decide(controller, schedule: RouteScheduleV1, player: int):
        candidates = build_full_core_candidates_v1(
            states,
            controller,
            tables,
            player,
            EMPTY_ROUTE_TASK_CARD_V1,
            route_profile=True,
        )
        candidates = apply_route_schedule_v1(
            states, candidates, schedule, player
        )
        feasibility = evaluate_full_core_feasibility_v1(
            states, candidates, tables, player
        )
        # RouteSchedule already owns the long-horizon commitment.  The generic
        # E5 econ layer reserves future animal cash against free structure
        # builds and was empirically unable to reproduce the accepted gold
        # production skeleton.  Keep exact legality/resource arbitration, but
        # select by the route-conditioned E4 priority.
        selection = select_full_core_candidates_v1(
            states, candidates, feasibility, controller, player
        )
        return selection, full_core_priority_v1(candidates, feasibility)

    selection0, score0 = decide(controller0, schedule0, 0)
    selection1, score1 = decide(controller1, schedule1, 1)
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
    updated0 = _clear_terminal_controller_v1(updated0, next_states.done)
    updated1 = _clear_terminal_controller_v1(updated1, next_states.done)
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
    updated0 = chain_animal_care_after_collection_v1(
        updated0, schedule0, next_states, 0, states.step
    )
    updated1 = chain_animal_care_after_collection_v1(
        updated1, schedule1, next_states, 1, states.step
    )
    updated0 = chain_animal_service_after_action_v1(
        updated0,
        selection0.controller,
        schedule0,
        next_states,
        0,
        states.step,
    )
    updated1 = chain_animal_service_after_action_v1(
        updated1,
        selection1.controller,
        schedule1,
        next_states,
        1,
        states.step,
    )
    updated0 = chain_crop_harvest_after_action_v1(
        updated0,
        selection0.controller,
        schedule0,
        states,
        next_states,
        0,
        states.step,
    )
    updated1 = chain_crop_harvest_after_action_v1(
        updated1,
        selection1.controller,
        schedule1,
        states,
        next_states,
        1,
        states.step,
    )
    updated0 = batch_route_collection_tasks_v1(
        updated0, schedule0, next_states, 0, states.step
    )
    updated1 = batch_route_collection_tasks_v1(
        updated1, schedule1, next_states, 1, states.step
    )
    day_end = ((next_states.step % TURNS_PER_DAY) == 0) & (~next_states.done)
    updated0 = cleanup_full_controller_day_end_v1(
        updated0, next_states.unit_active[:, 0], day_end
    )
    updated1 = cleanup_full_controller_day_end_v1(
        updated1, next_states.unit_active[:, 1], day_end
    )
    following = FullRuleCarryV1(
        environment_state=next_states,
        player0_controller=updated0,
        player1_controller=updated1,
        current_events=carry.current_events,
        diagnostics=diagnostics,
    )
    return following, bundle.action


def route_rule_step_v1(
    carry: FullRuleCarryV1,
    tables: StaticTables,
    schedule0: RouteScheduleV1,
    schedule1: RouteScheduleV1,
    *,
    include_expected_risk: bool = True,
    include_scenario_risk: bool = True,
    audit: bool = True,
) -> tuple[FullRuleCarryV1, None]:
    """Advance one exact simulator step with independently batched routes."""

    following, _ = route_rule_step_with_action_v1(
        carry,
        tables,
        schedule0,
        schedule1,
        include_expected_risk=include_expected_risk,
        include_scenario_risk=include_scenario_risk,
        audit=audit,
    )
    return following, None


def make_route_rule_arena_rollout_v1(
    *,
    rollout_steps: int = EPISODE_STEPS - 1,
    include_expected_risk: bool = True,
    include_scenario_risk: bool = True,
    audit: bool = True,
):
    """Return a JIT-compatible full-season route-vs-route rollout."""

    def rollout(
        initial_carry: FullRuleCarryV1,
        tables: StaticTables,
        schedule0: RouteScheduleV1,
        schedule1: RouteScheduleV1,
    ) -> FullRuleArenaResultV1:
        def body(carry, _):
            return route_rule_step_v1(
                carry,
                tables,
                schedule0,
                schedule1,
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

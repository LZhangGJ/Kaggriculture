from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from kaggriculture_jax import load_event_bank, load_tables, reset
from kaggriculture_jax.constants import TileKind
from kaggriculture_jax.simulator import batched_step_sync
from strategic_v5 import (
    FERTILIZER_ITEM,
    TaskStatusV1,
    build_e2_candidates_v1,
    compile_e2_action_bundle_v1,
    evaluate_e2_feasibility_v1,
    initialize_e2_ledger_v1,
    reset_controller_state_v1,
    select_e2_candidates_v1,
    update_e2_controller_from_effects_v1,
)


def test_e2_batched_legal_task_fuzz_finishes_without_safety_failures() -> None:
    batch_size = 64
    seeds = jnp.arange(batch_size, dtype=jnp.int32) + 300
    states = jax.vmap(reset)(seeds)
    positions = jnp.asarray([[4, 4], [0, 0], [4, 0]], dtype=jnp.int8)
    states = states._replace(
        unit_active=states.unit_active.at[:, 0, :3].set(True),
        unit_pos=states.unit_pos.at[:, 0, :3].set(
            jnp.broadcast_to(positions, (batch_size, 3, 2))
        ),
    )
    for x, y in ((3, 3), (0, 1), (4, 1)):
        states = states._replace(
            tile_kind=states.tile_kind.at[:, 0, y, x].set(TileKind.PLANT),
            tile_crop=states.tile_crop.at[:, 0, y, x].set(0),
            tile_origin_day=states.tile_origin_day.at[:, 0, y, x].set(0),
        )
    pattern = jnp.arange(batch_size) % 3
    for unit in range(3):
        states = states._replace(
            unit_inventory=states.unit_inventory.at[
                :, 0, unit, FERTILIZER_ITEM
            ].set((pattern == unit).astype(jnp.int16))
        )
    states = states._replace(
        shed=states.shed.at[:, 0, FERTILIZER_ITEM].set(
            (jnp.arange(batch_size) % 4).astype(jnp.int16)
        )
    )
    one = reset_controller_state_v1()
    controller0 = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )
    controller1 = controller0
    tables = load_tables()
    candidates = build_e2_candidates_v1(states, controller0, tables, 0)
    feasibility = evaluate_e2_feasibility_v1(states, candidates, tables, 0)
    selection = select_e2_candidates_v1(
        states,
        candidates,
        feasibility,
        controller0,
        initialize_e2_ledger_v1(states, controller0, 0),
        0,
    )
    controller0 = selection.controller
    _, event_bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:batch_size], event_bank)

    def body(_, carry):
        current, c0, c1, counters = carry
        bundle = compile_e2_action_bundle_v1(current, c0, c1)
        following = batched_step_sync(current, bundle.action, events, tables)
        c0, effects = update_e2_controller_from_effects_v1(
            current, following, c0, bundle.player0, 0
        )
        counters = counters + jnp.stack(
            (
                bundle.player0.diagnostics.invalid_raw_action_count,
                bundle.player0.diagnostics.unexpected_pass_count,
                effects.effect_mismatch_count,
                effects.owner_inactive_count,
                effects.deadline_missed_count,
                effects.resource_unavailable_count,
            ),
            axis=-1,
        )
        return following, c0, c1, counters

    # Worst case in this corpus is 8 moves to a shed, PICKUP, 7 moves to the
    # plant, then FERTILIZE: 17 transitions.  Twenty stays before day reset.
    run = jax.jit(lambda carry: lax.fori_loop(0, 20, body, carry))
    counters = jnp.zeros((batch_size, 6), dtype=jnp.int32)
    _, controller0, _, counters = run((states, controller0, controller1, counters))
    assert int(jnp.sum(counters)) == 0
    assert int(jnp.sum(selection.internal_resource_conflict)) == 0
    unit_status = controller0.unit_tasks.status
    market_status = controller0.market_tasks.status
    assert bool(jnp.all((unit_status == TaskStatusV1.EMPTY) | (unit_status == TaskStatusV1.DONE)))
    assert bool(jnp.all((market_status == TaskStatusV1.EMPTY) | (market_status == TaskStatusV1.DONE)))

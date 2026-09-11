from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from kaggriculture_jax import empty_action, load_event_bank, load_tables, reset
from kaggriculture_jax.constants import (
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    SHED_CAPACITY,
    TileKind,
)
from kaggriculture_jax.simulator import batched_step_sync
from strategic_v5 import (
    TaskStatusV1,
    TaskTypeV1,
    build_e3_candidates_v1,
    compile_e3_action_bundle_v1,
    evaluate_e3_feasibility_v1,
    reset_controller_state_v1,
    select_e3_candidates_v1,
    update_e3_controller_from_effects_v1,
)


TABLES = load_tables()


def _states(seeds):
    return jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))


def _controllers(batch_size: int):
    value = reset_controller_state_v1()
    return jax.tree.map(
        lambda item: jnp.broadcast_to(item, (batch_size,) + item.shape), value
    )


def _events(batch_size: int):
    _, bank = load_event_bank()
    return jax.tree.map(lambda value: value[:batch_size], bank)


def _empty_batched_action(batch_size: int):
    one = empty_action()
    return jax.tree.map(
        lambda item: jnp.broadcast_to(item, (batch_size,) + item.shape), one
    )


def test_second_unfed_day_causes_escape_and_fertilizer_refreshes_for_survivor() -> None:
    states = _states([501, 502])
    states = states._replace(
        step=jnp.asarray([23, 23], dtype=jnp.int16),
        tile_kind=states.tile_kind.at[:, 0, 3, 3].set(TileKind.COOP),
        tile_animal=states.tile_animal.at[:, 0, 3, 3].set(0),
        tile_origin_day=states.tile_origin_day.at[:, 0, 3, 3].set(0),
        tile_neglect=states.tile_neglect.at[:, 0, 3, 3].set(
            jnp.asarray([1, 0], dtype=jnp.int8)
        ),
    )
    following = batched_step_sync(states, _empty_batched_action(2), _events(2), TABLES)
    assert int(following.tile_animal[0, 0, 3, 3]) == -1
    assert int(following.tile_animal[1, 0, 3, 3]) == 0
    assert int(following.tile_flags[1, 0, 3, 3]) & FLAG_FERTILIZER_AVAILABLE


def test_care_bonus_is_banked_then_paid_on_next_fed_production_day() -> None:
    states = _states([503])
    states = states._replace(
        step=jnp.asarray([95], dtype=jnp.int16),
        tile_kind=states.tile_kind.at[0, 0, 3, 3].set(TileKind.COOP),
        tile_animal=states.tile_animal.at[0, 0, 3, 3].set(0),
        tile_origin_day=states.tile_origin_day.at[0, 0, 3, 3].set(0),
        tile_flags=states.tile_flags.at[0, 0, 3, 3].set(FLAG_FED | FLAG_CARED),
    )
    following = batched_step_sync(states, _empty_batched_action(1), _events(1), TABLES)
    assert int(following.tile_yield[0, 0, 3, 3]) == 1
    assert int(following.tile_pending_care[0, 0, 3, 3]) == 1
    following = following._replace(
        step=jnp.asarray([119], dtype=jnp.int16),
        tile_flags=following.tile_flags.at[0, 0, 3, 3].set(FLAG_FED),
    )
    paid = batched_step_sync(following, _empty_batched_action(1), _events(1), TABLES)
    assert int(paid.tile_yield[0, 0, 3, 3]) == 3
    assert int(paid.tile_pending_care[0, 0, 3, 3]) == 0


def test_animal_product_max_held_drops_excess_production() -> None:
    states = _states([504])
    states = states._replace(
        step=jnp.asarray([119], dtype=jnp.int16),
        tile_kind=states.tile_kind.at[0, 0, 3, 3].set(TileKind.COOP),
        tile_animal=states.tile_animal.at[0, 0, 3, 3].set(0),
        tile_origin_day=states.tile_origin_day.at[0, 0, 3, 3].set(0),
        tile_yield=states.tile_yield.at[0, 0, 3, 3].set(4),
        tile_flags=states.tile_flags.at[0, 0, 3, 3].set(FLAG_FED),
    )
    following = batched_step_sync(states, _empty_batched_action(1), _events(1), TABLES)
    assert int(following.tile_yield[0, 0, 3, 3]) == 4


def test_full_shed_masks_product_collection_reservation() -> None:
    states = _states([505])
    states = states._replace(
        unit_pos=states.unit_pos.at[0, 0, 0].set(jnp.asarray([3, 3], dtype=jnp.int8)),
        shed=states.shed.at[0, 0, 0].set(SHED_CAPACITY),
        tile_kind=states.tile_kind.at[0, 0, 3, 3].set(TileKind.COOP),
        tile_animal=states.tile_animal.at[0, 0, 3, 3].set(0),
        tile_yield=states.tile_yield.at[0, 0, 3, 3].set(2),
    )
    controller = _controllers(1)
    candidates = build_e3_candidates_v1(states, controller, TABLES, 0)
    feasibility = evaluate_e3_feasibility_v1(states, candidates, TABLES, 0)
    selected = select_e3_candidates_v1(states, candidates, feasibility, controller, 0)
    indices = selected.selected_candidate_indices[0]
    safe = jnp.clip(indices, 0, candidates.task_type.shape[1] - 1)
    assert not bool(
        jnp.any(
            (indices >= 0)
            & (candidates.task_type[0, safe] == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        )
    )


def test_two_units_cannot_double_feed_one_animal_or_spend_one_wheat_twice() -> None:
    states = _states([506])
    states = states._replace(
        unit_active=states.unit_active.at[0, 0, 1].set(True),
        unit_pos=states.unit_pos.at[0, 0, :2].set(
            jnp.asarray([[3, 3], [3, 4]], dtype=jnp.int8)
        ),
        shed=states.shed.at[0, 0, 0].set(1),
        tile_kind=states.tile_kind.at[0, 0, 3, 3].set(TileKind.COOP),
        tile_animal=states.tile_animal.at[0, 0, 3, 3].set(0),
    )
    controller = _controllers(1)
    candidates = build_e3_candidates_v1(states, controller, TABLES, 0)
    feasibility = evaluate_e3_feasibility_v1(states, candidates, TABLES, 0)
    selected = select_e3_candidates_v1(states, candidates, feasibility, controller, 0)
    indices = selected.selected_candidate_indices[0]
    safe = jnp.clip(indices, 0, candidates.task_type.shape[1] - 1)
    feed_count = jnp.sum(
        (indices >= 0) & (candidates.task_type[0, safe] == TaskTypeV1.ANIMAL_FEED)
    )
    assert int(feed_count) == 1
    assert int(selected.ledger.shed_reserved_out[0, 0]) == 1
    assert int(selected.internal_resource_conflict[0]) == 0


def test_terminal_too_late_animal_route_is_not_generated() -> None:
    states = _states([507])
    states = states._replace(
        step=jnp.asarray([718], dtype=jnp.int16),
        shed=states.shed.at[0, 0, 0].set(1),
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(TileKind.COOP),
        tile_animal=states.tile_animal.at[0, 0, 0, 0].set(0),
    )
    controller = _controllers(1)
    candidates = build_e3_candidates_v1(states, controller, TABLES, 0)
    assert not bool(jnp.any(candidates.present))


def test_e3_batched_task_fuzz_has_no_conflict_noop_or_effect_mismatch() -> None:
    batch_size = 64
    states = _states(jnp.arange(batch_size) + 600)
    positions = jnp.asarray([[4, 3], [3, 4], [3, 3]], dtype=jnp.int8)
    states = states._replace(
        unit_active=states.unit_active.at[:, 0, :3].set(True),
        unit_pos=states.unit_pos.at[:, 0, :3].set(
            jnp.broadcast_to(positions, (batch_size, 3, 2))
        ),
        shed=states.shed.at[:, 0, 0].set(3),
    )
    for animal, (x, y), kind in zip(
        range(3), ((4, 3), (3, 4), (3, 3)), (TileKind.COOP, TileKind.PASTURE, TileKind.PASTURE), strict=True
    ):
        states = states._replace(
            tile_kind=states.tile_kind.at[:, 0, y, x].set(kind),
            tile_animal=states.tile_animal.at[:, 0, y, x].set(animal),
            tile_origin_day=states.tile_origin_day.at[:, 0, y, x].set(0),
        )
    controller0 = _controllers(batch_size)
    controller1 = _controllers(batch_size)
    candidates = build_e3_candidates_v1(states, controller0, TABLES, 0)
    feasibility = evaluate_e3_feasibility_v1(states, candidates, TABLES, 0)
    selection = select_e3_candidates_v1(states, candidates, feasibility, controller0, 0)
    controller0 = selection.controller
    events = _events(batch_size)

    def body(_, carry):
        current, c0, counters = carry
        bundle = compile_e3_action_bundle_v1(current, c0, controller1)
        following = batched_step_sync(current, bundle.action, events, TABLES)
        c0, effects = update_e3_controller_from_effects_v1(
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
        return following, c0, counters

    run = jax.jit(lambda carry: lax.fori_loop(0, 12, body, carry))
    _, controller0, counters = run(
        (states, controller0, jnp.zeros((batch_size, 6), dtype=jnp.int32))
    )
    assert int(jnp.sum(counters)) == 0
    assert int(jnp.sum(selection.internal_resource_conflict)) == 0
    assert bool(
        jnp.all(
            (controller0.unit_tasks.status == TaskStatusV1.EMPTY)
            | (controller0.unit_tasks.status == TaskStatusV1.DONE)
        )
    )
    assert bool(
        jnp.all(
            (controller0.market_tasks.status == TaskStatusV1.EMPTY)
            | (controller0.market_tasks.status == TaskStatusV1.DONE)
        )
    )

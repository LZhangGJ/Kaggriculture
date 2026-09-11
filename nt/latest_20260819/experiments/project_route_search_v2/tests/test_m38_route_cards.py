from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.state import load_tables, reset
from kaggriculture_jax.types import Events
from strategic_v5.e4_executor import compile_full_core_player_action_v1

from project_route_search_v2.lifecycle import (
    initialize_project_controller_v2,
    snapshot_project_controller_v2,
)
from project_route_search_v2.m35_controller import (
    m35_player_action_dict_v2,
    update_m35_controller_from_effects_v2,
)
from project_route_search_v2.m35_genome import default_m35_farm_genome_v2
from project_route_search_v2.m36_calendar import empty_route_calendar_v3
from project_route_search_v2.constants import M26FertilizerPolicyV2
from project_route_search_v2.m3_constants import M3AnimalCarePolicyV2
from project_route_search_v2.m38_route_cards import (
    RouteCardModeV4,
    RouteCardStatusV3,
    RouteCardTypeV3,
    active_route_target_mask_v3,
    clear_expired_route_cards_v3,
    materialize_route_card_unit_tasks_v3,
    materialize_route_cards_v3,
    update_route_cards_from_effects_v3,
)
from project_route_search_v2.null_opponent import combine_with_null_opponent


def _events(batch_size: int = 1) -> Events:
    return Events(
        jnp.zeros((batch_size, 30, 100), dtype=jnp.bool_),
        jnp.zeros((batch_size, 30, 101), dtype=jnp.int8),
    )


def _execute_route_step(states, controller, genome):
    planned = materialize_route_card_unit_tasks_v3(states, controller, 0)
    action = compile_full_core_player_action_v1(states, planned, 0)
    joint = combine_with_null_opponent(
        m35_player_action_dict_v2(action), player_seat=0
    )
    next_states = batched_step_sync(
        states, joint, _events(states.step.shape[0]), load_tables()
    )
    updated, diagnostics = update_m35_controller_from_effects_v2(
        states, next_states, planned, action, genome, 0
    )
    updated = update_route_cards_from_effects_v3(
        states, next_states, planned, updated, action, 0
    )
    updated = snapshot_project_controller_v2(next_states, updated, 0)
    return next_states, updated, action, diagnostics


def test_feed_tour_picks_exact_batch_then_feeds_four_targets() -> None:
    states = jax.vmap(reset)(jnp.asarray((8101,), dtype=jnp.int32))
    kind = states.tile_kind
    animal = states.tile_animal
    neglect = states.tile_neglect
    for x, y in ((3, 3), (4, 3), (3, 4), (4, 4)):
        kind = kind.at[0, 0, y, x].set(jnp.int8(TileKind.PASTURE))
        animal = animal.at[0, 0, y, x].set(jnp.int8(1))
        neglect = neglect.at[0, 0, y, x].set(jnp.int8(1))
    states = states._replace(
        step=jnp.asarray((24,), dtype=jnp.int16),
        tile_kind=kind,
        tile_animal=animal,
        tile_neglect=neglect,
        shed=states.shed.at[0, 0, 0].set(jnp.int16(4)),
    )
    genome = default_m35_farm_genome_v2(1)
    controller = initialize_project_controller_v2(states, 0)
    controller, diagnostics = materialize_route_cards_v3(
        states,
        controller,
        genome,
        empty_route_calendar_v3(1),
        jnp.ones((1, 10, 10), dtype=jnp.bool_),
        0,
    )
    cards = controller.route_cards
    assert int(diagnostics.created_feed_card_count[0]) == 1
    assert int(cards.card_type[0, 0]) == RouteCardTypeV3.FEED_TOUR
    assert int(cards.route_length[0, 0]) == 4
    assert int(cards.pickup_quantity[0, 0]) == 4
    assert int(jnp.sum(active_route_target_mask_v3(cards)[0])) == 4

    ops: list[int] = []
    amounts: list[int] = []
    mismatch = 0
    for _ in range(40):
        states, controller, action, effect = _execute_route_step(
            states, controller, genome
        )
        ops.append(int(action.unit_op[0, 0]))
        amounts.append(int(action.unit_amount[0, 0]))
        mismatch += int(effect.effect_mismatch_count[0])
        if int(controller.route_cards.status[0, 0]) == RouteCardStatusV3.DONE:
            break

    assert ops.count(UnitOp.PICKUP) == 1
    assert amounts[ops.index(UnitOp.PICKUP)] == 4
    assert ops.count(UnitOp.FEED) == 4
    assert mismatch == 0
    assert int(states.unit_inventory[0, 0, 0, 0]) == 0


def test_crop_harvest_tour_harvests_three_before_single_drop() -> None:
    states = jax.vmap(reset)(jnp.asarray((8102,), dtype=jnp.int32))
    kind = states.tile_kind
    crop = states.tile_crop
    origin = states.tile_origin_day
    tile_yield = states.tile_yield
    flags = states.tile_flags
    for x, y in ((0, 0), (1, 0), (2, 0)):
        kind = kind.at[0, 0, y, x].set(jnp.int8(TileKind.PLANT))
        crop = crop.at[0, 0, y, x].set(jnp.int8(0))
        origin = origin.at[0, 0, y, x].set(jnp.int8(0))
        tile_yield = tile_yield.at[0, 0, y, x].set(jnp.int16(1))
        flags = flags.at[0, 0, y, x].set(jnp.uint8(FLAG_WATERED))
    states = states._replace(
        step=jnp.asarray((48,), dtype=jnp.int16),
        tile_kind=kind,
        tile_crop=crop,
        tile_origin_day=origin,
        tile_yield=tile_yield,
        tile_flags=flags,
        seeds=states.seeds.at[0, 0, 0].set(jnp.int16(0)),
        unit_pos=states.unit_pos.at[0, 0, 0].set(
            jnp.asarray((0, 0), dtype=jnp.int8)
        ),
    )
    genome = default_m35_farm_genome_v2(1)
    controller = initialize_project_controller_v2(states, 0)
    controller, diagnostics = materialize_route_cards_v3(
        states,
        controller,
        genome,
        empty_route_calendar_v3(1),
        jnp.ones((1, 10, 10), dtype=jnp.bool_),
        0,
    )
    assert int(diagnostics.created_harvest_card_count[0]) == 1
    assert (
        int(controller.route_cards.card_type[0, 0])
        == RouteCardTypeV3.CROP_HARVEST_TOUR
    )
    assert int(controller.route_cards.route_length[0, 0]) == 3

    ops: list[int] = []
    mismatch = 0
    for _ in range(35):
        states, controller, action, effect = _execute_route_step(
            states, controller, genome
        )
        ops.append(int(action.unit_op[0, 0]))
        mismatch += int(effect.effect_mismatch_count[0])
        if int(controller.route_cards.status[0, 0]) == RouteCardStatusV3.DONE:
            break

    harvest_positions = [i for i, op in enumerate(ops) if op == UnitOp.HARVEST]
    drop_positions = [i for i, op in enumerate(ops) if op == UnitOp.DROP]
    assert len(harvest_positions) == 3
    assert len(drop_positions) == 1
    assert drop_positions[0] > harvest_positions[-1]
    assert mismatch == 0
    assert int(states.shed[0, 0, 0]) == 3


def test_feed_tour_collects_due_animal_product_on_same_tile() -> None:
    states = jax.vmap(reset)(jnp.asarray((8107,), dtype=jnp.int32))
    kind = states.tile_kind
    animal = states.tile_animal
    neglect = states.tile_neglect
    tile_yield = states.tile_yield
    for x, y in ((4, 4), (4, 3)):
        kind = kind.at[0, 0, y, x].set(jnp.int8(TileKind.PASTURE))
        animal = animal.at[0, 0, y, x].set(jnp.int8(1))
        neglect = neglect.at[0, 0, y, x].set(jnp.int8(1))
        tile_yield = tile_yield.at[0, 0, y, x].set(jnp.int16(2))
    states = states._replace(
        step=jnp.asarray((24,), dtype=jnp.int16),
        tile_kind=kind,
        tile_animal=animal,
        tile_neglect=neglect,
        tile_yield=tile_yield,
        shed=states.shed.at[0, 0, 0].set(jnp.int16(2)),
    )
    genome = default_m35_farm_genome_v2(1)
    controller = initialize_project_controller_v2(states, 0)
    controller, _ = materialize_route_cards_v3(
        states,
        controller,
        genome,
        empty_route_calendar_v3(1),
        jnp.ones((1, 10, 10), dtype=jnp.bool_),
        0,
    )
    assert int(controller.route_cards.route_length[0, 0]) == 2

    ops: list[int] = []
    mismatch = 0
    for _ in range(20):
        states, controller, action, effect = _execute_route_step(
            states, controller, genome
        )
        ops.append(int(action.unit_op[0, 0]))
        mismatch += int(effect.effect_mismatch_count[0])
        if int(controller.route_cards.status[0, 0]) == RouteCardStatusV3.DONE:
            break

    assert ops.count(UnitOp.FEED) == 2
    assert ops.count(UnitOp.HARVEST) == 2
    assert mismatch == 0
    assert int(states.unit_inventory[0, 0, 0, 6]) == 4


def test_feed_route_reserves_final_day_transition_for_settlement() -> None:
    states = jax.vmap(reset)(jnp.asarray((8108,), dtype=jnp.int32))
    kind = states.tile_kind
    animal = states.tile_animal
    neglect = states.tile_neglect
    for x, y in ((4, 4), (4, 0), (0, 0), (0, 6)):
        kind = kind.at[0, 0, y, x].set(jnp.int8(TileKind.PASTURE))
        animal = animal.at[0, 0, y, x].set(jnp.int8(1))
        neglect = neglect.at[0, 0, y, x].set(jnp.int8(1))
    states = states._replace(
        step=jnp.asarray((73,), dtype=jnp.int16),
        tile_kind=kind,
        tile_animal=animal,
        tile_neglect=neglect,
        shed=states.shed.at[0, 0, 0].set(jnp.int16(4)),
    )
    genome = default_m35_farm_genome_v2(1)
    animal_genome = genome.animal._replace(
        care_policy=jnp.full(
            (1, 3), M3AnimalCarePolicyV2.CAPACITY_AWARE_EVERY_CYCLE,
            dtype=jnp.int8,
        ),
        first_cycle_care_bonus_target=jnp.ones((1, 3), dtype=jnp.int8),
        steady_cycle_care_bonus_target=jnp.ones((1, 3), dtype=jnp.int8),
    )
    genome = genome._replace(animal=animal_genome)
    controller = initialize_project_controller_v2(states, 0)
    controller, _ = materialize_route_cards_v3(
        states,
        controller,
        genome,
        empty_route_calendar_v3(1),
        jnp.ones((1, 10, 10), dtype=jnp.bool_),
        0,
    )
    # Four stops would cost exactly 23 primitive transitions and would place
    # the final CARE on step 95, where the official compiler emits PASS.  The
    # corrected 22-transition budget admits only the safe prefix.
    assert int(controller.route_cards.route_length[0, 0]) == 3


def test_feed_route_prioritizes_distant_survival_over_near_care_bonus() -> None:
    states = jax.vmap(reset)(jnp.asarray((8109,), dtype=jnp.int32))
    kind = states.tile_kind
    animal = states.tile_animal
    neglect = states.tile_neglect
    for x, y in ((9, 4), (4, 4), (4, 3)):
        kind = kind.at[0, 0, y, x].set(jnp.int8(TileKind.PASTURE))
        animal = animal.at[0, 0, y, x].set(jnp.int8(1))
    neglect = neglect.at[0, 0, 4, 9].set(jnp.int8(1))
    states = states._replace(
        step=jnp.asarray((672,), dtype=jnp.int16),
        tile_kind=kind,
        tile_animal=animal,
        tile_neglect=neglect,
        shed=states.shed.at[0, 0, 0].set(jnp.int16(3)),
    )
    genome = default_m35_farm_genome_v2(1)
    animal_genome = genome.animal._replace(
        care_policy=jnp.full(
            (1, 3), M3AnimalCarePolicyV2.CAPACITY_AWARE_EVERY_CYCLE,
            dtype=jnp.int8,
        ),
        first_cycle_care_bonus_target=jnp.ones((1, 3), dtype=jnp.int8),
        steady_cycle_care_bonus_target=jnp.ones((1, 3), dtype=jnp.int8),
        liquidation_start_step=jnp.asarray((718,), dtype=jnp.int16),
    )
    genome = genome._replace(animal=animal_genome)
    controller = initialize_project_controller_v2(states, 0)
    controller, _ = materialize_route_cards_v3(
        states,
        controller,
        genome,
        empty_route_calendar_v3(1),
        jnp.ones((1, 10, 10), dtype=jnp.bool_),
        0,
    )
    assert int(controller.route_cards.target_ids[0, 0, 0]) == 49


def test_route_card_materialization_is_jittable() -> None:
    states = jax.vmap(reset)(jnp.asarray((8103, 8104), dtype=jnp.int32))
    controller = initialize_project_controller_v2(states, 0)
    genome = default_m35_farm_genome_v2(2)
    calendar = empty_route_calendar_v3(2)
    run = jax.jit(materialize_route_cards_v3, static_argnums=(5,))
    updated, diagnostics = run(
        states,
        controller,
        genome,
        calendar,
        jnp.ones((2, 10, 10), dtype=jnp.bool_),
        0,
    )
    jax.block_until_ready((updated, diagnostics))
    np.testing.assert_array_equal(diagnostics.active_card_count, 0)


def test_route_card_clear_broadcasts_over_fixed_stop_axis() -> None:
    states = jax.vmap(reset)(jnp.asarray((8105, 8106), dtype=jnp.int32))
    controller = initialize_project_controller_v2(states, 0)
    cards = controller.route_cards._replace(
        status=controller.route_cards.status.at[0, 0].set(
            jnp.int8(RouteCardStatusV3.DONE)
        ),
        target_ids=controller.route_cards.target_ids.at[0, 0, 0].set(
            jnp.int16(17)
        ),
    )
    controller = controller._replace(route_cards=cards)
    cleared = jax.jit(clear_expired_route_cards_v3, static_argnums=(2,))(
        states, controller, 0
    )
    jax.block_until_ready(cleared)
    assert int(cleared.route_cards.status[0, 0]) == RouteCardStatusV3.EMPTY
    assert int(cleared.route_cards.target_ids[0, 0, 0]) == -1
    assert int(cleared.route_cards.target_ids[1, 0, 0]) == -1


def test_crop_route_is_cancelled_if_a_future_stop_turns_into_weed() -> None:
    states = jax.vmap(reset)(jnp.asarray((8113,), dtype=jnp.int32))
    states = states._replace(
        seeds=states.seeds.at[0, 0, 0].set(jnp.int16(2)),
    )
    genome = default_m35_farm_genome_v2(1)
    target = genome.crop.crop_target.at[0, 0].set(
        jnp.asarray((2, 0, 0, 0, 0), dtype=jnp.int16)
    )
    genome = genome._replace(crop=genome.crop._replace(crop_target=target))
    controller = initialize_project_controller_v2(states, 0)
    controller, _ = materialize_route_cards_v3(
        states,
        controller,
        genome,
        empty_route_calendar_v3(1),
        jnp.ones((1, 10, 10), dtype=jnp.bool_),
        0,
        RouteCardModeV4.CROP_ONLY,
    )
    owner = int(
        np.flatnonzero(
            np.asarray(controller.route_cards.card_type[0])
            == int(RouteCardTypeV3.CROP_FIELD_TOUR)
        )[0]
    )
    tile = int(controller.route_cards.target_ids[0, owner, 0])
    x, y = tile % 10, tile // 10
    changed = states._replace(
        tile_kind=states.tile_kind.at[0, 0, y, x].set(
            jnp.int8(TileKind.WEED)
        )
    )
    cleared = clear_expired_route_cards_v3(changed, controller, 0)
    assert int(cleared.route_cards.status[0, owner]) == RouteCardStatusV3.EMPTY
    assert int(cleared.route_cards.target_ids[0, owner, 0]) == -1


def test_crop_field_tour_plants_and_waters_three_tiles() -> None:
    states = jax.vmap(reset)(jnp.asarray((8110,), dtype=jnp.int32))
    states = states._replace(
        seeds=states.seeds.at[0, 0, 0].set(jnp.int16(3)),
    )
    genome = default_m35_farm_genome_v2(1)
    target = genome.crop.crop_target.at[0, 0].set(
        jnp.asarray((3, 0, 0, 0, 0), dtype=jnp.int16)
    )
    genome = genome._replace(crop=genome.crop._replace(crop_target=target))
    controller = initialize_project_controller_v2(states, 0)
    controller, diagnostics = materialize_route_cards_v3(
        states,
        controller,
        genome,
        empty_route_calendar_v3(1),
        jnp.ones((1, 10, 10), dtype=jnp.bool_),
        0,
        RouteCardModeV4.CROP_ONLY,
    )
    assert int(diagnostics.created_crop_field_card_count[0]) == 1
    assert (
        int(controller.route_cards.card_type[0, 0])
        == RouteCardTypeV3.CROP_FIELD_TOUR
    )
    assert int(controller.route_cards.route_length[0, 0]) == 3

    ops: list[int] = []
    mismatch = 0
    for _ in range(40):
        states, controller, action, effect = _execute_route_step(
            states, controller, genome
        )
        ops.append(int(action.unit_op[0, 0]))
        mismatch += int(effect.effect_mismatch_count[0])
        if int(controller.route_cards.status[0, 0]) == RouteCardStatusV3.DONE:
            break

    assert ops.count(UnitOp.PLANT) == 3
    assert ops.count(UnitOp.WATER) == 3
    assert ops.count(UnitOp.DROP) == 0
    assert mismatch == 0


def test_crop_field_tour_waters_harvests_replants_and_banks() -> None:
    states = jax.vmap(reset)(jnp.asarray((8111,), dtype=jnp.int32))
    kind = states.tile_kind
    crop = states.tile_crop
    origin = states.tile_origin_day
    tile_yield = states.tile_yield
    flags = states.tile_flags
    for x, y in ((0, 0), (1, 0), (2, 0)):
        kind = kind.at[0, 0, y, x].set(jnp.int8(TileKind.PLANT))
        crop = crop.at[0, 0, y, x].set(jnp.int8(0))
        origin = origin.at[0, 0, y, x].set(jnp.int8(0))
        tile_yield = tile_yield.at[0, 0, y, x].set(jnp.int16(1))
        flags = flags.at[0, 0, y, x].set(jnp.uint8(0))
    states = states._replace(
        step=jnp.asarray((48,), dtype=jnp.int16),
        tile_kind=kind,
        tile_crop=crop,
        tile_origin_day=origin,
        tile_yield=tile_yield,
        tile_flags=flags,
        seeds=states.seeds.at[0, 0, 0].set(jnp.int16(3)),
        unit_pos=states.unit_pos.at[0, 0, 0].set(
            jnp.asarray((0, 0), dtype=jnp.int8)
        ),
    )
    genome = default_m35_farm_genome_v2(1)
    target = genome.crop.crop_target.at[0, 0].set(
        jnp.asarray((3, 0, 0, 0, 0), dtype=jnp.int16)
    )
    genome = genome._replace(crop=genome.crop._replace(crop_target=target))
    controller = initialize_project_controller_v2(states, 0)
    controller, diagnostics = materialize_route_cards_v3(
        states,
        controller,
        genome,
        empty_route_calendar_v3(1),
        jnp.ones((1, 10, 10), dtype=jnp.bool_),
        0,
        RouteCardModeV4.CROP_ONLY,
    )
    assert int(diagnostics.created_crop_field_card_count[0]) == 1
    assert int(controller.route_cards.route_length[0, 0]) == 3

    ops: list[int] = []
    mismatch = 0
    for _ in range(50):
        states, controller, action, effect = _execute_route_step(
            states, controller, genome
        )
        ops.append(int(action.unit_op[0, 0]))
        mismatch += int(effect.effect_mismatch_count[0])
        if int(controller.route_cards.status[0, 0]) == RouteCardStatusV3.DONE:
            break

    assert ops.count(UnitOp.HARVEST) == 3
    assert ops.count(UnitOp.PLANT) == 3
    assert ops.count(UnitOp.WATER) == 6
    assert mismatch == 0
    # Watering a mature, previously-unwatered wheat tile raises the harvested
    # yield from one to two units, so the three-stop route banks six units.
    assert int(states.shed[0, 0, 0]) == 6


def test_mixed_route_collects_animal_fertilizer_then_services_crop() -> None:
    states = jax.vmap(reset)(jnp.asarray((8112,), dtype=jnp.int32))
    kind = states.tile_kind
    animal = states.tile_animal
    crop = states.tile_crop
    origin = states.tile_origin_day
    neglect = states.tile_neglect
    flags = states.tile_flags
    for x, y in ((3, 3), (4, 3)):
        kind = kind.at[0, 0, y, x].set(jnp.int8(TileKind.PASTURE))
        animal = animal.at[0, 0, y, x].set(jnp.int8(1))
        origin = origin.at[0, 0, y, x].set(jnp.int8(0))
        neglect = neglect.at[0, 0, y, x].set(jnp.int8(1))
        flags = flags.at[0, 0, y, x].set(
            jnp.uint8(FLAG_FERTILIZER_AVAILABLE)
        )
    kind = kind.at[0, 0, 4, 3].set(jnp.int8(TileKind.PLANT))
    crop = crop.at[0, 0, 4, 3].set(jnp.int8(0))
    origin = origin.at[0, 0, 4, 3].set(jnp.int8(0))
    flags = flags.at[0, 0, 4, 3].set(jnp.uint8(0))
    states = states._replace(
        step=jnp.asarray((24,), dtype=jnp.int16),
        tile_kind=kind,
        tile_animal=animal,
        tile_crop=crop,
        tile_origin_day=origin,
        tile_neglect=neglect,
        tile_flags=flags,
        shed=states.shed.at[0, 0, 0].set(jnp.int16(2)),
        unit_active=states.unit_active.at[0, 0, 1].set(True),
        unit_pos=states.unit_pos.at[0, 0, 1].set(
            jnp.asarray((2, 4), dtype=jnp.int8)
        ),
    )
    genome = default_m35_farm_genome_v2(1)
    crop_target = genome.crop.crop_target.at[0, 0].set(
        jnp.asarray((1, 0, 0, 0, 0), dtype=jnp.int16)
    )
    fertilizer_policy = genome.crop.fertilizer_policy.at[0, 0].set(
        jnp.int8(M26FertilizerPolicyV2.ALWAYS_WHEN_AVAILABLE)
    )
    genome = genome._replace(
        crop=genome.crop._replace(
            crop_target=crop_target,
            fertilizer_policy=fertilizer_policy,
        )
    )
    controller = initialize_project_controller_v2(states, 0)
    controller, diagnostics = materialize_route_cards_v3(
        states,
        controller,
        genome,
        empty_route_calendar_v3(1),
        jnp.ones((1, 10, 10), dtype=jnp.bool_),
        0,
        RouteCardModeV4.MIXED_WITH_FALLBACK,
    )
    owners = np.flatnonzero(
        np.asarray(controller.route_cards.card_type[0])
        == int(RouteCardTypeV3.MIXED_FARM_TOUR)
    )
    assert owners.size == 1
    owner = int(owners[0])
    assert int(diagnostics.created_mixed_card_count[0]) == 1
    route_actions = np.asarray(controller.route_cards.action_masks[0, owner])
    assert np.any(route_actions & int(1))  # FEED
    assert np.any(route_actions & int(4))  # COLLECT_FERTILIZER
    assert np.any(route_actions & int(32))  # WATER
    assert np.any(route_actions & int(64))  # FERTILIZE
    assert int(np.count_nonzero(route_actions & int(4))) == 1

    ops: list[int] = []
    mismatch = 0
    for _ in range(40):
        states, controller, action, effect = _execute_route_step(
            states, controller, genome
        )
        ops.extend(int(value) for value in np.asarray(action.unit_op[0]))
        mismatch += int(effect.effect_mismatch_count[0])
        if (
            int(controller.route_cards.status[0, owner])
            == RouteCardStatusV3.DONE
        ):
            break

    assert ops.count(UnitOp.FEED) == 2
    assert ops.count(UnitOp.COLLECT_FERTILIZER) == 1
    assert ops.count(UnitOp.FERTILIZE) == 1
    assert ops.count(UnitOp.WATER) == 1
    assert mismatch == 0
    assert int(states.unit_inventory[0, 0, owner, 8]) == 0

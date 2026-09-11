from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from kaggriculture_jax.constants import (
    ANIMALS,
    CROPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    PRODUCTS,
    SHED_ITEMS,
    TileKind,
)
from strategic_v5 import (
    BUILD_PASTURE_SLOT,
    TaskTypeV1,
    load_replay_bc_episode_v2,
    observation_to_canonical_state_v2,
    opponent_primary_task_type_v2,
    strategic_candidate_labels_v2,
)
from strategic_v5.replay_bc_v2 import (
    ORDERED_STOP_SLOT_V3,
    strategic_ordered_labels_v3,
)


REPLAY = (
    Path(__file__).resolve().parents[3]
    / "replay"
    / "2026-08-12"
    / "92472865.json"
)


def test_ordered_turn_labels_preserve_unit_then_market_sequence() -> None:
    labels = strategic_ordered_labels_v3(
        {
            "farmer": ["DROP"],
            "hands": [["WATER"]],
            "market": [
                ["SELL", "WOOL", 3],
                ["BUY_ANIMAL", "COW", 1],
                ["HIRE"],
                ["HIRE"],
            ],
        }
    )
    assert labels.unit_count == 2
    assert int(labels.unit_op[0]) == 5  # UnitOp.DROP
    assert int(labels.unit_op[1]) == 9  # UnitOp.WATER
    np.testing.assert_array_equal(
        labels.market_candidate_slot[:5],
        np.asarray((19, 4, 11, 11, ORDERED_STOP_SLOT_V3), dtype=np.int16),
    )
    np.testing.assert_array_equal(
        labels.market_quantity[:5], np.asarray((3, 1, 1, 1, 0), dtype=np.int16)
    )
    np.testing.assert_array_equal(labels.market_valid[:5], np.ones((5,), dtype=np.bool_))
    assert labels.market_count == 4


@pytest.fixture(scope="module")
def replay_document():
    if not REPLAY.exists():
        pytest.skip("KAWASHIGI replay fixture is not present")
    return json.loads(REPLAY.read_text(encoding="utf-8"))


def _assert_public_and_own_private_match(state, observation, expert_seat):
    farms = [observation["farms"][expert_seat], observation["farms"][1 - expert_seat]]
    assert int(state.step) == int(observation.get("step", observation["day"] * 24 + observation["hour"]))
    for player, farm in enumerate(farms):
        assert int(state.money[player]) == farm["money"]
        assert int(state.hires_today[player]) == farm["hires_today"]
        assert int(state.unlocked_count[player]) == len(farm["unlocked_quadrants"])
        positions = [farm["farmer"], *farm["hands"]]
        np.testing.assert_array_equal(np.asarray(state.unit_pos[player, : len(positions)]), positions)
        assert int(np.asarray(state.unit_active[player]).sum()) == len(positions)
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                kind = int(state.tile_kind[player, y, x])
                if tile is None:
                    assert kind == TileKind.EMPTY
                    continue
                if tile == "LOCKED":
                    assert kind == TileKind.LOCKED
                    continue
                expected = {
                    "WEED": TileKind.WEED,
                    "PLANT": TileKind.PLANT,
                    "COOP": TileKind.COOP,
                    "PASTURE": TileKind.PASTURE,
                }[tile["kind"]]
                assert kind == expected
                if tile["kind"] == "PLANT":
                    assert int(state.tile_crop[player, y, x]) == CROPS.index(tile["crop"])
                    assert int(state.tile_origin_day[player, y, x]) == tile["planted_day"]
                    assert int(state.tile_yield[player, y, x]) == tile["yield_units"]
                    assert int(state.tile_neglect[player, y, x]) == tile["consecutive_unwatered"]
                    flags = int(state.tile_flags[player, y, x])
                    assert bool(flags & FLAG_WATERED) == tile["watered_today"]
                if "animal" in tile:
                    assert int(state.tile_animal[player, y, x]) == ANIMALS.index(tile["animal"])
                    assert int(state.tile_origin_day[player, y, x]) == tile["placed_day"]
                    assert int(state.tile_yield[player, y, x]) == tile["yield_units"]
                    assert int(state.tile_neglect[player, y, x]) == tile["consecutive_unfed"]
                    flags = int(state.tile_flags[player, y, x])
                    assert bool(flags & FLAG_FED) == tile["fed_today"]
                    assert bool(flags & FLAG_CARED) == tile["cared_today"]
                    assert bool(flags & FLAG_FERTILIZER_AVAILABLE) == tile["fertilizer_available"]

    private = observation["private"]
    np.testing.assert_array_equal(
        np.asarray(state.shed[0]), [private["shed"].get(name, 0) for name in SHED_ITEMS]
    )
    np.testing.assert_array_equal(
        np.asarray(state.seeds[0]), [private["seeds"].get(name, 0) for name in CROPS]
    )
    for unit, inventory in enumerate(private["inventories"]):
        np.testing.assert_array_equal(
            np.asarray(state.unit_inventory[0, unit]),
            [inventory.get(name, 0) for name in SHED_ITEMS],
        )
    assert not np.asarray(state.shed[1]).any()
    assert not np.asarray(state.seeds[1]).any()
    assert not np.asarray(state.unit_inventory[1]).any()
    np.testing.assert_array_equal(
        np.asarray(state.market_inventory),
        [observation["market"]["inventory"][name] for name in PRODUCTS],
    )
    np.testing.assert_array_equal(
        np.asarray(state.market_price),
        [observation["market"]["prices"][name] for name in PRODUCTS],
    )


def test_real_replay_observation_parser_matches_public_and_own_private(replay_document):
    seat = replay_document["info"]["TeamNames"].index("カワシギ")
    for source_step in (0, 96, 360, 600, 718):
        observation = replay_document["steps"][source_step][seat]["observation"]
        state = observation_to_canonical_state_v2(observation, seat)
        _assert_public_and_own_private_match(state, observation, seat)


def test_replay_parser_preserves_1327_hinge_price_above_int16(replay_document):
    seat = replay_document["info"]["TeamNames"].index("カワシギ")
    observation = copy.deepcopy(replay_document["steps"][0][seat]["observation"])
    observation["market"]["prices"]["TOMATO"] = 300_000
    state = observation_to_canonical_state_v2(observation, seat)
    assert np.asarray(state.market_price).dtype == np.int32
    assert int(state.market_price[PRODUCTS.index("TOMATO")]) == 300_000


def test_real_replay_alignment_uses_previous_observation_and_current_action(replay_document):
    seat = replay_document["info"]["TeamNames"].index("カワシギ")
    samples, source = load_replay_bc_episode_v2(REPLAY, ["カワシギ"])
    assert source["samples"] == 719
    assert len(samples) == 719
    for action_index, sample in enumerate(samples, start=1):
        expected_positive, expected_quantity = strategic_candidate_labels_v2(
            replay_document["steps"][action_index][seat]["action"]
        )
        assert sample.source_step == action_index - 1
        np.testing.assert_array_equal(sample.positive_candidate_mask, expected_positive)
        np.testing.assert_array_equal(sample.market_quantity, expected_quantity)


def test_movement_only_action_is_not_mislabeled_as_stop():
    action = {
        "farmer": ["NORTH"],
        "hands": [["WATER"], ["PASS"]],
        "market": [],
    }
    positive, quantity = strategic_candidate_labels_v2(action)
    assert not positive.any()
    assert not quantity.any()


def test_strategic_action_maps_to_fixed_slots_and_quantities():
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["BUY_SEED", "TOMATO", 3],
            ["SELL", "MILK", 4],
            ["BUY_LAND"],
        ],
    }
    positive, quantity = strategic_candidate_labels_v2(action)
    assert positive[0]
    assert positive[6 + CROPS.index("TOMATO")]
    assert positive[12 + PRODUCTS.index("MILK")]
    assert quantity[0] == 1
    assert quantity[6 + CROPS.index("TOMATO")] == 3
    assert quantity[12 + PRODUCTS.index("MILK")] == 4


def test_build_pasture_and_opponent_action_are_labels_only(replay_document):
    build = {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []}
    positive, _ = strategic_candidate_labels_v2(build)
    assert positive[BUILD_PASTURE_SLOT]

    seat = replay_document["info"]["TeamNames"].index("カワシギ")
    opponent = 1 - seat
    observation = replay_document["steps"][0][seat]["observation"]
    before = observation_to_canonical_state_v2(observation, seat)
    label_a = opponent_primary_task_type_v2(
        {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]},
        observation,
        opponent,
    )
    label_b = opponent_primary_task_type_v2(
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_LAND"]]},
        observation,
        opponent,
    )
    after = observation_to_canonical_state_v2(observation, seat)
    assert label_a == TaskTypeV1.HIRE_WORKER
    assert label_b == TaskTypeV1.BUY_LAND
    np.testing.assert_array_equal(np.asarray(before.money), np.asarray(after.money))
    np.testing.assert_array_equal(np.asarray(before.tile_kind), np.asarray(after.tile_kind))

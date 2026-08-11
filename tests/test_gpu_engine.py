from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.fast_env import resolve_agent
from kaggriculture_lab.gpu_engine import (
    ANIMAL_BASE,
    COOP,
    CudaKaggricultureEnv,
    EMPTY,
    GpuEngineConfig,
    LOCKED,
    PASTURE,
    PLANT_BASE,
    PRODUCT_INDEX,
    U_FERTILIZE,
    U_HARVEST,
    U_PLANT,
    U_BUILD_COOP,
    U_DROP,
    U_PICKUP,
    U_PLACE,
    U_WATER,
    WEED,
    encode_action_dicts,
)


def _official_tile_code(tile):
    if tile is None:
        return EMPTY
    if tile == "LOCKED":
        return LOCKED
    if tile["kind"] == "WEED":
        return WEED
    if tile["kind"] == "PLANT":
        return PLANT_BASE + ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON").index(tile["crop"])
    if "animal" in tile:
        return ANIMAL_BASE + ("GOOSE", "COW", "SHEEP").index(tile["animal"])
    return COOP if tile["kind"] == "COOP" else PASTURE


def _assert_public_state_matches(gpu, observations):
    public = observations[0]
    np.testing.assert_allclose(gpu.state.money[0].cpu().numpy(), [farm["money"] for farm in public["farms"]])
    np.testing.assert_array_equal(gpu.state.market_inventory[0].cpu().numpy(), [public["market"]["inventory"][item] for item in PRODUCT_INDEX])
    np.testing.assert_array_equal(gpu.state.market_prices[0].cpu().numpy(), [public["market"]["prices"][item] for item in PRODUCT_INDEX])
    for player in range(2):
        farm = public["farms"][player]
        private = observations[player]["private"]
        np.testing.assert_array_equal(gpu.state.seeds[0, player].cpu().numpy(), [private["seeds"][crop] for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")])
        np.testing.assert_array_equal(gpu.state.shed[0, player].cpu().numpy(), [private["shed"][item] for item in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP")])
        expected_tiles = np.asarray([[_official_tile_code(tile) for tile in row] for row in farm["tiles"]])
        np.testing.assert_array_equal(gpu.state.tile_type[0, player].cpu().numpy(), expected_tiles)
        assert int(gpu.state.hands_count[0, player]) == len(farm["hands"])
        assert int(gpu.state.hires_today[0, player]) == farm["hires_today"]
        assert int(gpu.state.unlocked_count[0, player]) == len(farm["unlocked_quadrants"])
        expected_positions = [farm["farmer"], *farm["hands"]]
        np.testing.assert_array_equal(gpu.state.positions[0, player, : len(expected_positions)].cpu().numpy(), expected_positions)


def test_gpu_reset_and_pass_step():
    config = GpuEngineConfig(episode_steps=4, weed_spawn_chance=0, town_shop_unlock_interval=1000)
    env = CudaKaggricultureEnv(3, device="cpu", config=config, seeds=[1, 2, 3])
    assert env.state.tile_type.shape == (3, 2, 10, 10)
    assert (env.state.tile_type[:, :, :5, :5] == EMPTY).all()
    assert (env.state.tile_type[:, :, 5:, 5:] == LOCKED).all()
    result = env.step(env.empty_actions())
    assert result.step == 1
    assert (env.state.market_inventory[:, :8] == 9999).all()
    assert (env.state.market_inventory[:, PRODUCT_INDEX["FERTILIZER"]] == 10000).all()


def test_gpu_starter_trajectory_matches_official_without_random_events():
    official_config = {
        "episodeSteps": 96,
        "weedSpawnChance": 0.0,
        "townShopUnlockInterval": 1000,
    }
    tensor_config = GpuEngineConfig(
        episode_steps=96,
        weed_spawn_chance=0.0,
        town_shop_unlock_interval=1000,
        max_market_quantity=8,
    )
    official = FastKaggricultureEnv(configuration=official_config)
    observations = official.reset(seed=17)
    gpu = CudaKaggricultureEnv(1, device="cpu", config=tensor_config, seeds=[17])
    cuda_gpu = CudaKaggricultureEnv(1, device="cuda", config=tensor_config, seeds=[17]) if torch.cuda.is_available() else None
    starter = resolve_agent("starter")

    while not official.done:
        pair = [starter(observations[player], official.configuration) for player in range(2)]
        official_result = official.step(pair)
        gpu_result = gpu.step(encode_action_dicts([pair], device="cpu", config=tensor_config))
        cuda_result = cuda_gpu.step(encode_action_dicts([pair], device="cuda", config=tensor_config)) if cuda_gpu else None
        observations = official_result.observations

        _assert_public_state_matches(gpu, observations)
        assert gpu_result.done == official_result.done
        if cuda_gpu:
            _assert_public_state_matches(cuda_gpu, observations)
            assert cuda_result.done == official_result.done


def test_gpu_hire_land_inventory_and_animal_path_matches_official():
    official_config = {
        "episodeSteps": 48,
        "weedSpawnChance": 0.0,
        "townShopUnlockInterval": 1000,
    }
    tensor_config = GpuEngineConfig(
        episode_steps=48,
        weed_spawn_chance=0.0,
        town_shop_unlock_interval=1000,
        max_market_quantity=16,
    )
    official = FastKaggricultureEnv(configuration=official_config)
    observations = official.reset(seed=23)
    gpu = CudaKaggricultureEnv(1, device="cpu", config=tensor_config, seeds=[23])

    def action_for(step):
        market = []
        farmer = ["PASS"]
        hands = [["PASS"]] if step < 24 and step > 0 else []
        if step == 0:
            market = [["HIRE"], ["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 16], ["BUY_LAND"]]
        elif step == 1:
            farmer = ["BUILD_COOP"]
            hands = [["PICKUP", "GOOSE", 1]]
        elif step == 2:
            farmer = ["PICKUP", "WHEAT", 16]
            hands = [["WEST"]]
        elif step == 3:
            hands = [["PLACE", "GOOSE", 1]]
        elif step == 4:
            farmer = ["FEED"]
            hands = [["CARE"]]
        elif step == 24:
            farmer = ["COLLECT_FERTILIZER"]
        elif step == 25:
            farmer = ["FEED"]
        return {"farmer": farmer, "hands": hands, "market": market}

    while not official.done:
        pair = [action_for(official.step_index), action_for(official.step_index)]
        official_result = official.step(pair)
        gpu_result = gpu.step(encode_action_dicts([pair], device="cpu", config=tensor_config))
        observations = official_result.observations
        _assert_public_state_matches(gpu, observations)
        assert gpu_result.done == official_result.done


def test_gpu_bulk_fixed_price_market_orders_match_official_at_money_limit():
    official_config = {
        "episodeSteps": 8,
        "weedSpawnChance": 0.0,
        "townShopUnlockInterval": 1000,
    }
    tensor_config = GpuEngineConfig(
        episode_steps=8,
        weed_spawn_chance=0.0,
        town_shop_unlock_interval=1000,
        max_market_quantity=100,
    )
    official = FastKaggricultureEnv(configuration=official_config)
    observations = official.reset(seed=31)
    gpu = CudaKaggricultureEnv(1, device="cpu", config=tensor_config, seeds=[31])

    pair = [
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 100]]},
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "GOOSE", 20]]},
    ]
    official_result = official.step(pair)
    gpu_result = gpu.step(encode_action_dicts([pair], device="cpu", config=tensor_config))
    observations = official_result.observations

    _assert_public_state_matches(gpu, observations)
    assert gpu_result.done == official_result.done


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_common_interactions_match_cpu_tensor_engine():
    config = GpuEngineConfig(
        episode_steps=48,
        weed_spawn_chance=0.0,
        town_shop_unlock_interval=1000,
        max_market_quantity=16,
    )
    cpu = CudaKaggricultureEnv(2, device="cpu", config=config, seeds=[41, 42])
    gpu = CudaKaggricultureEnv(2, device="cuda", config=config, seeds=[41, 42])

    def action_for(step):
        market = []
        farmer = ["PASS"]
        hands = [["PASS"]] if 0 < step < 24 else []
        if step == 0:
            market = [["HIRE"], ["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 8]]
            farmer = ["BUILD_COOP"]
        elif step == 1:
            farmer = ["DIG"]
            hands = [["PICKUP", "GOOSE", 1]]
        elif step == 2:
            farmer = ["BUILD_COOP"]
            hands = [["WEST"]]
        elif step == 3:
            farmer = ["PICKUP", "WHEAT", 8]
            hands = [["PLACE", "GOOSE", 1]]
        elif step == 4:
            farmer = ["FEED"]
            hands = [["CARE"]]
        elif step == 24:
            farmer = ["COLLECT_FERTILIZER"]
        return {"farmer": farmer, "hands": hands, "market": market}

    for step in range(30):
        pairs = [[action_for(step), action_for(step)] for _ in range(2)]
        cpu.step(encode_action_dicts(pairs, device="cpu", config=config))
        gpu.step(encode_action_dicts(pairs, device="cuda", config=config))
        for field in fields(type(cpu.state)):
            expected = getattr(cpu.state, field.name)
            actual = getattr(gpu.state, field.name).cpu()
            assert torch.equal(actual, expected), f"state mismatch in {field.name} at step {step}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_crop_lifecycle_matches_cpu_tensor_engine():
    config = GpuEngineConfig(episode_steps=96, weed_spawn_chance=0.0, town_shop_unlock_interval=1000)
    cpu = CudaKaggricultureEnv(2, device="cpu", config=config, seeds=[51, 52])
    gpu = CudaKaggricultureEnv(2, device="cuda", config=config, seeds=[51, 52])

    def compare(label):
        for field in fields(type(cpu.state)):
            expected = getattr(cpu.state, field.name)
            actual = getattr(gpu.state, field.name).cpu()
            assert torch.equal(actual, expected), f"{label}: mismatch in {field.name}"

    cpu.state.seeds[:, :, 0] = 1
    gpu.state.seeds[:, :, 0] = 1
    cpu_plant, gpu_plant = cpu.empty_actions(), gpu.empty_actions()
    cpu_plant.unit_ops[:, :, 0] = U_PLANT
    gpu_plant.unit_ops[:, :, 0] = U_PLANT
    cpu.step(cpu_plant)
    gpu.step(gpu_plant)
    compare("plant")

    cpu.step_index = gpu.step_index = 48  # day two: wheat is mature and in its bonus window
    cpu.state.unit_inventory[:, :, 0, PRODUCT_INDEX["FERTILIZER"]] = 1
    gpu.state.unit_inventory[:, :, 0, PRODUCT_INDEX["FERTILIZER"]] = 1
    cpu_fertilize, gpu_fertilize = cpu.empty_actions(), gpu.empty_actions()
    cpu_fertilize.unit_ops[:, :, 0] = U_FERTILIZE
    gpu_fertilize.unit_ops[:, :, 0] = U_FERTILIZE
    cpu.step(cpu_fertilize)
    gpu.step(gpu_fertilize)
    compare("fertilize")

    cpu_water, gpu_water = cpu.empty_actions(), gpu.empty_actions()
    cpu_water.unit_ops[:, :, 0] = U_WATER
    gpu_water.unit_ops[:, :, 0] = U_WATER
    cpu.step(cpu_water)
    gpu.step(gpu_water)
    compare("water")

    cpu_harvest, gpu_harvest = cpu.empty_actions(), gpu.empty_actions()
    cpu_harvest.unit_ops[:, :, 0] = U_HARVEST
    gpu_harvest.unit_ops[:, :, 0] = U_HARVEST
    cpu.step(cpu_harvest)
    gpu.step(gpu_harvest)
    compare("harvest")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_inventory_and_animal_placement_match_cpu_tensor_engine():
    config = GpuEngineConfig(episode_steps=48, weed_spawn_chance=0.0, town_shop_unlock_interval=1000)
    cpu = CudaKaggricultureEnv(2, device="cpu", config=config, seeds=[61, 62])
    gpu = CudaKaggricultureEnv(2, device="cuda", config=config, seeds=[61, 62])

    def compare(label):
        for field in fields(type(cpu.state)):
            expected = getattr(cpu.state, field.name)
            actual = getattr(gpu.state, field.name).cpu()
            assert torch.equal(actual, expected), f"{label}: mismatch in {field.name}"

    cpu.state.shed[:, :, 0] = 10
    gpu.state.shed[:, :, 0] = 10
    cpu_pickup, gpu_pickup = cpu.empty_actions(), gpu.empty_actions()
    cpu_pickup.unit_ops[:, :, 0] = U_PICKUP
    gpu_pickup.unit_ops[:, :, 0] = U_PICKUP
    cpu_pickup.unit_quantities[:, :, 0] = 6
    gpu_pickup.unit_quantities[:, :, 0] = 6
    cpu.step(cpu_pickup)
    gpu.step(gpu_pickup)
    compare("pickup")

    cpu.state.unit_inventory[:, :, 0, 1:4] = 3
    gpu.state.unit_inventory[:, :, 0, 1:4] = 3
    cpu_drop, gpu_drop = cpu.empty_actions(), gpu.empty_actions()
    cpu_drop.unit_ops[:, :, 0] = U_DROP
    gpu_drop.unit_ops[:, :, 0] = U_DROP
    cpu.step(cpu_drop)
    gpu.step(gpu_drop)
    compare("drop")

    cpu.state.shed[:, :, 9] = 1
    gpu.state.shed[:, :, 9] = 1
    cpu_build, gpu_build = cpu.empty_actions(), gpu.empty_actions()
    cpu_build.unit_ops[:, :, 0] = U_BUILD_COOP
    gpu_build.unit_ops[:, :, 0] = U_BUILD_COOP
    cpu.step(cpu_build)
    gpu.step(gpu_build)
    cpu_goose, gpu_goose = cpu.empty_actions(), gpu.empty_actions()
    cpu_goose.unit_ops[:, :, 0] = U_PICKUP
    gpu_goose.unit_ops[:, :, 0] = U_PICKUP
    cpu_goose.unit_args[:, :, 0] = 9
    gpu_goose.unit_args[:, :, 0] = 9
    cpu.step(cpu_goose)
    gpu.step(gpu_goose)
    cpu_place, gpu_place = cpu.empty_actions(), gpu.empty_actions()
    cpu_place.unit_ops[:, :, 0] = U_PLACE
    gpu_place.unit_ops[:, :, 0] = U_PLACE
    cpu_place.unit_args[:, :, 0] = 9
    gpu_place.unit_args[:, :, 0] = 9
    cpu.step(cpu_place)
    gpu.step(gpu_place)
    compare("place animal")

    cpu.state.unit_inventory[:, :, 0, 1] = 5
    gpu.state.unit_inventory[:, :, 0, 1] = 5
    cpu_place_shed, gpu_place_shed = cpu.empty_actions(), gpu.empty_actions()
    cpu_place_shed.unit_ops[:, :, 0] = U_PLACE
    gpu_place_shed.unit_ops[:, :, 0] = U_PLACE
    cpu_place_shed.unit_args[:, :, 0] = 1
    gpu_place_shed.unit_args[:, :, 0] = 1
    cpu_place_shed.unit_quantities[:, :, 0] = 3
    gpu_place_shed.unit_quantities[:, :, 0] = 3
    cpu.step(cpu_place_shed)
    gpu.step(gpu_place_shed)
    compare("place shed")

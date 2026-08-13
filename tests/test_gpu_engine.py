from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.fast_env import resolve_agent
from kaggriculture_lab.gpu_engine import (
    ANIMAL_BASE,
    CROPS,
    COOP,
    CudaKaggricultureEnv,
    EMPTY,
    GpuEngineConfig,
    ITEMS,
    LOCKED,
    M_BUY_ANIMAL,
    M_BUY_LAND,
    M_BUY_PRODUCT,
    M_BUY_SEED,
    M_HIRE,
    M_SELL,
    PASTURE,
    PLANT_BASE,
    PRODUCT_INDEX,
    SHOP_NAMES,
    U_FERTILIZE,
    U_HARVEST,
    U_PLANT,
    U_BUILD_COOP,
    U_DROP,
    U_EAST,
    U_NORTH,
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


def _assert_public_state_matches(gpu, observations, *, batch=0):
    public = observations[0]
    assert int(public["step"]) == gpu.step_index, "observation.step"
    assert int(public["day"]) == gpu.step_index // gpu.config.turns_per_day, "observation.day"
    assert int(public["hour"]) == gpu.step_index % gpu.config.turns_per_day, "observation.hour"
    np.testing.assert_array_equal(
        gpu.state.money[batch].cpu().numpy(),
        [farm["money"] for farm in public["farms"]],
        err_msg="farms[*].money",
    )
    np.testing.assert_array_equal(
        gpu.state.market_inventory[batch].cpu().numpy(),
        [public["market"]["inventory"][item] for item in PRODUCT_INDEX],
        err_msg="market.inventory",
    )
    np.testing.assert_array_equal(
        gpu.state.market_prices[batch].cpu().numpy(),
        [public["market"]["prices"][item] for item in PRODUCT_INDEX],
        err_msg="market.prices",
    )
    expected_shop_counts = [public["town"]["unlocked_shops"].count(shop) for shop in SHOP_NAMES]
    np.testing.assert_array_equal(
        gpu.state.shop_counts[batch].cpu().numpy(),
        expected_shop_counts,
        err_msg="town.unlocked_shops (multiset)",
    )
    expected_shop_sequence = [SHOP_NAMES.index(shop) for shop in public["town"]["unlocked_shops"]]
    np.testing.assert_array_equal(
        gpu.state.shop_sequence[batch].cpu().numpy(),
        expected_shop_sequence + [-1] * (len(SHOP_NAMES) - len(expected_shop_sequence)),
        err_msg="town.unlocked_shops (order)",
    )
    for player in range(2):
        farm = public["farms"][player]
        private = observations[player]["private"]
        assert int(observations[player]["player"]) == player, f"player {player}: observation.player"
        assert int(observations[player]["step"]) == gpu.step_index, f"player {player}: observation.step"
        assert int(observations[player]["day"]) == gpu.step_index // gpu.config.turns_per_day, f"player {player}: observation.day"
        assert int(observations[player]["hour"]) == gpu.step_index % gpu.config.turns_per_day, f"player {player}: observation.hour"
        np.testing.assert_array_equal(
            gpu.state.seeds[batch, player].cpu().numpy(),
            [private["seeds"][crop] for crop in CROPS],
            err_msg=f"player {player}: private.seeds",
        )
        np.testing.assert_array_equal(
            gpu.state.shed[batch, player].cpu().numpy(),
            [private["shed"][item] for item in ITEMS],
            err_msg=f"player {player}: private.shed",
        )
        expected_tiles = np.asarray([[_official_tile_code(tile) for tile in row] for row in farm["tiles"]])
        np.testing.assert_array_equal(
            gpu.state.tile_type[batch, player].cpu().numpy(),
            expected_tiles,
            err_msg=f"player {player}: farms.tiles[*].kind",
        )
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                prefix = f"player {player}: farms.tiles[{y}][{x}]"
                if tile["kind"] == "PLANT":
                    assert int(gpu.state.planted_day[batch, player, y, x]) == tile["planted_day"], f"{prefix}.planted_day"
                    assert bool(gpu.state.watered[batch, player, y, x]) == tile["watered_today"], f"{prefix}.watered_today"
                    assert int(gpu.state.consecutive[batch, player, y, x]) == tile["consecutive_unwatered"], f"{prefix}.consecutive_unwatered"
                    assert int(gpu.state.yield_units[batch, player, y, x]) == tile["yield_units"], f"{prefix}.yield_units"
                    assert int(gpu.state.max_lifespan_step[batch, player, y, x]) == tile["max_lifespan_step"], f"{prefix}.max_lifespan_step"
                    assert int(gpu.state.fertilized_until_day[batch, player, y, x]) == tile["fertilized_until_day"], f"{prefix}.fertilized_until_day"
                elif "animal" in tile:
                    assert int(gpu.state.placed_day[batch, player, y, x]) == tile["placed_day"], f"{prefix}.placed_day"
                    assert bool(gpu.state.fed[batch, player, y, x]) == tile["fed_today"], f"{prefix}.fed_today"
                    assert bool(gpu.state.cared[batch, player, y, x]) == tile["cared_today"], f"{prefix}.cared_today"
                    assert int(gpu.state.consecutive[batch, player, y, x]) == tile["consecutive_unfed"], f"{prefix}.consecutive_unfed"
                    assert int(gpu.state.yield_units[batch, player, y, x]) == tile["yield_units"], f"{prefix}.yield_units"
                    assert bool(gpu.state.fertilizer_available[batch, player, y, x]) == tile["fertilizer_available"], f"{prefix}.fertilizer_available"
                    assert int(gpu.state.pending_care_bonus[batch, player, y, x]) == tile["pending_care_bonus"], f"{prefix}.pending_care_bonus"
        assert int(gpu.state.hands_count[batch, player]) == len(farm["hands"]), f"player {player}: farms.hands length"
        assert int(gpu.state.hires_today[batch, player]) == farm["hires_today"], f"player {player}: farms.hires_today"
        assert int(gpu.state.unlocked_count[batch, player]) == len(farm["unlocked_quadrants"]), f"player {player}: farms.unlocked_quadrants"
        expected_positions = [farm["farmer"], *farm["hands"]]
        np.testing.assert_array_equal(
            gpu.state.positions[batch, player, : len(expected_positions)].cpu().numpy(),
            expected_positions,
            err_msg=f"player {player}: farmer/hands positions",
        )
        assert len(private["inventories"]) == len(expected_positions), f"player {player}: private.inventories length"
        np.testing.assert_array_equal(
            gpu.state.unit_active[batch, player].cpu().numpy(),
            [True] * len(expected_positions) + [False] * (gpu.max_units - len(expected_positions)),
            err_msg=f"player {player}: active units",
        )
        for unit, inventory in enumerate(private["inventories"]):
            np.testing.assert_array_equal(
                gpu.state.unit_inventory[batch, player, unit].cpu().numpy(),
                [inventory.get(item, 0) for item in ITEMS],
                err_msg=f"player {player}: private.inventories[{unit}]",
            )


def _assert_step_result_matches(gpu_result, official_result, *, batch=0):
    assert gpu_result.step == official_result.step, "step result: step"
    assert gpu_result.done == official_result.done, "step result: done"
    expected_status = "DONE" if gpu_result.done else "ACTIVE"
    assert official_result.statuses == (expected_status, expected_status), "step result: statuses"
    np.testing.assert_array_equal(
        gpu_result.rewards[batch].cpu().numpy(),
        official_result.rewards,
        err_msg="step result: rewards",
    )


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
        "episodeSteps": 720,
        "weedSpawnChance": 0.0,
        "townShopUnlockInterval": 1000,
    }
    tensor_config = GpuEngineConfig(
        episode_steps=720,
        weed_spawn_chance=0.0,
        town_shop_unlock_interval=1000,
        max_market_quantity=8,
    )
    official = FastKaggricultureEnv(configuration=official_config)
    observations = official.reset(seed=17)
    gpu = CudaKaggricultureEnv(1, device="cpu", config=tensor_config, seeds=[17])
    cuda_gpu = CudaKaggricultureEnv(1, device="cuda", config=tensor_config, seeds=[17]) if torch.cuda.is_available() else None
    starter = resolve_agent("starter")
    _assert_public_state_matches(gpu, observations)
    if cuda_gpu:
        _assert_public_state_matches(cuda_gpu, observations)

    while not official.done:
        pair = [starter(observations[player], official.configuration) for player in range(2)]
        official_result = official.step(pair)
        gpu_result = gpu.step(encode_action_dicts([pair], device="cpu", config=tensor_config))
        cuda_result = cuda_gpu.step(encode_action_dicts([pair], device="cuda", config=tensor_config)) if cuda_gpu else None
        observations = official_result.observations

        _assert_public_state_matches(gpu, observations)
        _assert_step_result_matches(gpu_result, official_result)
        if cuda_gpu:
            _assert_public_state_matches(cuda_gpu, observations)
            _assert_step_result_matches(cuda_result, official_result)


def test_gpu_default_random_events_match_official_step_by_step():
    seeds = (0, 42)
    config = GpuEngineConfig()
    officials = [FastKaggricultureEnv() for _ in seeds]
    observations = [official.reset(seed=seed) for official, seed in zip(officials, seeds)]
    cpu = CudaKaggricultureEnv(len(seeds), device="cpu", config=config, seeds=seeds)
    cuda = CudaKaggricultureEnv(len(seeds), device="cuda", config=config, seeds=seeds) if torch.cuda.is_available() else None
    starter = resolve_agent("starter")

    for batch, batch_observations in enumerate(observations):
        _assert_public_state_matches(cpu, batch_observations, batch=batch)
        if cuda:
            _assert_public_state_matches(cuda, batch_observations, batch=batch)

    while not officials[0].done:
        action_pairs = [
            [starter(observations[batch][player], official.configuration) for player in range(2)]
            for batch, official in enumerate(officials)
        ]
        official_results = [official.step(pair) for official, pair in zip(officials, action_pairs)]
        cpu_result = cpu.step(encode_action_dicts(action_pairs, device="cpu", config=config))
        cuda_result = cuda.step(encode_action_dicts(action_pairs, device="cuda", config=config)) if cuda else None

        for batch, official_result in enumerate(official_results):
            _assert_public_state_matches(cpu, official_result.observations, batch=batch)
            _assert_step_result_matches(cpu_result, official_result, batch=batch)
            if cuda:
                _assert_public_state_matches(cuda, official_result.observations, batch=batch)
                _assert_step_result_matches(cuda_result, official_result, batch=batch)
        observations = [result.observations for result in official_results]


def test_gpu_dense_random_event_stream_matches_official_across_seeds():
    seeds = (
        *range(16),
        *range(1_000_000, 1_000_016),
        *range(123_456_789, 123_456_805),
        *range(2**31 - 16, 2**31),
    )
    official_config = {
        "episodeSteps": 12,
        "turnsPerDay": 1,
        "weedSpawnChance": 0.25,
        "townShopUnlockInterval": 1,
    }
    config = GpuEngineConfig(
        episode_steps=12,
        turns_per_day=1,
        weed_spawn_chance=0.25,
        town_shop_unlock_interval=1,
    )
    officials = [FastKaggricultureEnv(configuration=official_config) for _ in seeds]
    observations = [official.reset(seed=seed) for official, seed in zip(officials, seeds)]
    cpu = CudaKaggricultureEnv(len(seeds), device="cpu", config=config, seeds=seeds)
    cuda = CudaKaggricultureEnv(len(seeds), device="cuda", config=config, seeds=seeds) if torch.cuda.is_available() else None
    pass_action = {"farmer": ["PASS"], "hands": [], "market": []}

    while not officials[0].done:
        action_pairs = [[pass_action, pass_action] for _ in seeds]
        official_results = [official.step(pair) for official, pair in zip(officials, action_pairs)]
        cpu_result = cpu.step(encode_action_dicts(action_pairs, device="cpu", config=config))
        cuda_result = cuda.step(encode_action_dicts(action_pairs, device="cuda", config=config)) if cuda else None
        for batch, official_result in enumerate(official_results):
            _assert_public_state_matches(cpu, official_result.observations, batch=batch)
            _assert_step_result_matches(cpu_result, official_result, batch=batch)
            if cuda:
                _assert_public_state_matches(cuda, official_result.observations, batch=batch)
                _assert_step_result_matches(cuda_result, official_result, batch=batch)


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
    _assert_public_state_matches(gpu, observations)

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
        _assert_step_result_matches(gpu_result, official_result)


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
    _assert_public_state_matches(gpu, observations)

    pair = [
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 100]]},
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "GOOSE", 20]]},
    ]
    official_result = official.step(pair)
    gpu_result = gpu.step(encode_action_dicts([pair], device="cpu", config=tensor_config))
    observations = official_result.observations

    _assert_public_state_matches(gpu, observations)
    _assert_step_result_matches(gpu_result, official_result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_fixed_market_matches_cpu_tensor_engine_across_orders():
    config = GpuEngineConfig(
        episode_steps=8,
        starting_money=10_000,
        weed_spawn_chance=0.0,
        town_shop_unlock_interval=1000,
        max_market_orders=4,
        max_market_quantity=100,
        shed_capacity=20,
    )
    cpu = CudaKaggricultureEnv(2, device="cpu", config=config, seeds=[81, 82])
    gpu = CudaKaggricultureEnv(2, device="cuda", config=config, seeds=[81, 82])
    ops = torch.tensor(
        (
            ((M_HIRE, M_HIRE, M_BUY_SEED, M_BUY_ANIMAL), (M_BUY_LAND,) * 4),
            (
                (M_BUY_SEED, M_BUY_ANIMAL, M_HIRE, M_BUY_LAND),
                (M_BUY_ANIMAL, M_BUY_ANIMAL, M_BUY_ANIMAL, M_BUY_SEED),
            ),
        ),
        dtype=torch.int64,
    )
    args = torch.tensor(
        (
            ((0, 0, 4, 0), (0, 0, 0, 0)),
            ((2, 2, 0, 0), (0, 1, 2, 0)),
        ),
        dtype=torch.int64,
    )
    quantities = torch.tensor(
        (
            ((1, 1, 100, 20), (1, 1, 1, 1)),
            ((100, 20, 1, 1), (20, 20, 20, 100)),
        ),
        dtype=torch.int64,
    )

    for step in range(2):
        cpu_actions, gpu_actions = cpu.empty_actions(), gpu.empty_actions()
        cpu_actions.market_ops.copy_(ops)
        cpu_actions.market_args.copy_(args)
        cpu_actions.market_quantities.copy_(quantities)
        gpu_actions.market_ops.copy_(ops.cuda())
        gpu_actions.market_args.copy_(args.cuda())
        gpu_actions.market_quantities.copy_(quantities.cuda())
        cpu.step(cpu_actions)
        gpu.step(gpu_actions)
        for field in fields(type(cpu.state)):
            expected = getattr(cpu.state, field.name)
            actual = getattr(gpu.state, field.name).cpu()
            assert torch.equal(actual, expected), f"fixed market step {step}: mismatch in {field.name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_fused_route_moves_and_scans_wide_market_tensor():
    config = GpuEngineConfig(
        episode_steps=8,
        starting_money=10_000,
        weed_spawn_chance=0.0,
        town_shop_unlock_interval=1000,
        max_market_orders=20,
        max_market_quantity=16,
    )
    cpu = CudaKaggricultureEnv(2, device="cpu", config=config, seeds=[83, 84])
    gpu = CudaKaggricultureEnv(2, device="cuda", config=config, seeds=[83, 84])
    cpu_actions, gpu_actions = cpu.empty_actions(), gpu.empty_actions()
    cpu_actions.unit_ops[:, 0, 0] = U_EAST
    gpu_actions.unit_ops[:, 0, 0] = U_EAST
    cpu_actions.unit_ops[:, 1, 0] = U_NORTH
    gpu_actions.unit_ops[:, 1, 0] = U_NORTH
    cpu_actions.market_ops[:, :, -1] = M_BUY_SEED
    gpu_actions.market_ops[:, :, -1] = M_BUY_SEED
    cpu_actions.market_args[:, :, -1] = 4
    gpu_actions.market_args[:, :, -1] = 4
    cpu_actions.market_quantities[:, :, -1] = 3
    gpu_actions.market_quantities[:, :, -1] = 3

    cpu.step(cpu_actions)
    gpu.step(gpu_actions)
    for field in fields(type(cpu.state)):
        expected = getattr(cpu.state, field.name)
        actual = getattr(gpu.state, field.name).cpu()
        assert torch.equal(actual, expected), f"fused route mismatch in {field.name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_single_kernel_market_matches_cpu_across_fixed_and_dynamic_orders():
    config = GpuEngineConfig(
        episode_steps=8,
        starting_money=10_000,
        weed_spawn_chance=0.0,
        town_shop_unlock_interval=1000,
        max_market_orders=2,
        max_market_quantity=16,
        shed_capacity=100,
        fuse_market_kernel=True,
    )
    cpu = CudaKaggricultureEnv(2, device="cpu", config=config, seeds=[91, 92])
    gpu = CudaKaggricultureEnv(2, device="cuda", config=config, seeds=[91, 92])
    cpu.state.shed[:, :, 0] = 8
    gpu.state.shed[:, :, 0] = 8

    steps = (
        (
            (((M_HIRE, 0, 1), (M_BUY_SEED, 4, 16)), ((M_BUY_LAND, 0, 1), (M_BUY_ANIMAL, 0, 2))),
            (((M_BUY_SEED, 2, 7), (M_BUY_ANIMAL, 1, 3)), ((M_HIRE, 0, 1), (M_BUY_LAND, 0, 1))),
        ),
        (
            (((M_HIRE, 0, 1), (M_HIRE, 0, 1)), ((M_BUY_LAND, 0, 1), (M_BUY_LAND, 0, 1))),
            (((M_BUY_SEED, 1, 3), (M_BUY_SEED, 1, 4)), ((M_BUY_ANIMAL, 2, 2), (M_BUY_ANIMAL, 2, 3))),
        ),
        (
            (((M_SELL, 0, 4), (M_BUY_PRODUCT, 8, 3)), ((M_BUY_PRODUCT, 0, 5), (M_SELL, 0, 2))),
            (((M_BUY_PRODUCT, 0, 6), (M_SELL, 0, 3)), ((M_SELL, 0, 5), (M_BUY_PRODUCT, 8, 4))),
        ),
    )

    for step_index, batch_orders in enumerate(steps):
        cpu_actions, gpu_actions = cpu.empty_actions(), gpu.empty_actions()
        for batch, players in enumerate(batch_orders):
            for player, orders in enumerate(players):
                for order, (op, arg, quantity) in enumerate(orders):
                    cpu_actions.market_ops[batch, player, order] = op
                    cpu_actions.market_args[batch, player, order] = arg
                    cpu_actions.market_quantities[batch, player, order] = quantity
                    gpu_actions.market_ops[batch, player, order] = op
                    gpu_actions.market_args[batch, player, order] = arg
                    gpu_actions.market_quantities[batch, player, order] = quantity
        cpu.step(cpu_actions)
        gpu.step(gpu_actions)
        for field in fields(type(cpu.state)):
            expected = getattr(cpu.state, field.name)
            actual = getattr(gpu.state, field.name).cpu()
            assert torch.equal(actual, expected), f"single market kernel step {step_index}: mismatch in {field.name}"


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_dynamic_market_matches_cpu_tensor_engine():
    config = GpuEngineConfig(
        episode_steps=48,
        starting_money=10_000,
        weed_spawn_chance=0.0,
        town_shop_unlock_interval=1000,
        max_market_orders=2,
        max_market_quantity=16,
    )
    cpu = CudaKaggricultureEnv(2, device="cpu", config=config, seeds=[71, 72])
    gpu = CudaKaggricultureEnv(2, device="cuda", config=config, seeds=[71, 72])
    cpu.state.shed[:, :, 0] = 24
    gpu.state.shed[:, :, 0] = 24
    cpu.state.shed[:, :, PRODUCT_INDEX["FERTILIZER"]] = 16
    gpu.state.shed[:, :, PRODUCT_INDEX["FERTILIZER"]] = 16

    def compare(label):
        for field in fields(type(cpu.state)):
            expected = getattr(cpu.state, field.name)
            actual = getattr(gpu.state, field.name).cpu()
            assert torch.equal(actual, expected), f"{label}: mismatch in {field.name}"

    for step, product in enumerate((0, PRODUCT_INDEX["FERTILIZER"], 0)):
        cpu_actions, gpu_actions = cpu.empty_actions(), gpu.empty_actions()
        cpu_actions.market_ops[:, 0, 0] = M_SELL
        gpu_actions.market_ops[:, 0, 0] = M_SELL
        cpu_actions.market_ops[:, 1, 0] = M_BUY_PRODUCT
        gpu_actions.market_ops[:, 1, 0] = M_BUY_PRODUCT
        cpu_actions.market_args[:, :, 0] = product
        gpu_actions.market_args[:, :, 0] = product
        cpu_actions.market_quantities[:, :, 0] = 16
        gpu_actions.market_quantities[:, :, 0] = 16
        cpu.step(cpu_actions)
        gpu.step(gpu_actions)
        compare(f"dynamic market step {step}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_end_of_day_matches_cpu_with_weeds_unlocks_and_overflow():
    config = GpuEngineConfig(
        episode_steps=16,
        turns_per_day=4,
        max_hands=2,
        shed_capacity=5,
        weed_spawn_chance=0.25,
        town_shop_unlock_interval=1,
    )
    cpu = CudaKaggricultureEnv(2, device="cpu", config=config, seeds=[81, 82])
    gpu = CudaKaggricultureEnv(2, device="cuda", config=config, seeds=[81, 82])

    # Producing tomato, dying carrot, producing/cared goose, and escaping cow.
    cpu.state.tile_type[:, :, 0, 0] = PLANT_BASE + 2
    cpu.state.planted_day[:, :, 0, 0] = -7
    cpu.state.watered[:, :, 0, 0] = True
    cpu.state.fertilized_until_day[:, :, 0, 0] = 0
    cpu.state.tile_type[:, :, 0, 1] = PLANT_BASE + 1
    cpu.state.consecutive[:, :, 0, 1] = 1
    cpu.state.tile_type[:, :, 0, 2] = ANIMAL_BASE
    cpu.state.placed_day[:, :, 0, 2] = -3
    cpu.state.fed[:, :, 0, 2] = True
    cpu.state.cared[:, :, 0, 2] = True
    cpu.state.pending_care_bonus[:, :, 0, 2] = 2
    cpu.state.tile_type[:, :, 0, 3] = ANIMAL_BASE + 1
    cpu.state.consecutive[:, :, 0, 3] = 1

    # Item order must fill the remaining three shed slots with wheat first.
    cpu.state.shed[:, :, 2] = 2
    cpu.state.unit_inventory[:, :, :, 0] = 2
    cpu.state.unit_inventory[:, :, :, 1] = 2
    cpu.state.unit_active.fill_(True)
    cpu.state.hands_count.fill_(2)
    cpu.state.positions.fill_(3)
    for field in fields(type(cpu.state)):
        getattr(gpu.state, field.name).copy_(getattr(cpu.state, field.name).cuda())

    cpu.step_index = gpu.step_index = config.turns_per_day - 1
    cpu.step(cpu.empty_actions())
    gpu.step(gpu.empty_actions())
    for field in fields(type(cpu.state)):
        expected = getattr(cpu.state, field.name)
        actual = getattr(gpu.state, field.name).cpu()
        assert torch.equal(actual, expected), f"end-of-day mismatch in {field.name}"

# Licensed under the Apache License, Version 2.0.
"""Step-for-step differential checks against kaggle-environments 1.32.7."""
import json
import random

import pytest

from fast_kaggriculture import Config, FastEnv

kaggle_environments = pytest.importorskip("kaggle_environments")


def canonical(value):
    return json.loads(json.dumps(value))


def official_observation(state, player):
    obs = dict(state[player].observation)
    obs.pop("remainingOverageTime", None)
    return canonical(obs)


def assert_same(state, fast, step):
    for player in range(2):
        expected = official_observation(state, player)
        actual = canonical(fast.observation(player))
        assert actual == expected, f"state mismatch at step={step}, player={player}"


def run_trace(seed, actions):
    official = kaggle_environments.make(
        "kaggriculture", configuration={"seed": seed}, debug=True
    )
    state = official.reset(2)
    fast = FastEnv(Config(), seed)
    assert_same(state, fast, 0)
    for index, joint_action in enumerate(actions, 1):
        state = official.step(joint_action)
        fast.step(joint_action)
        assert_same(state, fast, index)
    return state, fast


PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def test_pass_trace_covers_town_day_rng_and_terminal():
    state, fast = run_trace(90210, [[PASS, PASS] for _ in range(719)])
    assert fast.done
    assert tuple(fast.rewards) == tuple(float(s.reward) for s in state)


def test_market_lockstep_dynamic_prices_town_and_atomic_orders():
    actions = [
        [
            {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 15], ["HIRE"], ["BUY_LAND"], ["BUY_PRODUCT", "WHEAT", 8]]},
            {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "CARROT", 10], ["HIRE"], ["BUY_PRODUCT", "WHEAT", 6]]},
        ],
        [
            {"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]], "market": [["SELL", "WHEAT", 4], ["BUY_ANIMAL", "GOOSE", 1]]},
            {"farmer": ["PLANT", "CARROT"], "hands": [["PLANT", "CARROT"]], "market": [["SELL", "WHEAT", 4], ["BUY_ANIMAL", "COW", 1]]},
        ],
    ]
    actions.extend([[PASS, PASS] for _ in range(50)])
    run_trace(71, actions)


def test_same_turn_place_inventory_is_visible_to_market_sell():
    """Unit PLACE commits before market slots in both official and fast envs."""
    actions = [
        [
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_PRODUCT", "FERTILIZER", 3]],
            },
            PASS,
        ],
        [
            {
                "farmer": ["PICKUP", "FERTILIZER", 3],
                "hands": [],
                "market": [],
            },
            PASS,
        ],
        [
            {
                "farmer": ["PLACE", "FERTILIZER", 3],
                "hands": [],
                "market": [["SELL", "FERTILIZER", 2]],
            },
            PASS,
        ],
    ]
    state, fast = run_trace(20260826, actions)
    assert fast.last_market_fills[0] == [2]
    assert official_observation(state, 0)["private"]["shed"]["FERTILIZER"] == 1


def test_end_of_day_full_shed_uses_inventory_dict_insertion_order():
    """Overflow keeps the same item Python's insertion-ordered dict keeps."""
    official = kaggle_environments.make(
        "kaggriculture",
        configuration={"seed": 31337, "shedCapacity": 3},
        debug=True,
    )
    state = official.reset(2)
    config = Config()
    config.shed_capacity = 3
    fast = FastEnv(config, 31337)
    actions = [[dict(PASS), dict(PASS)] for _ in range(24)]
    actions[0][0] = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["BUY_PRODUCT", "FERTILIZER", 1], ["BUY_PRODUCT", "WHEAT", 2]],
    }
    actions[1][0] = {
        "farmer": ["PICKUP", "FERTILIZER", 1], "hands": [], "market": []
    }
    actions[2][0] = {
        "farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []
    }
    actions[3][0] = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["BUY_PRODUCT", "WHEAT", 1]],
    }
    for index, joint_action in enumerate(actions, 1):
        state = official.step(joint_action)
        fast.step(joint_action)
        assert_same(state, fast, index)
    shed = canonical(fast.observation(0))["private"]["shed"]
    assert shed["WHEAT"] == 2
    assert shed["FERTILIZER"] == 1
    assert fast.last_end_of_day_overflow == (1, 0)


def test_crop_and_animal_care_production_escape_and_decay():
    trace = [[dict(PASS), dict(PASS)] for _ in range(240)]
    # Player 0: animal care and production across multiple daily refreshes.
    trace[0][0] = {"farmer": ["BUILD_COOP"], "hands": [], "market": [["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 12], ["HIRE"]]}
    trace[1][0] = {"farmer": ["PICKUP", "GOOSE", 1], "hands": [["PICKUP", "WHEAT", 12]], "market": []}
    trace[2][0] = {"farmer": ["PLACE", "GOOSE"], "hands": [["FEED"]], "market": []}
    trace[3][0] = {"farmer": ["CARE"], "hands": [["PASS"]], "market": []}
    for start in range(24, 216, 24):
        trace[start][0] = {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}
        trace[start + 1][0] = {"farmer": ["FEED"], "hands": [], "market": []}
        trace[start + 2][0] = {"farmer": ["CARE"], "hands": [], "market": []}
    # Player 1: ongoing crop watering and production.
    trace[0][1] = {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "TOMATO", 1], ["HIRE"]]}
    trace[1][1] = {"farmer": ["PLANT", "TOMATO"], "hands": [["WATER"]], "market": []}
    for start in range(24, 240, 24):
        trace[start][1] = {"farmer": ["WATER"], "hands": [], "market": []}
    run_trace(411, trace)


def _random_trace(seed, steps=240):
    rng = random.Random(seed)
    unit_ops = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"]
    market_ops = ["HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"]
    crops = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
    products = crops + ["EGG", "MILK", "WOOL", "FERTILIZER"]
    animals = ["GOOSE", "COW", "SHEEP"]

    def unit():
        op = rng.choice(unit_ops)
        if op == "PLANT": return [op, rng.choice(crops)]
        if op in ("PICKUP", "PLACE"): return [op, rng.choice(products + animals), rng.randint(1, 5)]
        return [op]

    def order():
        op = rng.choice(market_ops)
        if op in ("HIRE", "BUY_LAND"): return [op]
        choices = crops if op == "BUY_SEED" else (["WHEAT", "FERTILIZER"] if op == "BUY_PRODUCT" else (animals if op == "BUY_ANIMAL" else products))
        return [op, rng.choice(choices), rng.randint(1, 8)]

    trace = []
    for _ in range(steps):
        joint = []
        for _player in range(2):
            joint.append({"farmer": unit(), "hands": [unit() for _ in range(rng.randrange(4))], "market": [order() for _ in range(rng.randrange(4))]})
        trace.append(joint)
    return trace


@pytest.mark.parametrize("seed", [0, 1, 17, 123456789])
def test_seeded_mixed_action_fuzz(seed):
    run_trace(seed, _random_trace(seed))


def test_packed_single_and_segment_batch_match_scalar_core():
    np = pytest.importorskip("numpy")
    from fast_kaggriculture import FastBatchEnv, Item, Op

    scalar = FastEnv(Config(), 3)
    batch = FastBatchEnv(2, Config(), 3)
    units = np.zeros((2, 2, 1, 3), dtype=np.int32)
    units[..., 0] = int(Op.PASS)
    units[..., 1] = int(Item.NONE)
    uc = np.ones((2, 2), dtype=np.int32)
    market = np.zeros((2, 2, 1, 3), dtype=np.int32)
    mc = np.zeros((2, 2), dtype=np.int32)
    pass_joint = [PASS, PASS]
    for _ in range(31):
        scalar.step_raw(pass_joint)
        batch.step_packed(units, uc, market, mc)
    assert canonical(batch.observation(0, 0)) == canonical(scalar.observation(0))

    # Segment execution creates one OpenMP region for many consecutive turns.
    batch.reset([3, 4])
    turns = 31
    su = np.broadcast_to(units[:, None], (2, turns, 2, 1, 3)).copy()
    suc = np.broadcast_to(uc[:, None], (2, turns, 2)).copy()
    sm = np.broadcast_to(market[:, None], (2, turns, 2, 1, 3)).copy()
    smc = np.broadcast_to(mc[:, None], (2, turns, 2)).copy()
    batch.run_packed_segment(su, suc, sm, smc)
    assert canonical(batch.observation(0, 0)) == canonical(scalar.observation(0))

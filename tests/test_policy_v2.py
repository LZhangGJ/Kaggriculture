from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.gpu_policy import MAX_UNITS, encode_batch
from kaggriculture_lab.policy_v2 import (
    MAX_MARKET_ORDERS,
    StructuredKaggriculturePolicy,
    decode_structured_actions,
    policy_from_checkpoint,
    structured_action_targets,
    structured_policy_batch,
)


def test_structured_codec_preserves_multiple_orders_and_quantities():
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=8))
    actions = [
        {
            "farmer": ["PICKUP", "WHEAT", 7],
            "hands": [],
            "market": [
                ["SELL", "WHEAT", 23],
                ["BUY_SEED", "STRAWBERRY", 4],
                ["HIRE"],
            ],
        },
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]
    targets = structured_action_targets(observations, actions)
    decoded = decode_structured_actions(
        targets["unit_targets"],
        targets["unit_quantity_targets"],
        targets["market_targets"],
        targets["market_quantity_targets"],
        observations,
    )
    assert decoded == actions
    assert targets["market_targets"].shape == (2, MAX_MARKET_ORDERS)
    assert targets["unit_targets"].shape == (2, MAX_UNITS)


def test_structured_policy_shapes_and_environment_step():
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=9))
    model = StructuredKaggriculturePolicy(hidden_size=64)
    batch = structured_policy_batch(model, observations, "cpu", deterministic=True)
    assert batch.unit_indices.shape == (2, MAX_UNITS)
    assert batch.market_indices.shape == (2, MAX_MARKET_ORDERS)
    assert len(batch.actions) == 2
    result = env.step(batch.actions)
    assert result.step == 1


def test_structured_quantity_is_clipped_to_codec_limit():
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observation = list(env.reset(seed=10))[0]
    action = {
        "farmer": ["PICKUP", "WHEAT", 999],
        "hands": [],
        "market": [["SELL", "WHEAT", 999]],
    }
    targets = structured_action_targets([observation], [action])
    assert int(targets["unit_quantity_targets"][0, 0]) == 100
    assert int(targets["market_quantity_targets"][0, 0]) == 100


def test_route_prior_is_zero_initialized_and_checkpoint_loads():
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=11))
    plain = StructuredKaggriculturePolicy(hidden_size=64)
    routed = StructuredKaggriculturePolicy(hidden_size=64, route_prior=True)
    routed.load_state_dict(plain.state_dict(), strict=False)
    features, unit_context, _ = encode_batch(observations)
    plain_outputs = plain(
        torch.as_tensor(features), torch.as_tensor(unit_context)
    )
    routed_outputs = routed(
        torch.as_tensor(features), torch.as_tensor(unit_context)
    )
    for plain_output, routed_output in zip(plain_outputs, routed_outputs, strict=True):
        assert torch.equal(plain_output, routed_output)

    loaded = policy_from_checkpoint(
        {
            "hidden_size": 64,
            "route_prior": True,
            "model": routed.state_dict(),
        },
        "cpu",
    )
    assert loaded.route_prior

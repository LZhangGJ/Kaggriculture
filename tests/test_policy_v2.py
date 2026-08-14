from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.gpu_policy import FARM_FEATURES, MAX_UNITS, encode_batch
from kaggriculture_lab.policy_v2 import (
    FARM_FEATURE_OFFSET,
    MAX_MARKET_ORDERS,
    MARKET_TOKEN_INDEX,
    StructuredKaggriculturePolicy,
    canonicalize_seat_features,
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


def test_canonical_seat_features_swap_public_farms_and_clear_seat():
    feature0 = torch.arange(encode_batch(list(FastKaggricultureEnv().reset(seed=12)))[0].shape[1], dtype=torch.float32).unsqueeze(0)
    feature0[:, 3] = 0.0
    feature1 = feature0.clone()
    feature1[:, 3] = 1.0
    first = slice(FARM_FEATURE_OFFSET, FARM_FEATURE_OFFSET + FARM_FEATURES)
    second = slice(FARM_FEATURE_OFFSET + FARM_FEATURES, FARM_FEATURE_OFFSET + 2 * FARM_FEATURES)
    feature1[:, first] = feature0[:, second]
    feature1[:, second] = feature0[:, first]

    canonical0 = canonicalize_seat_features(feature0)
    canonical1 = canonicalize_seat_features(feature1)
    assert torch.equal(canonical0, canonical1)
    assert canonical0[0, 3] == 0.0


def test_canonical_seat_policy_is_invariant_to_absolute_seat():
    observations = list(FastKaggricultureEnv().reset(seed=13))
    features, unit_context, _ = encode_batch(observations)
    paired = torch.as_tensor(features[:1]).repeat(2, 1)
    paired[1, 3] = 1.0
    first = slice(FARM_FEATURE_OFFSET, FARM_FEATURE_OFFSET + FARM_FEATURES)
    second = slice(FARM_FEATURE_OFFSET + FARM_FEATURES, FARM_FEATURE_OFFSET + 2 * FARM_FEATURES)
    paired[1, first] = paired[0, second].clone()
    paired[1, second] = paired[0, first].clone()
    contexts = torch.as_tensor(unit_context[:1]).repeat(2, 1, 1)
    model = StructuredKaggriculturePolicy(hidden_size=64, canonical_seat=True).eval()

    outputs = model(paired, contexts)
    for output in outputs:
        torch.testing.assert_close(output[0], output[1], rtol=1e-5, atol=1e-6)

    loaded = policy_from_checkpoint(
        {
            "hidden_size": 64,
            "canonical_seat": True,
            "model": model.state_dict(),
        },
        "cpu",
    )
    assert loaded.canonical_seat


def test_autoregressive_market_teacher_forcing_is_causal_and_loads():
    observations = list(FastKaggricultureEnv().reset(seed=14))
    features, unit_context, _ = encode_batch(observations)
    model = StructuredKaggriculturePolicy(
        hidden_size=64, canonical_seat=True, autoregressive_market=True
    ).eval()
    teacher_tokens = torch.zeros((2, MAX_MARKET_ORDERS), dtype=torch.long)
    teacher_quantities = torch.zeros((2, MAX_MARKET_ORDERS), dtype=torch.long)
    changed_tokens = teacher_tokens.clone()
    changed_tokens[:, 0] = MARKET_TOKEN_INDEX[("HIRE", None)]

    outputs0 = model(
        torch.as_tensor(features),
        torch.as_tensor(unit_context),
        teacher_tokens,
        teacher_quantities,
    )
    outputs1 = model(
        torch.as_tensor(features),
        torch.as_tensor(unit_context),
        changed_tokens,
        teacher_quantities,
    )
    torch.testing.assert_close(outputs0[2][:, 0], outputs1[2][:, 0])
    assert not torch.equal(outputs0[2][:, 1], outputs1[2][:, 1])

    torch.manual_seed(15)
    sampled = model(
        torch.as_tensor(features),
        torch.as_tensor(unit_context),
        sample_market=True,
        return_market_choices=True,
    )
    assert sampled[5].shape == (2, MAX_MARKET_ORDERS)
    assert sampled[6].shape == (2, MAX_MARKET_ORDERS)
    replayed = model(
        torch.as_tensor(features),
        torch.as_tensor(unit_context),
        sampled[5],
        sampled[6],
    )
    torch.testing.assert_close(sampled[2], replayed[2])
    torch.testing.assert_close(sampled[3], replayed[3])

    loaded = policy_from_checkpoint(
        {
            "hidden_size": 64,
            "canonical_seat": True,
            "autoregressive_market": True,
            "model": model.state_dict(),
        },
        "cpu",
    )
    assert loaded.autoregressive_market

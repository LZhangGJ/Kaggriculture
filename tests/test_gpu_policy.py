from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.gpu_policy import (
    FEATURE_DIM,
    ITEMS,
    MAX_UNITS,
    UNIT_CONTEXT_BASE_DIM,
    UNIT_CONTEXT_INVENTORY_DIM,
    KaggriculturePolicy,
    action_masks,
    decode_actions,
    encode_batch,
    encode_observation,
    policy_batch,
)


def test_encoder_masks_and_policy_produce_valid_shapes():
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=3))
    features, unit_context, active = encode_batch(observations)
    unit_masks, market_masks = action_masks(observations)
    assert features.shape == (2, FEATURE_DIM)
    assert unit_context.shape == (2, MAX_UNITS, 3)
    assert active[:, 0].all()
    assert unit_masks[:, 0].any(axis=1).all()
    assert market_masks.any(axis=1).all()

    model = KaggriculturePolicy(hidden_size=64)
    batch = policy_batch(model, observations, "cpu", deterministic=True)
    result = env.step(batch.actions)
    assert result.step == 1

    unmasked = policy_batch(model, observations, "cpu", deterministic=True, mask_actions=False)
    assert len(unmasked.actions) == 2


def test_decode_respects_number_of_hands():
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=4))
    unit_indices = np.zeros((2, MAX_UNITS), dtype=np.int64)
    market_indices = np.zeros(2, dtype=np.int64)
    actions = decode_actions(unit_indices, market_indices, observations)
    assert actions == [
        {"farmer": ["PASS"], "hands": [], "market": []},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]


def test_encoder_can_preserve_inventory_for_each_unit():
    observation = list(FastKaggricultureEnv().reset(seed=5))[0]
    observation.farms[0].hands.append([1, 1])
    observation.private.inventories = [{"WHEAT": 3}, {"WOOL": 7}]

    _, legacy_context, _ = encode_observation(observation)
    _, inventory_context, active = encode_observation(
        observation,
        include_unit_inventory=True,
    )

    assert legacy_context.shape == (MAX_UNITS, UNIT_CONTEXT_BASE_DIM)
    assert inventory_context.shape == (MAX_UNITS, UNIT_CONTEXT_INVENTORY_DIM)
    assert active[:2].all()
    assert inventory_context[0, UNIT_CONTEXT_BASE_DIM + ITEMS.index("WHEAT")] == pytest.approx(
        0.03
    )
    assert inventory_context[1, UNIT_CONTEXT_BASE_DIM + ITEMS.index("WOOL")] == pytest.approx(
        0.07
    )
    assert inventory_context[0, UNIT_CONTEXT_BASE_DIM + ITEMS.index("WOOL")] == 0.0

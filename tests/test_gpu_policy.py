from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.gpu_policy import (
    FEATURE_DIM,
    MAX_UNITS,
    KaggriculturePolicy,
    action_masks,
    decode_actions,
    encode_batch,
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

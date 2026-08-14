from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.gpu_policy import FEATURE_DIM, MAX_UNITS, UNIT_INDEX
from kaggriculture_lab.replay_bc import encode_replay


def test_encode_replay_applies_action_shift_and_value_targets():
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = copy.deepcopy(env.reset(seed=7))
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    replay = {
        "rewards": [10.0, 5.0],
        "steps": [
            [
                {"status": "ACTIVE", "observation": observations[0], "action": action},
                {"status": "ACTIVE", "observation": observations[1], "action": action},
            ],
            [
                {"status": "ACTIVE", "observation": observations[0], "action": action},
                {"status": "ACTIVE", "observation": observations[1], "action": action},
            ],
        ],
    }

    arrays, stats = encode_replay(replay)

    assert arrays["features"].shape == (2, FEATURE_DIM)
    assert arrays["unit_context"].shape == (2, MAX_UNITS, 3)
    assert arrays["unit_targets"][:, 0].tolist() == [UNIT_INDEX[("PASS", None)]] * 2
    assert arrays["value_targets"].tolist() == [1.0, -1.0]
    assert stats["examples"] == 2
    assert stats["unit_mask_invalid"] == 0
    assert stats["market_mask_invalid"] == 0

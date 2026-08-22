from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.board_policy import (
    BOARD_CHANNELS,
    BOARD_SIZE,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
    BoardKaggriculturePolicy,
    board_policy_batch,
    encode_board_batch,
)
from kaggriculture_lab.gpu_policy import MAX_UNITS, action_masks


def test_board_encoder_is_canonical_and_actor_visible() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=7))
    boards, global_features, units, active = encode_board_batch(observations)
    assert boards.shape == (2, 2, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert global_features.shape == (2, GLOBAL_FEATURES)
    assert units.shape == (2, 2, MAX_UNITS, UNIT_FEATURES)
    assert active.shape == (2, 2, MAX_UNITS)
    assert np.isfinite(boards).all()
    assert np.isfinite(global_features).all()
    np.testing.assert_allclose(boards[0, 0], boards[1, 1])
    np.testing.assert_allclose(boards[0, 1], boards[1, 0])
    assert units[:, 0, 0, -1].tolist() == [1.0, 1.0]
    assert units[:, 1, 0, -1].tolist() == [0.0, 0.0]


def test_board_policy_shapes_masks_and_environment_step() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=8))
    boards, global_features, units, active = encode_board_batch(observations)
    model = BoardKaggriculturePolicy(hidden_size=64, board_width=16)
    unit_logits, market_logits, values = model(
        torch.as_tensor(boards),
        torch.as_tensor(global_features),
        torch.as_tensor(units),
        torch.as_tensor(active),
    )
    unit_masks, market_masks = action_masks(observations)
    assert unit_logits.shape[:2] == (2, MAX_UNITS)
    assert unit_logits.shape == torch.as_tensor(unit_masks).shape
    assert market_logits.shape == torch.as_tensor(market_masks).shape
    assert values.shape == (2,)
    loss = unit_logits[:, 0].square().mean() + market_logits.square().mean() + values.square().mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    batch = board_policy_batch(model.eval(), observations, "cpu", deterministic=True)
    result = env.step(batch.actions)
    assert result.step == 1

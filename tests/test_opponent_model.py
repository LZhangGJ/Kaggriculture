from __future__ import annotations

import numpy as np

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.hierarchical_schema import HierarchicalMemory
from kaggriculture_lab.opponent_model import (
    OPPONENT_FEATURES,
    OPPONENT_HISTORY_LAGS,
    build_opponent_history_sequences,
    opponent_public_features,
    runtime_opponent_history,
)


def test_opponent_history_is_observable_finite_and_lag_aligned() -> None:
    environment = FastKaggricultureEnv(
        configuration={"episodeSteps": 4}, copy_observations=True
    )
    observations = [environment.reset(seed=801)[0]]
    passive = {"farmer": ["PASS"], "hands": [], "market": []}
    while not environment.done:
        result = environment.step([passive, passive])
        if not environment.done:
            observations.append(result.observations[0])
    histories, mask = build_opponent_history_sequences(observations)

    assert histories.shape == (
        len(observations),
        len(OPPONENT_HISTORY_LAGS),
        OPPONENT_FEATURES,
    )
    assert np.isfinite(histories).all()
    assert mask[:, -1].all()
    np.testing.assert_allclose(histories[0, -1], opponent_public_features(observations[0]))


def test_runtime_history_does_not_duplicate_the_same_step() -> None:
    observation = FastKaggricultureEnv(configuration={"episodeSteps": 4}).reset(
        seed=802
    )[0]
    memory = HierarchicalMemory()
    first, first_mask = runtime_opponent_history(memory, observation)
    second, second_mask = runtime_opponent_history(memory, observation)
    assert len(memory.opponent_observation_history) == 1
    np.testing.assert_allclose(first, second)
    np.testing.assert_array_equal(first_mask, second_mask)

from __future__ import annotations

import copy
import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

from kaggriculture_lab import FastKaggricultureEnv, VectorFastEnv, run_fast_episode


def official_result(agent_a: str, agent_b: str, seed: int):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        from kaggle_environments import make

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": seed},
        debug=False,
    )
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    return (
        tuple(state.reward for state in final),
        tuple(state.status for state in final),
        copy.deepcopy(final[0].observation.farms),
        copy.deepcopy(final[0].observation.market),
        copy.deepcopy(final[0].observation.town),
        tuple(copy.deepcopy(state.observation.private) for state in final),
    )


def fast_result(agent_a: str, agent_b: str, seed: int):
    env = FastKaggricultureEnv()
    result = env.run((agent_a, agent_b), seed=seed)
    observations = env.observations()
    return (
        result.rewards,
        result.statuses,
        copy.deepcopy(observations[0].farms),
        copy.deepcopy(observations[0].market),
        copy.deepcopy(observations[0].town),
        tuple(copy.deepcopy(observation.private) for observation in observations),
    )


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("agents", [("pass", "starter"), ("starter", "starter")])
def test_fast_runner_matches_official_final_state(seed, agents):
    assert fast_result(*agents, seed) == official_result(*agents, seed)


def test_episode_length_and_terminal_state():
    result = run_fast_episode("starter", "pass", seed=7)
    assert result.steps == 719
    assert result.statuses == ("DONE", "DONE")
    assert all(reward is not None for reward in result.rewards)


def test_vector_environment_steps_in_lockstep():
    vector = VectorFastEnv(3, configuration={"episodeSteps": 4})
    observations = vector.reset([10, 11, 12])
    assert [pair[0].step for pair in observations] == [0, 0, 0]

    pass_action = {"farmer": ["PASS"], "hands": [], "market": []}
    results = vector.step([[pass_action, pass_action] for _ in range(3)])
    assert [result.step for result in results] == [1, 1, 1]
    assert not any(result.done for result in results)


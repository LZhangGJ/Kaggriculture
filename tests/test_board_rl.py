from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.board_policy import BoardKaggriculturePolicy
from kaggriculture_lab.board_rl import (
    actor_critic_batch,
    generalized_advantage_estimates,
    liquidatable_net_asset,
    multi_horizon_transition_targets,
    paired_liquidatable_net_asset_potentials,
    potential_shaped_value_targets,
    productive_resource_value,
    relative_money_potential,
)


def test_actor_critic_batch_is_legal_and_zero_sum_potential() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=11))
    assert relative_money_potential(observations[0]) == pytest.approx(
        -relative_money_potential(observations[1])
    )

    model = BoardKaggriculturePolicy(hidden_size=64, board_width=16)
    batch = actor_critic_batch(model, observations, "cpu")
    assert batch.log_probs.shape == (2,)
    assert batch.entropies.shape == (2,)
    assert batch.values.shape == (2,)
    assert torch.isfinite(batch.log_probs).all()
    assert torch.isfinite(batch.entropies).all()
    result = env.step(batch.actions)
    assert result.step == 1


def test_generalized_advantage_estimates_respect_terminal_steps() -> None:
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dones = torch.tensor([[False, False], [True, True]])
    values = torch.zeros_like(rewards)
    advantages, returns = generalized_advantage_estimates(
        rewards,
        dones,
        values,
        torch.zeros(2),
        gamma=1.0,
        gae_lambda=1.0,
    )
    expected = torch.tensor([[4.0, 6.0], [3.0, 4.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_liquidatable_net_asset_marks_products_and_pairs_antisymmetrically() -> None:
    products = (
        "WHEAT",
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
        "MELON",
        "EGG",
        "MILK",
        "WOOL",
        "FERTILIZER",
    )
    farms = [
        {"money": 1000, "tiles": [[None]]},
        {"money": 1000, "tiles": [[None]]},
    ]
    market = {"inventory": {item: 10_000 for item in products}}
    left = {
        "step": 100,
        "player": 0,
        "farms": farms,
        "private": {
            "shed": {"WHEAT": 4},
            "inventories": [{}],
            "seeds": {},
        },
        "market": market,
    }
    right = {
        "step": 100,
        "player": 1,
        "farms": farms,
        "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        "market": market,
    }
    assert liquidatable_net_asset(left) > liquidatable_net_asset(right)
    potentials = paired_liquidatable_net_asset_potentials([left, right])
    assert potentials[0] > 0
    assert potentials[0] == pytest.approx(-potentials[1])


def test_potential_shaped_value_targets_are_per_step_and_zero_sum() -> None:
    left = potential_shaped_value_targets(
        [0.0, 0.1, 0.4, 0.3],
        gamma=1.0,
        reward_scale=10.0,
        terminal_outcome=1.0,
        win_bonus=2.0,
    )
    right = potential_shaped_value_targets(
        [0.0, -0.1, -0.4, -0.3],
        gamma=1.0,
        reward_scale=10.0,
        terminal_outcome=-1.0,
        win_bonus=2.0,
    )
    assert left.tolist() == pytest.approx([5.0, 4.0, 1.0])
    assert left.tolist() != pytest.approx([left[-1]] * len(left))
    assert left == pytest.approx(-right)


def test_resource_growth_auxiliary_targets_do_not_cross_episode_resets() -> None:
    before = torch.tensor([[0.0], [1.0], [10.0], [11.0]])
    after = torch.tensor([[1.0], [2.0], [11.0], [12.0]])
    dones = torch.tensor([[False], [True], [False], [False]])
    targets, mask = multi_horizon_transition_targets(
        before, after, dones, horizons=(1, 3)
    )
    assert targets.shape == (4, 1, 2)
    assert mask[:, 0, 0].all()
    assert not mask[0, 0, 1]
    assert not mask[1, 0, 1]
    assert targets[1, 0, 1].item() == pytest.approx(11.0)


def test_productive_resource_value_includes_non_liquidatable_investment() -> None:
    observation = {
        "step": 10,
        "player": 0,
        "farms": [
            {"money": 1000, "hands": [{}, {}], "unlocked_quadrants": [0], "tiles": [[None]]},
            {"money": 1000, "hands": [], "unlocked_quadrants": [0], "tiles": [[None]]},
        ],
        "private": {"shed": {}, "inventories": [], "seeds": {"WHEAT": 5}},
        "market": {"inventory": {}},
    }
    assert productive_resource_value(observation) > liquidatable_net_asset(observation)

"""
action_utility_engine.py

Action Utility Engine for the Kaggriculture AI Agent.

Combines expected reward and action cost into
a single utility score.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.action_cost_estimator import (
    ActionCostEstimator,
)
from src.expected_reward_estimator import (
    ExpectedRewardEstimator,
)


class ActionUtilityEngine:
    """
    Compute action utility.
    """

    def __init__(self):

        self.reward_estimator = (
            ExpectedRewardEstimator()
        )

        self.cost_estimator = (
            ActionCostEstimator()
        )

    # ---------------------------------------------------------

    def utility(
        self,
        *,
        reward: float,
        probability: float,
        money_cost: float = 0.0,
        energy_cost: float = 0.0,
        time_cost: float = 0.0,
    ) -> float:
        """
        Compute overall action utility.
        """

        expected_reward = (
            self.reward_estimator.expected_reward(
                reward=reward,
                probability=probability,
            )
        )

        total_cost = (
            self.cost_estimator.estimate(
                money=money_cost,
                energy=energy_cost,
                time=time_cost,
            )
        )

        return expected_reward - total_cost

    # ---------------------------------------------------------

    def worthwhile(
        self,
        *,
        reward: float,
        probability: float,
        money_cost: float = 0.0,
        energy_cost: float = 0.0,
        time_cost: float = 0.0,
    ) -> bool:
        """
        Return True if utility is positive.
        """

        return (
            self.utility(
                reward=reward,
                probability=probability,
                money_cost=money_cost,
                energy_cost=energy_cost,
                time_cost=time_cost,
            )
            > 0
        )
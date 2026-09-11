"""
expected_reward_estimator.py

Expected Reward Estimator for the Kaggriculture AI Agent.

Estimates the expected value of an action
using probability and reward.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ExpectedRewardEstimator:
    """
    Estimate expected rewards.
    """

    def expected_reward(
        self,
        *,
        reward: float,
        probability: float,
    ) -> float:
        """
        Compute expected reward.
        """

        probability = max(
            0.0,
            min(
                1.0,
                probability,
            ),
        )

        return reward * probability

    # ---------------------------------------------------------

    def expected_utility(
        self,
        *,
        reward: float,
        probability: float,
        cost: float,
    ) -> float:
        """
        Compute expected utility.
        """

        return (
            self.expected_reward(
                reward=reward,
                probability=probability,
            )
            - cost
        )

    # ---------------------------------------------------------

    def worthwhile(
        self,
        *,
        reward: float,
        probability: float,
        cost: float,
    ) -> bool:
        """
        Return True if expected utility is positive.
        """

        return (
            self.expected_utility(
                reward=reward,
                probability=probability,
                cost=cost,
            )
            > 0
        )
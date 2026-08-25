"""
action_cost_estimator.py

Action Cost Estimator for the Kaggriculture AI Agent.

Estimates the total cost of performing
an action.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ActionCostEstimator:
    """
    Estimate action costs.
    """

    def estimate(
        self,
        *,
        money: float = 0.0,
        energy: float = 0.0,
        time: float = 0.0,
    ) -> float:
        """
        Return the weighted action cost.
        """

        return (
            money
            + (energy * 2.0)
            + (time * 0.5)
        )

    # ---------------------------------------------------------

    def affordable(
        self,
        *,
        available_money: float,
        estimated_cost: float,
    ) -> bool:
        """
        Return True if the action is affordable.
        """

        return available_money >= estimated_cost

    # ---------------------------------------------------------

    def efficiency(
        self,
        *,
        reward: float,
        cost: float,
    ) -> float:
        """
        Return reward-to-cost ratio.
        """

        if cost <= 0:
            return reward

        return reward / cost
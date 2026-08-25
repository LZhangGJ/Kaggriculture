"""
execution_plan_evaluator.py

Execution Plan Evaluator for the Kaggriculture AI Agent.

Evaluates the quality of an execution plan
using reward, cost, and efficiency metrics.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ExecutionPlanEvaluator:
    """
    Evaluate execution plans.
    """

    # ---------------------------------------------------------

    def score(
        self,
        *,
        reward: float,
        cost: float,
    ) -> float:
        """
        Compute plan score.
        """

        return reward - cost

    # ---------------------------------------------------------

    def efficiency(
        self,
        *,
        reward: float,
        cost: float,
    ) -> float:
        """
        Compute reward-to-cost ratio.
        """

        if cost <= 0:
            return reward

        return reward / cost

    # ---------------------------------------------------------

    def worthwhile(
        self,
        *,
        reward: float,
        cost: float,
    ) -> bool:
        """
        Return True if the plan has positive value.
        """

        return self.score(
            reward=reward,
            cost=cost,
        ) > 0
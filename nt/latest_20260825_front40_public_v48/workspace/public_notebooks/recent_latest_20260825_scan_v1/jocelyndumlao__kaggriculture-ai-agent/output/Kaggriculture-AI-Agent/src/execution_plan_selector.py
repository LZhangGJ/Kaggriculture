"""
execution_plan_selector.py

Execution Plan Selector for the Kaggriculture AI Agent.

Selects the best execution plan from
multiple candidate plans.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ExecutionPlanSelector:
    """
    Select the best execution plan.
    """

    # ---------------------------------------------------------

    def select(
        self,
        plans: list[dict],
    ) -> dict | None:
        """
        Return the highest-scoring plan.
        """

        if not plans:
            return None

        return max(
            plans,
            key=lambda plan: plan.get(
                "score",
                0.0,
            ),
        )

    # ---------------------------------------------------------

    def rank(
        self,
        plans: list[dict],
    ) -> list[dict]:
        """
        Return plans sorted by score.
        """

        return sorted(
            plans,
            key=lambda plan: plan.get(
                "score",
                0.0,
            ),
            reverse=True,
        )

    # ---------------------------------------------------------

    def top_n(
        self,
        plans: list[dict],
        *,
        limit: int,
    ) -> list[dict]:
        """
        Return the top N plans.
        """

        return self.rank(plans)[:limit]
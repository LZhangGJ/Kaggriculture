"""
decision_executor.py

Decision Executor for the Kaggriculture AI Agent.

Executes the complete decision pipeline
and returns the final action.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.decision_pipeline import DecisionPipeline


class DecisionExecutor:
    """
    Execute the decision pipeline.
    """

    def __init__(self):

        self.pipeline = DecisionPipeline()

    # ---------------------------------------------------------

    def execute(
        self,
        observation: dict,
    ):
        """
        Execute one decision cycle.
        """

        return self.pipeline.best_action(
            observation,
        )

    # ---------------------------------------------------------

    def has_action(
        self,
        observation: dict,
    ) -> bool:
        """
        Return True if an action exists.
        """

        return (
            self.execute(observation)
            is not None
        )

    # ---------------------------------------------------------

    def context(
        self,
        observation: dict,
    ) -> dict:
        """
        Return decision context.
        """

        return self.pipeline.build_context(
            observation,
        )

    # ---------------------------------------------------------

    def ranked_actions(
        self,
        observation: dict,
    ) -> list:
        """
        Return ranked actions.
        """

        return self.pipeline.rank_actions(
            observation,
        )

    # ---------------------------------------------------------

    def action_count(
        self,
        observation: dict,
    ) -> int:
        """
        Return number of legal actions.
        """

        return len(
            self.ranked_actions(
                observation,
            )
        )
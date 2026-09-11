"""
decision_learning_engine.py

Decision Learning Engine for the Kaggriculture AI Agent.

Learns from previous decisions by tracking
their observed outcomes.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class DecisionLearningEngine:
    """
    Learns action performance over time.
    """

    def __init__(self):

        self._memory = {}

    # ---------------------------------------------------------

    def record(
        self,
        action: str,
        reward: float,
    ) -> None:
        """
        Record the observed reward for an action.
        """

        action = action.upper()

        self._memory.setdefault(
            action,
            [],
        ).append(
            float(reward)
        )

    # ---------------------------------------------------------

    def average_reward(
        self,
        action: str,
    ) -> float:
        """
        Return the average reward for an action.
        """

        action = action.upper()

        rewards = self._memory.get(
            action,
            [],
        )

        if not rewards:
            return 0.0

        return sum(
            rewards
        ) / len(rewards)

    # ---------------------------------------------------------

    def adjustment(
        self,
        action: str,
    ) -> float:
        """
        Return a small adjustment based on
        historical performance.
        """

        return (
            self.average_reward(action)
            * 0.10
        )

    # ---------------------------------------------------------

    def total_records(self) -> int:
        """
        Return total number of stored rewards.
        """

        return sum(
            len(v)
            for v in self._memory.values()
        )

    # ---------------------------------------------------------

    def clear(self) -> None:
        """
        Remove all learned history.
        """

        self._memory.clear()
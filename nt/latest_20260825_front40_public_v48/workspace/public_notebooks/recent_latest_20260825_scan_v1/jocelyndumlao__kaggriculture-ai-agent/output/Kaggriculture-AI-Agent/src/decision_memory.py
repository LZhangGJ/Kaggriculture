"""
decision_memory.py

Decision Memory for the Kaggriculture AI Agent.

Stores previous decisions together with their scores.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from collections import deque


class DecisionMemory:
    """
    Stores recent AI decisions.
    """

    def __init__(
        self,
        max_size: int = 50,
    ):
        self._memory = deque(maxlen=max_size)

    # ---------------------------------------------------------

    def add(
        self,
        action: dict,
        score: float,
    ) -> None:
        """
        Store a decision.
        """

        self._memory.append(
            {
                "action": action,
                "score": float(score),
            }
        )

    # ---------------------------------------------------------

    def last(self):
        """
        Return the latest decision.
        """

        if not self._memory:
            return None

        return self._memory[-1]

    # ---------------------------------------------------------

    def decisions(self):
        """
        Return all stored decisions.
        """

        return list(self._memory)

    # ---------------------------------------------------------

    def average_score(self) -> float:
        """
        Return average stored score.
        """

        if not self._memory:
            return 0.0

        return (
            sum(
                d["score"]
                for d in self._memory
            )
            / len(self._memory)
        )

    # ---------------------------------------------------------

    def clear(self):
        """
        Remove all stored decisions.
        """

        self._memory.clear()
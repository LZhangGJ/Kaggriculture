"""
decision_replay_engine.py

Decision Replay Engine for the Kaggriculture AI Agent.

Stores complete decision history for replay,
analysis, and future learning.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class DecisionReplayEngine:
    """
    Stores every evaluated decision.
    """

    def __init__(self):

        self._history = []

    # ---------------------------------------------------------

    def record(
        self,
        turn: int,
        action: str,
        score: float,
    ) -> None:
        """
        Record one decision.
        """

        self._history.append(
            {
                "turn": turn,
                "action": action.upper(),
                "score": float(score),
            }
        )

    # ---------------------------------------------------------

    def history(self) -> list[dict]:
        """
        Return recorded history.
        """

        return list(self._history)

    # ---------------------------------------------------------

    def latest(self) -> dict | None:
        """
        Return the newest decision.
        """

        if not self._history:
            return None

        return self._history[-1]

    # ---------------------------------------------------------

    def total(self) -> int:
        """
        Number of recorded decisions.
        """

        return len(self._history)

    # ---------------------------------------------------------

    def clear(self) -> None:
        """
        Remove all stored decisions.
        """

        self._history.clear()
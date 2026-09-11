"""
action_history_manager.py

Action History Manager for the Kaggriculture AI Agent.

Tracks recently executed actions for future planning
and decision making.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from collections import deque


class ActionHistoryManager:
    """
    Stores recently executed actions.
    """

    def __init__(
        self,
        max_history: int = 20,
    ):
        self._history = deque(maxlen=max_history)

    # ---------------------------------------------------------

    def record(
        self,
        action: dict,
    ) -> None:
        """
        Record an executed action.
        """

        self._history.append(action)

    # ---------------------------------------------------------

    def last_action(
        self,
    ) -> dict | None:
        """
        Return the most recent action.
        """

        if not self._history:
            return None

        return self._history[-1]

    # ---------------------------------------------------------

    def history(
        self,
    ) -> list[dict]:
        """
        Return the complete history.
        """

        return list(self._history)

    # ---------------------------------------------------------

    def clear(
        self,
    ) -> None:
        """
        Remove all recorded actions.
        """

        self._history.clear()

    # ---------------------------------------------------------

    def count(
        self,
    ) -> int:
        """
        Return number of stored actions.
        """

        return len(self._history)
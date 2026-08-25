"""
action_history_analyzer.py

Action History Analyzer for the Kaggriculture AI Agent.

Provides simple statistics over recently
executed actions.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ActionHistoryAnalyzer:
    """
    Analyze recent action history.
    """

    def __init__(self):

        self.history = []

    # ---------------------------------------------------------

    def record(
        self,
        action: str,
    ) -> None:
        """
        Store an executed action.
        """

        self.history.append(
            action.upper(),
        )

    # ---------------------------------------------------------

    def count(
        self,
        action: str,
    ) -> int:
        """
        Count occurrences of an action.
        """

        return self.history.count(
            action.upper(),
        )

    # ---------------------------------------------------------

    def most_common(
        self,
    ) -> str | None:
        """
        Return the most frequent action.
        """

        if not self.history:
            return None

        return max(
            set(self.history),
            key=self.history.count,
        )

    # ---------------------------------------------------------

    def total_actions(
        self,
    ) -> int:
        """
        Return total recorded actions.
        """

        return len(
            self.history,
        )

    # ---------------------------------------------------------

    def clear(
        self,
    ) -> None:
        """
        Clear action history.
        """

        self.history.clear()
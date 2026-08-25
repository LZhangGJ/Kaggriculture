"""
decision_facade.py

Decision Facade for the Kaggriculture AI Agent.

Provides a simple interface for the complete
decision workflow.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.decision_workflow_coordinator import (
    DecisionWorkflowCoordinator,
)


class DecisionFacade:
    """
    High-level interface for decision making.
    """

    def __init__(self):

        self.workflow = DecisionWorkflowCoordinator()

    # ---------------------------------------------------------

    def evaluate(
        self,
        *,
        state_id: str,
        game_state: dict,
        action: str,
        maximums: dict[str, float] | None = None,
        selected: list[str] | None = None,
        market_score: float = 0.0,
        strategy_score: float = 0.0,
        search_algorithm: str = "DEFAULT",
    ) -> float:
        """
        Evaluate a decision.
        """

        return self.workflow.evaluate(
            state_id=state_id,
            game_state=game_state,
            action=action,
            maximums=maximums,
            selected=selected,
            market_score=market_score,
            strategy_score=strategy_score,
            search_algorithm=search_algorithm,
        )

    # ---------------------------------------------------------

    def history(
        self,
    ):
        """
        Return decision history.
        """

        return self.workflow.history()

    # ---------------------------------------------------------

    def clear_cache(
        self,
    ) -> None:
        """
        Clear cached decision features.
        """

        self.workflow.clear_cache()
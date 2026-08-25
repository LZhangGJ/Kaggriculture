"""
decision_workflow_coordinator.py

Decision Workflow Coordinator for the Kaggriculture AI Agent.

Coordinates context building, decision evaluation,
execution, and replay into a unified workflow.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.decision_context_builder import (
    DecisionContextBuilder,
)
from src.decision_executor import (
    DecisionExecutor,
)
from src.decision_replay_engine import (
    DecisionReplayEngine,
)
from src.decision_service_coordinator import (
    DecisionServiceCoordinator,
)


from src.base_workflow_coordinator import (
    BaseWorkflowCoordinator,
)


class DecisionWorkflowCoordinator:

    """
    Coordinate the complete decision workflow.
    """

    def __init__(self):

        self.context_builder = DecisionContextBuilder()
        self.service = DecisionServiceCoordinator()
        self.executor = DecisionExecutor()
        self.replay = DecisionReplayEngine()

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
        Build context and evaluate an action.
        """

        context = self.context_builder.build(
            game_state=game_state,
            search_algorithm=search_algorithm,
        )

        return self.service.evaluate(
            state_id=state_id,
            game_state=context,
            action=action,
            maximums=maximums,
            selected=selected,
            market_score=market_score,
            strategy_score=strategy_score,
        )

    # ---------------------------------------------------------

    def record(
        self,
        *,
        turn: int,
        action: str,
        score: float,
    ) -> None:
        """
        Record a completed decision.
        """

        self.replay.record(
            turn=turn,
            action=action,
            score=score,
        )

    # ---------------------------------------------------------

    def history(self):
        """
        Return replay history.
        """

        return self.replay.history()

    # ---------------------------------------------------------

    def clear_cache(
        self,
    ) -> None:
        """
        Clear cached decision features.
        """

        self.service.clear_cache()
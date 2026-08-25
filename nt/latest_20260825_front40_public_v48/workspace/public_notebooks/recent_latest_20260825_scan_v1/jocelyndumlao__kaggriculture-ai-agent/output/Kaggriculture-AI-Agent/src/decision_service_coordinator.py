"""
decision_service_coordinator.py

Decision Service Coordinator for the Kaggriculture AI Agent.

Coordinates feature preparation and decision
evaluation into a unified decision workflow.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.decision_feature_service import (
    DecisionFeatureService,
)
from src.unified_action_evaluator import (
    UnifiedActionEvaluator,
)


class DecisionServiceCoordinator:
    """
    Coordinate decision services.
    """

    def __init__(self):

        self.feature_service = DecisionFeatureService()
        self.evaluator = UnifiedActionEvaluator()

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
    ) -> float:
        """
        Evaluate a candidate action.
        """

        features = self.feature_service.features(
            state_id=state_id,
            game_state=game_state,
            maximums=maximums,
            selected=selected,
        )

        return self.evaluator.evaluate(
            action,
            game_state=features,
            market_score=market_score,
            strategy_score=strategy_score,
        )

    # ---------------------------------------------------------

    def clear_cache(
        self,
    ) -> None:
        """
        Clear cached decision features.
        """

        self.feature_service.clear_cache()
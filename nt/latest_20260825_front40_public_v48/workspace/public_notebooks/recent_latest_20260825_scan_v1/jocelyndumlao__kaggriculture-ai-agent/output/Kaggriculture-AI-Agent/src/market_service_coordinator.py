"""
market_service_coordinator.py

Market Service Coordinator for the Kaggriculture AI Agent.

Coordinates market feature preparation and
market decision services into a unified
market workflow.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.market_feature_service import (
    MarketFeatureService,
)
from src.market_decision_engine import (
    MarketDecisionEngine,
)


class MarketServiceCoordinator:
    """
    Coordinate market services.
    """

    def __init__(self):

        self.feature_service = MarketFeatureService()
        self.market_engine = MarketDecisionEngine()

    # ---------------------------------------------------------

    def prepare_features(
        self,
        *,
        state_id: str,
        game_state: dict,
        maximums: dict[str, float] | None = None,
        selected: list[str] | None = None,
    ) -> dict:
        """
        Prepare market features.
        """

        return self.feature_service.features(
            state_id=state_id,
            game_state=game_state,
            maximums=maximums,
            selected=selected,
        )

    # ---------------------------------------------------------

    def should_sell(
        self,
        *,
        current_price: float,
        average_price: float,
    ) -> bool:
        """
        Delegate selling decision.
        """

        return self.market_engine.should_sell(
            current_price=current_price,
            average_price=average_price,
        )

    # ---------------------------------------------------------

    def clear_cache(
        self,
    ) -> None:
        """
        Clear cached market features.
        """

        self.feature_service.clear_cache()
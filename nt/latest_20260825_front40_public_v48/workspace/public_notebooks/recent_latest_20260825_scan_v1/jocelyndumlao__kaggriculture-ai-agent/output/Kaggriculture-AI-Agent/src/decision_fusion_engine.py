"""
decision_fusion_engine.py

Decision Fusion Engine for the Kaggriculture AI Agent.

Combines multiple evaluation signals into one
final decision score.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class DecisionFusionEngine:
    """
    Combines scoring components.
    """

    def fuse(
        self,
        *,
        action_score: float = 0.0,
        opportunity_score: float = 0.0,
        market_score: float = 0.0,
        strategy_score: float = 0.0,
        risk_score: float = 0.0,
    ) -> float:
        """
        Compute the final fused score.
        """

        return (
            action_score
            + opportunity_score
            + market_score
            + strategy_score
            - risk_score
        )

    # ---------------------------------------------------------

    def better(
        self,
        score_a: float,
        score_b: float,
    ) -> bool:
        """
        Return True if score_a is better.
        """

        return score_a > score_b

    # ---------------------------------------------------------

    def normalize(
        self,
        score: float,
        *,
        minimum: float = -100.0,
        maximum: float = 100.0,
    ) -> float:
        """
        Clamp score to a fixed range.
        """

        return max(
            minimum,
            min(
                maximum,
                score,
            ),
        )

    # ---------------------------------------------------------

    def rank(
        self,
        scores: dict[str, float],
    ) -> list[tuple[str, float]]:
        """
        Rank actions by fused score.
        """

        return sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

    # ---------------------------------------------------------

    def best(
        self,
        scores: dict[str, float],
    ) -> str | None:
        """
        Return the best action.
        """

        if not scores:
            return None

        return max(
            scores,
            key=scores.get,
        )
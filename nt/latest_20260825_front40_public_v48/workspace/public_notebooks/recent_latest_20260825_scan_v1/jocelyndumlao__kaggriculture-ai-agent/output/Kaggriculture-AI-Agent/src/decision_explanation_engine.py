"""
decision_explanation_engine.py

Decision Explanation Engine for the Kaggriculture AI Agent.

Produces human-readable explanations for AI decisions.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class DecisionExplanationEngine:
    """
    Explains why an action was selected.
    """

    def explain(
        self,
        *,
        action: str,
        action_score: float,
        opportunity_score: float,
        risk_score: float,
        market_score: float,
        strategy_score: float,
        final_score: float,
    ) -> dict:
        """
        Return a structured explanation.
        """

        return {
            "action": action,
            "action_score": float(action_score),
            "opportunity_score": float(opportunity_score),
            "risk_score": float(risk_score),
            "market_score": float(market_score),
            "strategy_score": float(strategy_score),
            "final_score": float(final_score),
        }

    # ---------------------------------------------------------

    def summary(
        self,
        explanation: dict,
    ) -> str:
        """
        Return a readable summary.
        """

        return (
            f"{explanation['action']} selected "
            f"(score={explanation['final_score']:.1f})"
        )

    # ---------------------------------------------------------

    def best_action(
        self,
        explanations: list[dict],
    ) -> dict | None:
        """
        Return explanation with highest score.
        """

        if not explanations:
            return None

        return max(
            explanations,
            key=lambda x: x["final_score"],
        )

    # ---------------------------------------------------------

    def average_score(
        self,
        explanations: list[dict],
    ) -> float:
        """
        Return average final score.
        """

        if not explanations:
            return 0.0

        return (
            sum(
                item["final_score"]
                for item in explanations
            )
            / len(explanations)
        )

    # ---------------------------------------------------------

    def compare(
        self,
        first: dict,
        second: dict,
    ) -> float:
        """
        Return score difference.
        """

        return (
            first["final_score"]
            - second["final_score"]
        )
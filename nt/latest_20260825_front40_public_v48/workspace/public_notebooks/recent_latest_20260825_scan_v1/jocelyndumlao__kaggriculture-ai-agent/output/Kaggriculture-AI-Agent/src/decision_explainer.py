"""
decision_explainer.py

Decision Explanation Engine for the Kaggriculture AI Agent.

Provides human-readable explanations for AI decisions.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class DecisionExplainer:
    """
    Explains why a decision was selected.
    """

    def explain(
        self,
        action: str,
        *,
        utility: float = 0.0,
        profit: float = 0.0,
        risk: float = 0.0,
        season: str = "",
        reason: str = "",
    ):
        """
        Build a structured explanation.
        """

        return {
            "action": action,
            "utility": round(utility, 2),
            "profit": round(profit, 2),
            "risk": round(risk, 2),
            "season": season,
            "reason": reason,
        }

    # ---------------------------------------------------------

    def summary(self, explanation):
        """
        Convert an explanation dictionary into readable text.
        """

        return (
            f"Action: {explanation['action']}\n"
            f"Utility: {explanation['utility']}\n"
            f"Profit: {explanation['profit']}\n"
            f"Risk: {explanation['risk']}\n"
            f"Season: {explanation['season']}\n"
            f"Reason: {explanation['reason']}"
        )

    # ---------------------------------------------------------

    def best_reason(self):
        """
        Default explanation when no specific reason is available.
        """

        return "Highest overall utility score."
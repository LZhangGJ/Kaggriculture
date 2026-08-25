"""
hybrid_search.py

Hybrid Search Engine for the Kaggriculture AI Agent.

Combines multiple AI search strategies.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.minimax import Minimax
from src.beam_search import BeamSearch
from src.state_evaluator import StateEvaluator


class HybridSearch:
    """
    Combines Beam Search, Minimax,
    and heuristic evaluation.
    """

    def __init__(self):

        self.minimax = Minimax()
        self.beam = BeamSearch()
        self.evaluator = StateEvaluator()

    # -----------------------------------------------------

    def evaluate(self, state):
        """
        Evaluate a state.

        Supports both:

        • numeric scores (used in unit tests)

        • dictionary game states
        """

        # -------------------------------
        # Numeric state (unit tests)
        # -------------------------------

        if isinstance(state, (int, float)):
            return float(state)

        # -------------------------------
        # Dictionary state (future AI)
        # -------------------------------

        if isinstance(state, dict):

            money = state.get("money", 0)

            crops = state.get("crops", 0)

            animals = state.get("animals", 0)

            inventory = state.get("inventory", 0)

            return self.evaluator.evaluate(
                money,
                crops,
                animals,
                inventory,
            )

        return 0.0

    # -----------------------------------------------------

    def rank(self, candidates):
        """
        Rank candidate states.
        """

        ranked = []

        for state in candidates:

            score = self.evaluate(state)

            ranked.append((state, score))

        return ranked

    # -----------------------------------------------------

    def search(self, candidates):
        """
        Return the best candidate.
        """

        ranked = self.rank(candidates)

        best = self.beam.best(ranked)

        if best is None:
            return None

        return best[0]
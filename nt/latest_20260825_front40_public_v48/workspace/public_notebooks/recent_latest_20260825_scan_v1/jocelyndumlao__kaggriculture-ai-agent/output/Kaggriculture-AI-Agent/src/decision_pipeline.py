"""
decision_pipeline.py

Decision Pipeline for the Kaggriculture AI Agent.

Coordinates parsing, context building,
search selection, action generation,
and action scheduling.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.observation_parser import ObservationParser
from src.game_state_adapter import GameStateAdapter
from src.decision_context_builder import DecisionContextBuilder
from src.search_controller import SearchController
from src.legal_action_generator import LegalActionGenerator
from src.action_scoring_engine import ActionScoringEngine


class DecisionPipeline:
    """
    End-to-end decision pipeline.
    """

    def __init__(self):

        self.parser = ObservationParser()
        self.adapter = GameStateAdapter()
        self.context_builder = DecisionContextBuilder()
        self.search = SearchController()
        self.generator = LegalActionGenerator()
        self.scorer = ActionScoringEngine()

    # ---------------------------------------------------------

    def build_context(
        self,
        observation: dict,
    ) -> dict:

        state = self.adapter.adapt(observation)

        algorithm = self.search.select_algorithm(
            turn=observation.get("day", 0),
        )

        return self.context_builder.build(
            game_state=state,
            search_algorithm=algorithm,
        )

    # ---------------------------------------------------------

    def legal_actions(
        self,
        observation: dict,
    ):

        return self.generator.generate(
            observation,
        )

    # ---------------------------------------------------------

    def score(
        self,
        action: dict,
    ) -> float:

        return self.scorer.score(
            action=action["action"],
        )

    # ---------------------------------------------------------

    def rank_actions(
        self,
        observation: dict,
    ) -> list:

        actions = self.legal_actions(
            observation,
        )

        return sorted(
            actions,
            key=self.score,
            reverse=True,
        )

    # ---------------------------------------------------------

    def best_action(
        self,
        observation: dict,
    ):

        actions = self.rank_actions(
            observation,
        )

        if not actions:
            return None

        return actions[0]
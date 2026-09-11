"""
game_phase_coordinator.py

Game Phase Coordinator for the Kaggriculture AI Agent.

Combines season and daily turn phases into
a unified game phase description.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.turn_phase_manager import TurnPhaseManager


class GamePhaseCoordinator:
    """
    Coordinates game phases.
    """

    def __init__(self):

        self.turn_manager = TurnPhaseManager()

    # ---------------------------------------------------------

    def season_phase(
        self,
        day: int,
    ) -> str:
        """
        Return season phase.
        """

        if day <= 8:
            return "EARLY"

        if day <= 20:
            return "MID"

        if day <= 26:
            return "LATE"

        return "END"

    # ---------------------------------------------------------

    def game_phase(
        self,
        *,
        day: int,
        hour: int,
    ) -> dict:
        """
        Return combined game phase.
        """

        return {
            "season": self.season_phase(day),
            "turn": self.turn_manager.phase(hour),
        }

    # ---------------------------------------------------------

    def is_endgame(
        self,
        day: int,
    ) -> bool:

        return self.season_phase(day) == "END"

    # ---------------------------------------------------------

    def should_force_sell(
        self,
        day: int,
    ) -> bool:

        return day >= 28

    # ---------------------------------------------------------

    def should_expand(
        self,
        day: int,
    ) -> bool:

        return day <= 15
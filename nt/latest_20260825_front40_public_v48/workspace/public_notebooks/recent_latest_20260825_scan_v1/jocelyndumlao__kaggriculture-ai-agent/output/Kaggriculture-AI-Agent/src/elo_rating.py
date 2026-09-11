"""
elo_rating.py

Elo Rating System for the Kaggriculture AI Agent.

Maintains dynamic ratings for AI agents.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from math import pow


class EloRating:
    """
    Elo rating manager.
    """

    DEFAULT_RATING = 1000

    def __init__(self, k_factor: float = 32):
        self.k = k_factor
        self._ratings = {}

    # ---------------------------------------------------------

    def register(self, player: str):
        """
        Register a player.
        """
        self._ratings.setdefault(player, self.DEFAULT_RATING)

    # ---------------------------------------------------------

    def rating(self, player: str) -> float:
        """
        Return current rating.
        """
        self.register(player)
        return self._ratings[player]

    # ---------------------------------------------------------

    def expected_score(
        self,
        player_a: str,
        player_b: str,
    ) -> float:
        """
        Expected score of player A.
        """
        ra = self.rating(player_a)
        rb = self.rating(player_b)

        return 1 / (1 + pow(10, (rb - ra) / 400))

    # ---------------------------------------------------------

    def update(
        self,
        winner: str,
        loser: str,
    ):
        """
        Update ratings after one match.
        """

        ea = self.expected_score(winner, loser)
        eb = self.expected_score(loser, winner)

        self._ratings[winner] += self.k * (1 - ea)
        self._ratings[loser] += self.k * (0 - eb)

    # ---------------------------------------------------------

    def leaderboard(self):
        """
        Return players ordered by rating.
        """
        return sorted(
            self._ratings.items(),
            key=lambda item: item[1],
            reverse=True,
        )

    # ---------------------------------------------------------

    def reset(self):
        """
        Clear all ratings.
        """
        self._ratings.clear()
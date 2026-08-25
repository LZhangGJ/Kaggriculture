"""
episode_learning_engine.py

Episode Learning Engine for the Kaggriculture AI Agent.

Stores complete game (episode) outcomes and
computes long-term performance statistics.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class EpisodeLearningEngine:
    """
    Learn from complete game episodes.
    """

    def __init__(self):

        self.episodes = []

    # ---------------------------------------------------------

    def record_episode(
        self,
        *,
        reward: float,
        win: bool,
    ) -> None:
        """
        Store a completed episode.
        """

        self.episodes.append(
            {
                "reward": float(reward),
                "win": bool(win),
            }
        )

    # ---------------------------------------------------------

    def total_episodes(
        self,
    ) -> int:
        """
        Return number of recorded episodes.
        """

        return len(
            self.episodes
        )

    # ---------------------------------------------------------

    def win_rate(
        self,
    ) -> float:
        """
        Return win percentage.
        """

        if not self.episodes:
            return 0.0

        wins = sum(
            episode["win"]
            for episode in self.episodes
        )

        return wins / len(self.episodes)

    # ---------------------------------------------------------

    def average_reward(
        self,
    ) -> float:
        """
        Return average episode reward.
        """

        if not self.episodes:
            return 0.0

        total = sum(
            episode["reward"]
            for episode in self.episodes
        )

        return total / len(self.episodes)

    # ---------------------------------------------------------

    def best_reward(
        self,
    ) -> float:
        """
        Return highest observed reward.
        """

        if not self.episodes:
            return 0.0

        return max(
            episode["reward"]
            for episode in self.episodes
        )
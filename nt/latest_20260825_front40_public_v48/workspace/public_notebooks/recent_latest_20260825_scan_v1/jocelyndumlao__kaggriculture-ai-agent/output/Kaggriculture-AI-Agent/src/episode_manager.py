"""
episode_manager.py

Episode Manager for the Kaggriculture AI Agent.

Coordinates reinforcement learning episodes.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class EpisodeManager:
    """
    Tracks reinforcement learning episodes.
    """

    def __init__(self):
        self.episode = 0
        self.steps = 0
        self.total_reward = 0.0
        self.done = False

    # ---------------------------------------------------------

    def start_episode(self):
        """
        Begin a new episode.
        """

        self.episode += 1
        self.steps = 0
        self.total_reward = 0.0
        self.done = False

    # ---------------------------------------------------------

    def add_reward(self, reward: float):
        """
        Add reward to the episode.
        """

        self.total_reward += reward

    # ---------------------------------------------------------

    def step(self):
        """
        Advance one step.
        """

        self.steps += 1

    # ---------------------------------------------------------

    def finish(self):
        """
        Mark the episode as complete.
        """

        self.done = True

    # ---------------------------------------------------------

    def summary(self):
        """
        Return episode statistics.
        """

        return {
            "episode": self.episode,
            "steps": self.steps,
            "reward": self.total_reward,
            "done": self.done,
        }

    # ---------------------------------------------------------

    def reset(self):
        """
        Reset all counters.
        """

        self.episode = 0
        self.steps = 0
        self.total_reward = 0.0
        self.done = False
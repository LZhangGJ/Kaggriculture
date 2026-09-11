"""
experience_replay.py

Experience Replay Memory for the Kaggriculture AI Agent.

Stores gameplay experiences for future learning.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

import random


class ExperienceReplay:
    """
    Replay memory for reinforcement learning.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory = []

    # ---------------------------------------------------------

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Store one experience.
        """

        if len(self.memory) >= self.capacity:
            self.memory.pop(0)

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    # ---------------------------------------------------------

    def sample(self, batch_size: int):
        """
        Return a random batch of experiences.
        """

        batch_size = min(batch_size, len(self.memory))

        return random.sample(
            self.memory,
            batch_size,
        )

    # ---------------------------------------------------------

    def size(self):
        """
        Number of stored experiences.
        """

        return len(self.memory)

    # ---------------------------------------------------------

    def clear(self):
        """
        Remove every stored experience.
        """

        self.memory.clear()
        
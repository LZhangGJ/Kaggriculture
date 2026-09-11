"""
learning_rate_scheduler.py

Learning Rate Scheduler for the Kaggriculture AI Agent.

Gradually reduces the learning rate during training.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class LearningRateScheduler:
    """
    Exponential learning rate scheduler.
    """

    def __init__(
        self,
        initial_rate: float = 0.10,
        decay: float = 0.99,
        minimum_rate: float = 0.01,
    ):
        self.initial_rate = initial_rate
        self.rate = initial_rate
        self.decay = decay
        self.minimum_rate = minimum_rate

    # ---------------------------------------------------------

    def step(self) -> float:
        """
        Apply one decay step.
        """

        self.rate *= self.decay

        if self.rate < self.minimum_rate:
            self.rate = self.minimum_rate

        return self.rate

    # ---------------------------------------------------------

    def current(self) -> float:
        """
        Return current learning rate.
        """

        return self.rate

    # ---------------------------------------------------------

    def reset(self):
        """
        Reset to the initial learning rate.
        """

        self.rate = self.initial_rate
        
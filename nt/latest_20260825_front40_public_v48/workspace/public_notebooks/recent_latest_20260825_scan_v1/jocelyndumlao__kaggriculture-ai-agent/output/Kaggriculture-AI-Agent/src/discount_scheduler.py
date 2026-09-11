"""
discount_scheduler.py

Discount Factor Scheduler for the Kaggriculture AI Agent.

Gradually adjusts the discount factor (gamma)
during reinforcement learning.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

class DiscountScheduler:
    """
    Scheduler for gamma.
    """

    def __init__(
        self,
        initial_gamma: float = 0.80,
        maximum_gamma: float = 0.99,
        increment: float = 0.01,
    ):
        self.initial_gamma = initial_gamma
        self.gamma = initial_gamma
        self.maximum_gamma = maximum_gamma
        self.increment = increment

    # ---------------------------------------------------------

    def step(self) -> float:
        """
        Increase gamma.
        """

        self.gamma += self.increment

        if self.gamma > self.maximum_gamma:
            self.gamma = self.maximum_gamma

        # Avoid floating-point precision issues
        self.gamma = round(self.gamma, 10)

        return self.gamma

    # ---------------------------------------------------------

    def current(self) -> float:
        """
        Current gamma.
        """

        return self.gamma

    # ---------------------------------------------------------

    def reset(self):
        """
        Reset scheduler.
        """

        self.gamma = self.initial_gamma
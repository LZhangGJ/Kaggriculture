"""
decision_engine.py

Central decision engine for the Kaggriculture AI Agent.

Coordinates the planner and scheduler to determine
the next task for the agent.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.planner import Planner
from src.scheduler import Scheduler


class DecisionEngine:
    """
    Combines the planner and scheduler into one
    decision pipeline.
    """

    def __init__(self, parser, world, market):
        """
        Parameters
        ----------
        parser : ObservationParser
        world : World
        market : Market
        """

        self.parser = parser
        self.world = world
        self.market = market

        self.planner = Planner(
            parser,
            world,
            market,
        )

        self.scheduler = Scheduler()

    # ---------------------------------------------------------
    # Decision Pipeline
    # ---------------------------------------------------------

    def next_task(self):
        """
        Return the highest-priority task.
        """

        self.scheduler.clear()

        tasks = self.planner.plan()

        for task in tasks:

            self.scheduler.add(
                priority=task["priority"],
                action=task["task"],
                target=task.get("target"),
                crop=task.get("crop"),
                product=task.get("product"),
                amount=task.get("amount"),
            )

        return self.scheduler.next_task()
"""
agent.py

Main Kaggriculture AI Agent.

Connects every AI module into one complete pipeline.

Observation
    ↓
Parser
    ↓
World
    ↓
Market
    ↓
Planner
    ↓
Scheduler
    ↓
Decision Engine
    ↓
Action Builder
    ↓
Kaggle Action

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.parser import ObservationParser
from src.world import World
from src.market import Market
from src.decision_engine import DecisionEngine
from src.actions import ActionBuilder


class Agent:
    """
    Main AI Agent.

    Converts a Kaggle observation into
    one valid Kaggle action.
    """

    def __init__(self):
        self.builder = ActionBuilder()

    def act(self, observation):
        """
        Produce one Kaggle action.
        """

        parser = ObservationParser(observation)

        world = World(parser)

        market = Market(parser)

        engine = DecisionEngine(
            parser,
            world,
            market,
        )

        task = engine.next_task()

        return self.builder.build(task)
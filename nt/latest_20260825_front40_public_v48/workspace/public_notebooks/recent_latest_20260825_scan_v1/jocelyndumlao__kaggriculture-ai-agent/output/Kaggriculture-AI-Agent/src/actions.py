"""
actions.py

Converts planner decisions into valid Kaggle actions.

Author: Jocelyn Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from src.constants import Action


class ActionBuilder:
    """
    Convert planner tasks into Kaggle-compatible actions.
    """

    # Valid movement commands
    MOVES = {
        Action.NORTH.value,
        Action.SOUTH.value,
        Action.EAST.value,
        Action.WEST.value,
    }

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Main Builder
    # ---------------------------------------------------------

    def build(self, task: dict | None) -> dict:
        """
        Convert a planner task into the action dictionary
        expected by Kaggle.

        Supports both:
        - planner dictionaries
        - Scheduler Task dataclass objects
        """

        if task is None:
            return self.pass_turn()

        # -----------------------------------------------------
        # Support both dict and Task objects
        # -----------------------------------------------------

        if isinstance(task, dict):
            action = task.get("task")
            target = task.get("target")
            crop = task.get("crop")
            product = task.get("product")
            amount = task.get("amount")
        else:
            action = task.action
            target = task.target
            crop = task.crop
            product = task.product
            amount = task.amount

        # =====================================================
        # Movement
        # =====================================================

        if action in self.MOVES:

            return {
                "farmer": [action],
                "hands": [],
                "market": [],
            }

        # =====================================================
        # Pass
        # =====================================================

        if action == Action.PASS.value:

            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": [],
            }

        # =====================================================
        # Water
        # =====================================================

        if action == Action.WATER.value:

            return {
                "farmer": ["WATER"],
                "hands": [],
                "market": [],
            }

        # =====================================================
        # Harvest
        # =====================================================

        if action == Action.HARVEST.value:

            return {
                "farmer": ["HARVEST"],
                "hands": [],
                "market": [],
            }

        # =====================================================
        # Feed Animal
        # =====================================================

        if action == Action.FEED.value:

            return {
                "farmer": ["FEED"],
                "hands": [],
                "market": [],
            }

        # =====================================================
        # Care Animal
        # =====================================================

        if action == Action.CARE.value:

            return {
                "farmer": ["CARE"],
                "hands": [],
                "market": [],
            }

        # =====================================================
        # Fertilize
        # =====================================================

        if action == Action.FERTILIZE.value:

            return {
                "farmer": ["FERTILIZE"],
                "hands": [],
                "market": [],
            }

        # =====================================================
        # Collect Fertilizer
        # =====================================================

        if action == Action.COLLECT_FERTILIZER.value:

            return {
                "farmer": ["COLLECT_FERTILIZER"],
                "hands": [],
                "market": [],
            }

        # =====================================================
        # Dig
        # =====================================================

        if action == Action.DIG.value:

            return {
                "farmer": ["DIG"],
                "hands": [],
                "market": [],
            }

        # =====================================================
        # Plant Crop
        # =====================================================

        if action == Action.PLANT.value:

            crop = crop or "WHEAT"

            return {
                "farmer": ["PLANT", crop],
                "hands": [],
                "market": [],
            }

        # =====================================================
        # Sell Product
        # =====================================================

        if action == "SELL":

            return {
                "farmer": [],
                "hands": [],
                "market": [
                    [
                        "SELL",
                        product,
                        amount,
                    ]
                ],
            }

        # =====================================================
        # Buy Product
        # =====================================================

        if action == "BUY_PRODUCT":

            return {
                "farmer": [],
                "hands": [],
                "market": [
                    [
                        "BUY",
                        product,
                        amount,
                    ]
                ],
            }

        # =====================================================
        # Unknown Action
        # =====================================================

        return self.pass_turn()

    # ---------------------------------------------------------
    # Pass Turn
    # ---------------------------------------------------------

    def pass_turn(self) -> dict:
        """
        Return a PASS action.
        """

        return {
            "farmer": ["PASS"],
            "hands": [],
            "market": [],
        }
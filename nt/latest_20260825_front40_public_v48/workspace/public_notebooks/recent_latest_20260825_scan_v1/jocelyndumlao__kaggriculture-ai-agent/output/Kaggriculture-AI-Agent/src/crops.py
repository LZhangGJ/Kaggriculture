"""
crops.py

Crop database for the Kaggriculture AI Agent.

Stores all crop statistics used by the planner.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CropInfo:
    """Information describing one crop."""

    name: str

    seed_cost: int

    base_price: int

    first_yield_day: int

    max_yield_day: int | None

    yield_interval: int | None

    max_yield: int

    action_cost: int


# =====================================================
# Crop Database
# =====================================================

CROPS = {

    "WHEAT": CropInfo(
        name="WHEAT",
        seed_cost=10,
        base_price=25,
        first_yield_day=2,
        max_yield_day=4,
        yield_interval=None,
        max_yield=6,
        action_cost=1,
    ),

    "CARROT": CropInfo(
        name="CARROT",
        seed_cost=20,
        base_price=35,
        first_yield_day=2,
        max_yield_day=3,
        yield_interval=None,
        max_yield=4,
        action_cost=1,
    ),

    "TOMATO": CropInfo(
        name="TOMATO",
        seed_cost=50,
        base_price=60,
        first_yield_day=8,
        max_yield_day=None,
        yield_interval=1,
        max_yield=4,
        action_cost=1,
    ),

    "STRAWBERRY": CropInfo(
        name="STRAWBERRY",
        seed_cost=100,
        base_price=120,
        first_yield_day=10,
        max_yield_day=None,
        yield_interval=2,
        max_yield=4,
        action_cost=1,
    ),

    "MELON": CropInfo(
        name="MELON",
        seed_cost=80,
        base_price=250,
        first_yield_day=10,
        max_yield_day=12,
        yield_interval=None,
        max_yield=6,
        action_cost=1,
    ),
}
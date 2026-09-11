"""
config.py

Central configuration for the Kaggriculture AI Agent.

These values are the default competition settings and can be
overridden by the Kaggle environment configuration if needed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GameConfig:
    """Competition configuration."""

    # Board
    board_size: int = 10
    quadrant_size: int = 5

    # Time
    turns_per_day: int = 24
    total_days: int = 30
    episode_steps: int = 720

    # Economy
    starting_money: int = 3000

    # Storage
    shed_capacity: int = 100

    # Market
    max_market_orders: int = 10

    # Environment
    weed_spawn_chance: float = 0.005

    # Town
    town_shop_unlock_interval: int = 3
    town_shop_sell_interval: int = 4
    town_center_sell_interval: int = 12


DEFAULT_CONFIG = GameConfig()


def load_config(env_config: dict | None = None) -> GameConfig:
    """
    Create a GameConfig from the Kaggle environment configuration.

    Parameters
    ----------
    env_config : dict | None
        Environment configuration provided by Kaggle.

    Returns
    -------
    GameConfig
    """

    cfg = GameConfig()

    if env_config is None:
        return cfg

    cfg.board_size = env_config.get("boardSize", cfg.board_size)
    cfg.turns_per_day = env_config.get("turnsPerDay", cfg.turns_per_day)
    cfg.episode_steps = env_config.get("episodeSteps", cfg.episode_steps)
    cfg.shed_capacity = env_config.get("shedCapacity", cfg.shed_capacity)
    cfg.max_market_orders = env_config.get(
        "maxMarketOrdersPerTurn",
        cfg.max_market_orders,
    )
    cfg.weed_spawn_chance = env_config.get(
        "weedSpawnChance",
        cfg.weed_spawn_chance,
    )

    return cfg
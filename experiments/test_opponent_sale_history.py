#!/usr/bin/env python3
"""Minimal executable check for public opponent-sale accounting."""

from meta_agent.src.route_switch_features import OPPONENT_SALE_ITEMS, OpponentSaleHistory


def observation(step, inventory, shops=()):
    empty = [[None] * 10 for _ in range(10)]
    farm = {"money": 1000, "farmer": [0, 4], "hands": [], "tiles": empty,
            "unlocked_quadrants": [0], "hires_today": 0}
    return {"step": step, "day": step // 24, "hour": step % 24, "player": 0,
            "farms": [farm, farm], "private": {"shed": {"CARROT": 2}, "inventories": [{}]},
            "market": {"inventory": dict(inventory)},
            "town": {"unlocked_shops": list(shops)}}


def main():
    start = dict.fromkeys(OPPONENT_SALE_ITEMS, 10_000)
    history = OpponentSaleHistory()
    history.update(observation(216, start, ("PET_CAFE", "YARN_STORE")))
    history.record_action(observation(216, start, ("PET_CAFE", "YARN_STORE")),
                          {"farmer": ["PASS"], "hands": [],
                           "market": [["SELL", "CARROT", 2]]})
    # center + shops consume CARROT/WOOL by 3 and every other tracked item by 1.
    end = {item: value - (3 if item in {"CARROT", "WOOL"} else 1)
           for item, value in start.items()}
    end["CARROT"] += 2 + 4  # own 2, opponent 4
    history.update(observation(217, end, ("PET_CAFE", "YARN_STORE")))
    assert history.vector().tolist() == [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


if __name__ == "__main__":
    main()

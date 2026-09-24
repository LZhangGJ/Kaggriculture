#!/usr/bin/env python3
"""Small executable check for the replay capital-order dependency repair."""
from agent.main import sell_before_unfunded_land


def observation(step=219, wool=4):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    return {
        "step": step, "day": step // 24, "hour": step % 24, "player": 0,
        "farms": [{"money": 1886, "farmer": [4, 4], "hands": [], "tiles": tiles,
                   "unlocked_quadrants": ["NW", "NE"]}, {}],
        "private": {"shed": {}, "inventories": [{"WOOL": wool}]},
        "market": {"inventory": {"WHEAT": 10000, "CARROT": 10000,
                                   "TOMATO": 10000, "STRAWBERRY": 10000,
                                   "MELON": 10000, "EGG": 10000, "MILK": 10000,
                                   "WOOL": 10000, "FERTILIZER": 10000}},
    }


def main():
    action = {"farmer": ["PLACE", "WOOL", 4], "hands": [],
              "market": [["BUY_LAND"], ["SELL", "WOOL", 4]]}
    fixed = sell_before_unfunded_land(observation(), action)
    assert fixed["market"] == [["SELL", "WOOL", 4], ["BUY_LAND"]]
    assert sell_before_unfunded_land(observation(step=167), action) is action
    assert sell_before_unfunded_land(observation(wool=0), action) is action
    print("PASS")


if __name__ == "__main__":
    main()

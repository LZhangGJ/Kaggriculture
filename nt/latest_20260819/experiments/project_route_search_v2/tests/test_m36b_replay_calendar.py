from __future__ import annotations

import numpy as np

from project_route_search_v2.m36_schema import M36MarketTemplateV3
from project_route_search_v2.m36_calendar import validate_route_calendar_v3
from project_route_search_v2.m36_replay_calendar import (
    compile_gold_replay_calendar_v3,
    stack_route_calendars_v3,
)


def _tile(kind=None, **values):
    if kind is None:
        return None
    return {"kind": kind, **values}


def _row(day, hour, *, money=3000, hires=0, tiles=None, shed=None, inventories=None, market=None):
    board = [[None for _ in range(10)] for _ in range(10)]
    for y, x, value in tiles or ():
        board[y][x] = value
    observation = {
        "day": day,
        "hour": hour,
        "farms": [
            {"money": 3000, "hires_today": 0, "tiles": board, "unlocked_quadrants": ["NW"]},
            {
                "money": money,
                "hires_today": hires,
                "tiles": board,
                "unlocked_quadrants": ["NW"],
            },
        ],
        "private": {
            "shed": shed or {},
            "inventories": inventories or [{}, *("" for _ in range(0))],
            "seeds": {},
        },
    }
    return {"action": {"farmer": ["PASS"], "hands": [], "market": market or []}, "observation": observation}


def test_compile_daily_targets_without_raw_action_payload():
    crop = _tile("PLANT", crop="MELON")
    sheep = _tile("PASTURE", animal="SHEEP")
    empty = _row(0, 0)
    opening = _row(
        0,
        1,
        hires=5,
        shed={"COW": 2, "SHEEP": 1},
        market=[
            ["HIRE"],
            ["BUY_ANIMAL", "COW", 2],
            ["BUY_ANIMAL", "SHEEP", 2],
            ["BUY_PRODUCT", "WHEAT", 6],
        ],
    )
    day6 = _row(
        6,
        23,
        hires=7,
        tiles=[(0, 0, crop), (0, 1, crop), (4, 4, sheep)],
        shed={"COW": 2, "SHEEP": 0},
    )
    replay = {
        "id": 7,
        "info": {"TeamNames": ["other", "gold"]},
        "rewards": [0, 123],
        "steps": [
            [{}, empty],
            [{}, opening],
            [{}, day6],
        ],
    }

    calendar, diagnostics = compile_gold_replay_calendar_v3(
        replay, player=1, candidate_id=17
    )
    assert validate_route_calendar_v3(calendar) == []
    assert int(calendar.hand_target_by_day[0, 0]) == 5
    assert int(calendar.feed_stock_target_by_day[0, 0]) == 6
    assert np.asarray(calendar.crop_target_by_day)[0, 6].tolist() == [0, 0, 0, 0, 2]
    assert diagnostics.requested_animal_totals == (0, 2, 2)
    assert diagnostics.filled_animal_additions == (0, 2, 1)
    assert diagnostics.raw_action_payload_stored is False
    assert not any("action" in name for name in calendar._fields)
    assert calendar.animal_care_policy_by_day.shape == (1, 30, 3)


def test_stack_calendars_preserves_complete_day_axis():
    replay = {
        "id": 8,
        "info": {"TeamNames": ["a", "b"]},
        "rewards": [1, 2],
        "steps": [[_row(0, 0), _row(0, 0)]],
    }
    first, _ = compile_gold_replay_calendar_v3(replay, player=0, candidate_id=1)
    second, _ = compile_gold_replay_calendar_v3(replay, player=1, candidate_id=2)
    stacked = stack_route_calendars_v3((first, second))
    assert validate_route_calendar_v3(stacked) == []
    assert stacked.hand_target_by_day.shape == (2, 30)
    assert stacked.crop_target_by_day.shape == (2, 30, 5)
    assert stacked.animal_service_target_by_day.shape == (2, 30, 3)
    assert stacked.animal_care_policy_by_day.shape == (2, 30, 3)
    assert np.asarray(stacked.event_overflow_count).tolist() == [0, 0]


def test_market_template_ignores_empty_frames_and_uses_largest_transaction():
    empty = _row(3, 0)
    seed_only = _row(3, 1, market=[["BUY_SEED", "WHEAT", 2]])
    main_bundle = _row(
        3,
        2,
        market=[
            ["SELL", "FERTILIZER", 4],
            ["BUY_ANIMAL", "COW", 1],
            ["HIRE"],
            ["HIRE"],
        ],
    )
    replay = {
        "id": 9,
        "info": {"TeamNames": ["a", "gold"]},
        "rewards": [0, 1],
        "steps": [[{}, empty], [{}, seed_only], [{}, main_bundle]],
    }
    calendar, _ = compile_gold_replay_calendar_v3(
        replay, player=1, candidate_id=3
    )
    assert int(calendar.market_template_by_day[0, 3]) == int(
        M36MarketTemplateV3.SELL_HIRE_ANIMAL_SEED_FEED
    )

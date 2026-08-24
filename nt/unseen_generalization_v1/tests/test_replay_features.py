from __future__ import annotations

from unseen_generalization_v1.replay_features import reconstruct_observations, summarize_replay_payload


def _farm(money: int):
    tiles = [[None for _ in range(2)] for _ in range(2)]
    return {
        "money": money,
        "tiles": tiles,
        "farmer": [0, 0],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


def _full_obs(step: int, money0: int = 3000, money1: int = 3000):
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [_farm(money0), _farm(money1)],
        "private": {"shed": {}, "seeds": {}, "inventories": [[]]},
        "market": {
            "inventory": {"MILK": 100, "WOOL": 100},
            "prices": {"MILK": 160, "WOOL": 200},
        },
        "town": {"unlocked_shops": []},
    }


def _payload():
    return {
        "id": 123,
        "configuration": {"seed": 77},
        "agents": [{"submissionId": 11}, {"submissionId": 22}],
        "steps": [
            [
                {
                    "observation": _full_obs(0),
                    "action": {
                        "farmer": ["PASS"],
                        "hands": [],
                        "market": [["BUY_ANIMAL", "COW", 2]],
                    },
                    "reward": 0,
                    "status": "ACTIVE",
                },
                {
                    "observation": {
                        "player": 1,
                        "private": {"shed": {}, "seeds": {}, "inventories": [[]]},
                    },
                    "action": {
                        "farmer": ["PASS"],
                        "hands": [],
                        "market": [["BUY_ANIMAL", "SHEEP", 3]],
                    },
                    "reward": 0,
                    "status": "ACTIVE",
                },
            ],
            [
                {
                    "observation": _full_obs(1, 2500, 2400),
                    "action": {
                        "farmer": ["EAST"],
                        "hands": [],
                        "market": [["SELL", "MILK", 4]],
                    },
                    "reward": 0,
                    "status": "ACTIVE",
                },
                {
                    "observation": {"player": 1, "private": {"shed": {"WOOL": 5}}},
                    "action": {
                        "farmer": ["WEST"],
                        "hands": [],
                        "market": [["SELL", "WOOL", 5]],
                    },
                    "reward": 0,
                    "status": "ACTIVE",
                },
            ],
            [
                {
                    "observation": _full_obs(24, 5000, 5200),
                    "action": {"farmer": ["PASS"], "hands": [], "market": []},
                    "reward": 5000,
                    "status": "DONE",
                },
                {
                    "observation": {"player": 1},
                    "action": {"farmer": ["PASS"], "hands": [], "market": []},
                    "reward": 5200,
                    "status": "DONE",
                },
            ],
        ],
    }


def test_reconstructs_second_seat_public_delta():
    payload = _payload()
    observations, warnings = reconstruct_observations(payload["steps"])
    assert observations[1][1]["step"] == 1
    assert observations[1][1]["farms"][1]["money"] == 2400
    assert observations[1][1]["private"]["shed"]["WOOL"] == 5
    assert "seat_delta_public_fields_reconstructed" in warnings


def test_extracts_behavior_and_family_signatures():
    summary = summarize_replay_payload(_payload())
    assert summary.episode_id == "123"
    assert summary.episode_seed == 77
    assert summary.submission_ids == ("11", "22")
    assert summary.step_count == 3
    assert summary.players[0].market_items_bought["COW"] == 2
    assert summary.players[0].market_items_sold["MILK"] == 4
    assert summary.players[0].coarse_family == "MILK_PRIMARY"
    assert summary.players[1].market_items_bought["SHEEP"] == 3
    assert summary.players[1].market_items_sold["WOOL"] == 5
    assert summary.players[1].coarse_family == "WOOL_PRIMARY"
    assert len(summary.players[0].behavior_signature) == 64
    assert summary.players[0].final_reward == 5000.0

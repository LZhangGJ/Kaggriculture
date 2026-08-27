from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from import_nt_route_banks import collect_route_ids, decode_route  # noqa: E402


def _bank() -> dict[str, np.ndarray]:
    bank = {
        "unit_op": np.zeros((1, 719, 33), dtype=np.int8),
        "unit_item": np.full((1, 719, 33), -1, dtype=np.int8),
        "unit_amount": np.ones((1, 719, 33), dtype=np.int32),
        "unit_count": np.ones((1, 719), dtype=np.int8),
        "market_op": np.zeros((1, 719, 10), dtype=np.int8),
        "market_item": np.full((1, 719, 10), -1, dtype=np.int8),
        "market_amount": np.zeros((1, 719, 10), dtype=np.int32),
        "market_count": np.zeros((1, 719), dtype=np.int8),
    }
    bank["unit_count"][0, 0] = 2
    bank["unit_op"][0, 0, 0] = 8
    bank["unit_item"][0, 0, 0] = 0
    bank["unit_op"][0, 0, 1] = 6
    bank["unit_item"][0, 0, 1] = 9
    bank["unit_amount"][0, 0, 1] = 2
    bank["market_count"][0, 0] = 3
    bank["market_op"][0, 0, :3] = [1, 5, 6]
    bank["market_item"][0, 0, :3] = [-1, 11, 7]
    bank["market_amount"][0, 0, :3] = [0, 3, 4]
    return bank


def test_decode_route_preserves_farmer_hands_and_market_orders() -> None:
    tape = decode_route(_bank(), 0)

    assert len(tape) == 719
    assert tape[0] == {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [["PICKUP", "GOOSE", 2]],
        "market": [["HIRE"], ["BUY_ANIMAL", "SHEEP", 3], ["SELL", "WOOL", 4]],
    }
    assert tape[1] == {"farmer": ["PASS"], "hands": [], "market": []}


def test_collect_route_ids_ignores_scores_shop_indices_and_capacity() -> None:
    payload = {
        "route_ids": [4, 9],
        "route_capacity": 256,
        "seed": 1234,
        "routes": [
            {"first_shop": 7, "base_route": 11, "score": 0.92},
            {"second_routes": [[12, 13]], "games": 512},
        ],
        "metadata": {"source_route_count": 99},
    }

    assert collect_route_ids(payload) == {4, 9, 11, 12, 13}


def test_collect_route_ids_indexes_only_reachable_legacy_second_shop_rows() -> None:
    payload = {
        "first_route_ids": [1, 3],
        "second_route_ids": [
            [90, 90],
            [10, 11],
            [91, 91],
            [12, 13],
            [92, 92],
        ],
        "routes": [{"eligible_routes": [80], "selected_route": 12}],
    }

    assert collect_route_ids(payload) == {1, 3, 10, 11, 12, 13}

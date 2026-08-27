from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_causal_market_tree import _metrics, select_public_route  # noqa: E402


def test_public_route_selection_matches_causal_triggers() -> None:
    config = SimpleNamespace(
        yarn_first_start=88, yarn_second_start=153,
        yarn_third_enabled=True, yarn_third_start=216,
        yarn_third_prefixes=(("BRUNCH_SPOT", "PET_CAFE"),),
        yarn_third_pet_brunch_veto=True,
        bakery_capital_enabled=True, bakery_capital_start=160,
        bakery_capital_second_shops=("PIZZA_SHOP",),
        bakery_capital_minimum_cows=3, bakery_capital_minimum_sheep=2,
        bakery_capital_minimum_melons=10, bakery_capital_maximum_geese=0,
    )
    assets = {"COW": 3, "SHEEP": 2, "MELON": 10, "GOOSE": 0}
    assert select_public_route("YARN_STORE", None, None, assets, config) == (
        "yarn_first", 88
    )
    assert select_public_route("BAKERY", "YARN_STORE", None, assets, config) == (
        "yarn_second", 153
    )
    assert select_public_route("BAKERY", "PIZZA_SHOP", None, assets, config) == (
        "bakery_capital", 160
    )
    assert select_public_route(
        "BRUNCH_SPOT", "PET_CAFE", "YARN_STORE", assets, config
    ) == ("yarn_third", 216)
    assert select_public_route(
        "PET_CAFE", "BRUNCH_SPOT", "YARN_STORE", assets, config
    ) == ("default", -1)


def test_group_metrics_only_pair_complete_seed_rows() -> None:
    margins = np.asarray([10.0, 20.0, -5.0, 30.0, 40.0, -10.0])
    selected = np.asarray([True, True, True, False, False, True])

    metrics = _metrics(margins, selected)

    assert metrics["samples"] == 4
    assert metrics["paired_seed_samples"] == 1
    assert metrics["raw_win_rate"] == 0.5
    assert metrics["both_seats_win_rate"] == 1.0

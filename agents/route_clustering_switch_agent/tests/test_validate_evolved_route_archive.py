from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from validate_evolved_route_archive import _opponent_shortlist  # noqa: E402


def test_opponent_shortlist_keeps_specialists_outside_global_ranking() -> None:
    rows = [
        {
            "name": "robust",
            "opponent_raw_win_rates": {"A": .8, "B": .8},
            "opponent_mean_margins": {"A": 1, "B": 1},
        },
        {
            "name": "a-specialist",
            "opponent_raw_win_rates": {"A": 1.0, "B": .1},
            "opponent_mean_margins": {"A": 2, "B": -2},
        },
        {
            "name": "b-specialist",
            "opponent_raw_win_rates": {"A": .1, "B": 1.0},
            "opponent_mean_margins": {"A": -2, "B": 2},
        },
    ]

    selected = _opponent_shortlist(rows, ("A", "B"), 1)

    assert [row["name"] for row in selected] == ["a-specialist", "b-specialist"]

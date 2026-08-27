from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from analyze_compact_switch_search import summarize_by_opponent  # noqa: E402


def test_per_opponent_ranking_uses_raw_wins_and_builds_union() -> None:
    # target, opponent, seed, seat: B counters O0; A counters O1.
    outcome = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    outcome[1, 0] = 2
    outcome[0, 1] = 2
    margin = np.zeros_like(outcome, dtype=np.float32)

    rows, union = summarize_by_opponent(
        outcome, margin, ["A", "B"], ["O0", "O1"], stay_index=0, limit=1
    )

    assert rows[0]["top_static_targets"][0]["family"] == "B"
    assert rows[1]["top_static_targets"][0]["family"] == "A"
    assert rows[0]["oracle"]["raw_win_rate"] == 1.0
    assert rows[1]["static_routes_at_or_above_90pct"] == 1
    assert rows[0]["greedy_raw_win_portfolio"][0]["family"] == "B"
    assert rows[0]["greedy_raw_win_coverage"] == 1.0
    assert union == ["B", "A"]

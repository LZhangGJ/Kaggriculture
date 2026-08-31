from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "run_block_mvp_target_tail_oracle.py"
)
SPEC = importlib.util.spec_from_file_location("block_mvp_target_tail_oracle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
V2 = MODULE.v2


def test_tail_route_switch_changes_only_at_step_216():
    args = ("ROUTED", 10, 20, 30, 40)
    assert V2._own_route_index(95, *args) == 10
    assert V2._own_route_index(96, *args) == 30
    assert V2._own_route_index(215, *args) == 30
    assert V2._own_route_index(216, *args) == 40
    assert V2._own_route_index(718, *args) == 40
    assert V2._own_route_index(216, "RAW", 10, 20, None, 40) == 10


def _observation(seeds: int):
    farm = {
        "tiles": [[None for _ in range(10)] for _ in range(10)],
        "farmer": [1, 1],
        "hands": [],
        "unlocked_quadrants": ["NW"],
    }
    return {
        "player": 0,
        "farms": [farm, farm],
        "private": {
            "seeds": {"WHEAT": seeds},
            "shed": {},
            "inventories": [{}],
        },
    }


def test_post216_failure_locators_match_macro_semantics():
    plant = {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}
    assert V2._has_positional_unit_failure(_observation(0), plant)
    assert not V2._has_positional_unit_failure(_observation(1), plant)
    market = {
        "farmer": ["PASS"], "hands": [],
        "market": [["BUY_SEED", "WHEAT", 2]],
    }
    assert V2._has_market_failure(market, [1])
    assert not V2._has_market_failure(market, [2])


def _episode(target, tail, state, *, outcome, margin, unit, market, first, digest):
    return {
        "target": target,
        "tail": tail,
        "opponent": "O",
        "seed": state,
        "seat": 0,
        "outcome": outcome,
        "margin": margin,
        "macro_unit_failures": unit,
        "macro_market_failures": market,
        "first_post216_failure_step": first,
        "first_post216_unit_failure_step": first if unit else -1,
        "first_post216_market_failure_step": first if market else -1,
        "pre_tail_joint_trace_sha256": digest,
        "completed": True,
        "finite_rewards": True,
        "turns": V2.v1.HORIZON,
    }


def _rows():
    rows = []
    for target in V2.TARGET_IDS:
        rows.extend((
            _episode(
                target, MODULE.TAIL_B0, 1, outcome=0, margin=-10,
                unit=5, market=2, first=220, digest=f"{target}-1",
            ),
            _episode(
                target, MODULE.TAIL_TARGET, 1, outcome=2, margin=10,
                unit=3, market=1, first=-1, digest=f"{target}-1",
            ),
            _episode(
                target, MODULE.TAIL_B0, 2, outcome=2, margin=20,
                unit=1, market=0, first=-1, digest=f"{target}-2",
            ),
            _episode(
                target, MODULE.TAIL_TARGET, 2, outcome=2, margin=15,
                unit=2, market=2, first=230, digest=f"{target}-2",
            ),
        ))
    return rows


def test_tail_summary_reports_paired_oracle_and_target_selection():
    result = MODULE.summarize_tail_results(_rows(), "dev")
    assert result["states"] == 2
    assert result["games"] == 12
    assert result["all_finite_719"]
    row = result["targets"]["RB96_C01"]
    assert row["paired_target_minus_b0"]["post216_failure_repaired_states"] == 1
    assert row["paired_target_minus_b0"]["post216_failure_introduced_states"] == 1
    oracle = row["two_choice_oracle"]
    assert oracle["raw_win_gain_vs_b0_pp"] == 50
    assert oracle["target_tail_selection_rate"] == .5
    assert oracle["mean_margin_gain_vs_b0"] == 10
    pairs = MODULE.pair_tail_rows(_rows())
    state_two = next(
        row for row in pairs
        if row["target"] == "RB96_C01" and row["seed"] == 2
    )
    assert state_two["b0_outcome"] == state_two["target_outcome"] == 2
    assert state_two["oracle_choice"] == MODULE.TAIL_B0
    with patch.object(Path, "write_text") as write_text:
        MODULE._write_report(Path("REPORT.md"), result)
    rendered = write_text.call_args.args[0]
    assert "Only the >=216 tail differs" in rendered
    assert "select target" in rendered


def test_tail_pairing_rejects_pre216_divergence():
    rows = _rows()
    rows[1] = {**rows[1], "pre_tail_joint_trace_sha256": "different"}
    with pytest.raises(AssertionError, match="diverged before step 216"):
        MODULE.pair_tail_rows(rows)

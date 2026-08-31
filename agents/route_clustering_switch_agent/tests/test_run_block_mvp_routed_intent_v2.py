from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "run_block_mvp_routed_intent_v2.py"
)
SPEC = importlib.util.spec_from_file_location("block_mvp_routed_intent_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _pass_tape():
    return [
        {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}
        for _ in range(MODULE.v1.HORIZON)
    ]


def _observation(*, seeds=2, planted=()):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    for x, y in planted:
        tiles[y][x] = {"kind": "PLANT", "crop": "WHEAT"}
    farm = {
        "money": 1000,
        "tiles": tiles,
        "farmer": [1, 1],
        "hands": [[2, 2]],
        "unlocked_quadrants": [0],
    }
    return {
        "player": 0,
        "step": 0,
        "farms": [farm, dict(farm)],
        "private": {
            "shed": {},
            "seeds": {"WHEAT": seeds},
            "inventories": [{}, {}],
        },
        "market": {"prices": {"WHEAT": 10}},
    }


def _task(task_id, due, xy):
    return MODULE.RouteTask(
        task_id=task_id,
        due_step=due,
        activation_step=0,
        operation="PLANT",
        item="WHEAT",
        target_xy=xy,
        actor_index=None,
        grace_steps=24,
    )


def test_extract_masks_macro_actions_and_keeps_source_actor_as_provenance_only():
    tape = _pass_tape()
    tape[96]["farmer"] = ["PLANT", "WHEAT"]
    tape[96]["hands"][0] = ["BUILD_PASTURE"]
    positions = [((1, 1), (2, 2)) for _ in range(MODULE.v1.HORIZON)]
    masked, tasks, rows = MODULE.extract_positional_tasks(
        tape, positions, "T", lookahead=24, grace_steps=24
    )
    assert masked[96]["farmer"] == ["PASS"]
    assert masked[96]["hands"] == [["PASS"]]
    assert [task.target_xy for task in tasks] == [(1, 1), (2, 2)]
    assert all(task.actor_index is None for task in tasks)
    assert [row["source_actor"] for row in rows] == [0, 1]


def test_routed_carrier_uses_keep_units_and_target_static_market():
    baseline = _pass_tape()
    target = _pass_tape()
    baseline[96]["farmer"] = ["PLANT", "WHEAT"]
    baseline[96]["hands"][0] = ["NORTH"]
    baseline[96]["market"] = [["SELL", "WOOD", 1]]
    target[96]["farmer"] = ["SOUTH"]
    target[96]["hands"][0] = ["EAST"]
    target[96]["market"] = [["BUY_SEED", "WHEAT", 2]]
    routed = MODULE.compose_routed_carrier(baseline, target)
    assert routed[96]["farmer"] == ["PASS"]
    assert routed[96]["hands"] == [["NORTH"]]
    assert routed[96]["market"] == [["BUY_SEED", "WHEAT", 2]]


def test_jit_assignment_does_not_occupy_actor_at_activation():
    router = MODULE.ConcurrentIntentRouter((_task("late", 10, (5, 1)),))
    base = {"farmer": ["NORTH"], "hands": [["PASS"]], "market": []}
    assert router.route(_observation(), base, 5)["farmer"] == ["NORTH"]
    assert router.active == {}
    action = router.route(_observation(), base, 6)
    assert router.active
    assert action["farmer"] != ["NORTH"]


def test_concurrent_router_uses_two_actors_and_reserves_same_step_seeds():
    tasks = (_task("a", 0, (1, 1)), _task("b", 0, (2, 2)))
    base = {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}

    enough = MODULE.ConcurrentIntentRouter(tasks)
    action = enough.route(_observation(seeds=2), base, 0)
    assert action["farmer"][:2] == ["PLANT", "WHEAT"]
    assert action["hands"][0][:2] == ["PLANT", "WHEAT"]
    enough.observe_after(_observation(seeds=0, planted=((1, 1), (2, 2))), 0)
    assert set(enough.executed_steps) == {"a", "b"}

    scarce = MODULE.ConcurrentIntentRouter(tasks)
    static_market = [["BUY_SEED", "WHEAT", 2]]
    action = scarce.route(
        _observation(seeds=1), {**base, "market": static_market}, 0
    )
    unit_orders = [action["farmer"], *action["hands"]]
    assert sum(order[0] == "PLANT" for order in unit_orders) == 1
    assert sum(order[0] == "PASS" for order in unit_orders) == 1
    assert action["market"] == static_market
    assert scarce.snapshot()["market_orders_added"] == 0


def test_duplicate_satisfied_intent_is_not_counted_twice_or_forced_to_pass():
    tasks = (_task("first", 0, (1, 1)), _task("repeat", 1, (1, 1)))
    router = MODULE.ConcurrentIntentRouter(tasks)
    base = {"farmer": ["NORTH"], "hands": [["PASS"]], "market": []}
    observation = _observation(seeds=0, planted=((1, 1),))
    action = router.route(observation, base, 0)
    assert action["farmer"] == ["NORTH"]
    assert router.already_satisfied_steps == {"first": 0}
    router.route(observation, base, 1)
    assert "repeat" not in router.realized_steps
    router.finalize(MODULE.v1.STOP)
    assert router.unfinished_task_ids == ["repeat"]


def _episode(mode, target, unit, market, outcome, *, routed=False):
    return {
        "mode": mode,
        "target": target,
        "opponent": "O",
        "seed": 1,
        "seat": 0,
        "outcome": outcome,
        "margin": 1.0 if outcome == 2 else -1.0,
        "completed": True,
        "finite_rewards": True,
        "macro_unit_failures": unit,
        "macro_market_failures": market,
        "macro_failures": unit + market,
        "intent_total": 10 if routed else 0,
        "intent_realized": 10 if routed else 0,
        "intent_executed": 8 if routed else 0,
        "intent_already_satisfied": 2 if routed else 0,
        "intent_failed": 0,
        "intent_unfinished": 0,
        "key_intent_total": 2 if routed else 0,
        "key_intent_realized": 2 if routed else 0,
        "intent_delays": [0] * 10 if routed else [],
        "router_runtime": {
            "non_pass_base_unit_overrides": 3 if routed else 0,
        },
    }


def test_whole_episode_unit_gate_and_overlay_diagnostics_are_reported():
    rows = [_episode("KEEP", "B0_KEEP", 0, 0, 0)]
    for target in MODULE.TARGET_IDS:
        raw_unit = 0 if target == "RB96_C02" else 10
        rows.append(_episode("RAW", target, raw_unit, 0, 0))
        rows.append(_episode("ROUTED", target, 0, 99, 2, routed=True))
    report = MODULE.summarize_results(rows, "dev")
    assert report["status"] == "passed_positional_subgate"
    assert report["gates"][
        "R2_c00_c01_unit_failure_reduction_90pct_state_5pct_mean_0_25"
    ]
    diagnostic = report["targets"]["RB96_C00"]["non_gating_market_diagnostic"]
    assert diagnostic["routed_excess_failures_vs_keep"] == 99
    target = report["targets"]["RB96_C00"]
    assert target["gating_metric"] == (
        "whole_episode_paired_excess_positional_unit_failures"
    )
    overlay = target["non_gating_runtime_overlay_diagnostic"]
    assert overlay["non_pass_base_unit_overrides_total"] == 3
    assert overlay["mean_non_pass_base_unit_overrides_per_episode"] == 3
    assert report["deployment_eligible"] is False
    with patch.object(Path, "write_text") as write_text:
        MODULE._write_report(Path("REPORT.md"), report, {"pairing": "paired"})
    rendered = write_text.call_args.args[0]
    assert "whole-episode paired excess positional unit-macro failures" in rendered
    assert "non-PASS B0 overrides total / mean" in rendered
    assert "3 / 3.00" in rendered

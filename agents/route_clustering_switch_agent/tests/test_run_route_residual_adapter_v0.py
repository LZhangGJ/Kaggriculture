from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FAST_PACKAGE = Path(__file__).resolve().parents[1] / "fast_kaggriculture" / "python"
for path in (SCRIPTS, FAST_PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fast_kaggriculture import Config, FastEnv
import run_route_residual_adapter_v0 as residual
from meta_agent.src.route_switch_features import RouteSwitchHistory


def _pass() -> dict[str, object]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _tape() -> list[dict[str, object]]:
    return [_pass() for _ in range(residual.HORIZON)]


def test_six_residuals_have_exact_one_turn_semantics() -> None:
    step = 216
    base = _tape()
    baseline = _tape()
    base[step] = {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [["WATER"]],
        "market": [
            ["BUY_SEED", "WHEAT", 3],
            ["SELL", "EGG", 2],
        ],
    }
    baseline[step] = {
        "farmer": ["DIG"],
        "hands": [["PASS"]],
        "market": [["HIRE"]],
    }

    candidates = residual.residual_candidates(base, baseline, step)
    assert [candidate.code for candidate in candidates] == list(residual.EDIT_CODES)
    assert candidates[0].raw_action is None
    by_code = {candidate.code: candidate.effective_raw for candidate in candidates}
    assert by_code["B0_UNITS"]["farmer"] == ["DIG"]
    assert by_code["B0_UNITS"]["market"] == by_code["KEEP"]["market"]
    assert by_code["B0_MARKET"]["farmer"] == by_code["KEEP"]["farmer"]
    assert by_code["B0_MARKET"]["market"] == [["HIRE"]]
    assert by_code["B0_FULL"] == residual.normalize_player_action(baseline[step])
    assert by_code["DROP_BUYS"]["market"] == [["SELL", "EGG", 2]]
    assert by_code["CLEAR_MARKET"]["market"] == []

    units, unit_counts, market, market_counts = residual.pack_candidates(candidates)
    assert units.dtype == market.dtype == np.int32
    assert unit_counts.tolist()[0] == market_counts.tolist()[0] == -1
    assert unit_counts.tolist()[1:] == [2] * 5
    assert market_counts.tolist() == [-1, 2, 1, 1, 1, 0]


def test_identical_residuals_alias_to_keep() -> None:
    tape = _tape()
    candidates = residual.residual_candidates(tape, tape, 300)
    assert len(candidates) == 1
    assert candidates[0].code == "KEEP"
    assert candidates[0].aliases == residual.EDIT_CODES
    assert candidates[0].raw_action is None


def test_feature_panel_has_frozen_shape_and_both_boards() -> None:
    tape = _tape()
    tape[216]["market"] = [["BUY_SEED", "WHEAT", 2]]
    env = FastEnv(Config(), 20260828)
    observation = dict(env.observation(0))
    observation.update(step=216, day=9, hour=0, player=0)
    history = RouteSwitchHistory()
    history.update(observation)
    candidate = residual.residual_candidates(tape, _tape(), 216)[0]
    base = residual.normalize_player_action(tape[216])
    row = residual.candidate_features(
        observation, history, tape, base,
        residual.normalize_player_action(_pass()), candidate, 0, 216,
    )
    assert row.shape == (len(residual.FEATURE_NAMES),)
    assert len(residual.FEATURE_NAMES) == 556
    assert np.isfinite(row).all()
    vector = residual.action_vector(base, observation)
    assert vector.shape == (64,)


def test_composite_plan_crosses_the_239_240_route_boundary() -> None:
    baseline = _tape()
    left = _tape()
    right = _tape()
    right[240]["market"] = [["HIRE"]]
    genome = tuple(["left", "right", *(["left"] * 19)])
    composed = residual.scheduled_tape(
        baseline, {"left": left, "right": right}, genome,
    )
    assert composed[239] is left[239]
    assert composed[240] is right[240]
    assert residual.execution_route_ids(genome)[23:25] == ["left", "right"]

    env = FastEnv(Config(), 20260828)
    observation = dict(env.observation(0))
    observation.update(step=239, day=9, hour=23, player=0)
    history = RouteSwitchHistory()
    history.update(observation)
    from meta_agent.src.route_switch_features import route_switch_vector
    composite_vector = route_switch_vector(observation, history, composed)
    stale_gene_vector = route_switch_vector(observation, history, left)
    assert composite_vector[129] > stale_gene_vector[129]


def test_feature_panel_rejects_mismatched_observation_step() -> None:
    tape = _tape()
    env = FastEnv(Config(), 20260828)
    observation = dict(env.observation(0))
    observation.update(step=0, player=0)
    history = RouteSwitchHistory()
    history.update(observation)
    candidate = residual.residual_candidates(tape, tape, 216)[0]
    with np.testing.assert_raises(ValueError):
        residual.candidate_features(
            observation, history, tape,
            residual.normalize_player_action(_pass()),
            residual.normalize_player_action(_pass()),
            candidate, 0, 216,
        )


def test_oof_decision_metric_keeps_groups_contiguous() -> None:
    arrays = {
        "decision": np.asarray([0, 0, 1, 1], np.int32),
        "edit": np.asarray([0, 1, 0, 1], np.int8),
        "delta_margin": np.asarray([0.0, 5.0, 0.0, -3.0]),
    }
    mean = np.asarray([0.0, 2.0, 0.0, 1.0])
    std = np.zeros(4)
    positive = np.asarray([0.0, 1.0, 0.0, 1.0])
    metrics = residual._decision_metrics(
        arrays, mean, std, positive, beta=0.0,
        threshold=1.5, min_positive=0.5,
    )
    assert metrics["selected"] == 1
    assert metrics["mean_realized_delta_margin"] == 2.5
    assert metrics["harmful_rate"] == 0.0


def test_oof_decision_metric_rejects_noncontiguous_panel() -> None:
    arrays = {
        "decision": np.asarray([0, 1, 0], np.int32),
        "edit": np.asarray([0, 0, 1], np.int8),
        "delta_margin": np.zeros(3),
    }
    with np.testing.assert_raises(ValueError):
        residual._decision_metrics(
            arrays, np.zeros(3), np.zeros(3), np.zeros(3),
            beta=0.0, threshold=0.0, min_positive=0.5,
        )


def test_markdown_report_keeps_claim_and_evidence_boundary() -> None:
    def summary(wins: int, delta: float, edits: int) -> dict[str, object]:
        return {
            "states": 4, "raw_win_rate": wins / 4,
            "mean_margin": 1000.0 + delta,
            "mean_delta_margin_vs_fixed": delta,
            "loss_repairs_to_win": int(delta > 0),
            "outcome_regressions_vs_fixed": 0, "edits": edits,
        }

    text = residual._report_markdown({
        "status": "residual_adapter_feasible",
        "validation": {
            "fixed": summary(2, 0.0, 0),
            "learned": summary(3, 50.0, 8),
            "oracle": summary(4, 200.0, 20),
        },
        "splits": {
            "train_seeds": [1, 2], "validation_seeds": [3],
            "opponents": ["NR020"],
        },
        "gates": {"G0": True},
        "sealed_test_executed": False,
    })
    assert "| learned | 3/4 (75.00%)" in text
    assert "not the full-pool 90/80 championship claim" in text
    assert "do not retain a separate before/after identity snapshot" in text

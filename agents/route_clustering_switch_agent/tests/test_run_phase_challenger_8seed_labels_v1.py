from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_8seed_labels_v1 as runner


def _arm(shas: list[str], markets: list[int], best: str, margin: float) -> dict:
    return {
        "candidate_sha256": shas,
        "market_diff": markets,
        "all_candidates_oracle": {
            "best_sha256": best,
            "best_outcome": 2,
            "best_margin": margin,
        },
    }


def _decision(state_id: str = "NT0056:2026086300:0:216:1") -> dict:
    return {
        "state_id": state_id,
        "split": "train_proxy",
        "opponent": "NT0056",
        "seed": 2026086300,
        "seat": 0,
        "anchor": 216,
        "offset": 1,
        "decision_step": 217,
        "arms": {
            runner.preflight.R0: _arm(["keep", "a"], [0, 0], "a", 5.0),
            runner.preflight.PHASE: _arm(
                ["keep", "a", "p"], [0, 0, 1], "p", 6.0,
            ),
            "R2_all": _arm(
                ["keep", "a", "p"], [0, 0, 1], "p", 6.0,
            ),
        },
    }


def _candidate(
    state_id: str, index: int, sha: str, margin: float, market: int,
) -> dict:
    return {
        "state_id": state_id,
        "candidate_index": index,
        "candidate_sha256": sha,
        "outcome": 2 if sha != "keep" else 1,
        "margin": margin,
        "market_diff": market,
        "path_pure": market == 0,
    }


def test_fixed_formal_and_smoke_state_budgets() -> None:
    formal = runner.scenario_contract(False)
    smoke = runner.scenario_contract(True)
    assert formal["states"] == 8064
    assert len(formal["scenarios"]) == 96
    assert tuple(formal["seeds"]) == tuple(range(2026086300, 2026086308))
    assert smoke["states"] == 8
    assert len(smoke["scenarios"]) == 2


def test_eight_seeds_map_to_four_balanced_folds() -> None:
    mapping = runner.seed_fold_map(range(2026086300, 2026086308))
    assert list(mapping.values()) == [0, 1, 2, 3, 0, 1, 2, 3]


def test_scenario_keeps_only_r0_prefix_plus_phase_only_labels() -> None:
    decision = _decision()
    state_id = decision["state_id"]
    decision["arms"]["R2_all"] = _arm(
        ["keep", "a", "p", "r2-only"], [0, 0, 1, 0], "r2-only", 100.0,
    )
    candidates = [
        _candidate(state_id, 0, "keep", 0.0, 0),
        _candidate(state_id, 1, "a", 5.0, 0),
        _candidate(state_id, 2, "p", 6.0, 1),
        _candidate(state_id, 3, "r2-only", 100.0, 0),
    ]
    value = runner.summarize_scenario(
        [decision], candidates, {2026086300: 0},
    )
    assert [row["candidate_sha256"] for row in value["canonical_rows"]] == [
        "keep", "a", "p",
    ]
    assert value["counts"]["phase_only_rows"] == 1
    assert value["counts"]["positive_novelty_decisions"] == 1
    assert value["phase_positive_market_changing_rows"] == 1
    assert value["physical_R2_candidate_rows_audited_then_discarded"] == 4


def test_merge_summary_preserves_fold_and_opponent_coverage() -> None:
    decision = _decision()
    state_id = decision["state_id"]
    candidates = [
        _candidate(state_id, 0, "keep", 0.0, 0),
        _candidate(state_id, 1, "a", 5.0, 0),
        _candidate(state_id, 2, "p", 6.0, 1),
    ]
    value = runner.summarize_scenario(
        [decision], candidates, {2026086300: 0},
    )
    total = runner.empty_summary()
    runner.merge_summary(total, value)
    runner.merge_summary(total, value)
    assert total["counts"]["decisions"] == 2
    assert total["coverage"]["opponent"]["NT0056"][
        "positive_novelty_decisions"
    ] == 2
    assert total["positive_seed_folds"] == {0}


def test_cli_exposes_no_split_or_seed_override() -> None:
    destinations = {action.dest for action in runner.parser()._actions}
    assert "opponent" not in destinations
    assert "seed_start" not in destinations
    assert "seed_count" not in destinations


def test_loader_lists_are_both_pinned_to_train_to_avoid_heldout_fallback() -> None:
    load_args = runner.experiment_load_args(runner.parser().parse_args([]))
    included = tuple(dict.fromkeys((
        *(load_args.train_opponent or runner.path_v1.TRAIN_OPPONENTS),
        *(load_args.validation_opponent or runner.path_v1.HELDOUT_OPPONENTS),
    )))
    assert included == runner.TRAIN_OPPONENTS
    assert not set(included) & set(runner.path_v1.HELDOUT_OPPONENTS)

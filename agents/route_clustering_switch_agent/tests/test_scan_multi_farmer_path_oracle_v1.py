from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import scan_multi_farmer_path_oracle_v1 as scan


def test_path_pure_selection_ignores_better_market_changing_arm() -> None:
    rewards = np.asarray([[10, 10], [100, 0], [20, 0]], np.float64)
    assert scan.select_path_pure_index(rewards, np.asarray([0, 1, 0]), 0) == 2
    assert scan.select_path_pure_index(
        np.asarray([[10, 0], [9, 0]], np.float64), np.asarray([0, 0]), 0,
    ) == 0


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "split": "train_proxy", "genome_id": "NR020", "anchor": 216,
        "offset": 1, "opponent": "NT0056", "seed": 1, "seat": 0,
        "candidate_count": 4,
        "path_pure_candidate_count": 3, "market_diff_candidate_count": 1,
        "market_diff_step_count": 2, "keep_equivalent": True,
        "strict_headroom": False, "best_delta_margin": 0.0,
        "delta_outcome": 0, "loss_to_win_repair": False,
        "delta_self_reward": 0.0,
        "best_kind": "KEEP", "best_donor_id": None,
        "pair_beats_components_codes": [],
    }
    row.update(updates)
    return row


def test_summary_counts_headroom_market_filter_and_groups() -> None:
    summary = scan.summarize_decisions([
        _row(),
        _row(
            split="heldout_proxy", anchor=240, offset=2, opponent="NT0557",
            seat=1, strict_headroom=True, best_delta_margin=12.0,
            best_kind="PAIR", best_donor_id="d1",
            pair_beats_components_codes=["p1"],
            delta_outcome=2, loss_to_win_repair=True, delta_self_reward=7.0,
            candidate_count=5, path_pure_candidate_count=5,
            market_diff_candidate_count=0, market_diff_step_count=0,
        ),
    ])
    assert summary["overall"]["decisions"] == 2
    assert summary["overall"]["candidate_rollouts"] == 9
    assert summary["overall"]["headroom_rate"] == 0.5
    assert summary["overall"]["market_diff_candidates"] == 1
    assert summary["overall"]["pair_beats_components_events"] == 1
    assert summary["overall"]["pair_beats_components_decisions"] == 1
    assert summary["overall"]["outcome_improvement_decisions"] == 1
    assert summary["overall"]["loss_to_win_repair_decisions"] == 1
    assert summary["overall"]["loss_to_win_repair_scenarios"] == 1
    assert summary["overall"]["max_self_reward_delta"] == 7.0
    assert summary["headroom_best_kind"] == {"PAIR": 1}
    assert summary["anchors_with_headroom"] == [240]
    assert summary["by_split"]["heldout_proxy"]["headroom_decisions"] == 1

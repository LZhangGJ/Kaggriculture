from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_contract_retrieval_oracle_ablation_v1 as ablation


def test_nearest_distinct_scans_past_duplicate_prototypes() -> None:
    ids, inspected, distances = ablation.nearest_distinct_blocks(
        ["a", "a", "b", "c", "d", "e", "f", "g", "h"],
        np.arange(9, dtype=np.float32), 8,
    )
    assert ids == list("abcdefgh")
    assert inspected == 9
    assert distances["a"] == 0.0


def test_retrieval_is_at_anchor_and_unit_decision_is_anchor_plus_one() -> None:
    assert ablation.retrieval_and_decision_steps(216) == (216, 217)
    assert ablation.retrieval_and_decision_steps(696) == (696, 697)


def _arm(headroom: bool, gain: float) -> dict[str, object]:
    return {
        "strict_headroom": headroom, "oracle_gain": gain,
        "loss_to_win": False, "candidate_count": 2,
        "market_diff_candidate_count": 0, "market_diff_step_count": 0,
    }


def test_summary_uses_r2_positive_states_as_recall_denominator() -> None:
    rows = [
        {
            "anchor": 216, "opponent": "x", "split": "train_proxy",
            "arms": {
                "R0_active8": _arm(False, 0.0),
                "R1_contract8": _arm(True, 4.0),
                "R2_all": _arm(True, 8.0),
            },
        },
        {
            "anchor": 240, "opponent": "x", "split": "train_proxy",
            "arms": {
                "R0_active8": _arm(False, 0.0),
                "R1_contract8": _arm(False, 0.0),
                "R2_all": _arm(False, 0.0),
            },
        },
    ]
    summary = ablation.summarize(rows)["overall"]
    assert summary["r2_positive_states"] == 1
    assert summary["arms"]["R0_active8"]["positive_state_recall"] == 0.0
    assert summary["arms"]["R1_contract8"]["positive_state_recall"] == 1.0
    assert summary["arms"]["R1_contract8"]["oracle_gain_retention"] == 0.5
    assert summary["arms"]["R0_active8"]["candidate_count_one_states"] == 0
    assert summary["arms"]["R1_contract8"]["effective_nonkeep_p90"] == 1.0


def test_contract_distance_ignores_market_environment_slice() -> None:
    query = np.zeros(147, np.float32)
    contracts = np.zeros((2, 147), np.float32)
    contracts[0, 86:104] = 9999
    contracts[1, 0] = 1
    distance = ablation.contract_distances(
        query, np.zeros(100, np.uint8), np.zeros(4, np.uint8),
        contracts, np.zeros((2, 100), np.uint8),
        np.zeros((2, 4), np.uint8), np.ones(147),
    )
    assert distance[0] == 0.0
    assert distance[1] > 0.0


def test_r2_builder_visits_every_anchor_block(monkeypatch) -> None:
    visited: list[str] = []

    def fake_variants(base, donor, actor_count, horizon, start_offset, rank, include_all):
        del base, actor_count, horizon, start_offset, rank, include_all
        visited.append(donor.block_id)
        return []

    monkeypatch.setattr(ablation.path_v1, "_donor_variants", fake_variants)
    donors = [
        ablation.path_v1.DonorBlock(
            block_id=f"d{index}", anchor=216, cluster_id="C00",
            source_provenance_id="p", source_route_id="r", support=1,
            actions=tuple({} for _ in range(24)), team_name="t",
        )
        for index in range(5)
    ]
    candidates = ablation._canonical_candidates(
        [{} for _ in range(719)], donors, 216, 1, 217,
    )
    assert visited == [donor.block_id for donor in donors]
    assert [candidate.code for candidate in candidates] == ["KEEP"]


def test_fast_diversity_selection_uses_union_coverage(monkeypatch) -> None:
    coverage = {
        "d0": ("a", "b"), "d1": ("b", "c"),
        "d2": ("d",), "d3": ("e",),
    }

    def fake_variants(base, donor, actor_count, horizon, start_offset, rank, include_all):
        del base, actor_count, horizon, start_offset, rank, include_all
        return [
            (name, "SINGLE", ((0, 0),), donor.block_id, "C00", 1, 0, 1, ((name,),))
            for name in coverage[donor.block_id]
        ]

    monkeypatch.setattr(ablation.path_v1, "_donor_variants", fake_variants)
    donors = tuple(
        ablation.path_v1.DonorBlock(
            block_id=f"d{index}", anchor=216, cluster_id="C00",
            source_provenance_id="p", source_route_id="r", support=1,
            actions=tuple({} for _ in range(24)), team_name="t",
        )
        for index in range(4)
    )
    selected, audit = ablation._select_diverse_fast(
        [{} for _ in range(719)], donors, 216, 1, 3, 217,
    )
    assert [donor.block_id for donor in selected] == ["d0", "d1", "d2"]
    assert audit["diversity_unique_residuals"] == 4


from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_dataset_preflight_v1 as preflight


def _oracle(sha: str, outcome: int, margin: float) -> dict:
    return {
        "best_sha256": sha,
        "best_outcome": outcome,
        "best_margin": margin,
    }


def _arm(shas: list[str], markets: list[int], best: str, margin: float) -> dict:
    return {
        "candidate_sha256": shas,
        "market_diff": markets,
        "all_candidates_oracle": _oracle(best, 2, margin),
    }


def _decision(
    *, phase_markets: list[int] | None = None,
) -> dict:
    return {
        "state_id": "opp:1:0:216:1",
        "split": "train_proxy",
        "opponent": "opp",
        "seed": 1,
        "seat": 0,
        "anchor": 216,
        "offset": 1,
        "decision_step": 217,
        "arms": {
            preflight.R0: _arm(["keep", "a"], [0, 1], "a", 5.0),
            preflight.PHASE: _arm(
                ["keep", "a", "p"],
                phase_markets or [0, 1, 0],
                "a",
                5.0,
            ),
        },
    }


def _labels() -> dict[str, dict]:
    return {
        "keep": {"outcome": 1, "margin": 0.0, "market_diff": 0, "path_pure": True},
        "a": {"outcome": 2, "margin": 5.0, "market_diff": 1, "path_pure": False},
        "p": {"outcome": 2, "margin": 5.0, "market_diff": 0, "path_pure": True},
    }


def test_canonical_union_is_exact_r0_prefix_plus_phase_only_suffix() -> None:
    state = preflight.canonical_union_index(_decision())
    assert state["r0_sha256"] == ["keep", "a"]
    assert state["phase_only_sha256"] == ["p"]
    assert state["union_sha256"] == ["keep", "a", "p"]


def test_shared_sha_market_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="market mask"):
        preflight.canonical_union_index(_decision(phase_markets=[0, 0, 0]))


def test_equal_rank_phase_only_cannot_displace_r0_first_winner() -> None:
    state = preflight.canonical_union_index(_decision())
    rows, audit = preflight.validate_state_labels(state, _labels())
    assert audit["best_union_sha256"] == "a"
    assert audit["phase_only_ties_with_best_R0"] == 1
    assert [row["membership"] for row in rows] == ["R0", "R0", "phase_only"]


def test_tied_phase_row_does_not_hide_another_strictly_better_phase_row() -> None:
    decision = _decision()
    decision["arms"][preflight.PHASE] = _arm(
        ["keep", "a", "p", "q"], [0, 1, 0, 0], "q", 6.0,
    )
    labels = _labels()
    labels["q"] = {
        "outcome": 2, "margin": 6.0, "market_diff": 0, "path_pure": True,
    }
    state = preflight.canonical_union_index(decision)
    _, audit = preflight.validate_state_labels(state, labels)
    assert audit["phase_only_ties_with_best_R0"] == 1
    assert audit["best_union_sha256"] == "q"


def test_feature_memory_uses_one_base_encoding_and_lazy_pair_chunk() -> None:
    value = preflight.memory_estimate(10, 4, 2, width=3)
    assert value["dense_materialization"]["total_bytes"] == (10 + 4 + 8) * 3 * 4
    assert value["shared_candidate_encoding_plus_lazy_pair"]["candidate_base_memmap_bytes"] == 14 * 3 * 4
    assert value["shared_candidate_encoding_plus_lazy_pair"]["maximum_pair_chunk_bytes"] == 2 * 3 * 2 * 4


def test_two_seed_panel_is_data_insufficient_for_four_fold_signal_gate() -> None:
    counts = preflight._new_counts()
    counts["positive_novelty_decisions"] = 12
    coverage = {
        "opponent": {
            str(index): {**preflight._new_counts(), "positive_novelty_decisions": 2, "deployment_target_rows": 3}
            for index in range(6)
        },
        "anchor": {
            "216": {**preflight._new_counts(), "positive_novelty_decisions": 6},
            "240": {**preflight._new_counts(), "positive_novelty_decisions": 6},
        },
        "offset": {},
        "seed": {},
    }
    failed = preflight.signal_gate(counts, coverage, [0, 1])
    assert failed["passed"] is False
    assert failed["status"] == "data_insufficient"
    assert failed["recommend_train_seeds"] == 8
    passed = preflight.signal_gate(counts, coverage, [0, 1, 2, 3])
    assert passed["passed"] is True

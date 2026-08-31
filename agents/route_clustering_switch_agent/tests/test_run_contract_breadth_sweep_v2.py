from __future__ import annotations

import copy
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_contract_breadth_sweep_v2 as sweep


def test_nearest_distinct_scans_duplicates_and_caps() -> None:
    prefixes, inspected = sweep.nearest_distinct_prefixes(
        ["a", "a", "b", "c", "d"],
        [0.0, 0.1, 0.2, 0.3, 0.4],
        (2, 8),
    )
    assert prefixes == {2: ["a", "b"], 8: ["a", "b", "c", "d"]}
    assert inspected == {2: 3, 8: 5}


def test_exact_bitset_selection_matches_objective_and_lex_tie() -> None:
    audit = sweep.exact_action_diverse_three(
        ["d", "c", "b", "a"],
        {"a": 0b0011, "b": 0b0100, "c": 0b1000, "d": 0b0001},
        {"a": 1, "b": 1, "c": 1, "d": 99},
    )
    assert audit.selected_ids == ("a", "b", "c")
    assert audit.unique_payloads == 4
    assert audit.combinations_checked == 4

    tied = sweep.exact_action_diverse_three(
        ["d", "c", "b", "a"],
        {block_id: 1 for block_id in "abcd"},
        {block_id: 0 for block_id in "abcd"},
    )
    assert tied.selected_ids == ("a", "b", "c")


def test_bitset_selected_ids_and_audit_match_slow_set_union() -> None:
    payloads = {
        "a": {"p0", "p1"},
        "b": {"p2"},
        "c": {"p3"},
        "d": {"p0"},
    }
    support = {"a": 1, "b": 1, "c": 1, "d": 99}
    universe = {value: index for index, value in enumerate(sorted(set().union(*payloads.values())))}
    masks = {
        block_id: sum(1 << universe[value] for value in values)
        for block_id, values in payloads.items()
    }
    selected, audit = sweep.select_diverse_ids(payloads, masks, support)
    slow = max(
        __import__("itertools").combinations(sorted(payloads), 3),
        key=lambda combo: (len(set().union(*(payloads[x] for x in combo))), sum(support[x] for x in combo)),
    )
    assert selected == slow == ("a", "b", "c")
    assert audit["diversity_ids"] == list(slow)
    assert audit["diversity_unique_residuals"] == 4
    assert audit["diversity_support"] == 3
    assert audit["support_top_ids"] == ["d", "a", "b"]
    assert audit["support_top_unique_residuals"] == 3


def _parity_rows() -> tuple[dict, dict]:
    arm = {field: 0 for field in sweep.K8_PARITY_FIELDS}
    arm["selected_donor_ids"] = ["x", "y", "z"]
    current = {
        "opponent": "opp",
        "seed": 1,
        "seat": 0,
        "anchor": 216,
        "keep_rewards": [10, 9],
        "k8_unique_block_ids": ["x", "y", "z"],
        "arms": {"K8_contract": copy.deepcopy(arm)},
    }
    old = {
        "opponent": "opp",
        "seed": 1,
        "seat": 0,
        "anchor": 216,
        "keep_rewards": [10, 9],
        "r1_unique_block_ids": ["x", "y", "z"],
        "arms": {"R1_contract8": copy.deepcopy(arm)},
    }
    return current, old


def test_k8_parity_accepts_equal_and_reports_value_drift() -> None:
    current, old = _parity_rows()
    assert sweep.k8_parity_mismatches([current], [old]) == []
    current["arms"]["K8_contract"]["candidate_count"] = 7
    mismatch = sweep.k8_parity_mismatches([current], [old])
    assert len(mismatch) == 1
    assert "candidate_count" in mismatch[0]


def _selection_row(split: str, anchor: int, gains: dict[int, float]) -> dict:
    arms = {"R0_active8": {"oracle_gain": 0.0}}
    arms.update({f"K{k}_contract": {"oracle_gain": gains[k]} for k in sweep.KS})
    return {"split": split, "anchor": anchor, "arms": arms}


def test_global_k_selection_uses_train_only_and_smaller_k_tie() -> None:
    rows = [
        _selection_row("train_proxy", 216, {8: 1, 16: 2, 24: 2, 32: 0}),
        _selection_row("heldout_proxy", 216, {8: 0, 16: 0, 24: 0, 32: 999}),
    ]
    selection = sweep.choose_global_k_train_only(rows)
    assert selection["chosen_k"] == 16
    assert selection["heldout_used_for_selection"] is False


def test_per_anchor_router_is_frozen_from_train() -> None:
    rows = [
        _selection_row("train_proxy", 216, {8: 1, 16: 3, 24: 0, 32: 0}),
        _selection_row("heldout_proxy", 216, {8: 0, 16: 0, 24: 0, 32: 999}),
    ]
    assert sweep.choose_per_anchor_router_train_only(rows) == {216: "K16_contract"}

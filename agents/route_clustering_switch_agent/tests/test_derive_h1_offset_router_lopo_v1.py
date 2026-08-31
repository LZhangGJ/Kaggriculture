from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import derive_h1_offset_router_lopo_v1 as router


def _view(
    gain: float,
    sha: str,
    *,
    loss_to_win: bool = False,
) -> dict[str, object]:
    return {
        "strict_headroom": gain > 0,
        "oracle_gain": gain if gain > 0 else 0.0,
        "best_outcome": 2 if gain > 0 else 1,
        "best_margin": gain,
        "best_sha256": sha,
        "loss_to_win": loss_to_win,
    }


def _arm(gain: float, shas: list[str], pure: list[bool] | None = None) -> dict:
    return {
        "candidate_sha256": shas,
        "path_pure_mask": pure if pure is not None else [True] * len(shas),
        router.PRIMARY_VIEW: _view(gain, shas[-1]),
    }


def _row(
    opponent: str,
    split: str,
    anchor: int,
    offset: int,
    r0_gain: float,
    phase_gain: float,
    *,
    phase_loss_to_win: bool = False,
) -> dict:
    r0 = _arm(r0_gain, ["keep", "shared"])
    phase = _arm(phase_gain, ["keep", "shared", "phase-only"])
    phase[router.PRIMARY_VIEW]["loss_to_win"] = phase_loss_to_win
    r2_gain = max(r0_gain, phase_gain, 1.0)
    return {
        "state_id": f"{opponent}:{anchor}:{offset}",
        "opponent": opponent,
        "split": split,
        "anchor": anchor,
        "offset": offset,
        "phase_pool_arm": "K24_contract",
        "arms": {
            router.R0: r0,
            router.PHASE: phase,
            router.R2: _arm(r2_gain, ["keep", "shared", "phase-only", "r2"]),
        },
    }


def test_pareto_choice_rejects_tradeoff_and_tie_but_accepts_non_worse_strict() -> None:
    tradeoff = [
        _row("a", "train_proxy", 216, 1, 100, 1),
        _row("b", "train_proxy", 216, 1, 0, 1),
    ]
    assert router.pareto_choice(tradeoff)[0] == router.R0
    tie = [_row("a", "train_proxy", 216, 1, 5, 5)]
    assert router.pareto_choice(tie)[0] == router.R0
    strict_gain = [_row("a", "train_proxy", 216, 1, 5, 6)]
    assert router.pareto_choice(strict_gain)[0] == router.PHASE


def test_lopo_requires_five_phase_votes_and_falls_back_to_r0() -> None:
    rows = []
    unstable = [35, 35, 1, 1, 1, 1]
    for index in range(6):
        opponent = f"t{index}"
        rows.append(_row(opponent, "train_proxy", 216, 1, 10, 12))
        rows.append(_row(opponent, "train_proxy", 216, 2, 10, unstable[index]))
    frozen, stability, _ = router.lopo_mapping(rows, lambda row: row["offset"])
    assert frozen == {1: router.PHASE, 2: router.R0}
    assert stability["1"]["phase_same_arm_fold_count"] == 6
    assert stability["2"]["full_train_choice"] == router.PHASE
    assert stability["2"]["phase_same_arm_fold_count"] == 4
    assert stability["2"]["fallback_reason"] == "phase_lopo_below_5_of_6"


def test_union_is_canonical_r0_first_and_shared_mask_must_match() -> None:
    row = _row("a", "train_proxy", 216, 1, 5, 5)
    audit = router.union_candidate_audit(row)
    assert audit["candidate_sha256"] == ["keep", "shared", "phase-only"]
    assert audit["extra_vs_r0"] == 1
    assert router._union_view(row)["source_arm"] == router.R0

    row["arms"][router.PHASE][router.PRIMARY_VIEW] = _view(6, "phase-only")
    assert router._union_view(row)["source_arm"] == router.PHASE
    assert router._union_view(row)["phase_only_challenger_won"] is True

    row["arms"][router.PHASE]["path_pure_mask"][0] = False
    with pytest.raises(ValueError, match="shared canonical SHA"):
        router.union_candidate_audit(row)


def test_union_has_recovery_without_regression_and_metrics_keep_loss_to_win_separate() -> None:
    recovery = _row(
        "a", "heldout_proxy", 216, 1, 0, 5, phase_loss_to_win=True,
    )
    regression_for_phase = _row("a", "heldout_proxy", 240, 1, 4, 0)
    rows = [recovery, regression_for_phase]
    phase = router.metrics(rows, router.PHASE)
    union = router.metrics(rows, router.UNION)
    assert phase["coverage_recoveries_vs_R0"] == 1
    assert phase["coverage_regressions_vs_R0"] == 1
    assert phase["loss_to_win_states"] == 1
    assert union["coverage_recoveries_vs_R0"] == 1
    assert union["coverage_regressions_vs_R0"] == 0
    assert router.union_no_regression_audit(rows)["passed"] is True


def test_heldout_rows_cannot_change_train_only_mapping() -> None:
    train = [
        _row(f"t{i}", "train_proxy", 216, 1, 10, 12)
        for i in range(6)
    ]
    hostile_heldout = [
        _row("h", "heldout_proxy", 216, 1, 1000, 0)
        for _ in range(20)
    ]
    before = router.lopo_mapping(train, lambda row: row["offset"])[0]
    filtered = [row for row in [*train, *hostile_heldout] if row["split"] == "train_proxy"]
    after = router.lopo_mapping(filtered, lambda row: row["offset"])[0]
    assert before == after == {1: router.PHASE}
    assert router._fine_key(train[0]) == "216:1"

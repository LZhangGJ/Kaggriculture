from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import derive_phase_breadth_lopo_stability_v1 as lopo


def _arm(gain: float, loss_to_win: bool = False) -> dict[str, object]:
    return {
        "oracle_gain": gain,
        "strict_headroom": gain > 0,
        "loss_to_win": loss_to_win,
    }


def _row(
    opponent: str,
    split: str,
    anchor: int,
    gains: dict[str, float],
    loss_to_win: str | None = None,
) -> dict[str, object]:
    arms = {
        arm: _arm(gains.get(arm, 0.0), arm == loss_to_win)
        for arm in lopo.ARMS
    }
    arms["R2_all"] = _arm(max(float(value["oracle_gain"]) for value in arms.values()))
    return {
        "opponent": opponent,
        "split": split,
        "anchor": anchor,
        "arms": arms,
    }


def test_arm_choice_ties_prefer_r0_then_smaller_k() -> None:
    choice, _ = lopo.choose_anchor_arms([
        _row("a", "train_proxy", 216, {
            lopo.R0: 5, "K8_contract": 5, "K16_contract": 5,
        }),
        _row("a", "train_proxy", 240, {
            lopo.R0: 1, "K8_contract": 5, "K16_contract": 5,
        }),
    ])
    assert choice == {216: lopo.R0, 240: "K8_contract"}


def _six_opponent_train() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    opponents = [f"t{i}" for i in range(6)]
    unstable_k8 = [35, 35, 0, 0, 0, 0]
    for index, opponent in enumerate(opponents):
        rows.append(_row(opponent, "train_proxy", 216, {
            lopo.R0: 10,
            "K16_contract": 12,
        }))
        rows.append(_row(opponent, "train_proxy", 240, {
            lopo.R0: 10,
            "K8_contract": unstable_k8[index],
        }))
    return rows


def test_stability_freezes_6_of_6_non_r0_and_falls_back_at_4_of_6() -> None:
    frozen, stability, _ = lopo.lopo_stable_mapping(_six_opponent_train())
    assert frozen == {216: "K16_contract", 240: lopo.R0}
    assert stability["216"]["candidate_same_arm_fold_count"] == 6
    assert stability["216"]["stable"] is True
    assert stability["240"]["full_train_choice"] == "K8_contract"
    assert stability["240"]["candidate_same_arm_fold_count"] == 4
    assert stability["240"]["fallback_reason"] == "lopo_same_arm_below_5_of_6"


def test_full_train_tie_with_r0_cannot_freeze_non_r0() -> None:
    rows = [
        _row(f"t{i}", "train_proxy", 216, {
            lopo.R0: 10, "K8_contract": 10,
        })
        for i in range(6)
    ]
    frozen, stability, _ = lopo.lopo_stable_mapping(rows)
    assert frozen == {216: lopo.R0}
    assert stability["216"]["full_train_choice"] == lopo.R0


def test_coverage_recoveries_and_regressions_are_separate_from_loss_to_win() -> None:
    rows = [
        _row("h", "heldout_proxy", 216, {
            lopo.R0: 0, "K8_contract": 5,
        }, loss_to_win="K8_contract"),
        _row("h", "heldout_proxy", 240, {
            lopo.R0: 4, "K8_contract": 0,
        }),
    ]
    value = lopo.metrics(rows, "K8_contract")
    assert value["coverage_recoveries_vs_R0"] == 1
    assert value["coverage_regressions_vs_R0"] == 1
    assert value["loss_to_win_states"] == 1


def test_heldout_cannot_change_mapping_and_equal_metrics_reject_hard_router() -> None:
    train = _six_opponent_train()
    heldout = [
        _row("h", "heldout_proxy", 216, {
            lopo.R0: 5, "K16_contract": 5, "K32_contract": 1000,
        }),
        _row("h", "heldout_proxy", 240, {
            lopo.R0: 5, "K8_contract": 1000,
        }),
    ]
    result = lopo.derive([*train, *heldout])
    assert result["frozen_stable_mapping"] == {
        "216": "K16_contract",
        "240": lopo.R0,
    }
    assert result["selection_contract"]["heldout_used_for_rule_or_threshold"] is False
    assert result["heldout_double_metric_beats_r0"] is False
    assert result["status"] == "reject_hard_phase_router"
    assert result["next_direction"] == "shared-parameter stage-conditioned learned retriever"

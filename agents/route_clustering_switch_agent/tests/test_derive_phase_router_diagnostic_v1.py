from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import derive_phase_router_diagnostic_v1 as diagnostic


def _arm(gain: float) -> dict[str, object]:
    return {
        "oracle_gain": gain, "strict_headroom": gain > 0,
        "loss_to_win": False,
    }


def _row(split: str, anchor: int, r0: float, r1: float) -> dict[str, object]:
    return {
        "split": split, "anchor": anchor,
        "arms": {
            "R0_active8": _arm(r0), "R1_contract8": _arm(r1),
            "R2_all": _arm(max(r0, r1)),
        },
    }


def test_mapping_uses_train_only_and_r0_wins_ties() -> None:
    rows = [
        _row("train_proxy", 216, 5, 1),
        _row("train_proxy", 240, 2, 2),
        _row("heldout_proxy", 216, 0, 1000),
        _row("heldout_proxy", 240, 0, 1000),
    ]
    result = diagnostic.derive(rows)
    assert result["frozen_anchor_arm"] == {
        "216": "R0_active8", "240": "R0_active8",
    }
    assert result["heldout_used_for_selection"] is False


def test_phase_router_heldout_metrics_use_frozen_anchor_mapping() -> None:
    rows = [
        _row("train_proxy", 216, 5, 1),
        _row("train_proxy", 240, 1, 4),
        _row("heldout_proxy", 216, 8, 0),
        _row("heldout_proxy", 240, 0, 6),
    ]
    heldout = diagnostic.derive(rows)["comparison"]["heldout_proxy"]
    assert heldout["phase_router"]["positive_state_recall"] == 1.0
    assert heldout["phase_router"]["oracle_gain_retention"] == 1.0

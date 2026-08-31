from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_sha_diagnostic_v2 as diagnostic


def _smoke_report(path: Path, decisions: int = 1) -> None:
    path.write_text(json.dumps({
        "schema": "phase-challenger-residual-group-quantile-real-feature-smoke-v1",
        "status": "real_feature_nested_residual_smoke_complete",
        "evidence_boundary": {
            "train_only": True, "validation_opened": False,
            "sealed_opened": False, "candidate_committed": False,
        },
        "panels": {"compact1078": {"decisions": decisions}},
    }), encoding="utf-8")


def test_sign_groups_include_zero_in_nonpositive_risk() -> None:
    result = diagnostic.sign_group_summary(
        [b"a", b"a", b"b", b"b"],
        np.asarray([2.0, 0.0, 3.0, -1.0]),
    )
    assert result["mixed_positive_negative_groups"] == 1
    assert result["mixed_positive_nonpositive_groups"] == 2


def test_sha_must_be_canonical_lowercase_hex() -> None:
    assert diagnostic._hex("a" * 64, "x") == "a" * 64
    with pytest.raises(ValueError, match="canonical"):
        diagnostic._hex("G" * 64, "x")


def test_choices_reject_negative_index(tmp_path: Path) -> None:
    choices = tmp_path / "choices_compact1078.npz"
    np.savez(choices, decision=np.asarray([0]), A_materialized_row=np.asarray([-1]))
    _smoke_report(tmp_path / "SMOKE_REPORT.json")
    metadata = {
        "decision": np.asarray([0]),
        "phase_only": np.asarray([False]),
    }
    with pytest.raises(ValueError, match="row bounds"):
        diagnostic.load_choices(choices, metadata)


def test_choices_reject_wrong_decision(tmp_path: Path) -> None:
    choices = tmp_path / "choices_compact1078.npz"
    np.savez(choices, decision=np.asarray([7]), A_materialized_row=np.asarray([0]))
    _smoke_report(tmp_path / "SMOKE_REPORT.json")
    metadata = {
        "decision": np.asarray([0]),
        "phase_only": np.asarray([False]),
    }
    with pytest.raises(ValueError, match="do not align"):
        diagnostic.load_choices(choices, metadata)


def test_materialized_boundary_rejects_validation(tmp_path: Path) -> None:
    label = tmp_path / "labels"
    materialized = tmp_path / "materialized"
    report = {
        "schema": "phase-challenger-direct-u-phase-compact-et-ab-v1",
        "status": "formal_materialization_complete_training_not_requested",
        "training_executed": False,
        "feature_allocation_executed": True,
        "input_label_root": str(label),
        "runtime_boundary": {
            "validation_opened": True, "sealed_opened": False,
            "candidate_committed": False,
        },
    }
    with pytest.raises(ValueError, match="train-only"):
        diagnostic.validate_materialized_boundary(
            report, label.resolve(), materialized.resolve(),
        )

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_bilinear_listwise_A_v2 as runner


def _sha(index: int) -> str:
    return hashlib.sha256(f"canonical-block-{index}".encode()).hexdigest()


def _payload(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray], np.ndarray, list[dict]]:
    metadata = {
        "decision": np.asarray([0, 0, 0, 1, 1, 1]),
        "union_index": np.asarray([0, 1, 2, 0, 1, 2]),
        "seed": np.asarray([2026086300] * 6),
        "seat": np.asarray([0, 0, 0, 1, 1, 1]),
        "opponent": np.zeros(6, np.int8),
        "step": np.full(6, 217),
        "phase_only": np.asarray([False, False, True, False, False, True]),
        "path_pure": np.ones(6, np.bool_),
        "delta_margin": np.asarray([0.0, 2.0, 3.0, 0.0, -1.0, 4.0]),
        "edit": np.asarray([0, 1, 1, 0, 1, 1]),
    }
    margins = (10.0, 12.0, 13.0, 20.0, 19.0, 24.0)
    labels = []
    for row in range(6):
        decision, union = divmod(row, 3)
        labels.append({
            "state_id": f"NT0000:2026086300:{decision}:216:1",
            "union_index": union, "seed": 2026086300, "seat": decision,
            "step": 217, "opponent": "NT0000", "path_pure": True,
            "membership": "phase_only" if union == 2 else "R0",
            "split": "train_proxy", "margin": margins[row],
            "outcome": 2 if union == 0 else 1,
            "candidate_sha256": _sha(union),
        })
    path = tmp_path / "canonical_union_labels.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels), encoding="utf-8")
    selected = ~metadata["phase_only"]
    return path, metadata, selected, labels


def test_canonical_join_is_state_union_row_strict_and_train_only(tmp_path: Path) -> None:
    path, metadata, selected, _ = _payload(tmp_path)
    arrays, audit = runner.join_canonical_labels(
        path, runner._sha256_file(path), metadata, selected,
        expected_rows=6, expected_decisions=2,
    )
    assert arrays["candidate_sha256"].shape == (4,)
    assert arrays["state_id"].tolist() == [
        "NT0000:2026086300:0:216:1", "NT0000:2026086300:0:216:1",
        "NT0000:2026086300:1:216:1", "NT0000:2026086300:1:216:1",
    ]
    assert audit["state_id_union_index_physical_order_verified"] is True
    assert audit["candidate_sha_semantic"] == "frozen executable four-step block"


def test_canonical_byte_tamper_fails_before_semantic_join(tmp_path: Path) -> None:
    path, metadata, selected, labels = _payload(tmp_path)
    frozen = runner._sha256_file(path)
    labels[1]["candidate_sha256"] = _sha(99)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        runner.join_canonical_labels(path, frozen, metadata, selected)


def test_canonical_wrong_union_order_fails_even_when_bytes_are_rehashed(tmp_path: Path) -> None:
    path, metadata, selected, labels = _payload(tmp_path)
    labels[1], labels[2] = labels[2], labels[1]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels), encoding="utf-8")
    with pytest.raises(ValueError, match="row order"):
        runner.join_canonical_labels(
            path, runner._sha256_file(path), metadata, selected,
        )


def test_formal_config_locks_steps_folds_features_versions_and_sources() -> None:
    audit = runner.verify_pre_registered_environment()
    assert audit["decision_steps_sha256"] == runner.EXPECTED_STEPS_HASH
    assert audit["seed_folds_sha256"] == runner.EXPECTED_FOLDS_HASH
    assert audit["feature_names_sha256"] == runner.EXPECTED_FEATURE_HASH
    assert audit["projection_sha256"] == runner.EXPECTED_PROJECTION_HASH
    assert audit["versions"] == runner.EXPECTED_VERSIONS
    assert runner.EXPECTED_CONFIG["features"]["primary_width"] == 74
    assert runner.EXPECTED_CONFIG["trainer"]["optimizer"] == "scipy.optimize.L-BFGS-B"


def test_existing_output_refuses_before_any_input_read(tmp_path: Path) -> None:
    output = tmp_path / "already_exists"
    output.mkdir()
    args = argparse.Namespace(
        output_root=output,
        label_root=tmp_path / "missing_labels",
        materialized_root=tmp_path / "missing_materialized",
    )
    with pytest.raises(FileExistsError):
        runner.run(args)



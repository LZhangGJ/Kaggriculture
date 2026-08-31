"""Run the train-only paired-seat DRO two-head residual smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

import run_phase_challenger_residual_nested_smoke_v1 as runner
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_residual_paired_seat_two_head_v0 as paired


SCHEMA = "phase-challenger-residual-paired-seat-two-head-smoke-v0"
DEFAULT_FINGERPRINTS = runner.DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_state_action_diagnostic_v2_20260829a"
)
DEFAULT_OUTPUT = runner.DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_residual_paired_seat_two_head_smoke_v0_20260829a"
)
_ORIGINAL_LOAD = runner.load_arrays
_FINGERPRINT_ROOT = DEFAULT_FINGERPRINTS


def load_arrays_with_groups(root, representation, decision_steps):
    arrays, panel, source_indices = _ORIGINAL_LOAD(
        root, representation, decision_steps,
    )
    report_path = _FINGERPRINT_ROOT / "DIAGNOSTIC_REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status")
        != "validated_train_only_state_action_diagnostic_complete"
        or report.get("evidence_boundary", {}).get("train_only") is not True
        or report.get("evidence_boundary", {}).get("validation_opened") is not False
        or report.get("evidence_boundary", {}).get("sealed_opened") is not False
    ):
        raise ValueError("paired-seat fingerprints crossed the train-only boundary")
    artifact = report["artifacts"]["derived_fingerprints"]
    fingerprint_path = _FINGERPRINT_ROOT / "derived_fingerprints.npz"
    if residual._sha256_file(fingerprint_path) != artifact["sha256"]:
        raise ValueError("paired-seat fingerprint artifact hash changed")
    with np.load(fingerprint_path, allow_pickle=False) as fingerprints:
        groups = fingerprints["observable_state337_action"]
    if np.any(source_indices < 0) or np.any(source_indices >= len(groups)):
        raise ValueError("materialized source row escaped fingerprint panel")
    metadata_path = root / "metadata.npz"
    with np.load(metadata_path, allow_pickle=False) as metadata:
        seat = np.asarray(metadata["seat"], np.int8)[source_indices]
    arrays[paired.GROUP_KEY] = groups[source_indices]
    arrays["seat"] = seat
    panel.update({
        "paired_seat_group_fingerprint_sha256": artifact["sha256"],
        "paired_seat_group_semantics": "observable state337 plus executable action SHA",
        "paired_seat_group_unique": len(set(map(bytes, groups[source_indices]))),
    })
    return arrays, panel, source_indices


def parser():
    result = runner.parser()
    result.add_argument(
        "--fingerprint-root", type=Path, default=DEFAULT_FINGERPRINTS,
    )
    result.set_defaults(
        output_root=DEFAULT_OUTPUT,
        representations=("compact1078",),
        trees=24,
    )
    return result


def run(args):
    global _FINGERPRINT_ROOT
    _FINGERPRINT_ROOT = args.fingerprint_root.resolve()
    runner.nested = paired
    runner.SCHEMA = SCHEMA
    runner.load_arrays = load_arrays_with_groups
    try:
        report = runner.run(args)
    finally:
        runner.load_arrays = _ORIGINAL_LOAD
    report["pipeline"] = (
        "paired-seat worst-case utility head + P(any-seat nonpositive) risk head; "
        "outer-train exact harm cache; phase rejection returns exact A"
    )
    report["fingerprint_contract"] = {
        "root": str(_FINGERPRINT_ROOT),
        "report_sha256": residual._sha256_file(
            _FINGERPRINT_ROOT / "DIAGNOSTIC_REPORT.json"
        ),
        "hash_is_not_a_model_feature": True,
    }
    residual._atomic_text(
        args.output_root.resolve() / "SMOKE_REPORT.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    for representation in args.representations:
        choices = args.output_root.resolve() / f"choices_{representation}.npz"
        if choices.is_file():
            residual._atomic_text(
                args.output_root.resolve()
                / f"choices_{representation}_PROVENANCE.json",
                json.dumps({
                    "schema": "paired-seat-two-head-choices-provenance-v0",
                    "choices_sha256": residual._sha256_file(choices),
                    "smoke_report_sha256_before_provenance_sidecar": None,
                    "materialized_report_sha256": residual._sha256_file(
                        args.materialized_root.resolve() / "FINAL_REPORT.json"
                    ),
                    "fingerprint_report_sha256": report[
                        "fingerprint_contract"
                    ]["report_sha256"],
                    "train_only": True,
                    "validation_opened": False,
                    "sealed_opened": False,
                }, ensure_ascii=False, indent=2) + "\n",
            )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

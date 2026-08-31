#!/usr/bin/env python3
"""Run the locked optimization-only stability study for formal control views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (Path(__file__).resolve().parent, CODE_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_phase_challenger_bilinear_listwise_A_v2 as frozen_runner
import train_phase_challenger_bilinear_listwise_optimizer_stability_v2_1_locked as base_locked
import train_phase_challenger_control_view_optimizer_stability_v2_1_locked as control


SCHEMA = "control-view-optimizer-stability-v2.1-locked-run"
DEFAULT_OUTPUT = (
    frozen_runner.DEFAULT_PARENT
    / "control_view_optimizer_stability_v2_1_locked_20260829a"
)
EXPECTED_CONTROL_TRAINER_SHA256 = "ceb3157690ee0a38f0ec9dc1457dfabf477d28e8f550e0bf76b2236ec579a438"
EXPECTED_BASE_LOCKED_SHA256 = "e9c81b4419e5c353f2416240ec71b9690d356324aa62bf533a32ec51a796c429"
EXPECTED_RUNNER_NORMALIZED_SHA256 = "ff50ca55906a9e879fe4bcda73959bedc90c4a8b875ffd28a64a3c4f434b2fb8"
FORBIDDEN_RESULT_KEYS = {
    "acceptance_gate", "all_decision_metrics", "A_choices", "outer_metrics",
    "rank_metrics", "screen_result", "selected_A_row",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = _sha256(resolved)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"locked artifact SHA mismatch: {resolved}")
    return {"path": str(resolved), "sha256": observed, "bytes": resolved.stat().st_size}


def _runner_normalized_sha256(path: Path) -> str:
    prefix = "EXPECTED_RUNNER_NORMALIZED_SHA256 = "
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1]:
        lines.pop()
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("runner normalized-hash marker changed")
    lines[matches[0]] = prefix + '"<normalized-self-hash>"'
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _all_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        result.update(map(str, value))
        for item in value.values():
            result.update(_all_keys(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            result.update(_all_keys(item))
    return result


def verify_locked_environment(args: argparse.Namespace) -> dict[str, Any]:
    sources = {
        "control_trainer": _artifact(
            Path(control.__file__), EXPECTED_CONTROL_TRAINER_SHA256,
        ),
        "base_locked_trainer": _artifact(
            Path(base_locked.__file__), EXPECTED_BASE_LOCKED_SHA256,
        ),
    }
    observed_runner = _runner_normalized_sha256(Path(__file__).resolve())
    if observed_runner != EXPECTED_RUNNER_NORMALIZED_SHA256:
        raise ValueError("control runner normalized source hash changed")
    dependency_environment = frozen_runner.verify_pre_registered_environment()
    label_root = args.label_root.resolve()
    materialized_root = args.materialized_root.resolve()
    reports = frozen_runner.verify_frozen_reports(label_root, materialized_root)
    inputs = {
        "canonical_labels": _artifact(
            label_root / "canonical_union_labels.jsonl",
            frozen_runner.EXPECTED_HASHES["canonical_union_labels.jsonl"],
        ),
        "metadata": _artifact(
            materialized_root / "metadata.npz",
            frozen_runner.EXPECTED_HASHES["metadata.npz"],
        ),
        "compact1078": _artifact(
            materialized_root / "compact_1078.npy",
            frozen_runner.EXPECTED_HASHES["compact_1078.npy"],
        ),
    }
    return {
        "sources": sources,
        "runner_normalized_sha256": observed_runner,
        "dependency_environment": dependency_environment,
        "frozen_reports": reports,
        "frozen_inputs": inputs,
        "all_locks_verified_before_array_open": True,
    }


def verify_control_result(result: Mapping[str, Any]) -> None:
    views = list(result.get("views", []))
    if len(views) != 16 or int(result.get("optimizer_runs", -1)) != 64:
        raise RuntimeError("control-view optimization coverage changed")
    if any(
        bool(view.get("heldout_feature_constructed"))
        or bool(view.get("heldout_prediction_executed"))
        or bool(view.get("heldout_outcome_metric_computed"))
        for view in views
    ):
        raise RuntimeError("control-view study crossed the held-out boundary")
    if result.get("formal_integration_gate", {}).get("heldout_outcome_metrics_used") is not False:
        raise RuntimeError("control-view gate did not prove its optimization-only boundary")
    forbidden = _all_keys(result) & FORBIDDEN_RESULT_KEYS
    if forbidden:
        raise RuntimeError(f"forbidden outcome-screen keys entered control report: {sorted(forbidden)}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    locks = verify_locked_environment(args)
    arrays, _, input_provenance = frozen_runner.load_formal_arrays(args)
    result = control.optimization_only_control_study(arrays)
    verify_control_result(result)
    report = {
        "schema": SCHEMA, "status": result["status"],
        "evidence_class": "train-proxy control-view optimization-only; no selector outcome evidence",
        "boundary": {
            "train_only": True, "heldout_features": False,
            "heldout_predictions": False, "heldout_outcome_acceptance": False,
            "validation_opened": False, "sealed_opened": False,
            "choices_generated": False, "candidate_committed": False,
        },
        "locks": locks, "input_provenance": input_provenance,
        "result": result, "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "FINAL_REPORT.json"
    _write_text(
        report_path,
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    markdown_path = output / "FINAL_REPORT.md"
    _write_text(
        markdown_path,
        "\n".join((
            "# Control-view optimizer stability v2.1 (locked)", "",
            f"Status: `{report['status']}`.",
            "This is optimization-only evidence for 16 formal control views and 64 fits.",
            "No held-out features, predictions, outcome acceptance, or choices were produced.",
            f"Formal integration ready: `{result['formal_integration_gate']['ready']}`.", "",
        )),
    )
    provenance = {
        "schema": f"{SCHEMA}-provenance",
        "final_report": _artifact(report_path),
        "final_report_markdown": _artifact(markdown_path),
        "locks": locks, "input_provenance": input_provenance,
        "validation_opened": False, "sealed_opened": False,
        "content_hash_chain_complete": True,
    }
    _write_text(
        output / "PROVENANCE.json",
        json.dumps(_jsonable(provenance), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps({
        "event": report["status"], "output": str(output),
        "formal_integration_gate": result["formal_integration_gate"],
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--label-root", type=Path, default=frozen_runner.DEFAULT_LABEL_ROOT)
    result.add_argument("--materialized-root", type=Path, default=frozen_runner.DEFAULT_MATERIALIZED_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

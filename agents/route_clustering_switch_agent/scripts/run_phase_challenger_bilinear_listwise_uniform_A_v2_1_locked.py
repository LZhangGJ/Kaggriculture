#!/usr/bin/env python3
"""Formal train-only runner for uniform top-outcome selector A/v2.1 locked."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
import train_phase_challenger_bilinear_listwise_A_v2 as frozen_trainer
import train_phase_challenger_bilinear_listwise_optimizer_stability_v2_1_locked as optimizer_locked
import train_phase_challenger_bilinear_listwise_uniform_A_v2_1_locked as trainer


SCHEMA = "phase-challenger-bilinear-listwise-uniform-A-v2.1-locked-formal-run"
DEFAULT_OUTPUT = (
    frozen_runner.DEFAULT_PARENT
    / "phase_challenger_bilinear_listwise_uniform_A_v2_1_locked_20260829a"
)
DEFAULT_CONTROL_REPORT = (
    frozen_runner.DEFAULT_PARENT
    / "control_view_optimizer_stability_v2_1_locked_20260829a"
    / "FINAL_REPORT.json"
)
EXPECTED_NEW_TRAINER_SHA256 = "8160d9e1eee41bdce98a25c6e0ef0832feaa695a33679911793eb6854a86b12b"
EXPECTED_FROZEN_TRAINER_SHA256 = "26d959198e74283f4aa49d1eb50621e8cca5265a1f7faf821f77a3530d73a0ee"
EXPECTED_OPTIMIZER_LOCKED_SHA256 = "e9c81b4419e5c353f2416240ec71b9690d356324aa62bf533a32ec51a796c429"
EXPECTED_FROZEN_RUNNER_SHA256 = "aa269947ea5bcfe3a3476c996b29632462828790e07159c66d9736e5604f7195"
EXPECTED_CONTROL_REPORT_SHA256 = "158ea4ffb3c176bd6b2535281d4dd28e0af663efc14de2011a09a0db4922d676"
EXPECTED_RUNNER_NORMALIZED_SHA256 = "53a5b172e3659b675fcc5e5be3050eac3bbc6ddefcbea576146ca127e7fa08c9"

EXPECTED_CONFIG: dict[str, Any] = {
    "schema": trainer.SCHEMA,
    "decision_steps": frozen_runner.DEFAULT_STEPS,
    "seed_folds": frozen_trainer.SEED_FOLDS,
    "features": frozen_runner.EXPECTED_CONFIG["features"],
    "trainer": dict(trainer.DEFAULT_CONFIG),
    "controls": frozen_runner.EXPECTED_CONFIG["controls"],
    "formal_counts": frozen_runner.EXPECTED_CONFIG["formal_counts"],
    "evidence_boundary": frozen_runner.EXPECTED_CONFIG["evidence_boundary"],
    "frozen_flow": {
        "folds": "unchanged from frozen v2",
        "threshold_source": "outer-train inner seed-fold OOF only",
        "controls": "unchanged from frozen v2",
        "acceptance": "unchanged from frozen v2",
    },
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


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _final_artifact(staged_path: Path, final_path: Path) -> dict[str, Any]:
    artifact = _artifact(staged_path)
    artifact["path"] = str(final_path.resolve())
    return artifact


def _verify_control_report(path: Path) -> dict[str, Any]:
    artifact = _artifact(path, EXPECTED_CONTROL_REPORT_SHA256)
    report = json.loads(path.read_text(encoding="utf-8"))
    gate = report.get("result", {}).get("formal_integration_gate", {})
    boundary = report.get("boundary", {})
    if (
        report.get("schema") != "control-view-optimizer-stability-v2.1-locked-run"
        or report.get("status") != "control_views_optimization_stable"
        or gate.get("all_16_views_both_heads_stable") is not True
        or gate.get("all_64_optimizer_runs_accounted") is not True
        or gate.get("heldout_outcome_metrics_used") is not False
        or gate.get("ready") is not True
        or boundary.get("heldout_predictions") is not False
        or boundary.get("heldout_outcome_acceptance") is not False
        or boundary.get("validation_opened") is not False
        or boundary.get("sealed_opened") is not False
        or boundary.get("choices_generated") is not False
        or boundary.get("candidate_committed") is not False
    ):
        raise ValueError("locked control-view optimization report semantics changed")
    return {"artifact": artifact, "formal_integration_gate": gate}


def verify_locked_environment(args: argparse.Namespace) -> dict[str, Any]:
    """Verify every source, dependency, report, and input before array opening."""

    sources = {
        "new_trainer": _artifact(Path(trainer.__file__), EXPECTED_NEW_TRAINER_SHA256),
        "frozen_trainer": _artifact(Path(frozen_trainer.__file__), EXPECTED_FROZEN_TRAINER_SHA256),
        "optimizer_locked_trainer": _artifact(
            Path(optimizer_locked.__file__), EXPECTED_OPTIMIZER_LOCKED_SHA256,
        ),
        "frozen_runner": _artifact(Path(frozen_runner.__file__), EXPECTED_FROZEN_RUNNER_SHA256),
    }
    observed_self = _runner_normalized_sha256(Path(__file__).resolve())
    if observed_self != EXPECTED_RUNNER_NORMALIZED_SHA256:
        raise ValueError("formal uniform runner normalized source hash changed")
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
    control = _verify_control_report(DEFAULT_CONTROL_REPORT)
    return {
        "sources": sources,
        "runner_normalized_sha256": observed_self,
        "dependency_environment": dependency_environment,
        "frozen_reports": reports, "frozen_inputs": inputs,
        "control_view_optimization_study": control,
        "EXPECTED_CONFIG_sha256": _json_hash(EXPECTED_CONFIG),
        "all_locks_verified_before_array_open": True,
    }


def verify_result(result: Mapping[str, Any]) -> None:
    if result.get("schema") != trainer.SCHEMA:
        raise RuntimeError("formal result schema changed")
    if set(result.get("acceptance_gate", {})) != trainer.EXPECTED_ACCEPTANCE_KEYS:
        raise RuntimeError("formal acceptance conditions changed")
    if set(result.get("controls", {}).get("rank_metrics", {})) != trainer.EXPECTED_CONTROL_NAMES:
        raise RuntimeError("formal control set changed")
    if result.get("outer_labels_used_for_model_or_threshold") is not False:
        raise RuntimeError("outer labels entered model fitting or threshold calibration")
    flow = result.get("frozen_flow_contract", {})
    if not all(flow.get(key) is True for key in (
        "folds_unchanged", "controls_unchanged", "threshold_source_unchanged",
        "acceptance_unchanged", "scoped_adapter_restored",
    )):
        raise RuntimeError("frozen flow contract was not preserved")


def implementation_provenance() -> dict[str, Any]:
    test_path = (
        Path(__file__).resolve().parents[1] / "tests"
        / "test_phase_challenger_bilinear_listwise_uniform_A_v2_1_locked.py"
    )
    return {
        "runner": _artifact(Path(__file__)),
        "trainer": _artifact(Path(trainer.__file__), EXPECTED_NEW_TRAINER_SHA256),
        "frozen_runner": _artifact(Path(frozen_runner.__file__), EXPECTED_FROZEN_RUNNER_SHA256),
        "frozen_trainer": _artifact(Path(frozen_trainer.__file__), EXPECTED_FROZEN_TRAINER_SHA256),
        "optimizer_locked_trainer": _artifact(
            Path(optimizer_locked.__file__), EXPECTED_OPTIMIZER_LOCKED_SHA256,
        ),
        "focused_test": _artifact(test_path) if test_path.is_file() else None,
        "scoped_adapter_only": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    locks = verify_locked_environment(args)
    arrays, source_rows, input_provenance = frozen_runner.load_formal_arrays(args)
    result = trainer.evaluate(arrays)
    verify_result(result)
    choices = np.asarray(result.pop("A_choices"), np.int64)
    groups = frozen_trainer.decision_slices(np.asarray(arrays["decision"]))
    if len(choices) != frozen_runner.EXPECTED_DECISIONS or any(
        not np.any(group == choice) for group, choice in zip(groups, choices, strict=True)
    ):
        raise RuntimeError("formal choices lost source-decision coverage")

    stage = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True, exist_ok=False)
    try:
        choices_name = "choices_bilinear_listwise_uniform_A_v2_1_locked.npz"
        choices_path = stage / choices_name
        np.savez_compressed(
            choices_path,
            decision=np.asarray([arrays["decision"][group[0]] for group in groups], np.int64),
            selected_A_row=choices,
            selected_materialized_row=source_rows[choices],
            selected_state_id=np.asarray(arrays["state_id"])[choices],
            selected_union_index=np.asarray(arrays["union_index"])[choices],
            selected_candidate_sha256=np.asarray(arrays["candidate_sha256"])[choices],
            choice_is_KEEP=(np.asarray(arrays["edit"])[choices] == frozen_trainer.KEEP_EDIT),
        )
        choices_artifact = _final_artifact(choices_path, output / choices_name)
        report = {
            "schema": SCHEMA,
            "status": "formal_train_only_uniform_v2_1_complete",
            "screen_result": result["status"],
            "EXPECTED_CONFIG": EXPECTED_CONFIG,
            "EXPECTED_CONFIG_sha256": locks["EXPECTED_CONFIG_sha256"],
            "evidence_boundary": {
                "train_only": True, "repair_on_dev": True,
                "outer_labels_model_or_threshold": False,
                "outer_labels_evaluation_acceptance_only": True,
                "validation_opened": False, "sealed_opened": False,
                "candidate_committed": False, "unbiased_estimate": False,
            },
            "locks": locks, "input_provenance": input_provenance,
            "implementation_provenance": implementation_provenance(),
            "result": result, "artifacts": {"choices": choices_artifact},
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = stage / "FINAL_REPORT.json"
        _write_text(
            report_path,
            json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        )
        markdown_path = stage / "FINAL_REPORT.md"
        _write_text(
            markdown_path,
            "\n".join((
                "# Bilinear listwise uniform A/v2.1 locked train-only screen", "",
                f"Status: `{report['status']}`; screen: `{report['screen_result']}`.",
                "Frozen folds, inner-OOF threshold source, controls, and acceptance remain unchanged.",
                "Outer labels are evaluation/acceptance only. Validation and sealed data remain unopened.", "",
            )),
        )
        provenance = {
            "schema": f"{SCHEMA}-provenance",
            "choices": choices_artifact,
            "final_report": _final_artifact(report_path, output / report_path.name),
            "final_report_markdown": _final_artifact(markdown_path, output / markdown_path.name),
            "locks": locks, "input_provenance": input_provenance,
            "implementation_provenance": report["implementation_provenance"],
            "content_hash_chain_complete": True,
            "validation_opened": False, "sealed_opened": False,
        }
        provenance_path = stage / "PROVENANCE.json"
        _write_text(
            provenance_path,
            json.dumps(_jsonable(provenance), ensure_ascii=False, indent=2) + "\n",
        )
        stage.replace(output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    print(json.dumps({
        "event": report["status"], "screen_result": report["screen_result"],
        "output": str(output),
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


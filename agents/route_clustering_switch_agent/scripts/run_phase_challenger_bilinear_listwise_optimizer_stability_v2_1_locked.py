#!/usr/bin/env python3
"""Run the locked train-only optimizer stability study for listwise v2.1."""

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
import train_phase_challenger_bilinear_listwise_A_v2 as frozen_trainer
import train_phase_challenger_bilinear_listwise_optimizer_stability_v2_1_locked as study


SCHEMA = "bilinear-listwise-optimizer-stability-v2.1-locked-run"
DEFAULT_OUTPUT = (
    frozen_runner.DEFAULT_PARENT
    / "phase_challenger_bilinear_listwise_optimizer_stability_v2_1_locked_20260829a"
)
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


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": _sha256(resolved), "bytes": resolved.stat().st_size}


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
    if isinstance(value, Mapping):
        return set(map(str, value)) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, (tuple, list)):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def verify_optimization_only_result(result: Mapping[str, Any]) -> None:
    targets = list(result.get("targets", []))
    if len(targets) != 10:
        raise RuntimeError("locked study did not cover ten unique training targets")
    if any(
        bool(target.get("heldout_prediction_executed"))
        or bool(target.get("heldout_outcome_metric_computed"))
        for target in targets
    ):
        raise RuntimeError("optimization-only target crossed the held-out boundary")
    if result.get("recommendation", {}).get("selection_used_heldout_outcome_metrics") is not False:
        raise RuntimeError("solver selection did not prove its optimization-only boundary")
    forbidden = _all_keys(result) & FORBIDDEN_RESULT_KEYS
    if forbidden:
        raise RuntimeError(f"forbidden outcome-screen keys entered optimizer report: {sorted(forbidden)}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    environment = frozen_runner.verify_pre_registered_environment()
    arrays, _, input_provenance = frozen_runner.load_formal_arrays(args)
    result = study.optimization_only_study(arrays)
    verify_optimization_only_result(result)
    implementation = {
        "locked_study": _artifact(Path(study.__file__)),
        "locked_runner": _artifact(Path(__file__)),
        "frozen_trainer": _artifact(Path(frozen_trainer.__file__)),
        "frozen_loader": _artifact(Path(frozen_runner.__file__)),
    }
    report = {
        "schema": SCHEMA, "status": result["status"],
        "evidence_class": "train-proxy optimization-only numerical stability; no selector outcome evidence",
        "boundary": {
            "train_only": True, "heldout_predictions": False,
            "heldout_outcome_acceptance": False,
            "validation_opened": False, "sealed_opened": False,
            "choices_generated": False, "candidate_committed": False,
        },
        "environment": environment,
        "input_provenance": input_provenance,
        "implementation_provenance": implementation,
        "result": result,
        "elapsed_seconds": time.perf_counter() - started,
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
            "# Bilinear listwise optimizer stability v2.1 (locked)", "",
            f"Status: `{report['status']}`.",
            "Evidence is train-proxy optimization-only; no held-out prediction or outcome acceptance was computed.",
            "The listwise target is top-outcome-set classification, not a full ordinal objective.",
            f"Recommended listwise target: `{result['recommendation']['listwise_objective']}`.",
            f"Recommended listwise solver: `{result['recommendation']['listwise_solver']}`.",
            f"Recommended risk solver: `{result['recommendation']['risk_solver']}`.", "",
        )),
    )
    provenance = {
        "schema": f"{SCHEMA}-provenance",
        "final_report": _artifact(report_path),
        "final_report_markdown": _artifact(markdown_path),
        "implementation": implementation,
        "input_provenance": input_provenance,
        "validation_opened": False, "sealed_opened": False,
        "content_hash_chain_complete": True,
    }
    _write_text(
        output / "PROVENANCE.json",
        json.dumps(_jsonable(provenance), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps({
        "event": report["status"], "output": str(output),
        "recommendation": result["recommendation"],
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

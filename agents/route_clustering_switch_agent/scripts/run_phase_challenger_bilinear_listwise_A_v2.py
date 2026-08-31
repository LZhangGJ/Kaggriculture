#!/usr/bin/env python3
"""Formal train-only runner for the independent bilinear listwise A/v2 screen.

This command intentionally has no model hyperparameter flags.  It first locks
the complete experiment configuration, source dependencies, package versions,
and frozen artifacts.  The canonical candidate SHA is the executable four-step
block identity; it is joined by state_id + union_index + physical row order and
is only a cross-fitted soft prior in the trainer, never an eligibility gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (Path(__file__).resolve().parent, CODE_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_phase_challenger_compact_et_ab_v1 as compact_v1
import train_phase_challenger_bilinear_listwise_A_v2 as trainer


SCHEMA = "phase-challenger-bilinear-listwise-A-v2-formal-run"
DEFAULT_PARENT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
)
DEFAULT_LABEL_ROOT = DEFAULT_PARENT / "phase_challenger_8seed_labels_v1_formal_20260829a"
DEFAULT_MATERIALIZED_ROOT = DEFAULT_PARENT / "phase_challenger_compact_et_ab_v1_materialized_20260829a"
DEFAULT_OUTPUT = DEFAULT_PARENT / "phase_challenger_bilinear_listwise_A_v2_20260829a"
DEFAULT_STEPS = (217, 218, 219, 266, 267, 268, 435, 436, 457, 482, 483, 484, 529, 530, 531, 553)
EXPECTED_ROWS = 29_088
EXPECTED_DECISIONS = 1_536
EXPECTED_CANONICAL_ROWS = 179_310
EXPECTED_CANONICAL_DECISIONS = 8_064

EXPECTED_HASHES = {
    "canonical_union_labels.jsonl": "46ce99e9e6eb8d6ae25a06b4a110e8272bdcaacff546f71204f0a3fc1c1c297f",
    "label_FINAL_REPORT.json": "05f7ece70adbc19459d30df7198b3e3e38ea4ed7f6363cf6bbf9807f1fa0c5b6",
    "materialized_FINAL_REPORT.json": "afb1c6d93c456322cdfba5382a92d851edde957c4c00a0846d8aebd3957ce604",
    "compact_1078.npy": "1da4b0af5536e92d861322057ccd19a97fdbc13671d949e1d9d985e904af4cd7",
    "metadata.npz": "f6dcb85e8a6ddddda4051e7fd50df88e5e9121e74d64343ad99335cc828ee269",
    "trainer.py": "26d959198e74283f4aa49d1eb50621e8cca5265a1f7faf821f77a3530d73a0ee",
    "compact_dependency.py": "c7c8a990e9731e6bc080255040f6a84ec3188acdc03188c2dbd21415100d67d2",
}
EXPECTED_VERSIONS = {"numpy": "2.5.2", "scipy": "1.18.1", "scikit-learn": "1.9.0"}
EXPECTED_FEATURE_HASH = "2d8104e478da08a80662dba103dc27ded18cffb12fc26c2fac4760cc69b54079"
EXPECTED_PROJECTION_HASH = "749e5455b46173d3cb1d47a02e5089181acf4890dca340014a57a57029fc69f7"
EXPECTED_STEPS_HASH = "4725c1d522dcb18b28894d493e15cd7fe5a027f3f9c6cd1ecf30bd067383f712"
EXPECTED_FOLDS_HASH = "79e33cf5ec5190e7de664c7fd9cb1a4bbc4f41743c81ea6d843334a72b24a794"
EXPECTED_RUNNER_NORMALIZED_SHA256 = "8be544394b4a0a877d2e541f1751fd6e8f2ae50e3dbbe89f68bd4dd714976ba7"

EXPECTED_CONFIG: dict[str, Any] = {
    "schema": trainer.SCHEMA,
    "decision_steps": DEFAULT_STEPS,
    "seed_folds": trainer.SEED_FOLDS,
    "features": {
        "observable_state": "compact1078[0:337]",
        "relative_action": "candidate-minus-KEEP 154 causal action fields",
        "state_width": 337, "action_width": 154,
        "projection_width_each": 8, "bilinear_width": 64,
        "action_main_effect_width": 8, "SHA_prior_width": 2,
        "primary_width": 74,
    },
    "trainer": dict(trainer.DEFAULT_CONFIG),
    "controls": (
        "KEEP", "SHA-prior-only", "bilinear-no-SHA", "primary",
        "state-shuffle-2026082911", "state-shuffle-2026082923",
        "state-shuffle-2026082937",
    ),
    "formal_counts": {
        "rows": EXPECTED_ROWS, "decisions": EXPECTED_DECISIONS,
        "canonical_rows": EXPECTED_CANONICAL_ROWS,
        "canonical_decisions": EXPECTED_CANONICAL_DECISIONS,
    },
    "evidence_boundary": "train-only repair-on-dev; outer labels evaluation/acceptance only",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runner_normalized_sha256(path: Path) -> str:
    prefix = "EXPECTED_RUNNER_NORMALIZED_SHA256 = "
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1]:
        lines.pop()
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("runner normalized-hash marker changed")
    lines[matches[0]] = prefix + "\"<normalized-self-hash>\""
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _artifact(path: Path, *, expected_sha: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = _sha256_file(resolved)
    if expected_sha is not None and observed != expected_sha:
        raise ValueError(f"frozen artifact SHA mismatch: {resolved}")
    return {"path": str(resolved), "sha256": observed, "bytes": resolved.stat().st_size}


def feature_schema() -> dict[str, Any]:
    names = tuple(compact_v1.COMPACT_FEATURE_NAMES)
    if len(names) != 1078:
        raise ValueError("compact feature width changed")
    donor = {
        index for index, name in enumerate(names)
        if name == "donor_support_log" or name.startswith("donor_rank_") or name.startswith("donor_cluster_")
    }
    derivable = {names.index(name) for name in (
        "runtime_actor_count", "source_actor_count", "segment", "segment_offset",
    )}
    excluded = donor | derivable | {names.index("seat")}
    relative = tuple(index for index in range(907, len(names)) if index not in excluded)
    if len(donor) != 12 or len(relative) != trainer.ACTION_WIDTH:
        raise ValueError("the frozen causal action154 partition changed")
    payload = {
        "state_names": names[:trainer.STATE_WIDTH],
        "relative_action_names": tuple(
            f"candidate_minus_KEEP_{names[index]}" for index in relative
        ),
    }
    observed_hash = _json_hash(payload)
    if observed_hash != EXPECTED_FEATURE_HASH:
        raise ValueError("state337/action154 feature-name hash changed")
    return {
        "relative_indices": np.asarray(relative, np.int64),
        "feature_names_sha256": observed_hash,
        "state_width": trainer.STATE_WIDTH,
        "relative_action_width": len(relative),
        "donor_provenance_excluded": True,
        "observed_market_price_is_feature": False,
    }


def verify_pre_registered_environment() -> dict[str, Any]:
    versions = {
        "numpy": np.__version__, "scipy": scipy.__version__,
        "scikit-learn": importlib.metadata.version("scikit-learn"),
    }
    if versions != EXPECTED_VERSIONS:
        raise ValueError(f"formal package versions changed: {versions}")
    sources = {
        "trainer": _artifact(Path(trainer.__file__), expected_sha=EXPECTED_HASHES["trainer.py"]),
        "compact_feature_dependency": _artifact(
            Path(compact_v1.__file__), expected_sha=EXPECTED_HASHES["compact_dependency.py"],
        ),
    }
    if _runner_normalized_sha256(Path(__file__).resolve()) != EXPECTED_RUNNER_NORMALIZED_SHA256:
        raise ValueError("formal runner normalized source hash changed")
    if trainer.projection_hash() != EXPECTED_PROJECTION_HASH:
        raise ValueError("fixed signed projection hash changed")
    if _json_hash(DEFAULT_STEPS) != EXPECTED_STEPS_HASH:
        raise ValueError("pre-registered decision-step hash changed")
    if _json_hash(trainer.SEED_FOLDS) != EXPECTED_FOLDS_HASH:
        raise ValueError("pre-registered seed-fold hash changed")
    feature = feature_schema()
    return {
        "versions": versions, "sources": sources,
        "runner_normalized_sha256": EXPECTED_RUNNER_NORMALIZED_SHA256,
        "projection_sha256": EXPECTED_PROJECTION_HASH,
        "decision_steps_sha256": EXPECTED_STEPS_HASH,
        "seed_folds_sha256": EXPECTED_FOLDS_HASH,
        "feature_names_sha256": feature["feature_names_sha256"],
        "EXPECTED_CONFIG_sha256": _json_hash(EXPECTED_CONFIG),
        "dependency_semantics": (
            "trainer source is the complete NumPy/SciPy model/evaluator; compact dependency "
            "provides only the frozen feature-name partition and upstream schema"
        ),
    }


def verify_frozen_reports(label_root: Path, materialized_root: Path) -> dict[str, Any]:
    label_report_path = label_root / "FINAL_REPORT.json"
    materialized_report_path = materialized_root / "FINAL_REPORT.json"
    label_artifact = _artifact(
        label_report_path, expected_sha=EXPECTED_HASHES["label_FINAL_REPORT.json"],
    )
    materialized_artifact = _artifact(
        materialized_report_path, expected_sha=EXPECTED_HASHES["materialized_FINAL_REPORT.json"],
    )
    label_report = json.loads(label_report_path.read_text(encoding="utf-8"))
    materialized_report = json.loads(materialized_report_path.read_text(encoding="utf-8"))
    if (
        label_report.get("schema") != "phase-challenger-8seed-labels-v1"
        or label_report.get("status") != "label_signal_ready_for_residual_gate_training"
        or materialized_report.get("schema") != compact_v1.SCHEMA
        or materialized_report.get("status") != "formal_materialization_complete_training_not_requested"
        or materialized_report.get("training_executed") is not False
        or materialized_report.get("runtime_boundary", {}).get("validation_opened") is not False
        or materialized_report.get("runtime_boundary", {}).get("sealed_opened") is not False
        or materialized_report.get("runtime_boundary", {}).get("candidate_committed") is not False
        or int(materialized_report.get("panel", {}).get("rows", -1)) != EXPECTED_CANONICAL_ROWS
        or int(materialized_report.get("panel", {}).get("decisions", -1)) != EXPECTED_CANONICAL_DECISIONS
        or materialized_report.get("panel", {}).get("candidate_order_sha256")
        != "69e45374b4c9ebd9f0a822b55d227cd5987adbab905abe6b45c309a25d507c88"
        or Path(materialized_report.get("input_label_root", "")).resolve() != label_root.resolve()
    ):
        raise ValueError("frozen label/materialization report semantics changed")
    return {
        "label_report": label_artifact,
        "materialized_report": materialized_artifact,
        "reports_verified_before_referenced_array_open": True,
    }


def _hex_sha(value: Any) -> str:
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError("canonical candidate SHA is not lowercase SHA-256")
    return text


def join_canonical_labels(
    path: Path, expected_sha: str, metadata: Mapping[str, np.ndarray],
    selected: np.ndarray, *, expected_rows: int | None = None,
    expected_decisions: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Hash first, then join every canonical row by state_id+union_index+order."""

    artifact = _artifact(path, expected_sha=expected_sha)
    n = len(np.asarray(metadata["decision"]))
    selected = np.asarray(selected, np.bool_)
    if selected.shape != (n,):
        raise ValueError("canonical selection mask is not metadata-aligned")
    for name in ("decision", "union_index", "seed", "seat", "opponent", "step", "phase_only", "path_pure", "delta_margin", "edit"):
        if name not in metadata or len(np.asarray(metadata[name])) != n:
            raise ValueError(f"canonical join is missing metadata field {name}")
    collected: dict[str, list[Any]] = {
        "candidate_sha256": [], "state_id": [], "outcome": [], "margin": [],
    }
    current_state: str | None = None
    expected_union = 0
    decision_index = -1
    closed_states: set[str] = set()
    opponent_map: dict[str, int] = {}
    decision_keep_margin = 0.0
    decision_shas: set[str] = set()
    rows_read = 0
    with path.open("r", encoding="utf-8") as source:
        for row, raw in enumerate(source):
            if row >= n:
                raise ValueError("canonical labels have more rows than metadata")
            label = json.loads(raw)
            state_id = str(label.get("state_id", ""))
            union = int(label.get("union_index", -1))
            if union == 0:
                if state_id in closed_states or state_id == current_state:
                    raise ValueError("canonical state_id reappeared or restarted")
                if current_state is not None:
                    closed_states.add(current_state)
                current_state = state_id
                expected_union = 0
                decision_index += 1
                decision_keep_margin = float(label["margin"])
                decision_shas = set()
            if state_id != current_state or union != expected_union:
                raise ValueError("canonical state_id/union_index row order changed")
            expected_union += 1
            if int(metadata["decision"][row]) != decision_index or int(metadata["union_index"][row]) != union:
                raise ValueError("canonical row order disagrees with materialized metadata")
            if (
                int(label["seed"]) != int(metadata["seed"][row])
                or int(label["seat"]) != int(metadata["seat"][row])
                or int(label["step"]) != int(metadata["step"][row])
                or bool(label["path_pure"]) != bool(metadata["path_pure"][row])
                or (str(label["membership"]) == "phase_only") != bool(metadata["phase_only"][row])
                or str(label.get("split")) != "train_proxy"
            ):
                raise ValueError("canonical row provenance disagrees with materialized metadata")
            opponent_name = str(label["opponent"])
            opponent_value = int(metadata["opponent"][row])
            previous = opponent_map.setdefault(opponent_name, opponent_value)
            if previous != opponent_value or len(set(opponent_map.values())) != len(opponent_map):
                raise ValueError("canonical opponent identity mapping changed")
            margin = float(label["margin"])
            if not np.isclose(
                float(metadata["delta_margin"][row]), margin - decision_keep_margin,
                rtol=0.0, atol=1e-9,
            ):
                raise ValueError("canonical margin no longer matches materialized delta_margin")
            sha = _hex_sha(label["candidate_sha256"])
            if sha in decision_shas:
                raise ValueError("canonical decision contains a candidate SHA collision")
            decision_shas.add(sha)
            if selected[row]:
                collected["candidate_sha256"].append(sha.encode("ascii"))
                collected["state_id"].append(state_id)
                collected["outcome"].append(int(label["outcome"]))
                collected["margin"].append(margin)
            rows_read += 1
    if rows_read != n:
        raise ValueError("canonical labels ended before materialized metadata")
    decisions_read = decision_index + 1
    if expected_rows is not None and rows_read != expected_rows:
        raise ValueError("canonical row count changed")
    if expected_decisions is not None and decisions_read != expected_decisions:
        raise ValueError("canonical decision count changed")
    selected_rows = int(np.count_nonzero(selected))
    if any(len(values) != selected_rows for values in collected.values()):
        raise RuntimeError("canonical selected labels are incomplete")
    arrays = {
        "candidate_sha256": np.asarray(collected["candidate_sha256"], "S64"),
        "state_id": np.asarray(collected["state_id"], "U64"),
        "outcome": np.asarray(collected["outcome"], np.int8),
        "margin": np.asarray(collected["margin"], np.float64),
    }
    return arrays, {
        "canonical_labels": artifact, "rows": rows_read,
        "decisions": decisions_read, "selected_rows": selected_rows,
        "state_id_union_index_physical_order_verified": True,
        "train_proxy_only": True, "opponent_identity_map": opponent_map,
        "candidate_sha_semantic": "frozen executable four-step block",
    }


def load_formal_arrays(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    label_root = args.label_root.resolve()
    materialized_root = args.materialized_root.resolve()
    report_audit = verify_frozen_reports(label_root, materialized_root)
    metadata_path = materialized_root / "metadata.npz"
    compact_path = materialized_root / "compact_1078.npy"
    metadata_artifact = _artifact(metadata_path, expected_sha=EXPECTED_HASHES["metadata.npz"])
    compact_artifact = _artifact(compact_path, expected_sha=EXPECTED_HASHES["compact_1078.npy"])
    with np.load(metadata_path, allow_pickle=False) as source:
        metadata = {name: source[name] for name in source.files}
    rows = len(np.asarray(metadata.get("decision", [])))
    if rows != EXPECTED_CANONICAL_ROWS or any(len(value) != rows for value in metadata.values()):
        raise ValueError("materialized metadata row contract changed")
    selected = np.isin(np.asarray(metadata["step"]), DEFAULT_STEPS) & ~np.asarray(metadata["phase_only"], np.bool_)
    source_rows = np.flatnonzero(selected)
    if len(source_rows) != EXPECTED_ROWS:
        raise ValueError("the fixed 16-step A/R0 panel row count changed")
    labels, label_audit = join_canonical_labels(
        label_root / "canonical_union_labels.jsonl",
        EXPECTED_HASHES["canonical_union_labels.jsonl"], metadata, selected,
        expected_rows=EXPECTED_CANONICAL_ROWS,
        expected_decisions=EXPECTED_CANONICAL_DECISIONS,
    )
    feature = feature_schema()
    compact = np.load(compact_path, mmap_mode="r", allow_pickle=False)
    if compact.shape != (EXPECTED_CANONICAL_ROWS, 1078) or compact.dtype != np.float32:
        raise ValueError("compact1078 storage contract changed")
    state = np.asarray(compact[source_rows, :trainer.STATE_WIDTH], np.float64)
    causal = np.asarray(compact[np.ix_(source_rows, feature["relative_indices"])], np.float64)
    decision = np.asarray(metadata["decision"])[source_rows]
    edit = np.asarray(metadata["edit"])[source_rows]
    action = np.empty_like(causal)
    for group in trainer.decision_slices(decision):
        keep = group[edit[group] == trainer.KEEP_EDIT]
        if len(keep) != 1:
            raise ValueError("formal selected decision lost KEEP")
        action[group] = causal[group] - causal[int(keep[0])]
    arrays = {
        "state": state, "action": action,
        "candidate_sha256": labels["candidate_sha256"],
        "state_id": labels["state_id"],
        "decision": np.asarray(decision, np.int64),
        "union_index": np.asarray(metadata["union_index"])[source_rows].astype(np.int16),
        "seed": np.asarray(metadata["seed"])[source_rows].astype(np.int64),
        "seat": np.asarray(metadata["seat"])[source_rows].astype(np.int8),
        "opponent": np.asarray(metadata["opponent"])[source_rows].astype(np.int16),
        "step": np.asarray(metadata["step"])[source_rows].astype(np.int16),
        "edit": np.asarray(edit, np.int16),
        "outcome": labels["outcome"], "margin": labels["margin"],
    }
    input_audit = trainer.validate_arrays(arrays)
    if input_audit["rows"] != EXPECTED_ROWS or input_audit["decisions"] != EXPECTED_DECISIONS:
        raise ValueError("formal A panel coverage changed")
    return arrays, source_rows, {
        **report_audit, **label_audit,
        "metadata": metadata_artifact, "compact1078": compact_artifact,
        "feature_schema": {key: value for key, value in feature.items() if key != "relative_indices"},
        "trainer_input_audit": input_audit,
    }


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def implementation_provenance() -> dict[str, Any]:
    tests = Path(__file__).resolve().parents[1] / "tests"
    candidates = (
        tests / "test_train_phase_challenger_bilinear_listwise_A_v2.py",
        tests / "test_train_phase_challenger_bilinear_listwise_A_v2_gradient.py",
        tests / "test_train_phase_challenger_bilinear_listwise_A_v2_convergence.py",
        tests / "test_train_phase_challenger_bilinear_listwise_A_v2_safety.py",
        tests / "test_train_phase_challenger_bilinear_listwise_A_v2_leakage.py",
        tests / "test_train_phase_challenger_bilinear_listwise_A_v2_sha_scope.py",
        tests / "test_run_phase_challenger_bilinear_listwise_A_v2.py",
    )
    return {
        "runner": _artifact(Path(__file__)),
        "trainer": _artifact(Path(trainer.__file__), expected_sha=EXPECTED_HASHES["trainer.py"]),
        "transitive_compact_feature_dependency": _artifact(
            Path(compact_v1.__file__), expected_sha=EXPECTED_HASHES["compact_dependency.py"],
        ),
        "focused_tests": [_artifact(path) for path in candidates if path.is_file()],
        "SHA_semantics": (
            "file content hashes lock code bytes; candidate SHA locks a four-step executable "
            "block and is cross-fitted soft input only"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    environment = verify_pre_registered_environment()
    arrays, source_rows, input_provenance = load_formal_arrays(args)
    result = trainer.evaluate(arrays)
    choices = np.asarray(result.pop("A_choices"), np.int64)
    groups = trainer.decision_slices(np.asarray(arrays["decision"]))
    if len(choices) != EXPECTED_DECISIONS or any(
        not np.any(group == choice) for group, choice in zip(groups, choices, strict=True)
    ):
        raise RuntimeError("formal choices lost source-decision coverage")
    output.mkdir(parents=True, exist_ok=False)
    choices_path = output / "choices_bilinear_listwise_A_v2.npz"
    np.savez_compressed(
        choices_path,
        decision=np.asarray([arrays["decision"][group[0]] for group in groups], np.int64),
        selected_A_row=choices,
        selected_materialized_row=source_rows[choices],
        selected_state_id=np.asarray(arrays["state_id"])[choices],
        selected_union_index=np.asarray(arrays["union_index"])[choices],
        selected_candidate_sha256=np.asarray(arrays["candidate_sha256"])[choices],
        choice_is_KEEP=(np.asarray(arrays["edit"])[choices] == trainer.KEEP_EDIT),
    )
    choices_artifact = _artifact(choices_path)
    report = {
        "schema": SCHEMA,
        "status": "formal_train_only_v2_complete",
        "screen_result": result["status"],
        "EXPECTED_CONFIG": EXPECTED_CONFIG,
        "EXPECTED_CONFIG_sha256": environment["EXPECTED_CONFIG_sha256"],
        "evidence_boundary": {
            "train_only": True, "repair_on_dev": True,
            "outer_labels_model_or_threshold": False,
            "outer_labels_evaluation_acceptance_only": True,
            "validation_opened": False, "sealed_opened": False,
            "candidate_committed": False, "unbiased_estimate": False,
        },
        "environment": environment, "input_provenance": input_provenance,
        "implementation_provenance": implementation_provenance(),
        "result": result, "artifacts": {"choices": choices_artifact},
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output / "FINAL_REPORT.json"
    _write_text(report_path, json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n")
    provenance = {
        "schema": "phase-challenger-bilinear-listwise-A-v2-provenance",
        "choices": choices_artifact, "final_report": _artifact(report_path),
        "input_provenance": input_provenance,
        "implementation_provenance": report["implementation_provenance"],
        "content_hash_chain_complete": True,
        "validation_opened": False, "sealed_opened": False,
    }
    _write_text(
        output / "choices_bilinear_listwise_A_v2_PROVENANCE.json",
        json.dumps(_jsonable(provenance), ensure_ascii=False, indent=2) + "\n",
    )
    _write_text(
        output / "FINAL_REPORT.md",
        "\n".join((
            "# Bilinear listwise A/v2 train-only screen", "",
            f"Status: `{report['status']}`; screen: `{report['screen_result']}`.",
            f"Rows: {EXPECTED_ROWS}; decisions: {EXPECTED_DECISIONS}; all include explicit KEEP.",
            "Candidate SHA is a four-step executable-block soft prior, never eligibility.",
            "Outer labels are evaluation/acceptance only. Validation and sealed data remain unopened.", "",
        )),
    )
    print(json.dumps({
        "event": report["status"], "screen_result": report["screen_result"],
        "output": str(output),
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    result.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

















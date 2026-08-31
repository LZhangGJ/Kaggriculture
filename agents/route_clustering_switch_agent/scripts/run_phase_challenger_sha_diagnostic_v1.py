"""Materialize action/query/feature fingerprints and diagnose label aliasing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (Path(__file__).resolve().parent, CODE_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_phase_challenger_compact_et_ab_v1 as compact_v1
import run_phase_challenger_direct_nested_smoke_v1 as direct_smoke
import run_route_residual_adapter_v0 as residual


SCHEMA = "phase-challenger-sha-diagnostic-v1"
DEFAULT_LABEL_ROOT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
    r"\phase_challenger_8seed_labels_v1_formal_20260829a"
)
DEFAULT_MATERIALIZED = direct_smoke.DEFAULT_MATERIALIZED
DEFAULT_CHOICES = DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_residual_group_quantile_real_feature_smoke_v1_20260829a"
) / "choices_compact1078.npz"
DEFAULT_OUTPUT = DEFAULT_MATERIALIZED.parent / (
    "phase_challenger_sha_diagnostic_v1_20260829a"
)


def _fold(hex_digest: str) -> int:
    return int(hex_digest[:8], 16) % 4


def sign_group_summary(
    keys: Sequence[bytes], delta: np.ndarray,
) -> dict[str, Any]:
    groups: dict[bytes, list[float]] = defaultdict(list)
    for key, value in zip(keys, np.asarray(delta, np.float64), strict=True):
        groups[bytes(key)].append(float(value))
    mixed = [
        values for values in groups.values()
        if any(value > 0 for value in values)
        and any(value < 0 for value in values)
    ]
    return {
        "unique_groups": len(groups),
        "positive_groups": sum(any(value > 0 for value in values) for values in groups.values()),
        "negative_groups": sum(any(value < 0 for value in values) for values in groups.values()),
        "mixed_positive_negative_groups": len(mixed),
        "rows_in_mixed_groups": sum(len(values) for values in mixed),
        "maximum_group_rows": max(map(len, groups.values()), default=0),
    }


def focus_rows(
    extended: Mapping[str, np.ndarray], metadata: Mapping[str, np.ndarray],
    choices_path: Path,
) -> list[dict[str, Any]]:
    if not choices_path.is_file():
        return []
    with np.load(choices_path, allow_pickle=False) as choices:
        selected = np.asarray(choices["A_materialized_row"], np.int64)
    delta = np.asarray(metadata["delta_margin"], np.float64)
    harmful = selected[delta[selected] < 0]
    result = []
    for row in harmful:
        candidate = extended["candidate_sha256"][row]
        query = extended["query_sha256"][row]
        feature = extended["compact_feature_blake2b128"][row]
        candidate_rows = np.flatnonzero(extended["candidate_sha256"] == candidate)
        pair_rows = np.flatnonzero(
            (extended["candidate_sha256"] == candidate)
            & (extended["query_sha256"] == query)
        )
        feature_rows = np.flatnonzero(
            extended["compact_feature_blake2b128"] == feature
        )
        def describe(rows: np.ndarray) -> dict[str, Any]:
            values = delta[rows]
            return {
                "rows": int(len(rows)),
                "positive": int(np.count_nonzero(values > 0)),
                "negative": int(np.count_nonzero(values < 0)),
                "zero": int(np.count_nonzero(values == 0)),
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
            }
        result.append({
            "materialized_row": int(row),
            "decision": int(metadata["decision"][row]),
            "step": int(metadata["step"][row]),
            "opponent": int(metadata["opponent"][row]),
            "seed": int(metadata["seed"][row]),
            "seat": int(metadata["seat"][row]),
            "delta_margin": float(delta[row]),
            "candidate_sha256": bytes(candidate).decode("ascii"),
            "query_sha256": bytes(query).decode("ascii"),
            "candidate_history": describe(candidate_rows),
            "query_candidate_history": describe(pair_rows),
            "exact_compact_feature_history": describe(feature_rows),
        })
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    contract = compact_v1.load_input_contract(args.label_root.resolve(), smoke=False)
    materialized_report = json.loads(
        (args.materialized_root / "FINAL_REPORT.json").read_text(encoding="utf-8")
    )
    storage = materialized_report["storage"]
    metadata_path = args.materialized_root / "metadata.npz"
    compact_path = args.materialized_root / "compact_1078.npy"
    direct_smoke._artifact_matches(metadata_path, storage["metadata"])
    direct_smoke._artifact_matches(compact_path, storage["compact1078"])
    with np.load(metadata_path, allow_pickle=False) as raw:
        metadata = {key: raw[key] for key in raw.files}
    features = np.load(compact_path, mmap_mode="r")
    rows = len(metadata["decision"])
    extended = {
        "candidate_sha256": np.empty(rows, "S64"),
        "query_sha256": np.empty(rows, "S64"),
        "compact_feature_blake2b128": np.empty(rows, "S16"),
        "outcome": np.empty(rows, np.int8),
        "anchor": np.empty(rows, np.int16),
        "offset": np.empty(rows, np.int8),
        "market_diff": np.empty(rows, np.bool_),
        "action_fold": np.empty(rows, np.int8),
        "query_fold": np.empty(rows, np.int8),
    }
    at = 0
    decisions = 0
    for decision, labels in compact_v1.iter_panels(
        contract["paths"]["h1_decisions.jsonl"],
        contract["paths"]["canonical_union_labels.jsonl"],
    ):
        stop = at + len(labels)
        expected_rows = np.arange(at, stop)
        if (
            not np.all(metadata["decision"][expected_rows] == decisions)
            or not np.all(metadata["step"][expected_rows] == int(decision["decision_step"]))
            or not np.array_equal(
                metadata["union_index"][expected_rows], np.arange(len(labels))
            )
        ):
            raise RuntimeError("materialized metadata row order changed")
        query = str(decision["query_sha256"])
        if len(query) != 64:
            raise ValueError("query SHA is not canonical hex")
        for local, label in enumerate(labels):
            row = at + local
            candidate = str(label["candidate_sha256"])
            if len(candidate) != 64:
                raise ValueError("candidate SHA is not canonical hex")
            keep_margin = float(labels[0]["margin"])
            delta = float(label["margin"]) - keep_margin
            if (
                int(metadata["union_index"][row]) != int(label["union_index"])
                or bool(metadata["phase_only"][row])
                != (str(label["membership"]) == "phase_only")
                or float(metadata["delta_margin"][row]) != delta
            ):
                raise RuntimeError("label and materialized metadata changed")
            extended["candidate_sha256"][row] = candidate.encode("ascii")
            extended["query_sha256"][row] = query.encode("ascii")
            extended["compact_feature_blake2b128"][row] = hashlib.blake2b(
                np.asarray(features[row], np.float32).tobytes(), digest_size=16,
            ).digest()
            extended["outcome"][row] = int(label["outcome"])
            extended["anchor"][row] = int(label["anchor"])
            extended["offset"][row] = int(label["offset"])
            extended["market_diff"][row] = bool(label["market_diff"])
            extended["action_fold"][row] = _fold(candidate)
            extended["query_fold"][row] = _fold(query)
        at = stop
        decisions += 1
    if at != rows or decisions != len(set(map(int, metadata["decision"]))):
        raise RuntimeError("extended metadata did not cover the panel")
    extended_path = output / "extended_metadata.npz"
    residual._atomic_npz(extended_path, extended)
    delta = np.asarray(metadata["delta_margin"], np.float64)
    pair_keys = [
        bytes(query) + bytes(candidate)
        for query, candidate in zip(
            extended["query_sha256"], extended["candidate_sha256"], strict=True,
        )
    ]
    phase = np.asarray(metadata["phase_only"], np.bool_)
    report = {
        "schema": SCHEMA,
        "status": "train_only_sha_alias_diagnostic_complete",
        "evidence_boundary": {
            "train_only": True,
            "validation_opened": False,
            "sealed_opened": False,
            "model_trained": False,
        },
        "panel": {
            "rows": rows,
            "decisions": decisions,
            "feature_width": int(features.shape[1]),
        },
        "groups": {
            "candidate_action_sha": sign_group_summary(
                extended["candidate_sha256"], delta,
            ),
            "query_sha": sign_group_summary(extended["query_sha256"], delta),
            "query_candidate_pair": sign_group_summary(pair_keys, delta),
            "exact_compact_feature": sign_group_summary(
                extended["compact_feature_blake2b128"], delta,
            ),
            "phase_candidate_action_sha": sign_group_summary(
                extended["candidate_sha256"][phase], delta[phase],
            ),
        },
        "focus_harmful_A_choices": focus_rows(
            extended, metadata, args.choices.resolve(),
        ),
        "fold_counts": {
            "action": np.bincount(extended["action_fold"], minlength=4).tolist(),
            "query": np.bincount(extended["query_fold"], minlength=4).tolist(),
        },
        "artifacts": {
            "extended_metadata": {
                "path": str(extended_path),
                "bytes": extended_path.stat().st_size,
                "sha256": residual._sha256_file(extended_path),
            },
            "source_metadata_sha256": residual._sha256_file(metadata_path),
            "source_compact_sha256": residual._sha256_file(compact_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    residual._atomic_text(
        output / "DIAGNOSTIC_REPORT.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    residual._atomic_text(
        output / "DIAGNOSTIC_REPORT.md",
        "\n".join((
            "# PHASE-CHALLENGER SHA alias diagnostic",
            "", f"Status: `{report['status']}`",
            f"Rows: {rows}; decisions: {decisions}",
            "Candidate, query, query+candidate and exact compact-feature groups are checked for sign conflicts.",
            "No model, validation or sealed artifact was opened.",
            "",
        )),
    )
    print(json.dumps({
        "event": report["status"], "output": str(output),
        "rows": rows, "decisions": decisions,
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    result.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED)
    result.add_argument("--choices", type=Path, default=DEFAULT_CHOICES)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

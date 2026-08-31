#!/usr/bin/env python3
"""Generate one independent eight-seed fresh-market phase-label shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_contract_retrieval_oracle_ablation_v1 as retrieval
import run_phase_challenger_8seed_labels_v1 as base


SCHEMA = "fresh-market-trace-labels-v1"
LABEL_SCHEMA = "fresh-market-trace-canonical-label-v1"
R2_ROW_SCHEMA = "fresh-market-trace-r2-candidate-label-v1"
DEFAULT_SEED_START = 2026086400
SHARD_SIZE = 8
FORBIDDEN_SEED_RANGES = (
    (2026086100, 2026086115, "sealed"),
    (2026086300, 2026086307, "existing_train"),
    (2026086320, 2026086327, "validation"),
)
DEFAULT_OUTPUT_PARENT = (
    base.DEFAULT_OUTPUT_PARENT / "fresh_market_trace_labels_v1"
)


def _stable_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shard_seeds(seed_start: int, seed_count: int, shard_index: int) -> tuple[int, ...]:
    if int(seed_count) != SHARD_SIZE:
        raise ValueError("fresh labels v1 requires exactly eight seeds per shard")
    if int(shard_index) < 0:
        raise ValueError("shard index must be non-negative")
    first = int(seed_start) + int(shard_index) * int(seed_count)
    seeds = tuple(range(first, first + int(seed_count)))
    for seed in seeds:
        for low, high, name in FORBIDDEN_SEED_RANGES:
            if low <= seed <= high:
                raise ValueError(f"seed {seed} is in forbidden {name} range {low}..{high}")
    return seeds


def resolve_block_library(
    value: Path | None, smoke: bool,
) -> tuple[Path, str]:
    default = retrieval.DEFAULT_LIBRARY.resolve()
    if value is None:
        if not smoke:
            raise ValueError(
                "formal fresh shards require an explicit --block-library; "
                "the legacy R0 library is smoke-only"
            )
        root, mode = default, "legacy_default_smoke_only"
    else:
        root = value.resolve()
        mode = "explicit_smoke" if smoke else "explicit_frozen_path_base_formal"
        if not smoke and root == default:
            raise ValueError(
                "formal fresh shards reject the default legacy R0 library; "
                "pass an explicit expanded block library"
            )
    if not (root / "block_library_manifest.json").is_file():
        raise ValueError(f"block library manifest is missing: {root}")
    return root, mode


def default_output(seed_start: int, seed_count: int, shard_index: int) -> Path:
    first = int(seed_start) + int(seed_count) * int(shard_index)
    return DEFAULT_OUTPUT_PARENT / f"shard_{int(shard_index):03d}_seed_{first}"


def _candidate_counts(path: Path) -> dict[str, Any]:
    values: dict[str, list[int]] = {
        "R0_active8": [], "phase_frozen": [], "R2_all": [],
    }
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            for arm in values:
                shas = list(map(str, row["arms"][arm]["candidate_sha256"]))
                markets = list(map(int, row["arms"][arm]["market_diff"]))
                if not shas or len(shas) != len(markets) or len(shas) != len(set(shas)):
                    raise ValueError(f"invalid {arm} candidate panel at {row['state_id']}")
                values[arm].append(len(shas))
    return {
        "decisions": rows,
        "by_arm": {
            arm: {
                "min": min(counts), "max": max(counts),
                "total_candidate_rows": sum(counts),
            }
            for arm, counts in values.items()
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    shard = report["fresh_shard"]
    library = report["candidate_library"]
    return "\n".join((
        "# FRESH-MARKET-TRACE-LABELS-v1",
        "",
        f"- Status: `{report['status']}`",
        f"- Shard: {shard['shard_index']}; seeds: {shard['seeds']}",
        f"- Observed decisions: {report['scenario_contract']['states_observed']}",
        f"- Candidate-library mode: `{library['mode']}`",
        f"- Executed candidate layer: `{library['role']}`",
        f"- Candidate-library manifest SHA256: `{library['manifest_sha256']}`",
        f"- Input-contract SHA256: `{report['input_contract_sha256']}`",
        "",
        "The shard retains every scheduled decision, canonical R0-prefix plus "
        "phase-only labels, completed trajectories, and frozen input provenance.",
        "No model was trained and no candidate was committed.",
        "",
    ))


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = shard_seeds(args.seed_start, args.seed_count, args.shard_index)
    library, library_mode = resolve_block_library(args.block_library, args.smoke)
    manifest_path = library / "block_library_manifest.json"
    manifest_sha = base.residual._sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = args.expected_library_manifest_sha256
    if not args.smoke and expected is None:
        raise ValueError(
            "formal fresh shards require --expected-library-manifest-sha256"
        )
    if expected is not None and manifest_sha != str(expected).lower():
        raise ValueError("explicit block-library manifest SHA256 does not match")

    output = (
        args.output_root.resolve() if args.output_root is not None
        else default_output(args.seed_start, args.seed_count, args.shard_index)
    )
    delegated = argparse.Namespace(**vars(args))
    delegated.block_library = library
    delegated.output_root = output
    delegated.scenario_seeds = seeds
    delegated.report_schema = SCHEMA
    delegated.label_schema = LABEL_SCHEMA
    delegated.r2_row_schema = R2_ROW_SCHEMA
    report = base.run(delegated)

    counts = _candidate_counts(output / "h1_decisions.jsonl")
    r2_artifact = report["artifacts"].get("R2_candidate_labels.jsonl") or {}
    expected_r2_rows = counts["by_arm"]["R2_all"]["total_candidate_rows"]
    if (
        int(r2_artifact.get("rows", -1)) != expected_r2_rows
        or int(report["throughput"]["physical_R2_candidate_rollouts"])
        != expected_r2_rows
    ):
        raise AssertionError("complete physical R2 candidate labels were not retained")
    if counts["decisions"] != int(report["scenario_contract"]["states_observed"]):
        raise AssertionError("decision artifact count changed during fresh-shard audit")
    formal_expected = base.EXPECTED_FORMAL_STATES
    if not args.smoke and counts["decisions"] != formal_expected:
        raise AssertionError("formal shard did not retain its complete 84-step panel")

    input_contract = {
        "schema": SCHEMA,
        "train_opponents": list(base.TRAIN_OPPONENTS),
        "seats": list(base.SEATS),
        "anchors": list(map(int, base.path_v1.ANCHORS)),
        "offsets": list(map(int, base.OFFSETS)),
        "decisions_per_scenario": len(base.path_v1.ANCHORS) * len(base.OFFSETS),
        "frozen_genome": report["frozen_genome"],
        "frozen_phase_mapping": report["frozen_phase_mapping"],
        "candidate_library": report["strict_library"],
        "frozen_inputs": report["frozen_inputs"],
        "experiment_inputs": report["experiment_inputs"],
        "native_extension": report["implementation"]["native_extension"],
        "fresh_generator": retrieval._artifact(Path(__file__).resolve()),
    }
    input_contract["candidate_execution_contract"] = {
        "path_base": {
            "status": "executed",
            "native_api": "rollout_schedule_unit_override_sequence_batch",
        },
        "market_full_action_overlay": {
            "status": "not_executed_in_v1",
            "requires_composite_full_schedule_native_batch": True,
        },
    }
    input_sha = _stable_sha(input_contract)
    provenance = {
        "schema": "fresh-market-trace-shard-provenance-v1",
        "shard_index": int(args.shard_index),
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "seeds": list(seeds[:1] if args.smoke else seeds),
        "planned_formal_seeds": list(seeds),
        "smoke": bool(args.smoke),
        "input_contract_sha256": input_sha,
        "input_contract": input_contract,
        "candidate_library": {
            "root": str(library), "mode": library_mode, "role": "path_base",
            "manifest_sha256": manifest_sha,
            "deduplication_mode": manifest.get("deduplication_mode"),
            "block_count": int(report["strict_library"]["block_count"]),
            "prototype_count": int(report["strict_library"]["prototype_count"]),
        },
        "no_decision_filtering": True,
        "all_physical_R2_candidate_labels_retained": True,
    }
    provenance_path = output / "shard_provenance.json"
    base.residual._atomic_text(
        provenance_path, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
    )

    report["status"] = "fresh_shard_smoke_passed" if args.smoke else "fresh_shard_complete"
    report["decision"] = "Labels only; no selector trained and no route committed."
    report["fresh_shard"] = {
        "shard_index": int(args.shard_index),
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "seeds": list(seeds[:1] if args.smoke else seeds),
        "planned_formal_seeds": list(seeds),
        "formal_decisions_per_scenario": 84,
        "no_decision_filtering": True,
        "all_physical_R2_candidate_labels_retained": True,
    }
    report["candidate_library"] = provenance["candidate_library"]
    report["candidate_execution_contract"] = input_contract["candidate_execution_contract"]
    report["candidate_counts"] = counts
    report["input_contract_sha256"] = input_sha
    report["artifacts"][provenance_path.name] = retrieval._artifact(provenance_path)
    report["evidence_boundary"].update({
        "fresh_market_trace": True,
        "existing_train_seed_range_opened": False,
        "validation_seed_range_opened": False,
        "sealed_seed_range_opened": False,
        "all_scheduled_decisions_retained": True,
    })
    report_path = output / ("SMOKE_REPORT.json" if args.smoke else "FINAL_REPORT.json")
    markdown_path = output / ("SMOKE_REPORT.md" if args.smoke else "FINAL_REPORT.md")
    base.residual._atomic_text(
        report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    base.residual._atomic_text(markdown_path, _markdown(report))
    return report


def parser() -> argparse.ArgumentParser:
    result = base.parser()
    result.description = __doc__
    result.set_defaults(block_library=None, output_root=None)
    result.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    result.add_argument("--seed-count", type=int, default=SHARD_SIZE)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--expected-library-manifest-sha256")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

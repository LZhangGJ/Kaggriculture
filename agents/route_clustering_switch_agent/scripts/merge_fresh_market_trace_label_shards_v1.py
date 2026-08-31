#!/usr/bin/env python3
"""Strictly validate and merge independent fresh-market label shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO

import run_fresh_market_trace_labels_v1 as fresh


SCHEMA = "fresh-market-trace-label-shard-merge-v1"
FILES = (
    "h1_decisions.jsonl", "canonical_union_labels.jsonl",
    "R2_candidate_labels.jsonl",
    "frozen_trajectories.jsonl",
)


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.resolve()), "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = int(rows)
    return value


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{number}") from exc


def expected_state_ids(seeds: Sequence[int]) -> Iterator[str]:
    for opponent in fresh.base.TRAIN_OPPONENTS:
        for seed in seeds:
            for seat in fresh.base.SEATS:
                for anchor in fresh.base.path_v1.ANCHORS:
                    for offset in fresh.base.OFFSETS:
                        yield f"{opponent}:{int(seed)}:{seat}:{anchor}:{offset}"


def canonical_union(decision: Mapping[str, Any]) -> list[str]:
    return list(fresh.base.preflight.canonical_union_index(decision)["union_sha256"])


def _next_or_error(iterator: Iterator[dict[str, Any]], what: str) -> dict[str, Any]:
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ValueError(f"{what} ended before the registered Cartesian panel") from exc


def _assert_exhausted(iterator: Iterator[Any], what: str) -> None:
    try:
        next(iterator)
    except StopIteration:
        return
    raise ValueError(f"{what} has rows beyond the registered Cartesian panel")


def load_shard_report(root: Path) -> dict[str, Any]:
    root = root.resolve()
    report_path = root / "FINAL_REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != fresh.SCHEMA or report.get("status") != "fresh_shard_complete":
        raise ValueError(f"not a completed formal fresh shard: {root}")
    shard = report.get("fresh_shard") or {}
    if (
        bool(shard.get("no_decision_filtering")) is not True
        or int(shard.get("seed_count", -1)) != fresh.SHARD_SIZE
        or len(shard.get("seeds", ())) != fresh.SHARD_SIZE
        or report.get("candidate_library", {}).get("mode")
        != "explicit_frozen_path_base_formal"
    ):
        raise ValueError(f"shard contract or frozen path-base gate failed: {root}")
    report["_root"] = root
    report["_report_path"] = report_path
    return report


def validate_report_set(
    reports: Sequence[Mapping[str, Any]], expected_count: int,
) -> list[Mapping[str, Any]]:
    if len(reports) != int(expected_count) or expected_count < 1:
        raise ValueError("the number of shard reports differs from --expected-shard-count")
    ordered = sorted(reports, key=lambda row: int(row["fresh_shard"]["shard_index"]))
    indices = [int(row["fresh_shard"]["shard_index"]) for row in ordered]
    if indices != list(range(expected_count)):
        raise ValueError("shard indices must be unique and contiguous from zero")
    starts = {int(row["fresh_shard"]["seed_start"]) for row in ordered}
    hashes = {str(row["input_contract_sha256"]) for row in ordered}
    libraries = {
        str(row["candidate_library"]["manifest_sha256"]) for row in ordered
    }
    if len(starts) != 1 or len(hashes) != 1 or len(libraries) != 1:
        raise ValueError("shards disagree on seed base, frozen input hash, or library SHA")
    seed_start = next(iter(starts))
    seen: set[int] = set()
    for report in ordered:
        shard = report["fresh_shard"]
        expected = list(fresh.shard_seeds(
            seed_start, fresh.SHARD_SIZE, int(shard["shard_index"]),
        ))
        seeds = list(map(int, shard["seeds"]))
        if seeds != expected or seen.intersection(seeds):
            raise ValueError("shard seeds overlap or do not match index-derived order")
        seen.update(seeds)
    return ordered


def _verify_artifact(report: Mapping[str, Any], name: str) -> Path:
    root = Path(report["_root"])
    path = root / name
    frozen = report["artifacts"].get(name) or {}
    if not path.is_file() or _sha256_file(path) != str(frozen.get("sha256")):
        raise ValueError(f"shard artifact SHA mismatch: {path}")
    return path


def validate_and_copy_shard(
    report: Mapping[str, Any], outputs: Mapping[str, TextIO],
) -> dict[str, int]:
    paths = {name: _verify_artifact(report, name) for name in FILES}
    provenance_path = _verify_artifact(report, "shard_provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("input_contract_sha256") != report["input_contract_sha256"]
        or provenance.get("seeds") != report["fresh_shard"]["seeds"]
        or provenance.get("candidate_library", {}).get("manifest_sha256")
        != report["candidate_library"]["manifest_sha256"]
    ):
        raise ValueError("shard provenance disagrees with its final report")

    decisions = _rows(paths["h1_decisions.jsonl"])
    labels = _rows(paths["canonical_union_labels.jsonl"])
    r2_labels = _rows(paths["R2_candidate_labels.jsonl"])
    decision_count = label_count = r2_count = 0
    seeds = list(map(int, report["fresh_shard"]["seeds"]))
    r2_by_scenario: dict[tuple[str, int, int], int] = {}
    for expected_state in expected_state_ids(seeds):
        decision = _next_or_error(decisions, "decisions")
        if str(decision.get("state_id")) != expected_state:
            raise ValueError(
                f"decision order changed: expected {expected_state}, "
                f"found {decision.get('state_id')}"
            )
        union = canonical_union(decision)
        r0_count = len(decision["arms"][fresh.base.preflight.R0]["candidate_sha256"])
        for index, sha in enumerate(union):
            label = _next_or_error(labels, "canonical labels")
            expected_membership = "R0" if index < r0_count else "phase_only"
            if (
                str(label.get("state_id")) != expected_state
                or int(label.get("union_index", -1)) != index
                or str(label.get("candidate_sha256")) != sha
                or str(label.get("membership")) != expected_membership
            ):
                raise ValueError(f"canonical union order/SHA changed at {expected_state}")
            outputs["canonical_union_labels.jsonl"].write(
                json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n"
            )
            label_count += 1
        r2_arm = decision["arms"]["R2_all"]
        r2_shas = list(map(str, r2_arm["candidate_sha256"]))
        r2_markets = list(map(int, r2_arm["market_diff"]))
        if len(r2_shas) != len(r2_markets) or len(r2_shas) != len(set(r2_shas)):
            raise ValueError(f"invalid physical R2 arm at {expected_state}")
        scenario_key = (
            str(decision["opponent"]), int(decision["seed"]), int(decision["seat"]),
        )
        for candidate_index, (sha, market) in enumerate(
            zip(r2_shas, r2_markets, strict=True)
        ):
            row = _next_or_error(r2_labels, "R2 candidate labels")
            expected_row = {
                "schema": fresh.R2_ROW_SCHEMA,
                "state_id": expected_state,
                "opponent": str(decision["opponent"]),
                "seed": int(decision["seed"]),
                "seat": int(decision["seat"]),
                "anchor": int(decision["anchor"]),
                "offset": int(decision["offset"]),
                "step": int(decision["decision_step"]),
                "candidate_index": candidate_index,
                "candidate_sha256": sha,
                "market_diff": market,
            }
            try:
                int(row["outcome"])
                float(row["margin"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("R2 candidate outcome/margin is invalid") from exc
            if (
                any(row.get(key) != value for key, value in expected_row.items())
                or "path_pure" not in row
                or bool(row["path_pure"]) != (market == 0)
            ):
                raise ValueError(f"R2 candidate order/SHA/market changed at {expected_state}")
            outputs["R2_candidate_labels.jsonl"].write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            r2_count += 1
        r2_by_scenario[scenario_key] = r2_by_scenario.get(scenario_key, 0) + len(r2_shas)
        outputs["h1_decisions.jsonl"].write(
            json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n"
        )
        decision_count += 1
    _assert_exhausted(decisions, "decisions")
    _assert_exhausted(labels, "canonical labels")
    _assert_exhausted(r2_labels, "R2 candidate labels")
    if decision_count != fresh.base.EXPECTED_FORMAL_STATES:
        raise AssertionError("formal shard Cartesian size changed")

    trajectories = _rows(paths["frozen_trajectories.jsonl"])
    trajectory_count = trajectory_r2_rows = 0
    for opponent in fresh.base.TRAIN_OPPONENTS:
        for seed in seeds:
            for seat in fresh.base.SEATS:
                row = _next_or_error(trajectories, "trajectories")
                scenario_key = (opponent, int(seed), int(seat))
                expected_r2_rows = r2_by_scenario[scenario_key]
                if (
                    str(row.get("opponent")) != opponent
                    or int(row.get("seed", -1)) != seed
                    or int(row.get("seat", -1)) != seat
                    or int(row.get("decisions", -1))
                    != len(fresh.base.path_v1.ANCHORS) * len(fresh.base.OFFSETS)
                    or int(row.get("physical_candidate_rollouts", -1))
                    != expected_r2_rows
                    or row.get("completed") is not True
                ):
                    raise ValueError("trajectory order or completion contract changed")
                outputs["frozen_trajectories.jsonl"].write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                trajectory_count += 1
                trajectory_r2_rows += int(row["physical_candidate_rollouts"])
    _assert_exhausted(trajectories, "trajectories")
    if trajectory_r2_rows != r2_count:
        raise ValueError("trajectory physical candidate rollouts differ from R2 rows")
    return {
        "decisions": decision_count, "labels": label_count,
        "r2_labels": r2_count,
        "trajectories": trajectory_count,
    }


def merge(args: argparse.Namespace) -> dict[str, Any]:
    reports = validate_report_set(
        [load_shard_report(path) for path in args.shard_root],
        args.expected_shard_count,
    )
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    final_paths = {name: output / name for name in FILES}
    temp_paths = {
        name: path.with_name(path.name + f".tmp.{os.getpid()}")
        for name, path in final_paths.items()
    }
    totals = {"decisions": 0, "labels": 0, "r2_labels": 0, "trajectories": 0}
    with (
        temp_paths[FILES[0]].open("w", encoding="utf-8", newline="\n") as decisions,
        temp_paths[FILES[1]].open("w", encoding="utf-8", newline="\n") as labels,
        temp_paths[FILES[2]].open("w", encoding="utf-8", newline="\n") as r2_labels,
        temp_paths[FILES[3]].open("w", encoding="utf-8", newline="\n") as trajectories,
    ):
        handles = {
            FILES[0]: decisions, FILES[1]: labels, FILES[2]: r2_labels, FILES[3]: trajectories,
        }
        for report in reports:
            counts = validate_and_copy_shard(report, handles)
            for key, value in counts.items():
                totals[key] += int(value)
    for name in FILES:
        os.replace(temp_paths[name], final_paths[name])

    source_path = output / "source_shards.jsonl"
    with source_path.open("w", encoding="utf-8", newline="\n") as handle:
        for report in reports:
            handle.write(json.dumps({
                "schema": "fresh-market-trace-source-shard-v1",
                "shard_index": report["fresh_shard"]["shard_index"],
                "seeds": report["fresh_shard"]["seeds"],
                "input_contract_sha256": report["input_contract_sha256"],
                "candidate_library": report["candidate_library"],
                "report": _artifact(Path(report["_report_path"])),
            }, ensure_ascii=False, sort_keys=True) + "\n")

    seeds = [
        seed for report in reports for seed in report["fresh_shard"]["seeds"]
    ]
    report = {
        "schema": SCHEMA,
        "status": "fresh_shards_merged_complete",
        "shard_count": len(reports),
        "shard_indices": list(range(len(reports))),
        "seeds": seeds,
        "input_contract_sha256": reports[0]["input_contract_sha256"],
        "candidate_library": reports[0]["candidate_library"],
        "strict_order": "shard, opponent, seed, seat, anchor, offset, then candidate_index/union_index",
        "complete_cartesian_per_shard": True,
        "no_decision_filtering": True,
        "counts": totals,
        "artifacts": {
            FILES[0]: _artifact(final_paths[FILES[0]], totals["decisions"]),
            FILES[1]: _artifact(final_paths[FILES[1]], totals["labels"]),
            FILES[2]: _artifact(final_paths[FILES[2]], totals["r2_labels"]),
            FILES[3]: _artifact(final_paths[FILES[3]], totals["trajectories"]),
            source_path.name: _artifact(source_path, len(reports)),
        },
    }
    fresh.base.residual._atomic_text(
        output / "FINAL_REPORT.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--shard-root", type=Path, action="append", required=True)
    result.add_argument("--expected-shard-count", type=int, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    merge(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

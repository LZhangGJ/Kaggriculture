#!/usr/bin/env python3
"""Re-identify RouteGenome records and build one-row-per-genome aggregates."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))

from meta_agent.src.route_genome import ROUTE_GENOME_SCHEMA_VERSION, RouteGenome


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8", newline="\n")


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def _upgrade(payload: Mapping[str, Any]) -> RouteGenome:
    previous = RouteGenome.from_dict(payload)
    if previous.schema_version == ROUTE_GENOME_SCHEMA_VERSION:
        return previous
    return RouteGenome.create(
        source=previous.source,
        anchor_targets=previous.anchor_targets,
        phase_macro_counts=previous.phase_macro_counts,
        structural_events=previous.structural_events,
        market_profile=previous.market_profile,
        cash_profile=previous.cash_profile,
    )


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "p75": percentile(0.75),
        "max": ordered[-1],
    }


def _dataset_labels(source: Mapping[str, Any]) -> list[str]:
    values = source.get("datasets", []) or []
    return [str(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input RouteGenome JSONL(.gz)")
    parser.add_argument(
        "--output-records",
        type=Path,
        required=True,
        help="Re-identified execution-level JSONL(.gz)",
    )
    parser.add_argument(
        "--output-groups",
        type=Path,
        required=True,
        help="One-row-per-genome aggregate JSONL(.gz)",
    )
    parser.add_argument("--summary", type=Path, help="Defaults beside OUTPUT_GROUPS")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    records_path = args.output_records.resolve()
    groups_path = args.output_groups.resolve()
    summary_path = (
        args.summary or groups_path.with_name(groups_path.name + ".summary.json")
    ).resolve()
    if not input_path.is_file():
        parser.error(f"input does not exist: {input_path}")
    if len({input_path, records_path, groups_path, summary_path}) != 4:
        parser.error("input and output paths must be distinct")
    for path in (records_path, groups_path, summary_path):
        if path.exists() and not args.overwrite:
            parser.error(f"output exists; pass --overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[str, dict[str, Any]] = {}
    input_schema_counts: Counter[int] = Counter()
    input_ids: set[str] = set()
    record_count = 0

    with _open_text(records_path, "w") as output_handle:
        for payload in _json_lines(input_path):
            input_schema_counts[int(payload.get("schema_version", 1))] += 1
            input_ids.add(str(payload["genome_id"]))
            genome = _upgrade(payload)
            output_handle.write(json.dumps(
                genome.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n")
            record_count += 1

            group = groups.setdefault(genome.genome_id, {
                "genetic_payload": genome.genetic_payload(),
                "source_ids": [],
                "teams": Counter(),
                "datasets": Counter(),
                "outcomes": Counter(),
                "own_rewards": [],
                "opponent_rewards": [],
                "reward_margins": [],
                "cash": defaultdict(list),
                "market_prices": defaultdict(list),
            })
            source = genome.source
            group["source_ids"].append(str(source["source_id"]))
            group["teams"][str(source.get("team_name", ""))] += 1
            group["datasets"].update(_dataset_labels(source))
            group["outcomes"][str(source.get("result", "unknown"))] += 1
            own_reward = float(source.get("final_reward", 0.0) or 0.0)
            opponent_reward = float(source.get("opponent_reward", 0.0) or 0.0)
            group["own_rewards"].append(own_reward)
            group["opponent_rewards"].append(opponent_reward)
            group["reward_margins"].append(own_reward - opponent_reward)
            for key, value in genome.cash_profile.items():
                if isinstance(value, (int, float)):
                    group["cash"][str(key)].append(float(value))
            for row in genome.market_profile:
                observed = row.get("observed_price") or {}
                if "median" in observed:
                    key = (str(row.get("operation", "")), str(row.get("item") or ""))
                    group["market_prices"][key].append(float(observed["median"]))

    group_size_histogram: Counter[int] = Counter()
    with _open_text(groups_path, "w") as output_handle:
        for genome_id in sorted(groups):
            group = groups[genome_id]
            source_ids = sorted(group["source_ids"])
            sample_count = len(source_ids)
            group_size_histogram[sample_count] += 1
            market_environment = [
                {
                    "operation": operation,
                    "item": item or None,
                    "observed_median_price": _numeric_summary(values),
                }
                for (operation, item), values in sorted(group["market_prices"].items())
            ]
            row = {
                "schema_version": ROUTE_GENOME_SCHEMA_VERSION,
                "kind": f"route_genome_v{ROUTE_GENOME_SCHEMA_VERSION}_aggregate",
                "genome_id": genome_id,
                "genetic_payload": group["genetic_payload"],
                "sample_count": sample_count,
                "source_ids": source_ids,
                "team_counts": dict(group["teams"].most_common()),
                "dataset_counts": dict(group["datasets"].most_common()),
                "outcome_counts": dict(sorted(group["outcomes"].items())),
                "reward_summary": {
                    "own": _numeric_summary(group["own_rewards"]),
                    "opponent": _numeric_summary(group["opponent_rewards"]),
                    "margin": _numeric_summary(group["reward_margins"]),
                },
                "cash_summary": {
                    key: _numeric_summary(values)
                    for key, values in sorted(group["cash"].items())
                },
                "market_environment_summary": market_environment,
            }
            output_handle.write(json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n")

    unique_genomes = len(groups)
    duplicate_groups = sum(count for size, count in group_size_histogram.items() if size > 1)
    records_in_duplicate_groups = sum(
        size * count for size, count in group_size_histogram.items() if size > 1
    )
    summary = {
        "schema_version": ROUTE_GENOME_SCHEMA_VERSION,
        "kind": f"route_genome_v{ROUTE_GENOME_SCHEMA_VERSION}_aggregation_summary",
        "input": str(input_path),
        "output_records": str(records_path),
        "output_groups": str(groups_path),
        "input_schema_counts": {
            str(key): value for key, value in sorted(input_schema_counts.items())
        },
        "records": record_count,
        "input_unique_genome_ids": len(input_ids),
        "unique_genomes": unique_genomes,
        "collapsed_records": record_count - unique_genomes,
        "aggregation_rate": (
            (record_count - unique_genomes) / record_count if record_count else 0.0
        ),
        "duplicate_groups": duplicate_groups,
        "records_in_duplicate_groups": records_in_duplicate_groups,
        "maximum_group_size": max(group_size_histogram, default=0),
        "group_size_histogram": {
            str(size): count for size, count in sorted(group_size_histogram.items())
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

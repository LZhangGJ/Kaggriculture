#!/usr/bin/env python3
"""Merge compatible macro-intent artifact directories at seat grain."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def seat_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["episode_id"]), str(row["player_index"]), str(row["submission_id"])


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.input_dir]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        json.loads((directory / "summary_v1.json").read_text(encoding="utf-8"))
        for directory in input_dirs
    ]
    feature_files = [np.load(directory / "intent_features_v1.npz") for directory in input_dirs]
    schema_fields = ("anchors", "intent_categories", "macro_actions", "stage_width")
    for field in schema_fields:
        reference = feature_files[0][field]
        if any(not np.array_equal(reference, features[field]) for features in feature_files[1:]):
            raise ValueError(f"incompatible feature schema: {field}")

    selected_rows: list[dict[str, Any]] = []
    selected_indices: list[list[int]] = []
    selected_keys_by_source: list[set[tuple[str, str, str]]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    duplicate_seats = 0
    metadata_header: list[str] = []
    for source_index, directory in enumerate(input_dirs):
        rows = read_csv(directory / "intent_seats_v1.csv")
        if rows and not metadata_header:
            metadata_header = list(rows[0])
        indices: list[int] = []
        source_keys: set[tuple[str, str, str]] = set()
        for row_index, row in enumerate(rows):
            key = seat_key(row)
            if key in seen_keys:
                duplicate_seats += 1
                continue
            seen_keys.add(key)
            source_keys.add(key)
            indices.append(row_index)
            merged = dict(row)
            merged["source_artifact"] = directory.name
            merged["seat_index"] = len(selected_rows)
            selected_rows.append(merged)
        selected_indices.append(indices)
        selected_keys_by_source.append(source_keys)

    layout = np.concatenate(
        [features["layout"][indices] for features, indices in zip(feature_files, selected_indices)],
        axis=0,
    )
    stage_actions = np.concatenate(
        [features["stage_actions"][indices] for features, indices in zip(feature_files, selected_indices)],
        axis=0,
    )
    cumulative_actions = np.concatenate(
        [features["cumulative_actions"][indices] for features, indices in zip(feature_files, selected_indices)],
        axis=0,
    )
    seat_ids = np.asarray(
        [f"{row['episode_id']}:{row['player_index']}:{row['submission_id']}" for row in selected_rows],
        dtype=str,
    )
    temporary_npz = output_dir / "intent_features_v1.partial.npz"
    np.savez_compressed(
        temporary_npz,
        seat_id=seat_ids,
        layout=layout,
        stage_actions=stage_actions,
        cumulative_actions=cumulative_actions,
        anchors=feature_files[0]["anchors"],
        intent_categories=feature_files[0]["intent_categories"],
        macro_actions=feature_files[0]["macro_actions"],
        stage_width=feature_files[0]["stage_width"],
    )
    os.replace(temporary_npz, output_dir / "intent_features_v1.npz")
    write_csv(
        output_dir / "intent_seats_v1.csv",
        selected_rows,
        [*metadata_header, "source_artifact"],
    )

    event_header: list[str] = []
    event_count = 0
    operation_counts: Counter[str] = Counter()
    spatial_count = 0
    missing_coordinate = 0
    success_sum = 0
    success_count = 0
    event_temporary = output_dir / "macro_events_v1.csv.partial"
    with event_temporary.open("w", encoding="utf-8-sig", newline="") as output_stream:
        writer: csv.DictWriter[str] | None = None
        for directory, selected_keys in zip(input_dirs, selected_keys_by_source):
            with (directory / "macro_events_v1.csv").open(
                encoding="utf-8-sig", newline=""
            ) as input_stream:
                reader = csv.DictReader(input_stream)
                if not event_header:
                    event_header = list(reader.fieldnames or [])
                    writer = csv.DictWriter(output_stream, fieldnames=event_header)
                    writer.writeheader()
                assert writer is not None
                for row in reader:
                    if seat_key(row) not in selected_keys:
                        continue
                    writer.writerow(row)
                    event_count += 1
                    operation = str(row["operation"])
                    operation_counts[operation] += 1
                    if operation in {"PLANT", "PLACE", "BUILD_COOP", "BUILD_PASTURE"}:
                        spatial_count += 1
                        missing_coordinate += int(row["row"] == "")
                        if row["observed_success"] != "":
                            success_sum += int(row["observed_success"])
                            success_count += 1
    os.replace(event_temporary, output_dir / "macro_events_v1.csv")

    failure_rows: list[dict[str, Any]] = []
    for directory in input_dirs:
        for row in read_csv(directory / "failures_v1.csv"):
            value = dict(row)
            value["source_artifact"] = directory.name
            failure_rows.append(value)
    write_csv(
        output_dir / "failures_v1.csv",
        failure_rows,
        ["episode_id", "failure_kind", "detail", "source_artifact"],
    )

    submissions: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        key = str(row["submission_id"])
        value = submissions.setdefault(
            key,
            {
                "rank": int(row["rank"]),
                "team_id": row["team_id"],
                "team": row["team"],
                "seat_count": 0,
                "WIN": 0,
                "LOSS": 0,
                "TIE": 0,
            },
        )
        value["seat_count"] += 1
        value[row["result"]] += 1

    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "merged_macro_intent_artifacts",
        "input_artifacts": [str(directory) for directory in input_dirs],
        "output_grain": "one row per (episode_id, player_index, submission_id)",
        "input_seats": sum(summary["output_seats"] for summary in summaries),
        "duplicate_seats_removed": duplicate_seats,
        "output_seats": len(selected_rows),
        "parsed_episodes": len({row["episode_id"] for row in selected_rows}),
        "failures": len(failure_rows),
        "failure_kinds": dict(Counter(row["failure_kind"] for row in failure_rows)),
        "mapping_methods": dict(Counter(row["mapping_method"] for row in selected_rows)),
        "replay_step_lengths": dict(Counter(row["replay_steps"] for row in selected_rows)),
        "anchors": feature_files[0]["anchors"].astype(int).tolist(),
        "stage_width": int(feature_files[0]["stage_width"]),
        "stage_count": int(stage_actions.shape[1]),
        "intent_categories": feature_files[0]["intent_categories"].astype(str).tolist(),
        "macro_actions": feature_files[0]["macro_actions"].astype(str).tolist(),
        "feature_shapes": {
            "layout": list(layout.shape),
            "stage_actions": list(stage_actions.shape),
            "cumulative_actions": list(cumulative_actions.shape),
        },
        "macro_event_rows": event_count,
        "macro_event_operations": dict(operation_counts),
        "spatial_intent_rows": spatial_count,
        "spatial_missing_coordinate": missing_coordinate,
        "spatial_observed_success_rate": success_sum / success_count if success_count else None,
        "submissions": submissions,
    }
    (output_dir / "summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select a diverse, auditable parent panel for the first genome experiment."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO


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


def _outcome_score(counts: dict[str, Any]) -> float:
    wins = int(counts.get("win", 0))
    ties = int(counts.get("tie", 0))
    games = sum(int(value) for value in counts.values())
    return (wins + 0.5 * ties) / games if games else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--focus-dataset", default="our_latest")
    parser.add_argument("--focus-team", default="QQ Farming")
    parser.add_argument("--focus-count", type=int, default=4)
    parser.add_argument("--support-count", type=int, default=8)
    parser.add_argument("--performance-count", type=int, default=4)
    parser.add_argument("--novel-count", type=int, default=4)
    parser.add_argument("--minimum-performance-support", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = {str(row["genome_id"]): row for row in _json_lines(args.groups.resolve())}
    mappings = {str(row["genome_id"]): row for row in _json_lines(args.family_map.resolve())}
    family_payload = json.loads(args.families.resolve().read_text(encoding="utf-8"))
    families = {str(row["family"]): row for row in family_payload["families"]}
    records_by_genome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    focus_records = []
    for row in _json_lines(args.records.resolve()):
        genome_id = str(row["genome_id"])
        records_by_genome[genome_id].append(row)
        source = row["source"]
        if (
            args.focus_dataset in (source.get("datasets", []) or [])
            and str(source.get("team_name", "")) == args.focus_team
        ):
            focus_records.append(row)

    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()

    def add(genome_id: str, role: str, detail: str) -> bool:
        if genome_id in selected_ids:
            return False
        if genome_id not in groups or genome_id not in mappings:
            raise KeyError(f"missing group or family mapping for {genome_id}")
        selected_ids.add(genome_id)
        selected.append({"genome_id": genome_id, "role": role, "detail": detail})
        return True

    focus_by_family: dict[str, set[str]] = defaultdict(set)
    for record in focus_records:
        genome_id = str(record["genome_id"])
        focus_by_family[str(mappings[genome_id]["family"])].add(genome_id)
    focus_candidates = []
    for family, genome_ids in focus_by_family.items():
        best = min(
            genome_ids,
            key=lambda genome_id: (
                float(mappings[genome_id]["distance_to_medoid"]),
                genome_id,
            ),
        )
        focus_candidates.append((
            -sum(
                1 for record in focus_records
                if str(mappings[str(record["genome_id"])]["family"]) == family
            ),
            family,
            best,
        ))
    for negative_count, family, genome_id in sorted(focus_candidates)[: args.focus_count]:
        add(
            genome_id,
            "focus_family_nearest",
            f"nearest focus-team genome in {family}; focus executions={-negative_count}",
        )

    ordered_families = list(family_payload["families"])
    support_added = 0
    for family in ordered_families:
        if support_added >= args.support_count:
            break
        if add(
            str(family["medoid_genome_id"]),
            "support_medoid",
            f"{family['family']} support={family['execution_support']}",
        ):
            support_added += 1

    performance_candidates = sorted(
        (
            family for family in ordered_families
            if int(family["execution_support"]) >= args.minimum_performance_support
        ),
        key=lambda family: (
            -_outcome_score(family["historical_outcome_counts"]),
            -int(family["execution_support"]),
            str(family["family"]),
        ),
    )
    performance_added = 0
    for family in performance_candidates:
        if performance_added >= args.performance_count:
            break
        score = _outcome_score(family["historical_outcome_counts"])
        if add(
            str(family["medoid_genome_id"]),
            "historical_performance_medoid",
            f"{family['family']} historical score={score:.4f}; not a causal fitness claim",
        ):
            performance_added += 1

    novel_candidates = sorted(
        (
            family for family in ordered_families
            if int(family["genome_count"]) == 1
            and int(family["historical_outcome_counts"].get("win", 0)) > 0
        ),
        key=lambda family: (
            -_outcome_score(family["historical_outcome_counts"]),
            -float(family["historical_mean_reward"]),
            str(family["family"]),
        ),
    )
    novel_added = 0
    for family in novel_candidates:
        if novel_added >= args.novel_count:
            break
        if add(
            str(family["medoid_genome_id"]),
            "winning_singleton_novelty",
            f"{family['family']} is a winning singleton at distance threshold {family_payload['threshold']}",
        ):
            novel_added += 1

    def carrier(genome_id: str, prefer_focus: bool) -> dict[str, Any]:
        candidates = records_by_genome[genome_id]
        if not candidates:
            raise KeyError(f"no execution record for {genome_id}")
        if prefer_focus:
            focused = [
                row for row in candidates
                if args.focus_dataset in (row["source"].get("datasets", []) or [])
                and str(row["source"].get("team_name", "")) == args.focus_team
            ]
            if focused:
                candidates = focused
        chosen = max(
            candidates,
            key=lambda row: (
                {"win": 2, "tie": 1, "loss": 0}.get(str(row["source"].get("result")), -1),
                float(row["source"].get("final_reward", 0.0) or 0.0),
                float(row["cash_profile"].get("minimum", 0.0) or 0.0),
                str(row["source"]["source_id"]),
            ),
        )
        source = chosen["source"]
        replay_sources = list(source.get("replay_sources", []) or [])
        return {
            "source_id": str(source["source_id"]),
            "episode_id": int(source["episode_id"]),
            "player_index": int(source["player_index"]),
            "team_name": str(source.get("team_name", "")),
            "opponent_team_name": str(source.get("opponent_team_name", "")),
            "result": str(source.get("result", "unknown")),
            "final_reward": float(source.get("final_reward", 0.0) or 0.0),
            "replay_path": str(replay_sources[0]["absolute_path"]) if replay_sources else None,
        }

    parents = []
    for index, selection in enumerate(selected, 1):
        genome_id = selection["genome_id"]
        mapping = mappings[genome_id]
        family = families[str(mapping["family"])]
        parents.append({
            "parent_id": f"P{index:03d}",
            **selection,
            "family": str(mapping["family"]),
            "distance_to_family_medoid": float(mapping["distance_to_medoid"]),
            "is_family_medoid": bool(mapping["is_medoid"]),
            "genome_sample_count": int(groups[genome_id]["sample_count"]),
            "family_genome_count": int(family["genome_count"]),
            "family_execution_support": int(family["execution_support"]),
            "family_historical_outcome_counts": family["historical_outcome_counts"],
            "carrier": carrier(
                genome_id, selection["role"] == "focus_family_nearest"
            ),
        })

    payload = {
        "schema_version": 1,
        "kind": "route_genome_initial_parent_panel",
        "family_set_id": family_payload["family_set_id"],
        "focus": {
            "dataset": args.focus_dataset,
            "team": args.focus_team,
            "execution_records": len(focus_records),
            "unique_genomes": len({str(row["genome_id"]) for row in focus_records}),
            "families": len(focus_by_family),
        },
        "selection_policy": {
            "focus_family_nearest": args.focus_count,
            "support_medoid": args.support_count,
            "historical_performance_medoid": args.performance_count,
            "winning_singleton_novelty": args.novel_count,
            "warning": (
                "Replay outcomes are observational and opponent-confounded; this panel is for "
                "compiler and mutation coverage, not a final fitness ranking."
            ),
        },
        "parents": parents,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "parents": len(parents),
        "roles": {
            role: sum(parent["role"] == role for parent in parents)
            for role in sorted({parent["role"] for parent in parents})
        },
        "families": len({parent["family"] for parent in parents}),
    }, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

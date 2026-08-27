#!/usr/bin/env python3
"""Append a deterministic quality-diversity panel of replay tapes to a route library."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))

from meta_agent.src.route_genome import MACRO_KEYS, PRODUCTION_KINDS
from meta_agent.src.teammate_expanded_routes import load_action_tapes


def _open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(
        "r", encoding="utf-8"
    )


def _records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".gz":
        with _open_text(path) as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    byte_limit = path.stat().st_size
    with path.open("rb") as handle:
        while handle.tell() < byte_limit:
            line = handle.readline(byte_limit - handle.tell())
            if not line.endswith(b"\n"):
                break
            if line.strip():
                yield json.loads(line)


def _source_priority(row: Mapping[str, Any], focus_dataset: str) -> tuple[Any, ...]:
    source = row["source"]
    datasets = {str(value) for value in source.get("datasets", ())}
    own = float(source.get("final_reward", 0.0) or 0.0)
    other = float(source.get("opponent_reward", 0.0) or 0.0)
    return (
        str(source.get("result", "")) == "win",
        focus_dataset in datasets,
        float(row.get("cash_profile", {}).get("minimum", 0.0) or 0.0) >= 0.0,
        own - other,
        own,
        str(source.get("source_id", "")),
    )


def _feature(row: Mapping[str, Any], market_keys: Sequence[tuple[str, str]]) -> np.ndarray:
    values: list[float] = []
    anchors = {int(value["step"]): value for value in row.get("anchor_targets", ())}
    for step in (168, 288, 432, 576, 719):
        counts = anchors.get(step, {}).get("counts", {})
        values.extend(math.log1p(max(0, int(counts.get(key, 0)))) for key in PRODUCTION_KINDS)
    phases = {int(value["phase"]): value for value in row.get("phase_macro_counts", ())}
    for phase in range(10):
        counts = phases.get(phase, {}).get("counts", {})
        values.extend(math.log1p(max(0, int(counts.get(key, 0)))) for key in MACRO_KEYS)
    market = {
        (str(value.get("operation", "")), str(value.get("item") or "")): value
        for value in row.get("market_profile", ())
    }
    for key in market_keys:
        item = market.get(key, {})
        values.append(math.log1p(max(0, int(item.get("orders", 0) or 0))))
        values.append(math.log1p(max(0, int(item.get("quantity", 0) or 0))))
    return np.asarray(values, dtype=np.float32)


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(values, return_inverse=True)
    if len(unique) <= 1:
        return np.zeros(len(values), dtype=np.float32)
    return inverse.astype(np.float32) / float(len(unique) - 1)


def _reference(row: Mapping[str, Any]) -> Path:
    references = list(row["source"].get("replay_sources", ()) or ())
    existing = [Path(str(value["absolute_path"])) for value in references]
    return next((path for path in existing if path.is_file()), existing[0])


def _next_family_ordinal(entries: Sequence[Mapping[str, Any]], prefix: str) -> int:
    pattern = re.compile(re.escape(prefix) + r"(\d+)$")
    ordinals = [
        int(match.group(1))
        for entry in entries
        if (match := pattern.fullmatch(str(entry.get("family", ""))))
    ]
    return max(ordinals, default=0) + 1


def _load_replay(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        import orjson
    except ImportError:
        return json.loads(raw)
    return orjson.loads(raw)


def _tape(replay: Mapping[str, Any], player: int) -> list[dict[str, Any]]:
    steps = list(replay.get("steps", ()) or ())
    if len(steps) < 720:
        raise ValueError(f"incomplete replay with {len(steps)} steps")
    result = []
    for recorded_step in range(1, 720):
        action = steps[recorded_step][player].get("action") or {}
        result.append({
            "farmer": list(action.get("farmer") or ["PASS"]),
            "hands": [list(value or ["PASS"]) for value in action.get("hands", ()) or ()],
            "market": [list(value) for value in action.get("market", ()) or ()],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--base-actions", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--focus-dataset", default="our_latest")
    parser.add_argument("--focus-count", type=int, default=24)
    parser.add_argument("--team-seed-count", type=int, default=64)
    parser.add_argument("--team-cap", type=int, default=8)
    parser.add_argument("--quality-weight", type=float, default=0.15)
    parser.add_argument("--family-prefix", default="NR")
    args = parser.parse_args()

    if (
        args.count <= 0
        or args.focus_count < 0
        or args.team_seed_count < 0
        or args.team_cap <= 0
    ):
        parser.error(
            "count/team-cap must be positive and focus/team-seed counts non-negative"
        )
    metadata = json.loads(args.base_metadata.read_text(encoding="utf-8"))
    base_entries = list(metadata["opponent_routes"])
    first_family_ordinal = _next_family_ordinal(base_entries, args.family_prefix)
    used_sources = {
        str(value["route_id"])
        for value in base_entries
        if ":" in str(value.get("route_id", ""))
    }
    used_sources.update(
        str(value.get("provenance", {}).get("source_id", "")) for value in base_entries
    )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    execution_records = 0
    market_keys: set[tuple[str, str]] = set()
    for row in _records(args.records.resolve()):
        execution_records += 1
        source = row.get("source", {})
        if int(source.get("replay_steps", 0) or 0) < 720:
            continue
        if str(source.get("result", "")) == "loss":
            continue
        if str(source.get("source_id", "")) in used_sources:
            continue
        if not source.get("replay_sources"):
            continue
        groups[str(row["genome_id"])].append(row)
        for value in row.get("market_profile", ()):
            market_keys.add((str(value.get("operation", "")), str(value.get("item") or "")))
    representatives = [
        max(values, key=lambda row: _source_priority(row, args.focus_dataset))
        for _, values in sorted(groups.items())
    ]
    representatives = [row for row in representatives if _reference(row).is_file()]
    if len(representatives) < args.count:
        parser.error(f"only {len(representatives)} eligible unique genomes for {args.count} routes")

    market_order = sorted(market_keys)
    raw_features = np.stack([_feature(row, market_order) for row in representatives])
    median = np.median(raw_features, axis=0)
    scale = np.quantile(raw_features, .75, axis=0) - np.quantile(raw_features, .25, axis=0)
    scale = np.where(scale > 1e-6, scale, np.std(raw_features, axis=0))
    active = scale > 1e-6
    features = ((raw_features[:, active] - median[active]) / scale[active]).astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features /= np.where(norms > 1e-9, norms, 1.0)
    rewards = np.asarray(
        [float(row["source"].get("final_reward", 0.0) or 0.0) for row in representatives]
    )
    margins = np.asarray([
        float(row["source"].get("final_reward", 0.0) or 0.0)
        - float(row["source"].get("opponent_reward", 0.0) or 0.0)
        for row in representatives
    ])
    support = np.asarray([len(groups[str(row["genome_id"])]) for row in representatives])
    quality = (
        .45 * _percentile_ranks(rewards)
        + .35 * _percentile_ranks(margins)
        + .20 * _percentile_ranks(np.log1p(support))
    )
    teams = [str(row["source"].get("team_name", "") or "unknown") for row in representatives]
    datasets = [set(map(str, row["source"].get("datasets", ()) or ())) for row in representatives]

    selected: list[int] = []
    selected_set: set[int] = set()
    team_counts: Counter[str] = Counter()
    reasons: dict[int, str] = {}
    selection_distances: dict[int, float | None] = {}

    def add(
        index: int,
        reason: str,
        *,
        cap: int | None = None,
        distance: float | None = None,
    ) -> bool:
        if index in selected_set or len(selected) >= args.count:
            return False
        limit = cap if cap is not None else args.team_cap
        if team_counts[teams[index]] >= limit:
            return False
        selected.append(index)
        selected_set.add(index)
        team_counts[teams[index]] += 1
        reasons[index] = reason
        selection_distances[index] = distance
        return True

    team_seeds = []
    for team in sorted(set(teams)):
        candidates = [index for index, value in enumerate(teams) if value == team]
        team_seeds.append(max(
            candidates, key=lambda index: (quality[index], margins[index])
        ))
    team_seeds.sort(
        key=lambda index: (quality[index], margins[index], teams[index]), reverse=True
    )
    for index in team_seeds[:args.team_seed_count]:
        add(index, "best_per_team")
    focus = [index for index, values in enumerate(datasets) if args.focus_dataset in values]
    focus.sort(key=lambda index: (quality[index], margins[index]), reverse=True)
    for index in focus:
        if sum(args.focus_dataset in datasets[value] for value in selected) >= args.focus_count:
            break
        add(index, "focus_dataset", cap=max(args.focus_count, args.team_cap))

    if selected:
        similarities = features @ features[np.asarray(selected)].T
        minimum_distance = np.min(2.0 - 2.0 * similarities, axis=1)
    else:
        minimum_distance = np.full(len(representatives), 2.0, dtype=np.float32)
    while len(selected) < args.count:
        eligible = np.asarray([
            index not in selected_set and team_counts[teams[index]] < args.team_cap
            for index in range(len(representatives))
        ])
        if not bool(np.any(eligible)):
            parser.error("team cap prevents filling the requested panel")
        objective = minimum_distance + args.quality_weight * quality
        objective[~eligible] = -np.inf
        index = int(np.argmax(objective))
        add(
            index,
            "farthest_quality_diversity",
            distance=float(minimum_distance[index]),
        )
        distance = 2.0 - 2.0 * (features @ features[index])
        minimum_distance = np.minimum(minimum_distance, distance)

    tapes = load_action_tapes(args.base_actions)
    output_entries = list(base_entries)
    provenance = []
    for rank, index in enumerate(selected, 1):
        row = representatives[index]
        source = row["source"]
        family = f"{args.family_prefix}{first_family_ordinal + rank - 1:03d}"
        route_id = f"replay:{source['source_id']}"
        replay_path = _reference(row)
        tapes[route_id] = _tape(_load_replay(replay_path), int(source["player_index"]))
        entry = {
            "family": family,
            "alias": f"new-replay-{source['source_id']}",
            "route_id": route_id,
            "team": teams[index],
            "support": int(support[index]),
            "selected": True,
            "drop_reason": None,
            "source_execution_hard_failures": None,
            "provenance": {
                "source_id": str(source["source_id"]),
                "genome_id": str(row["genome_id"]),
                "datasets": sorted(datasets[index]),
                "replay_path": str(replay_path.resolve()),
                "historical_result": str(source.get("result", "")),
                "historical_reward": float(source.get("final_reward", 0.0) or 0.0),
                "historical_margin": float(margins[index]),
                "selection_reason": reasons[index],
                "quality_score": float(quality[index]),
                "nearest_selected_distance_when_added": selection_distances[index],
            },
        }
        output_entries.append(entry)
        provenance.append(entry)

    packed = zlib.compress(
        json.dumps(tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), level=9
    )
    output_metadata = dict(metadata)
    output_metadata["opponent_routes"] = output_entries
    output_metadata["selected"] = [value for value in output_entries if value.get("selected", True)]
    output_metadata["actions_file"] = str(args.output_actions.resolve())
    output_metadata["actions_sha256"] = hashlib.sha256(packed).hexdigest()
    for path in (args.output_actions, args.output_metadata, args.output_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_actions.write_bytes(packed)
    args.output_metadata.write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "replay-route-quality-diversity-panel-v1",
        "records": str(args.records.resolve()),
        "base_route_count": len(base_entries),
        "execution_records": execution_records,
        "eligible_unique_genomes": len(representatives),
        "selected_route_count": len(selected),
        "total_route_count": len(output_entries),
        "family_prefix": args.family_prefix,
        "first_family_ordinal": first_family_ordinal,
        "selection": {
            "focus_dataset": args.focus_dataset,
            "focus_count": args.focus_count,
            "team_seed_count": args.team_seed_count,
            "team_cap": args.team_cap,
            "quality_weight": args.quality_weight,
            "feature_dimensions": int(features.shape[1]),
            "observed_market_prices_used": False,
            "team_counts": dict(team_counts.most_common()),
            "reason_counts": dict(Counter(reasons.values())),
        },
        "actions_sha256": hashlib.sha256(packed).hexdigest(),
        "routes": provenance,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_actions": str(args.output_actions.resolve()),
        "output_metadata": str(args.output_metadata.resolve()),
        "selected_routes": len(selected),
        "total_routes": len(output_entries),
        "eligible_unique_genomes": len(representatives),
        "focus_selected": sum(args.focus_dataset in datasets[index] for index in selected),
        "teams": len(team_counts),
        "reason_counts": dict(Counter(reasons.values())),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Verify official V7 labels, raw replay hashes, and trace invariants."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from kaggriculture_lab.official_replay_v7 import (
    OFFICIAL_V7_MANIFEST_SCHEMA,
    read_v7_label_bundle,
    replay_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != OFFICIAL_V7_MANIFEST_SCHEMA:
        parser.error(f"unsupported manifest schema: {manifest.get('schema')!r}")
    dataset_dir = manifest_path.parent
    raw_dir = Path(manifest["source_replay_dir"])
    errors: list[str] = []
    quality: Counter[str] = Counter()
    teams: Counter[str] = Counter()
    clusters: Counter[int] = Counter()
    action_hashes: set[str] = set()
    samples = 0
    actors = 0
    verified_bytes = 0
    for row in manifest.get("episodes", []):
        episode_id = int(row["episode_id"])
        label_path = dataset_dir / row["label_file"]
        raw_path = raw_dir / row["raw_file"]
        if not label_path.is_file():
            errors.append(f"missing label: {label_path}")
            continue
        if not raw_path.is_file():
            errors.append(f"missing raw replay: {raw_path}")
            continue
        bundle = read_v7_label_bundle(label_path)
        if int(bundle["episode_id"]) != episode_id:
            errors.append(f"episode mismatch in {label_path}")
        digest = replay_sha256(raw_path)
        if digest != bundle.get("raw_sha256"):
            errors.append(f"raw SHA-256 mismatch: {raw_path}")
        verified_bytes += raw_path.stat().st_size
        quality[str(bundle["quality"]["tier"])] += 1
        for actor in bundle.get("actors", []):
            actors += 1
            teams[str(actor["teacher"])] += 1
            clusters[int(actor["teacher_cluster"])] += 1
            action_hashes.add(str(actor["action_hash"]))
            traces = list(actor.get("traces", []))
            values = list(actor.get("value_targets", []))
            samples += len(traces)
            if len(traces) != len(values):
                errors.append(
                    f"trace/value length mismatch: episode={episode_id} player={actor['player']}"
                )
            audit = actor.get("trace_audit", {})
            for field in (
                "invalid_probability_rows",
                "missing_chain_ids",
                "invalid_stage_indices",
                "discontinuous_routes",
            ):
                if int(audit.get(field, -1)) != 0:
                    errors.append(
                        f"trace audit {field}={audit.get(field)}: "
                        f"episode={episode_id} player={actor['player']}"
                    )
    daily_files = list(raw_dir.glob("*.json"))
    report: dict[str, Any] = {
        "schema": "kaggriculture.official-replay-v7-audit.v1",
        "manifest": str(manifest_path),
        "valid": not errors,
        "errors": errors,
        "episodes": len(manifest.get("episodes", [])),
        "actors": actors,
        "samples": samples,
        "unique_teams": len(teams),
        "unique_action_trajectories": len(action_hashes),
        "quality_tiers": dict(sorted(quality.items())),
        "clusters": {str(key): value for key, value in sorted(clusters.items())},
        "verified_selected_raw_bytes": verified_bytes,
        "source_daily_json_files": len(daily_files),
        "source_daily_json_bytes": sum(path.stat().st_size for path in daily_files),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

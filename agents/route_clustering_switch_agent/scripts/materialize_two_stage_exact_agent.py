#!/usr/bin/env python3
"""Materialize the six-route deployable exact-state agent assets."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import zlib
from pathlib import Path

import numpy as np


def _load_actions(path: Path) -> dict:
    return json.loads(zlib.decompress(path.read_bytes()))


def _compact_model(path: Path) -> tuple[bytes, str, list[str], list[int]]:
    values: dict[str, np.ndarray] = {}
    used: set[str] = set()
    state_counts: list[int] = []
    with np.load(path, allow_pickle=False) as saved:
        opening = str(saved["opening"])
        values["opening"] = np.asarray(opening)
        used.add(opening)
        for stage in ("early", "late"):
            states = saved[f"{stage}_states"].astype(np.float32)
            old_targets = saved[f"{stage}_targets"].astype(str)
            old_actions = saved[f"{stage}_actions"].astype(np.int32)
            chosen = old_targets[old_actions]
            targets = np.asarray([opening, *sorted(set(chosen) - {opening})])
            index = {family: position for position, family in enumerate(targets)}
            actions = np.asarray([index[str(family)] for family in chosen], dtype=np.int32)
            values[f"{stage}_states"] = states
            values[f"{stage}_actions"] = actions
            values[f"{stage}_targets"] = targets
            values[f"{stage}_checkpoint"] = np.asarray(
                int(saved[f"{stage}_checkpoint"])
            )
            used.update(targets.tolist())
            state_counts.append(len(states))
    packed = io.BytesIO()
    np.savez_compressed(packed, **values)
    return packed.getvalue(), opening, sorted(used), state_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-actions", type=Path, required=True)
    parser.add_argument("--parent-metadata", type=Path, required=True)
    parser.add_argument("--shared-actions", type=Path, required=True)
    parser.add_argument("--shared-manifest", type=Path, required=True)
    parser.add_argument("--extra-actions", type=Path, action="append", default=[])
    parser.add_argument("--extra-manifest", type=Path, action="append", default=[])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--holdout-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packed_model, opening, families, state_counts = _compact_model(args.model)
    parent_metadata = json.loads(args.parent_metadata.read_text(encoding="utf-8"))
    parent_rows = {
        str(row["family"]): row for row in parent_metadata["opponent_routes"]
    }
    if opening not in parent_rows:
        raise KeyError(f"opening route missing from parent library: {opening}")
    shared_manifest = json.loads(args.shared_manifest.read_text(encoding="utf-8"))
    shared_rows = {str(row["family"]): row for row in shared_manifest["routes"]}
    parent_actions = _load_actions(args.parent_actions)
    shared_actions = _load_actions(args.shared_actions)
    for path in args.extra_manifest:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        shared_rows.update({str(row["family"]): row for row in manifest["routes"]})
    for path in args.extra_actions:
        shared_actions.update(_load_actions(path))

    opening_row = dict(parent_rows[opening])
    rows = [opening_row]
    tapes = {str(opening_row["route_id"]): parent_actions[str(opening_row["route_id"])]}
    for family in families:
        if family == opening:
            continue
        source = shared_rows[family]
        rows.append({
            "family": family,
            "alias": f"{source['base']}@{source['checkpoint']}->{source['donor']}",
            "route_id": family,
            "team": "evolved-shared-prefix",
            "support": 1,
            "selected": True,
            "drop_reason": None,
            "provenance": source,
        })
        tapes[family] = shared_actions[family]

    actions_bytes = zlib.compress(
        json.dumps(tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    metadata = {
        "schema_version": 1,
        "taxonomy": "two-stage-exact-public-state-v1",
        "carrier_role": parent_metadata.get("carrier_role"),
        "opponent_routes": rows,
        "selected": rows,
        "actions_file": "route_actions.json.zlib",
        "actions_sha256": hashlib.sha256(actions_bytes).hexdigest(),
        "deployment_pruning": {
            "source_route_count": len(parent_metadata["opponent_routes"]),
            "deployed_route_count": len(rows),
            "families": families,
        },
    }
    checkpoints = [48, 96]
    policy = {
        "schema_version": 1,
        "kind": "two_stage_exact_public_state_route_runtime",
        "feature_schema": "semantic_route_switch_v1",
        "targets": families,
        "nodes": [
            {"selected": {"opening": opening, "checkpoint": checkpoint, "enabled": True}}
            for checkpoint in checkpoints
        ],
        "sequence_selection": {"checkpoints": checkpoints},
        "exact_state_model_npz_base64": base64.b64encode(packed_model).decode("ascii"),
        "exact_state_model": {
            "source": str(args.model.resolve()),
            "state_counts": state_counts,
            "fallback": opening,
            "public_state_only": True,
        },
    }
    nash = {
        "schema_version": 1,
        "value": 0.0,
        "opening_support": [{"family": opening, "weight": 1.0}],
    }
    holdout = json.loads(args.holdout_report.read_text(encoding="utf-8"))
    manifest = {
        "schema": "deployable-two-stage-exact-agent-assets-v1",
        "opening": opening,
        "routes": families,
        "route_count": len(families),
        "checkpoints": checkpoints,
        "state_counts": state_counts,
        "model_bytes": len(packed_model),
        "holdout": holdout.get("selector", holdout),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "route_actions.json.zlib").write_bytes(actions_bytes)
    for name, value in (
        ("route_library.json", metadata),
        ("route_policy.json", policy),
        ("opening_nash.json", nash),
        ("ASSET_MANIFEST.json", manifest),
    ):
        (args.output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "route_count": len(families),
        "routes": families,
        "actions_bytes": len(actions_bytes),
        "model_bytes": len(packed_model),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

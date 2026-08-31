#!/usr/bin/env python3
"""Materialize the routes referenced by one quantized route-Q policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path


def _load_actions(path: Path) -> dict[str, list[dict]]:
    return json.loads(zlib.decompress(path.read_bytes()))


def materialize(
    policy_path: Path, sources: list[tuple[Path, Path]], output: Path,
) -> dict:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    families = list(dict.fromkeys([
        *policy.get("openings", ()), *policy["targets"],
    ]))
    rows: dict[str, dict] = {}
    tapes: dict[str, list[dict]] = {}
    for actions_path, manifest_path in sources:
        actions = _load_actions(actions_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["routes"]:
            family = str(row["family"])
            if family in families:
                rows[family] = row
                tapes[family] = actions[family]
    missing = [family for family in families if family not in tapes]
    if missing:
        raise KeyError(f"route family is absent from supplied archives: {missing[0]}")

    packed = zlib.compress(
        json.dumps(tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    entries = [{
        "family": family,
        "alias": (
            f"{rows[family]['base']}@{rows[family]['checkpoint']}"
            f"->{rows[family]['donor']}"
        ),
        "route_id": family,
        "team": "replay-map-elites-shared-prefix",
        "support": 1,
        "selected": True,
        "drop_reason": None,
        "provenance": rows[family],
    } for family in families]
    metadata = {
        "schema_version": 1,
        "taxonomy": "quantized-public-state-route-q-v1",
        "opponent_routes": entries,
        "selected": entries,
        "actions_file": "route_actions.json.zlib",
        "actions_sha256": hashlib.sha256(packed).hexdigest(),
        "deployment_pruning": {
            "deployed_route_count": len(families), "families": families,
        },
    }
    nash = {
        "schema_version": 1,
        "value": 0.0,
        "opening_support": [{"family": str(policy["openings"][0]), "weight": 1.0}],
    }
    manifest = {
        "schema": "deployable-quantized-route-q-assets-v1",
        "opening": str(policy["openings"][0]),
        "routes": families,
        "route_count": len(families),
        "checkpoints": [int(value) for value in policy["checkpoints"]],
        "policy_bytes": policy_path.stat().st_size,
        "actions_bytes": len(packed),
        "actions_sha256": metadata["actions_sha256"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "route_actions.json.zlib").write_bytes(packed)
    for name, value in (
        ("route_library.json", metadata), ("route_policy.json", policy),
        ("opening_nash.json", nash), ("ASSET_MANIFEST.json", manifest),
    ):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--actions", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.actions) != len(args.manifest):
        parser.error("--actions and --manifest counts differ")
    result = materialize(
        args.policy, list(zip(args.actions, args.manifest)), args.output
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

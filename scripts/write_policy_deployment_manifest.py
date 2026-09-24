#!/usr/bin/env python3
"""Write a reproducible fingerprint for the deployed replay-to-R1 policy."""

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "agent/main.py",
    "agent/replay_deployment.json",
    "agent/route_policy.json",
    "agent/route_actions.json.zlib",
    "agent/route_library.json",
    "agent/teammate_base.py",
    "policy/r1/agent.py",
    "policy/r1/config.json",
    "policy/r1/agent.so",
    "meta_agent/src/search_route_policy.py",
    "meta_agent/src/teammate_expanded_routes.py",
    "meta_agent/src/route_switch_features.py",
    "meta_agent/src/market_manager.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    deployment = json.loads((ROOT / "agent/replay_deployment.json").read_text())
    for name, expected in deployment["sha256"].items():
        actual = sha256(ROOT / "agent" / name)
        if actual != expected:
            raise SystemExit(f"deployment hash mismatch for agent/{name}: {actual} != {expected}")

    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    route_policy = json.loads((ROOT / "agent/route_policy.json").read_text())
    payload = {
        "schema": "kaggriculture-policy-deployment-v1",
        "files": {name: sha256(ROOT / name) for name in FILES},
        "semantics": {
            "opening": deployment.get("opening"),
            "handoff_step": deployment.get("handoff_step"),
            "handoff_land": deployment.get("handoff_land"),
            "handoff_land_delay_days": deployment.get("handoff_land_delay_days"),
            "max_animals": config.get("max_animals"),
            "replant": config.get("replant"),
            "target_fallbacks": route_policy.get("target_fallbacks"),
        },
        "deployment_embedded_hashes_verified": True,
    }
    result = dict(
        payload,
        digest_semantics="canonical JSON payload excluding canonical_sha256 and digest_semantics",
        canonical_sha256=canonical_digest(payload),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(result["canonical_sha256"])


if __name__ == "__main__":
    main()

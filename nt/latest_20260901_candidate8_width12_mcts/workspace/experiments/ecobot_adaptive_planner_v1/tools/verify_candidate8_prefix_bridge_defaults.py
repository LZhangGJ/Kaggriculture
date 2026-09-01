#!/usr/bin/env python3
"""Verify that omitted prefix arguments equal explicit disabled-prefix values."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_candidate8_multifuture_oracle import load_genome
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def projection(result: dict) -> dict:
    return {
        "rewards": [float(value) for value in result["rewards"]],
        "decision_day": [int(value) for value in result["decision_day"]],
        "selected_rank": [int(value) for value in result["selected_rank"]],
        "selected_family": [int(value) for value in result["selected_family"]],
        "selected_signature": [int(value) for value in result["selected_signature"]],
        "expanded_nodes": int(result["expanded_nodes"]),
        "complete_continuations": int(result["complete_continuations"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    genome = load_genome(args.genomes, 0)
    common = (genome, bundle.index("G001"), 3051001, [3], 0, 8, 8, False, True)
    omitted = projection(bundle.adaptive_executor.candidate8_sequence_oracle(*common))
    explicit = projection(
        bundle.adaptive_executor.candidate8_sequence_oracle(
            *common, (), (), -1, 0
        )
    )
    passed = omitted == explicit
    payload = {
        "schema": "kaggriculture.candidate8-prefix-default-regression.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "omitted_prefix_arguments": omitted,
        "explicit_disabled_prefix": explicit,
        "exact": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

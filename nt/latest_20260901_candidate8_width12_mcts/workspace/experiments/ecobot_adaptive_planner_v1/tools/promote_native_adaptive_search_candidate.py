#!/usr/bin/env python3
"""Materialize validated native-adaptive search rows as a probe config."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--validation-rank",
        type=int,
        action="append",
        dest="validation_ranks",
        help="One-based validation rank; repeat to build a common-seed finalist grid.",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Apply NAME=VALUE to every promoted row; repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.receipt.read_text(encoding="utf-8"))
    ranking = source["validation"]["ranking"]
    validation_ranks = args.validation_ranks or [1]
    if len(set(validation_ranks)) != len(validation_ranks):
        raise ValueError("validation ranks must be unique")
    if any(not 1 <= rank <= len(ranking) for rank in validation_ranks):
        raise ValueError("validation rank out of range")
    rows = [ranking[rank - 1] for rank in validation_ranks]
    overrides: dict[str, float] = {}
    for raw_override in args.override:
        name, raw_value = raw_override.split("=", 1)
        overrides[name] = float(raw_value)
    labels = (
        [args.label]
        if len(rows) == 1
        else [f"{args.label}_rank{rank}" for rank in validation_ranks]
    )
    output = {
        "schema": "kaggriculture.native-adaptive-genomes.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "search_receipt": str(args.receipt.resolve()),
            "validation_ranks": validation_ranks,
            "screen_genome_indices": [
                int(row["screen_genome_index"]) for row in rows
            ],
            "search_objectives": [float(row["objective"]) for row in rows],
            "overrides": overrides,
        },
        "policy_semantics": source["policy_semantics"],
        "genomes": [
            {"label": label, "values": {**row["values"], **overrides}}
            for label, row in zip(labels, rows, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["provenance"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

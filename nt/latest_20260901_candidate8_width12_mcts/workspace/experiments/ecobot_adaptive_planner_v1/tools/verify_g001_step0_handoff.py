#!/usr/bin/env python3
"""Verify exact G001 prefix playback before the adaptive R6 handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int) -> list[float]:
    names = list(adaptive_genome_names())
    values = dict(zip(names, adaptive_default_genome(), strict=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    values.update({str(k): float(v) for k, v in payload.get("base_values", {}).items()})
    values.update({
        str(k): float(v)
        for k, v in payload["genomes"][index].get("values", {}).items()
    })
    return [float(values[name]) for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--prefix-steps", default="1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    genome = load_genome(args.genomes, args.genome_index)
    g001 = bundle.index("G001")
    prefix_steps = [int(value) for value in args.prefix_steps.split(",")]
    if any(value < 0 or value > 719 for value in prefix_steps):
        raise ValueError("prefix steps must be in [0, 719]")
    rows: list[dict] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        native = bundle.executor.play(g001, g001, seed, -1, -1, -1, -1, True)
        native_trace = native["trace"]
        for seat in (0, 1):
            for prefix in prefix_steps:
                hybrid = bundle.adaptive_executor.play_blend_trace(
                    genome, g001, g001, seed, seat, 6, prefix
                )
                hybrid_trace = hybrid["trace"]
                own_equal = all(
                    hybrid_trace[step][seat] == native_trace[step][seat]
                    for step in range(prefix)
                )
                opponent_equal = all(
                    hybrid_trace[step][1 - seat] == native_trace[step][1 - seat]
                    for step in range(prefix)
                )
                rows.append({
                    "seed": seed,
                    "seat": seat,
                    "prefix_steps": prefix,
                    "trace_steps": len(hybrid_trace),
                    "candidate_prefix_exact": own_equal,
                    "opponent_prefix_exact": opponent_equal,
                    "candidate_handoff_action_differs_from_g001": (
                        prefix < 719
                        and hybrid_trace[prefix][seat] != native_trace[prefix][seat]
                    ),
                })

    exact = all(
        row["candidate_prefix_exact"] and row["opponent_prefix_exact"]
        and row["trace_steps"] == 719
        for row in rows
    )
    payload = {
        "schema": "kaggriculture.g001-prefix-r6-handoff-verification.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if exact else "FAIL",
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "dual_seat": True,
        "states": len(rows),
        "prefix_steps": prefix_steps,
        "candidate_prefix_exact_count": sum(row["candidate_prefix_exact"] for row in rows),
        "opponent_prefix_exact_count": sum(row["opponent_prefix_exact"] for row in rows),
        "candidate_handoff_action_differs_from_g001_count": sum(
            row["candidate_handoff_action_differs_from_g001"] for row in rows
        ),
        "rows": rows,
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())

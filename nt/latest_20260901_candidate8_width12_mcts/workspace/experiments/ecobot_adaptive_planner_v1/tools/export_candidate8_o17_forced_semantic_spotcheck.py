#!/usr/bin/env python3
"""Export selected O1.7 forced-plan traces for official parity checking."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
META_AGENT = (
    ROOT
    / "research"
    / "team_mate"
    / "Kaggriculture_main_512631c"
    / "agents"
    / "route_clustering_switch_agent"
    / "src"
)
sys.path.insert(0, str(META_AGENT))

from audit_candidate8_multifuture_oracle import load_genome  # noqa: E402
from meta_agent.src.native_teammate_executor import NativeTeammateBundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--case-ids", type=int, nargs="+", default=(0, 32, 33))
    parser.add_argument(
        "--profiles", nargs="+", choices=("full", "target", "milestone"),
        default=("full", "milestone")
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rows = {int(row["case_id"]): row for row in audit["rows"]}
    genome = load_genome(args.genomes, args.genome_index)
    games = []
    for case_id in args.case_ids:
        row = rows[case_id]
        for profile in args.profiles:
            plan_path = args.plan_dir / f"case_{case_id:03d}_{profile}.npz"
            bundle = NativeTeammateBundle(
                args.source, args.actions, args.metadata, plan_path
            )
            result = bundle.adaptive_executor.play_forced_backbone(
                genome,
                bundle.index(str(row["opponent"])),
                int(row["seed"]),
                int(row["seat"]),
                True,
            )
            games.append(
                {
                    "opponent": str(row["opponent"]),
                    "seed": int(row["seed"]),
                    "candidate_seat": int(row["seat"]),
                    "decision_day": -1,
                    "arm": f"case{case_id}_{profile}",
                    "expected_signature": 0,
                    "actual_signature": 0,
                    "signature_exact": True,
                    "native_rewards": [float(value) for value in result["rewards"]],
                    "end_overflow": int(result["end_overflow"]),
                    "trace": list(result["trace"]),
                }
            )
            print(
                json.dumps(
                    {
                        "case": case_id,
                        "profile": profile,
                        "rewards": games[-1]["native_rewards"],
                    }
                ),
                flush=True,
            )
    payload = {
        "schema": "kaggriculture.candidate8-o17-forced-semantic-spotcheck.v1",
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

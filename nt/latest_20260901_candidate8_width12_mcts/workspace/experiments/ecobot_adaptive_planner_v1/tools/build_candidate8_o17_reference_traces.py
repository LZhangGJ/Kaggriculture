#!/usr/bin/env python3
"""Export the 36 O1.7 reference matches as temporary action traces.

The traces are an offline transport format used only to let the official
Python 1.32.7 environment reconstruct observations.  They are never consumed
by the forced-plan executor.  The durable O1.7 input is a semantic daily plan
that contains no action tape and no coordinates.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
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

from audit_candidate8_multifuture_oracle import load_genome, sha256  # noqa: E402
from meta_agent.src.native_teammate_executor import NativeTeammateBundle  # noqa: E402


CATEGORY = "SEMANTIC_CANDIDATE_OR_SOFT_COMPILER_GAP"


def _case_key(row: dict) -> tuple[str, int, int]:
    return str(row["opponent"]), int(row["seed"]), int(row["seat"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    attribution = json.loads(args.attribution.read_text(encoding="utf-8"))
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    cases = [row for row in attribution["rows"] if row["category"] == CATEGORY]
    if len(cases) != 36:
        raise RuntimeError(f"expected 36 O1.7 cases, observed {len(cases)}")
    portfolio = {_case_key(row): row for row in controls["portfolio_rows"]}

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    games: list[dict] = []
    started = time.perf_counter()
    for case_index, case in enumerate(cases, start=1):
        key = _case_key(case)
        reference = portfolio[key]
        reference_route = str(reference["best_route"])
        if reference_route != str(case["best_reference_route"]):
            raise RuntimeError(f"reference route mismatch for {key}")
        seat = int(case["seat"])
        result = bundle.adaptive_executor.play_blend_trace(
            genome,
            bundle.index(reference_route),
            bundle.index(str(case["opponent"])),
            int(case["seed"]),
            seat,
            3,  # full frozen route for both unit and market actions
            1,
        )
        rewards = [float(value) for value in result["rewards"]]
        expected = [0.0, 0.0]
        expected[seat] = float(reference["own_cash"])
        expected[1 - seat] = float(reference["opponent_cash"])
        if rewards != expected:
            raise RuntimeError(
                f"reference trace reward mismatch for {key}: {rewards} != {expected}"
            )
        trace = list(result["trace"])
        if len(trace) != 719:
            raise RuntimeError(f"incomplete reference trace for {key}: {len(trace)}")
        games.append(
            {
                "case_id": case_index - 1,
                "opponent": str(case["opponent"]),
                "seed": int(case["seed"]),
                "candidate_seat": seat,
                "reference_route": reference_route,
                "reference_route_index": int(reference["best_route_index"]),
                "native_rewards": rewards,
                "reference_margin": float(reference["margin"]),
                "best_width_margin": float(case["best_search"]["margin"]),
                "trace": trace,
            }
        )
        print(
            json.dumps(
                {
                    "progress": case_index,
                    "total": len(cases),
                    "opponent": case["opponent"],
                    "reference": reference_route,
                }
            ),
            flush=True,
        )

    payload = {
        "schema": "kaggriculture.candidate8-o17-reference-traces.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "boundary": (
            "Temporary offline transport only. Raw actions are used to reconstruct "
            "official observations, then discarded from the forced-plan input."
        ),
        "raw_action_payload_stored": True,
        "deployable_agent_input": False,
        "cases": len(games),
        "simulation_seconds": time.perf_counter() - started,
        "inputs": {
            "attribution": {"path": str(args.attribution), "sha256": sha256(args.attribution)},
            "controls": {"path": str(args.controls), "sha256": sha256(args.controls)},
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    print(
        json.dumps(
            {
                "status": "PASS",
                "cases": len(games),
                "output": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest().upper(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay an exported O1 trace bundle in official Python environment 1.32.7."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from kaggle_environments import make


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    version = importlib.metadata.version("kaggle-environments")
    if version != "1.32.7":
        raise RuntimeError(f"official package 1.32.7 required, got {version}")
    with gzip.open(args.trace_bundle, "rt", encoding="utf-8") as stream:
        bundle = json.load(stream)

    rows: list[dict[str, object]] = []
    for game in bundle["games"]:
        official = make(
            "kaggriculture",
            configuration={"episodeSteps": 720, "seed": int(game["seed"])},
            debug=False,
        )
        state = official.reset(2)
        for joint_action in game["trace"]:
            state = official.step(joint_action)
        rewards = [float(value.reward) for value in state]
        statuses = [str(value.status) for value in state]
        row = {
            key: game[key]
            for key in (
                "opponent",
                "seed",
                "candidate_seat",
                "decision_day",
                "arm",
                "expected_signature",
                "actual_signature",
                "signature_exact",
                "native_rewards",
                "end_overflow",
            )
        }
        row.update(
            {
                "trace_steps": len(game["trace"]),
                "statuses": statuses,
                "complete": len(game["trace"]) == 719
                and statuses == ["DONE", "DONE"],
                "official_rewards": rewards,
                "reward_exact": rewards == game["native_rewards"],
            }
        )
        rows.append(row)
        print(
            f"official {len(rows)}/{len(bundle['games'])} "
            f"day={row['decision_day']} seat={row['candidate_seat']} "
            f"signature={row['signature_exact']} reward={row['reward_exact']}",
            flush=True,
        )
    passed = all(
        row["signature_exact"]
        and row["complete"]
        and row["reward_exact"]
        and row["end_overflow"] == 0
        and all(math.isfinite(value) for value in row["official_rewards"])
        for row in rows
    )
    payload = {
        "schema": "kaggriculture.candidate8-o1-official-spotcheck.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "boundary": {
            "checked": "selected arm regeneration and complete trace engine parity",
            "not_checked": (
                "official Python cannot inject a counterfactual RNG suffix; "
                "Bank A/B value estimates remain C++ counterfactual measurements"
            ),
        },
        "official_package_version": version,
        "games": len(rows),
        "summary": {
            "complete_games": sum(int(row["complete"]) for row in rows),
            "signature_exact_games": sum(int(row["signature_exact"]) for row in rows),
            "reward_exact_games": sum(int(row["reward_exact"]) for row in rows),
            "end_overflow_events": sum(int(row["end_overflow"]) for row in rows),
        },
        "input": {
            "trace_bundle": str(args.trace_bundle),
            "sha256": sha256(args.trace_bundle),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

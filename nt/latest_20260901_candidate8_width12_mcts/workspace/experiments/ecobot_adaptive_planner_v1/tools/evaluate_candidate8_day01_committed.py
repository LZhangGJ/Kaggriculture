#!/usr/bin/env python3
"""Evaluate frozen Day0 plus model-selected Day1 actions without lookahead."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from audit_candidate8_multifuture_oracle import load_genome
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def outcome(own: float, rival: float) -> float:
    return 1.0 if own > rival else 0.5 if own == rival else 0.0


def summarize(rows: list[dict], key: str) -> dict:
    own = [row[f"{key}_own"] for row in rows]
    rival = [row[f"{key}_rival"] for row in rows]
    margins = [a - b for a, b in zip(own, rival)]
    return {
        "games": len(rows),
        "score_rate": statistics.fmean(outcome(a, b) for a, b in zip(own, rival)),
        "mean_margin": statistics.fmean(margins),
        "mean_cash": statistics.fmean(own),
        "minimum_cash": min(own),
        "hard_error_games": sum(
            row[f"{key}_hard_errors"] > 0 for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--day0-selection", required=True, type=Path)
    parser.add_argument("--day1-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    day0 = json.loads(args.day0_selection.read_text(encoding="utf-8"))
    day1 = json.loads(args.day1_selection.read_text(encoding="utf-8"))
    day0_ranks = {
        int(seat): int(value["rank"])
        for seat, value in day0["selected_by_seat"].items()
    }
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)

    rows = []
    started = time.perf_counter()
    for selection in day1["rows"]:
        opponent_name = selection["opponent"]
        opponent = bundle.index(opponent_name)
        seed = int(selection["prefix_seed"])
        seat = int(selection["seat"])
        day0_rank = day0_ranks[seat]
        day1_rank = int(selection["selected_rank"])

        baseline = bundle.adaptive_executor.play(
            genome, opponent, seed, seat, False
        )
        day0_play = bundle.adaptive_executor.candidate8_committed_sequence(
            genome, opponent, seed, [0], [day0_rank], seat, False
        )
        day01_play = bundle.adaptive_executor.candidate8_committed_sequence(
            genome, opponent, seed, [0, 1], [day0_rank, day1_rank], seat, False
        )

        def append_result(row: dict, key: str, result: dict) -> None:
            row[f"{key}_own"] = float(result["rewards"][seat])
            row[f"{key}_rival"] = float(result["rewards"][1 - seat])
            row[f"{key}_hard_errors"] = int(
                result.get("end_overflow", 0)
                + result.get("avoidable_crop_losses", 0)
                + result.get("avoidable_animal_losses", 0)
            )

        row = {
            "opponent": opponent_name,
            "seed": seed,
            "seat": seat,
            "day0_rank": day0_rank,
            "day1_rank": day1_rank,
            "day1_signature": int(selection["selected_signature"]),
            "day1_family": int(selection["selected_family"]),
            "day1_activated": bool(selection["activated"]),
        }
        append_result(row, "baseline", baseline)
        append_result(row, "day0", day0_play)
        append_result(row, "day01", day01_play)
        rows.append(row)

    elapsed = time.perf_counter() - started
    summaries = {
        key: summarize(rows, key) for key in ("baseline", "day0", "day01")
    }
    per_opponent = {}
    for opponent_name in sorted({row["opponent"] for row in rows}):
        local = [row for row in rows if row["opponent"] == opponent_name]
        per_opponent[opponent_name] = {
            key: summarize(local, key) for key in ("baseline", "day0", "day01")
        }
    payload = {
        "schema": "kaggriculture.candidate8-day01-committed-evaluation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "No future rollout is queried while choosing either action. "
            "Day0 is frozen by training-opponent expectation; Day1 ranks are "
            "read from a public-feature model selection manifest."
        ),
        "inputs": {
            "day0_selection": {
                "path": str(args.day0_selection), "sha256": sha256(args.day0_selection)
            },
            "day1_selection": {
                "path": str(args.day1_selection), "sha256": sha256(args.day1_selection)
            },
        },
        "elapsed_seconds": elapsed,
        "games_per_second": len(rows) * 3 / elapsed if elapsed else 0.0,
        "summaries": summaries,
        "per_opponent": per_opponent,
        "gains": {
            "day0_vs_baseline_score_rate": (
                summaries["day0"]["score_rate"] - summaries["baseline"]["score_rate"]
            ),
            "day0_vs_baseline_margin": (
                summaries["day0"]["mean_margin"] - summaries["baseline"]["mean_margin"]
            ),
            "day01_vs_day0_score_rate": (
                summaries["day01"]["score_rate"] - summaries["day0"]["score_rate"]
            ),
            "day01_vs_day0_margin": (
                summaries["day01"]["mean_margin"] - summaries["day0"]["mean_margin"]
            ),
        },
        "gate": {
            "no_hard_errors": summaries["day01"]["hard_error_games"] == 0,
            "day0_not_worse_than_baseline": (
                summaries["day0"]["score_rate"] >= summaries["baseline"]["score_rate"]
                and summaries["day0"]["mean_margin"] >= summaries["baseline"]["mean_margin"]
            ),
            "day1_improves_day0": (
                summaries["day01"]["score_rate"] >= summaries["day0"]["score_rate"]
                and summaries["day01"]["mean_margin"] > summaries["day0"]["mean_margin"]
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "summaries": summaries,
        "gains": payload["gains"],
        "gate": payload["gate"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

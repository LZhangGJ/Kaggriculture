#!/usr/bin/env python3
"""Merge the reliable C++ actual arena with confirmed Candidate8 Oracle runs.

The actual and Oracle samples intentionally use different seed banks and game
counts.  This script preserves both denominators instead of blending them into
one misleading win rate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--oracle-dir", required=True, action="append", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    actual_doc = load_json(args.actual)
    actual_rows = {row["opponent"]: row for row in actual_doc["opponents"]}
    metadata_rows = {}
    if args.metadata:
        metadata_doc = load_json(args.metadata)
        metadata_rows = {row["family"]: row for row in metadata_doc.get("selected", [])}

    oracle_docs: dict[str, tuple[Path, dict]] = {}
    duplicate_ids: list[str] = []
    for directory in args.oracle_dir:
        for path in sorted(directory.glob("*.json")):
            doc = load_json(path)
            opponent = doc["parameters"]["opponent"]
            if opponent in oracle_docs:
                duplicate_ids.append(opponent)
            oracle_docs[opponent] = (path, doc)

    actual_ids = set(actual_rows)
    oracle_ids = set(oracle_docs)
    rows = []
    total_oracle_wins = 0
    total_oracle_ties = 0
    total_oracle_losses = 0
    total_oracle_crop_losses = 0
    total_oracle_animal_losses = 0
    total_oracle_overflow = 0

    for opponent in sorted(actual_ids & oracle_ids):
        actual = actual_rows[opponent]
        source_path, oracle_doc = oracle_docs[opponent]
        summary = oracle_doc["summary"]
        competitive = summary["rolling_oracle_competitive"]
        states = oracle_doc["states"]
        oracle_games = int(competitive["wins"] + competitive["ties"] + competitive["losses"])
        oracle_score_rate = (
            (competitive["wins"] + 0.5 * competitive["ties"]) / oracle_games
            if oracle_games
            else 0.0
        )
        crop_losses = sum(int(state.get("avoidable_crop_losses", 0)) for state in states)
        animal_losses = sum(int(state.get("avoidable_animal_losses", 0)) for state in states)
        overflow = sum(int(state.get("end_overflow", 0)) for state in states)

        total_oracle_wins += competitive["wins"]
        total_oracle_ties += competitive["ties"]
        total_oracle_losses += competitive["losses"]
        total_oracle_crop_losses += crop_losses
        total_oracle_animal_losses += animal_losses
        total_oracle_overflow += overflow

        rows.append(
            {
                "opponent": opponent,
                "source": {
                    "family": metadata_rows.get(opponent, {}).get("family"),
                    "alias": metadata_rows.get(opponent, {}).get("alias"),
                    "team": metadata_rows.get(opponent, {}).get("team"),
                    "route_id": metadata_rows.get(opponent, {}).get("route_id"),
                    "support": metadata_rows.get(opponent, {}).get("support"),
                },
                "actual": {
                    "games": actual["games"],
                    "wins": actual["wins"],
                    "ties": actual["ties"],
                    "losses": actual["losses"],
                    "win_rate": actual["win_rate"],
                    "score_rate": actual["score_rate"],
                    "mean_margin": actual["mean_margin"],
                    "mean_cash": actual["mean_cash"],
                },
                "oracle": {
                    "games": oracle_games,
                    "wins": competitive["wins"],
                    "ties": competitive["ties"],
                    "losses": competitive["losses"],
                    "win_rate": competitive["win_rate"],
                    "score_rate": oracle_score_rate,
                    "mean_margin": summary["rolling_oracle_margin"]["mean"],
                    "mean_cash": summary["rolling_oracle_cash"]["mean"],
                    "avoidable_crop_losses": crop_losses,
                    "avoidable_animal_losses": animal_losses,
                    "end_overflow": overflow,
                },
                "oracle_minus_actual_win_rate": oracle_score_rate - actual["score_rate"],
                "oracle_receipt": str(source_path),
            }
        )

    oracle_games = total_oracle_wins + total_oracle_ties + total_oracle_losses
    route_bands = {
        "oracle_at_least_90pct": [r["opponent"] for r in rows if r["oracle"]["score_rate"] >= 0.90],
        "oracle_50_to_below_90pct": [
            r["opponent"] for r in rows if 0.50 <= r["oracle"]["score_rate"] < 0.90
        ],
        "oracle_below_50pct": [r["opponent"] for r in rows if r["oracle"]["score_rate"] < 0.50],
        "actual_at_least_90pct": [r["opponent"] for r in rows if r["actual"]["score_rate"] >= 0.90],
        "actual_at_least_50pct": [r["opponent"] for r in rows if r["actual"]["score_rate"] >= 0.50],
        "actual_zero_wins": [r["opponent"] for r in rows if r["actual"]["wins"] == 0],
    }

    output = {
        "schema": "kaggriculture.cxx-clean133-actual-vs-candidate8-oracle.v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "definitions": {
            "actual": "Deployable R6 planner; no future information.",
            "oracle": (
                "R7 Candidate8 rolling Oracle with the actual future revealed; "
                "an upper bound over the current candidate/executor stack, not deployable strength."
            ),
        },
        "inputs": {
            "actual_receipt": str(args.actual),
            "oracle_directories": [str(path) for path in args.oracle_dir],
            "route_metadata": str(args.metadata) if args.metadata else None,
        },
        "coverage": {
            "actual_opponents": len(actual_ids),
            "oracle_opponents": len(oracle_ids),
            "merged_opponents": len(rows),
            "missing_oracle": sorted(actual_ids - oracle_ids),
            "extra_oracle": sorted(oracle_ids - actual_ids),
            "duplicate_oracle_ids": sorted(set(duplicate_ids)),
        },
        "summary": {
            "actual_games": actual_doc["summary"]["games"],
            "actual_wins": actual_doc["summary"]["wins"],
            "actual_ties": actual_doc["summary"]["ties"],
            "actual_losses": actual_doc["summary"]["losses"],
            "actual_score_rate": actual_doc["summary"]["score_rate"],
            "oracle_games": oracle_games,
            "oracle_wins": total_oracle_wins,
            "oracle_ties": total_oracle_ties,
            "oracle_losses": total_oracle_losses,
            "oracle_score_rate": (
                (total_oracle_wins + 0.5 * total_oracle_ties) / oracle_games
                if oracle_games
                else 0.0
            ),
            "oracle_routes_at_least_90pct": len(route_bands["oracle_at_least_90pct"]),
            "oracle_routes_at_least_50pct": (
                len(route_bands["oracle_at_least_90pct"])
                + len(route_bands["oracle_50_to_below_90pct"])
            ),
            "oracle_routes_below_50pct": len(route_bands["oracle_below_50pct"]),
            "actual_routes_at_least_90pct": len(route_bands["actual_at_least_90pct"]),
            "actual_routes_at_least_50pct": len(route_bands["actual_at_least_50pct"]),
            "actual_routes_zero_wins": len(route_bands["actual_zero_wins"]),
            "oracle_avoidable_crop_losses": total_oracle_crop_losses,
            "oracle_avoidable_animal_losses": total_oracle_animal_losses,
            "oracle_end_overflow": total_oracle_overflow,
        },
        "route_bands": route_bands,
        "opponents": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"output": str(args.output), **output["coverage"], **output["summary"]}, indent=2))


if __name__ == "__main__":
    main()

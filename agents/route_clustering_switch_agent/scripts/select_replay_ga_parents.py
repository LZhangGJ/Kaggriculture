#!/usr/bin/env python3
"""Select a performant, team-diverse GA parent panel from a fixed-route screen."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prefix", default="NR")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--team-cap", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0 or args.team_cap <= 0:
        parser.error("count and team-cap must be positive")

    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    details = {
        str(value["family"]): value for value in metadata["opponent_routes"]
    }
    ranking = [
        dict(row)
        for row in screen["ranking"]
        if str(row["family"]).startswith(args.prefix)
    ]
    ranking.sort(key=lambda row: (
        -float(row["mean_score"]),
        -float(row["one_sided_95pct_margin_lower"]),
        -float(row["mean_margin"]),
        str(row["family"]),
    ))
    selected = []
    team_counts: Counter[str] = Counter()
    for row in ranking:
        family = str(row["family"])
        team = str(details[family].get("team", "") or "unknown")
        if team_counts[team] >= args.team_cap:
            continue
        selected.append({
            **row,
            "team": team,
            "route_id": str(details[family]["route_id"]),
            "provenance": details[family].get("provenance"),
        })
        team_counts[team] += 1
        if len(selected) >= args.count:
            break
    if len(selected) < args.count:
        parser.error(
            f"only {len(selected)} parents satisfy prefix={args.prefix!r} and team cap"
        )

    payload: dict[str, Any] = {
        "schema": "fixed-screen-diverse-ga-parent-panel-v1",
        "screen": str(args.screen.resolve()),
        "metadata": str(args.metadata.resolve()),
        "candidate_prefix": args.prefix,
        "requested_count": args.count,
        "team_cap": args.team_cap,
        "families": [row["family"] for row in selected],
        "families_csv": ",".join(str(row["family"]) for row in selected),
        "team_counts": dict(team_counts.most_common()),
        "parents": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "families_csv": payload["families_csv"],
        "teams": len(team_counts),
        "top": [
            {
                key: row[key]
                for key in ("family", "team", "mean_score", "mean_margin")
            }
            for row in selected[:5]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

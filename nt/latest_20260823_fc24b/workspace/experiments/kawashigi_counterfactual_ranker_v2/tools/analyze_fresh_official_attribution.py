#!/usr/bin/env python3
"""Paired attribution for the fresh official Challenger evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "games": len(rows),
        "wins": sum(bool(row["win"]) for row in rows),
        "ties": sum(bool(row["tie"]) for row in rows),
        "win_rate": sum(bool(row["win"]) for row in rows) / len(rows),
        "mean_margin": sum(float(row["margin"]) for row in rows) / len(rows),
    }


def exact_one_sided_sign_p(gains: int, losses: int) -> float:
    """P[X >= gains] for X~Binomial(gains+losses, 0.5)."""

    discordant = gains + losses
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, value) for value in range(gains, discordant + 1)) / (2 ** discordant)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--challenger-name", default="counterfactual_lgbm_v1")
    parser.add_argument("--fallback-name", default="rank01_centroid_fallback")
    args = parser.parse_args()
    arena_path = resolve(args.arena)
    arena = json.loads(arena_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in arena["rows"]:
        key = (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        grouped[key][str(row["candidate"])] = row
    paired = []
    for (opponent, seed, seat), candidates in sorted(grouped.items()):
        challenger = candidates[args.challenger_name]
        fallback = candidates[args.fallback_name]
        probe = challenger.get("candidate_route_probe", {}) or {}
        paired.append({
            "opponent": opponent,
            "seed": seed,
            "seat": seat,
            "fallback_route": str(probe.get("fallback", "")),
            "selected_route": str(probe.get("route", "")),
            "switched": str(probe.get("route", "")) != str(probe.get("fallback", "")),
            "predicted_gain": float(probe.get("predicted_gain", 0.0) or 0.0),
            "challenger_win": bool(challenger["win"]),
            "fallback_win": bool(fallback["win"]),
            "win_delta": int(bool(challenger["win"])) - int(bool(fallback["win"])),
            "challenger_margin": float(challenger["margin"]),
            "fallback_margin": float(fallback["margin"]),
            "margin_delta": float(challenger["margin"] - fallback["margin"]),
        })
    switch_rows = [row for row in paired if row["switched"]]
    transitions = {}
    for transition in sorted({f"{row['fallback_route']}->{row['selected_route']}" for row in switch_rows}):
        group = [row for row in switch_rows if f"{row['fallback_route']}->{row['selected_route']}" == transition]
        transitions[transition] = {
            "contexts": len(group),
            "net_wins": sum(row["win_delta"] for row in group),
            "improved_wins": sum(row["win_delta"] > 0 for row in group),
            "lost_wins": sum(row["win_delta"] < 0 for row in group),
            "mean_margin_delta": sum(row["margin_delta"] for row in group) / len(group),
            "mean_predicted_gain": sum(row["predicted_gain"] for row in group) / len(group),
        }
    by_opponent = {}
    for opponent in sorted({row["opponent"] for row in paired}):
        group = [row for row in paired if row["opponent"] == opponent]
        by_opponent[opponent] = {
            "contexts": len(group),
            "switches": sum(row["switched"] for row in group),
            "challenger_wins": sum(row["challenger_win"] for row in group),
            "fallback_wins": sum(row["fallback_win"] for row in group),
            "challenger_win_rate": sum(row["challenger_win"] for row in group) / len(group),
            "fallback_win_rate": sum(row["fallback_win"] for row in group) / len(group),
            "net_wins": sum(row["win_delta"] for row in group),
            "mean_margin_delta": sum(row["margin_delta"] for row in group) / len(group),
        }
    challenger_rows = [row for row in arena["rows"] if row["candidate"] == args.challenger_name]
    fallback_rows = [row for row in arena["rows"] if row["candidate"] == args.fallback_name]
    v8_rows = [row for row in arena["rows"] if row["candidate"] == "public_opening_router_v8"]
    challenger_summary = summarize(challenger_rows)
    fallback_summary = summarize(fallback_rows)
    v8_summary = summarize(v8_rows)
    net_wins = challenger_summary["wins"] - fallback_summary["wins"]
    gained_wins = sum(row["win_delta"] > 0 for row in paired)
    lost_wins = sum(row["win_delta"] < 0 for row in paired)
    sign_test_p = exact_one_sided_sign_p(gained_wins, lost_wins)
    challenger_minimum = min(value["challenger_win_rate"] for value in by_opponent.values())
    fallback_minimum = min(value["fallback_win_rate"] for value in by_opponent.values())
    opponent_safety_pass = challenger_minimum >= fallback_minimum
    directional_pass = bool(
        arena.get("all_done")
        and net_wins > 0
        and challenger_summary["mean_margin"] > fallback_summary["mean_margin"]
    )
    promotion_grade = bool(
        directional_pass
        and sign_test_p < 0.05
        and opponent_safety_pass
    )
    result = {
        "schema": "kawashigi-counterfactual-fresh-official-attribution-v1",
        "status": "PROMOTION_GRADE_PASS" if promotion_grade else ("CLOSED_LOOP_DIRECTIONAL_PASS_NOT_PROMOTED" if directional_pass else "FRESH_OFFICIAL_FAIL"),
        "truth_boundary": "fresh official 1.32.7 seeds; threshold and model were frozen before this panel",
        "arena": str(arena_path),
        "arena_sha256": sha256(arena_path),
        "all_done": bool(arena.get("all_done")),
        "challenger": challenger_summary,
        "fallback": fallback_summary,
        "v8_control": v8_summary,
        "delta_vs_fallback": {
            "wins": net_wins,
            "win_rate": challenger_summary["win_rate"] - fallback_summary["win_rate"],
            "mean_margin": challenger_summary["mean_margin"] - fallback_summary["mean_margin"],
        },
        "paired_sign_test": {
            "gained_wins": gained_wins,
            "lost_wins": lost_wins,
            "discordant_contexts": gained_wins + lost_wins,
            "one_sided_p_value": sign_test_p,
        },
        "opponent_safety": {
            "challenger_minimum_opponent_win_rate": challenger_minimum,
            "fallback_minimum_opponent_win_rate": fallback_minimum,
            "pass": opponent_safety_pass,
        },
        "switches": len(switch_rows),
        "switch_rate": len(switch_rows) / len(paired),
        "switch_net_wins": sum(row["win_delta"] for row in switch_rows),
        "switch_mean_margin_delta": sum(row["margin_delta"] for row in switch_rows) / len(switch_rows) if switch_rows else 0.0,
        "transitions": transitions,
        "by_opponent": by_opponent,
        "acceptance": {
            "directional_pass": directional_pass,
            "paired_one_sided_p_below_0p05": sign_test_p < 0.05,
            "mean_margin_improved": challenger_summary["mean_margin"] > fallback_summary["mean_margin"],
            "minimum_opponent_win_rate_not_reduced": opponent_safety_pass,
            "promotion_grade": promotion_grade,
        },
        "paired_rows": paired,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "challenger", "fallback", "v8_control", "delta_vs_fallback", "paired_sign_test", "opponent_safety", "switches", "switch_rate", "switch_net_wins", "switch_mean_margin_delta", "transitions", "by_opponent", "acceptance")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

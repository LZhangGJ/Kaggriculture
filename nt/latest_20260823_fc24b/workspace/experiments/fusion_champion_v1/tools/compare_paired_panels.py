#!/usr/bin/env python3
"""Compare two same-seed, seat-swapped panel receipts game by game.

The source receipt may contain a superset of the candidate's opponents.  This
lets a compact critical panel be compared against a previously frozen full
roster without manufacturing an intermediate receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def game_map(row: dict) -> dict[tuple[int, int], dict]:
    return {
        (int(game["seed"]), int(game["candidate_seat"])): game
        for game in row["per_game"]
    }


def score(game: dict) -> float:
    if "result" in game:
        return {"win": 1.0, "tie": 0.5, "loss": 0.0}[game["result"]]
    margin = int(game["margin"])
    return 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0


def result(game: dict) -> str:
    if "result" in game:
        return str(game["result"])
    margin = int(game["margin"])
    return "win" if margin > 0 else "tie" if margin == 0 else "loss"


def compare_row(candidate: dict, source: dict) -> tuple[dict, list[dict]]:
    candidate_games = game_map(candidate)
    source_games = game_map(source)
    if candidate_games.keys() != source_games.keys():
        raise ValueError(f"per-game key mismatch for {candidate['opponent']}")
    changes: list[dict] = []
    improved = regressed = unchanged = rescued = harmed = 0
    for key in sorted(candidate_games):
        new = candidate_games[key]
        old = source_games[key]
        delta = int(new["margin"]) - int(old["margin"])
        improved += delta > 0
        regressed += delta < 0
        unchanged += delta == 0
        rescued += result(old) == "loss" and result(new) == "win"
        harmed += result(old) == "win" and result(new) != "win"
        if delta or result(old) != result(new):
            changes.append(
                {
                    "seed": key[0],
                    "candidate_seat": key[1],
                    "source_result": result(old),
                    "candidate_result": result(new),
                    "source_margin": int(old["margin"]),
                    "candidate_margin": int(new["margin"]),
                    "margin_delta": delta,
                }
            )
    games = len(candidate_games)
    old_score = sum(score(game) for game in source_games.values())
    new_score = sum(score(game) for game in candidate_games.values())
    return {
        "opponent": candidate["opponent"],
        "games": games,
        "source_score_rate": old_score / games,
        "candidate_score_rate": new_score / games,
        "score_delta_games": new_score - old_score,
        "source_mean_margin": float(source["mean_margin"]),
        "candidate_mean_margin": float(candidate["mean_margin"]),
        "mean_margin_delta": float(candidate["mean_margin"] - source["mean_margin"]),
        "margin_improved_games": improved,
        "margin_regressed_games": regressed,
        "exact_margin_unchanged_games": unchanged,
        "source_losses_rescued_to_win": rescued,
        "source_wins_harmed": harmed,
        "changed_games": len(changes),
    }, changes


def render(payload: dict) -> str:
    lines = [
        f"# {payload['candidate_label']} 对 {payload['source_label']} 配对审计",
        "",
        payload["decision_zh"],
        "",
        f"固定事件库、双方换座；每个对手 {payload['games_per_opponent']} 局。",
        "",
        "| 对手 | 原版得分率 | 候选得分率 | 等价胜场变化 | 平均分差变化 | 救回败局 | 伤害原胜局 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {opponent} | {source_score_rate:.2%} | {candidate_score_rate:.2%} | "
            "{score_delta_games:+.1f} | {mean_margin_delta:+.1f} | "
            "{source_losses_rescued_to_win} | {source_wins_harmed} |".format(**row)
        )
    overall = payload["overall"]
    lines += [
        "",
        f"- 配对对局：{overall['games']}。",
        f"- 等价胜场变化：{overall['score_delta_games']:+.1f}。",
        f"- 救回败局/伤害原胜局：{overall['source_losses_rescued_to_win']} / {overall['source_wins_harmed']}。",
        f"- 分差改善/退化/不变：{overall['margin_improved_games']} / {overall['margin_regressed_games']} / {overall['exact_margin_unchanged_games']}。",
        f"- 候选最低逐对手得分率：{overall['candidate_min_score_rate']:.2%}。",
        "",
        f"机器结果：`{payload['output_json']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--source-label", default="source")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    candidate = load(args.candidate)
    source = load(args.source)
    for field in ("official_package_version", "seed_values", "games_per_opponent", "seat_protocol"):
        if candidate[field] != source[field]:
            raise ValueError(f"mismatched {field}")
    source_rows = {row["opponent"]: row for row in source["rows"]}
    missing = [row["opponent"] for row in candidate["rows"] if row["opponent"] not in source_rows]
    if missing:
        raise ValueError(f"source lacks opponents: {missing}")
    rows, changes = [], {}
    for candidate_row in candidate["rows"]:
        row, row_changes = compare_row(candidate_row, source_rows[candidate_row["opponent"]])
        rows.append(row)
        changes[row["opponent"]] = row_changes
    overall = {
        "games": sum(row["games"] for row in rows),
        "score_delta_games": sum(row["score_delta_games"] for row in rows),
        "margin_improved_games": sum(row["margin_improved_games"] for row in rows),
        "margin_regressed_games": sum(row["margin_regressed_games"] for row in rows),
        "exact_margin_unchanged_games": sum(row["exact_margin_unchanged_games"] for row in rows),
        "source_losses_rescued_to_win": sum(row["source_losses_rescued_to_win"] for row in rows),
        "source_wins_harmed": sum(row["source_wins_harmed"] for row in rows),
        "candidate_min_score_rate": min(row["candidate_score_rate"] for row in rows),
    }
    safe = overall["source_wins_harmed"] == 0 and overall["score_delta_games"] > 0
    payload = {
        "schema": "kaggriculture.fusion_champion.paired-panel-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if safe else "FAIL",
        "decision_zh": "候选通过当前配对安全门，可进入独立事件库复验。" if safe else "候选未通过当前配对安全门，不应晋升。",
        "candidate_label": args.candidate_label,
        "source_label": args.source_label,
        "official_package_version": candidate["official_package_version"],
        "seed_values": candidate["seed_values"],
        "games_per_opponent": candidate["games_per_opponent"],
        "seat_protocol": candidate["seat_protocol"],
        "candidate_input": {"path": str(args.candidate), "sha256": sha256(args.candidate)},
        "source_input": {"path": str(args.source), "sha256": sha256(args.source)},
        "output_json": str(args.output_json),
        "rows": rows,
        "overall": overall,
        "changed_games": changes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_report.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "overall": overall}, ensure_ascii=False))
    return 0 if safe else 2


if __name__ == "__main__":
    raise SystemExit(main())

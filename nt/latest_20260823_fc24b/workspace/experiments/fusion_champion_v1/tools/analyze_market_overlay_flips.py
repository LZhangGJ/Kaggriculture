#!/usr/bin/env python3
"""Audit a same-seed market-overlay panel against its frozen source.

The block runner deliberately writes a compact receipt.  This tool joins it
to the richer FC15 diagnostic receipt without rerunning games, records every
result flip, and groups Boatlee losses by public route state.  It is an audit
tool only; none of these diagnostics are policy inputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest().upper()


def outcome(margin: int) -> str:
    return "win" if margin > 0 else "tie" if margin == 0 else "loss"


def game_map(row: dict) -> dict[tuple[int, int], dict]:
    return {
        (int(game["seed"]), int(game["candidate_seat"])): game
        for game in row["per_game"]
    }


def summarize(candidate_row: dict, source_row: dict) -> tuple[dict, list[dict]]:
    candidate = game_map(candidate_row)
    source = game_map(source_row)
    if candidate.keys() != source.keys():
        raise ValueError(f"game-key mismatch: {candidate_row['opponent']}")
    changes = []
    rescued = spoiled = improved = regressed = unchanged = 0
    for key in sorted(candidate):
        new = int(candidate[key]["margin"])
        old = int(source[key]["margin"])
        delta = new - old
        improved += delta > 0
        regressed += delta < 0
        unchanged += delta == 0
        rescued += old < 0 < new
        spoiled += old > 0 > new
        if delta or outcome(old) != outcome(new):
            changes.append(
                {
                    "seed": key[0],
                    "candidate_seat": key[1],
                    "source_margin": old,
                    "candidate_margin": new,
                    "margin_delta": delta,
                    "source_result": outcome(old),
                    "candidate_result": outcome(new),
                }
            )
    games = len(candidate)
    source_wins = sum(int(game["margin"]) > 0 for game in source.values())
    candidate_wins = sum(int(game["margin"]) > 0 for game in candidate.values())
    return (
        {
            "opponent": candidate_row["opponent"],
            "games": games,
            "source_wins": source_wins,
            "candidate_wins": candidate_wins,
            "source_win_rate": source_wins / games,
            "candidate_win_rate": candidate_wins / games,
            "net_wins": candidate_wins - source_wins,
            "source_mean_margin": mean(int(game["margin"]) for game in source.values()),
            "candidate_mean_margin": mean(int(game["margin"]) for game in candidate.values()),
            "rescued_losses": rescued,
            "spoiled_wins": spoiled,
            "margin_improved": improved,
            "margin_regressed": regressed,
            "margin_unchanged": unchanged,
        },
        changes,
    )


def boatlee_groups(candidate_row: dict, diagnostic_row: dict) -> list[dict]:
    candidate = game_map(candidate_row)
    diagnostic = game_map(diagnostic_row)
    buckets: dict[tuple[int, int, bool, bool], list[int]] = defaultdict(list)
    for key, game in candidate.items():
        margin = int(game["margin"])
        if margin >= 0:
            continue
        detail = diagnostic[key]
        group = (
            int(detail["fc15_k320_route"]),
            int(detail["opponent_route"]),
            bool(detail["opponent_market_overlay"]),
            bool(detail["fc15_sheep_pressure"]),
        )
        buckets[group].append(margin)
    rows = []
    for group, margins in buckets.items():
        rows.append(
            {
                "fc15_route": group[0],
                "opponent_route": group[1],
                "opponent_market_overlay": group[2],
                "fc15_sheep_pressure": group[3],
                "losses": len(margins),
                "mean_loss_margin": mean(margins),
                "losses_within_500": sum(value >= -500 for value in margins),
                "losses_within_1000": sum(value >= -1000 for value in margins),
            }
        )
    return sorted(rows, key=lambda row: (-row["losses"], row["mean_loss_margin"]))


def render(payload: dict) -> str:
    lines = [
        f"# {payload['candidate_label']} 相对 {payload['source_label']} 配对审计",
        "",
        "同一随机事件、双方换座，逐局比较；诊断字段只用于离线归因，不进入策略。",
        "",
        "| 对手 | 原版胜率 | 候选胜率 | 净胜场 | 救回败局 | 伤害原胜局 | 平均分差变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {opponent} | {source_win_rate:.2%} | {candidate_win_rate:.2%} | "
            "{net_wins:+d} | {rescued_losses} | {spoiled_wins} | {margin_delta:+.1f} |".format(
                **row,
                margin_delta=row["candidate_mean_margin"] - row["source_mean_margin"],
            )
        )
    if payload["boatlee_loss_groups"]:
        lines += [
            "",
            "## Boatlee 剩余败局结构",
            "",
            "| 我方路线 | 对手路线 | 对手市场覆盖 | 羊压分支 | 败局 | 平均负分 | 500内 | 1000内 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in payload["boatlee_loss_groups"]:
            lines.append(
                "| {fc15_route} | {opponent_route} | {opponent_market_overlay} | "
                "{fc15_sheep_pressure} | {losses} | {mean_loss_margin:.1f} | "
                "{losses_within_500} | {losses_within_1000} |".format(**row)
            )
    lines += ["", f"机器凭证：`{payload['output_json']}`", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--boatlee-diagnostics", type=Path)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--source-label", default="FC15")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    candidate = load(args.candidate)
    source = load(args.source)
    if candidate["official_package_version"] != source["official_package_version"]:
        raise ValueError("official-package mismatch")
    source_rows = {row["opponent"]: row for row in source["rows"]}
    rows, changes = [], {}
    for candidate_row in candidate["rows"]:
        name = candidate_row["opponent"]
        row, changed = summarize(candidate_row, source_rows[name])
        rows.append(row)
        changes[name] = changed

    groups = []
    if args.boatlee_diagnostics:
        diagnostics = load(args.boatlee_diagnostics)
        diagnostic_row = next(
            row for row in diagnostics["rows"] if row["opponent"] == "boatlee_v21_latest"
        )
        candidate_row = next(
            (row for row in candidate["rows"] if row["opponent"] == "boatlee_v21_latest"),
            None,
        )
        if candidate_row:
            groups = boatlee_groups(candidate_row, diagnostic_row)

    payload = {
        "schema": "kaggriculture.fc15-market-overlay-paired-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_label": args.candidate_label,
        "source_label": args.source_label,
        "official_package_version": candidate["official_package_version"],
        "candidate_input": {"path": str(args.candidate), "sha256": digest(args.candidate)},
        "source_input": {"path": str(args.source), "sha256": digest(args.source)},
        "rows": rows,
        "boatlee_loss_groups": groups,
        "changed_games": changes,
        "output_json": str(args.output_json),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_report.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Paired comparison of the FC12G weed/hire guard against frozen FC2B.

The two inputs must use identical opponents, seeds, and seat-swap protocol.
This script deliberately compares every game rather than only aggregate rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _score(result: str) -> float:
    return {"win": 1.0, "tie": 0.5, "loss": 0.0}[result]


def _game_map(row: dict) -> dict[tuple[int, int], dict]:
    return {
        (int(game["seed"]), int(game["candidate_seat"])): game
        for game in row["per_game"]
    }


def _validate(candidate: dict, source: dict) -> None:
    fields = ("official_package_version", "seed_values", "games_per_opponent", "seat_protocol")
    for field in fields:
        if candidate[field] != source[field]:
            raise ValueError(f"mismatched {field}: {candidate[field]!r} != {source[field]!r}")
    candidate_names = [row["opponent"] for row in candidate["rows"]]
    source_names = [row["opponent"] for row in source["rows"]]
    if candidate_names != source_names:
        raise ValueError("opponent order/set mismatch")


def _compare_row(candidate_row: dict, source_row: dict) -> tuple[dict, list[dict]]:
    candidate_games = _game_map(candidate_row)
    source_games = _game_map(source_row)
    if candidate_games.keys() != source_games.keys():
        raise ValueError(f"per-game key mismatch for {candidate_row['opponent']}")

    deltas: list[dict] = []
    improved = regressed = unchanged = 0
    rescued = harmed = 0
    score_improved = score_regressed = 0
    for key in sorted(candidate_games):
        candidate_game = candidate_games[key]
        source_game = source_games[key]
        margin_delta = int(candidate_game["margin"]) - int(source_game["margin"])
        if margin_delta > 0:
            improved += 1
        elif margin_delta < 0:
            regressed += 1
        else:
            unchanged += 1

        source_score = _score(source_game["result"])
        candidate_score = _score(candidate_game["result"])
        if candidate_score > source_score:
            score_improved += 1
        elif candidate_score < source_score:
            score_regressed += 1
        if source_game["result"] == "loss" and candidate_game["result"] == "win":
            rescued += 1
        if source_game["result"] == "win" and candidate_game["result"] != "win":
            harmed += 1

        if margin_delta or source_game["result"] != candidate_game["result"]:
            deltas.append(
                {
                    "seed": key[0],
                    "candidate_seat": key[1],
                    "source_result": source_game["result"],
                    "candidate_result": candidate_game["result"],
                    "source_margin": int(source_game["margin"]),
                    "candidate_margin": int(candidate_game["margin"]),
                    "margin_delta": margin_delta,
                    "source_candidate_cash": int(source_game["candidate_cash"]),
                    "candidate_candidate_cash": int(candidate_game["candidate_cash"]),
                    "source_opponent_cash": int(source_game["opponent_cash"]),
                    "candidate_opponent_cash": int(candidate_game["opponent_cash"]),
                }
            )

    games = len(candidate_games)
    summary = {
        "opponent": candidate_row["opponent"],
        "games": games,
        "source_wins": int(source_row["wins"]),
        "candidate_wins": int(candidate_row["wins"]),
        "win_delta": int(candidate_row["wins"]) - int(source_row["wins"]),
        "source_score_rate": float(source_row["score_rate"]),
        "candidate_score_rate": float(candidate_row["score_rate"]),
        "score_rate_delta": float(candidate_row["score_rate"] - source_row["score_rate"]),
        "source_mean_margin": float(source_row["mean_margin"]),
        "candidate_mean_margin": float(candidate_row["mean_margin"]),
        "mean_margin_delta": float(candidate_row["mean_margin"] - source_row["mean_margin"]),
        "margin_improved_games": improved,
        "margin_regressed_games": regressed,
        "exact_margin_unchanged_games": unchanged,
        "outcome_improved_games": score_improved,
        "outcome_regressed_games": score_regressed,
        "source_losses_rescued_to_win": rescued,
        "source_wins_harmed": harmed,
        "changed_games": len(deltas),
    }
    return summary, deltas


def _render_report(payload: dict) -> str:
    lines = [
        "# FC12G 批量雇工前杂草补偿保护：配对验收",
        "",
        f"生成时间：{payload['generated_at_utc']}",
        "",
        "## 结论",
        "",
        payload["decision_zh"],
        "",
        "## 验收口径",
        "",
        f"- 官方规则版本：{payload['official_package_version']}。",
        f"- 每个对手：{payload['games_per_opponent']} 局；相同随机种子，双方换座。",
        "- 原版与候选逐局按 `(opponent, seed, seat)` 配对，比较胜负与终局现金差。",
        "- 该阶段只验收单一修复，不宣称已达到全池逐对手 90%。",
        "",
        "## 逐对手结果",
        "",
        "| 对手 | 原版胜率 | FC12G胜率 | 胜场变化 | 平均分差变化 | 救回败局 | 伤害原胜局 | 变化局数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {opponent} | {source_score_rate:.2%} | {candidate_score_rate:.2%} | "
            "{win_delta:+d} | {mean_margin_delta:+.1f} | {source_losses_rescued_to_win} | "
            "{source_wins_harmed} | {changed_games} |".format(**row)
        )
    overall = payload["overall"]
    lines.extend(
        [
            "",
            "## 总计",
            "",
            f"- 配对对局：{overall['games']}。",
            f"- 胜场：{overall['source_wins']} → {overall['candidate_wins']}（{overall['win_delta']:+d}）。",
            f"- 救回原败局：{overall['source_losses_rescued_to_win']}；伤害原胜局：{overall['source_wins_harmed']}。",
            f"- 现金差改善/退化/不变：{overall['margin_improved_games']} / {overall['margin_regressed_games']} / {overall['exact_margin_unchanged_games']}。",
            f"- 候选最低逐对手胜率：{overall['candidate_min_score_rate']:.2%}。",
            "",
            "## 证据文件",
            "",
            f"- FC12G：`{payload['candidate_input']['path']}`",
            f"- FC2B 原版：`{payload['source_input']['path']}`",
            f"- 机器可读配对结果：`{payload['output_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    candidate = _load(args.candidate)
    source = _load(args.source)
    _validate(candidate, source)

    source_rows = {row["opponent"]: row for row in source["rows"]}
    rows: list[dict] = []
    changes: dict[str, list[dict]] = {}
    for candidate_row in candidate["rows"]:
        row, row_changes = _compare_row(candidate_row, source_rows[candidate_row["opponent"]])
        rows.append(row)
        changes[candidate_row["opponent"]] = row_changes

    overall = {
        "games": sum(row["games"] for row in rows),
        "source_wins": sum(row["source_wins"] for row in rows),
        "candidate_wins": sum(row["candidate_wins"] for row in rows),
        "win_delta": sum(row["win_delta"] for row in rows),
        "margin_improved_games": sum(row["margin_improved_games"] for row in rows),
        "margin_regressed_games": sum(row["margin_regressed_games"] for row in rows),
        "exact_margin_unchanged_games": sum(row["exact_margin_unchanged_games"] for row in rows),
        "outcome_improved_games": sum(row["outcome_improved_games"] for row in rows),
        "outcome_regressed_games": sum(row["outcome_regressed_games"] for row in rows),
        "source_losses_rescued_to_win": sum(row["source_losses_rescued_to_win"] for row in rows),
        "source_wins_harmed": sum(row["source_wins_harmed"] for row in rows),
        "candidate_min_score_rate": min(row["candidate_score_rate"] for row in rows),
    }
    passed_local_safety = overall["source_wins_harmed"] == 0 and overall["outcome_regressed_games"] == 0
    decision_zh = (
        "该修复通过关键对手面板的配对安全门，可进入更大全池复验；"
        "但仍未达到最终逐对手 90% 总目标。"
        if passed_local_safety
        else "该修复未通过配对安全门，不应晋升；需要检查被伤害的原胜局。"
    )
    payload = {
        "schema": "kaggriculture.fusion_champion.fc12.paired_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed_local_safety else "FAIL",
        "decision_zh": decision_zh,
        "official_package_version": candidate["official_package_version"],
        "seed_values": candidate["seed_values"],
        "games_per_opponent": candidate["games_per_opponent"],
        "seat_protocol": candidate["seat_protocol"],
        "candidate_input": {"path": str(args.candidate), "sha256": _sha256(args.candidate)},
        "source_input": {"path": str(args.source), "sha256": _sha256(args.source)},
        "output_json": str(args.output_json),
        "rows": rows,
        "overall": overall,
        "changed_games": changes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with args.output_report.open("w", encoding="utf-8") as handle:
        handle.write(_render_report(payload))
    print(json.dumps({"status": payload["status"], "overall": overall}, ensure_ascii=False))


if __name__ == "__main__":
    main()

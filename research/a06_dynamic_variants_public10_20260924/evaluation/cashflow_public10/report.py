"""Audit and report the completed R14 Cashflow versus public top-ten panel."""

from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import statistics


HERE = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read(HERE / "PROTOCOL.json")
    opponents = protocol["opponents"]
    seeds = protocol["seeds"]
    rows = [json.loads(line) for line in (HERE / "games.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected = {(row["ref"], seed, seat) for row in opponents for seed in seeds for seat in (0, 1)}
    actual = {(row["public"], row["seed"], row["local_seat"]) for row in rows}
    if len(rows) != len(actual) or actual != expected or len(rows) != 1000:
        raise ValueError(f"incomplete/duplicate panel: {len(rows)} rows, {len(actual)} keys, {len(expected)} expected")
    if any(row["local"] != "r14_cashflow" or row["error"] or row.get("steps") != 719 for row in rows):
        raise ValueError("invalid policy or game result")

    by_ref = defaultdict(list)
    by_seed = defaultdict(list)
    for row in rows:
        by_ref[row["public"]].append(row)
        by_seed[row["seed"]].append(row)
    details = []
    for rank, opponent in enumerate(opponents, 1):
        group = by_ref[opponent["ref"]]
        if len(group) != 100:
            raise ValueError(f"not 100 games: {opponent['ref']}")
        wins = sum(row["margin"] > 0 for row in group)
        ties = sum(row["margin"] == 0 for row in group)
        losses = 100 - wins - ties
        details.append({"rank_in_frozen_top10": rank, "id": opponent["id"], "ref": opponent["ref"],
                        "url": f"https://www.kaggle.com/code/{opponent['ref']}",
                        "version": opponent["version"],
                        "main_sha256": opponent["files"]["main.py"],
                        "games": 100, "wins": wins, "ties": ties, "losses": losses,
                        "win_rate": wins / 100, "point_rate": (wins + 0.5 * ties) / 100,
                        "seat0_wins": sum(row["margin"] > 0 for row in group if row["local_seat"] == 0),
                        "seat1_wins": sum(row["margin"] > 0 for row in group if row["local_seat"] == 1),
                        "mean_cash_margin": statistics.mean(row["margin"] for row in group),
                        "mean_candidate_cash": statistics.mean(row["local_cash"] for row in group),
                        "mean_opponent_cash": statistics.mean(row["public_cash"] for row in group)})

    wins = sum(item["wins"] for item in details)
    ties = sum(item["ties"] for item in details)
    losses = sum(item["losses"] for item in details)
    rng = random.Random(2026093004)
    boot = []
    for _ in range(3000):
        sampled = [rng.choice(seeds) for _ in seeds]
        boot.append(sum(row["local_win"] + 0.5 * row["tie"] for seed in sampled for row in by_seed[seed]) / 1000)
    boot.sort()
    seen = {}
    duplicate_codes = []
    unique_details = []
    for item in details:
        if item["main_sha256"] in seen:
            duplicate_codes.append([seen[item["main_sha256"]], item["ref"]])
        else:
            seen[item["main_sha256"]] = item["ref"]
            unique_details.append(item)
    unique_games = 100 * len(unique_details)
    unique_points = sum(item["wins"] + 0.5 * item["ties"] for item in unique_details)
    outcome_vectors = {
        item["ref"]: tuple((row["local_cash"], row["public_cash"], row["margin"])
                           for row in sorted(by_ref[item["ref"]], key=lambda row: (row["seed"], row["local_seat"])))
        for item in details
    }
    same_outcome_pairs = [[left["ref"], right["ref"]] for left, right in combinations(details, 2)
                          if outcome_vectors[left["ref"]] == outcome_vectors[right["ref"]]]
    seen_outcomes = set()
    unique_outcome_details = []
    for item in details:
        outcome = outcome_vectors[item["ref"]]
        if outcome not in seen_outcomes:
            seen_outcomes.add(outcome)
            unique_outcome_details.append(item)
    outcome_unique_points = sum(item["wins"] + 0.5 * item["ties"] for item in unique_outcome_details)
    result = {"candidate": "r14_cashflow", "games": len(rows), "opponents": len(opponents),
              "unique_runnable_main_files": len(unique_details),
              "wins": wins, "ties": ties, "losses": losses,
              "win_rate": wins / len(rows), "point_rate": (wins + 0.5 * ties) / len(rows),
              "bootstrap_95_point_rate": [boot[74], boot[2924]],
              "unique_code_point_rate": unique_points / unique_games,
              "duplicate_main_sha_groups": duplicate_codes,
              "identical_100_game_result_pairs": same_outcome_pairs,
              "unique_result_vectors": len(unique_outcome_details),
              "unique_result_vector_point_rate": outcome_unique_points / (100 * len(unique_outcome_details)),
              "mean_cash_margin": statistics.mean(row["margin"] for row in rows),
              "games_with_observed_action_over_1s": sum(max(row["max_local_action_s"], row["max_public_action_s"]) > 1 for row in rows),
              "max_observed_action_seconds": max(max(row["max_local_action_s"], row["max_public_action_s"]) for row in rows),
              "per_opponent": details,
              "method": "50 new common seeds x both seats, official 1.32.7 Python referee, live policy actions"}
    (HERE / "RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (HERE / "PER_OPPONENT.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["rank_in_frozen_top10", "ref", "version", "games", "wins", "ties", "losses",
                  "point_rate", "seat0_wins", "seat1_wins", "mean_cash_margin"]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(details)

    lines = ["# R14 Cashflow vs 冻结公开前十", "",
             "对手是 2026-09-24 交付包中 `QUALIFIED_TOP20.json` 的前十，不代表此刻 Kaggle 的实时前十。",
             "双方均按实时观察运行；官方 Kaggriculture 1.32.7 Python 裁判，50 个新种子、每种子交换座位。",
             f"全部 {len(rows):,} 局完成 719 步且零错误。胜 {wins}、平 {ties}、负 {losses}；",
             f"整体胜率 **{wins / len(rows):.2%}**，胜一分、平半分的积分率 **{result['point_rate']:.2%}**。",
             f"按共同种子重采样的整体积分率 95% 区间：{boot[74]:.2%}–{boot[2924]:.2%}。", "",
             "| 冻结顺位 | 公开方案 | Cashflow 胜-平-负 | 积分率 | 座位 0 / 1 胜 | 平均终局现金差 |",
             "|---:|---|---:|---:|---:|---:|"]
    for item in details:
        lines.append(f"| {item['rank_in_frozen_top10']} | [{item['ref']}]({item['url']}) "
                     f"| {item['wins']}-{item['ties']}-{item['losses']} | {item['point_rate']:.1%} "
                     f"| {item['seat0_wins']} / {item['seat1_wins']} | {item['mean_cash_margin']:+,.1f} |")
    lines.extend(["", "## 核验与边界", "",
                  "`PROTOCOL.json` 固定双方文件哈希、官方规则哈希、种子和座位；`games.jsonl` 保留逐局结果。",
                  "`RESULTS.json` 和 `PER_OPPONENT.csv` 保留可机器读取的结果。",
                  "交付包中的公开方案为 2026-09-23/24 抓取版本；不能从本地胜率直接推断实时天梯胜率。"])
    if (HERE / "SERIAL_RECHECK.json").exists():
        serial = read(HERE / "SERIAL_RECHECK.json")
        if serial["exact_cash_and_result_matches"] != serial["games"]:
            raise ValueError("serial recheck mismatch")
        lines.append(f"单进程重跑 {serial['games']} 局，终局双方现金及胜负与正式面板逐局完全一致。")
    if duplicate_codes:
        lines.append(f"其中 {len(duplicate_codes)} 对方案的可运行 `main.py` 完全相同：" +
                     "；".join(f"`{a}` 与 `{b}`" for a, b in duplicate_codes) + "。")
        lines.append(f"去重后的 {len(unique_details)} 份可运行代码，{unique_games} 局积分率为 **{result['unique_code_point_rate']:.2%}**。")
    if same_outcome_pairs:
        lines.append("另外，在本次相同 50 种子双座位面板中，以下方案的逐局双方现金完全相同：" +
                     "；".join(f"`{a}` 与 `{b}`" for a, b in same_outcome_pairs) + "。")
        lines.append(f"按本次结果向量去重后有 {len(unique_outcome_details)} 组，积分率 **{result['unique_result_vector_point_rate']:.2%}**；"
                     "这只证明本次测试输出一致，不证明所有盘面下的策略相同。")
    lines.append("16 进程并发的单步耗时受机器争用影响，不能直接作为 Kaggle 容器超时判定。")
    lines.append("")
    (HERE / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"games": len(rows), "wins": wins, "ties": ties, "losses": losses,
                      "point_rate": round(result["point_rate"], 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

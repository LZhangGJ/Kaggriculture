"""Audit a complete 13-agent paired-seat panel and rank by strict match points."""

from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


HERE = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read(HERE / "PROTOCOL.json")
    agents = [row["id"] for row in protocol["agents"]]
    seeds = protocol["seeds"]
    pairs = [tuple(pair) for pair in protocol["pairs"]]
    expected = {(a, b, seed, seat) for a, b in pairs for seed in seeds for seat in (0, 1)}
    rows = [json.loads(line) for line in (HERE / "games.jsonl").read_text(encoding="utf-8").splitlines() if line]
    actual = {(r["a"], r["b"], r["seed"], r["seat_a"]) for r in rows}
    if len(rows) != len(actual) or actual != expected or len(rows) != 7800:
        raise ValueError(f"incomplete or duplicate panel: {len(rows)} rows, {len(actual)} keys, {len(expected)} expected")
    if any(r["error"] or r.get("steps") != 719 for r in rows):
        raise ValueError("error/incomplete games cannot count toward ranking")

    pair_data = defaultdict(list)
    summary = {name: {"id": name, "games": 0, "wins": 0, "losses": 0, "ties": 0,
                      "points": 0.0, "margin_sum": 0.0, "cash_sum": 0.0} for name in agents}
    seed_points = {name: {seed: 0.0 for seed in seeds} for name in agents}
    for row in rows:
        a, b, seed = row["a"], row["b"], row["seed"]
        margin = row["margin"]
        pa = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        pb = 1.0 - pa
        pair_data[(a, b)].append(row)
        for name, point, cash, signed_margin in ((a, pa, row["cash_a"], margin),
                                                  (b, pb, row["cash_b"], -margin)):
            item = summary[name]
            item["games"] += 1
            item["points"] += point
            item["wins"] += int(point == 1.0)
            item["losses"] += int(point == 0.0)
            item["ties"] += int(point == 0.5)
            item["cash_sum"] += cash
            item["margin_sum"] += signed_margin
            seed_points[name][seed] += point

    pair_results = []
    for a, b in pairs:
        group = pair_data[(a, b)]
        if len(group) != 100 or {r["seed"] for r in group} != set(seeds):
            raise ValueError(f"bad pair coverage: {a} vs {b}")
        wins_a = sum(r["margin"] > 0 for r in group)
        wins_b = sum(r["margin"] < 0 for r in group)
        ties = 100 - wins_a - wins_b
        pair_results.append({"a": a, "b": b, "games": 100, "wins_a": wins_a,
                             "wins_b": wins_b, "ties": ties,
                             "mean_margin_a": statistics.mean(r["margin"] for r in group),
                             "seat0_wins_a": sum(r["margin"] > 0 for r in group if r["seat_a"] == 0),
                             "seat1_wins_a": sum(r["margin"] > 0 for r in group if r["seat_a"] == 1)})

    bootstrap = {name: [] for name in agents}
    first_count = {name: 0 for name in agents}
    rng = random.Random(2026093001)
    for _ in range(3000):
        sampled = [rng.choice(seeds) for _ in seeds]
        rates = {name: sum(seed_points[name][seed] for seed in sampled) / (len(seeds) * 24)
                 for name in agents}
        for name, rate in rates.items():
            bootstrap[name].append(rate)
        first_count[max(agents, key=lambda name: rates[name])] += 1
    for name, item in summary.items():
        item["point_rate"] = item["points"] / item["games"]
        item["strict_win_rate"] = item["wins"] / item["games"]
        item["mean_margin"] = item.pop("margin_sum") / item["games"]
        item["mean_cash"] = item.pop("cash_sum") / item["games"]
        rates = sorted(bootstrap[name])
        item["bootstrap_95_pct"] = [rates[74], rates[2924]]
        item["bootstrap_first_probability"] = first_count[name] / 3000
    ranking = sorted(summary.values(), key=lambda item: (-item["point_rate"], -item["mean_margin"], item["id"]))
    for position, item in enumerate(ranking, 1):
        item["rank"] = position
    result = {"protocol": "PROTOCOL.json", "games": len(rows), "agents": len(agents),
              "pairs": len(pairs), "seeds": len(seeds), "ranking": ranking,
              "pair_results": pair_results,
              "max_observed_action_seconds": max(max(r["max_action_a_s"], r["max_action_b_s"]) for r in rows),
              "games_with_observed_action_over_1s": sum(max(r["max_action_a_s"], r["max_action_b_s"]) > 1 for r in rows),
              "ranking_method": "win=1, tie=0.5, loss=0; all 1200 games per agent; secondary mean cash margin",
              "uncertainty": "3000 bootstrap resamples of the 50 shared seeds, keeping both seats and every opponent together"}
    (HERE / "RANKING.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    matrix = {name: {other: "—" for other in agents} for name in agents}
    for row in pair_results:
        a, b = row["a"], row["b"]
        matrix[a][b] = f"{row['wins_a']}-{row['wins_b']}-{row['ties']}"
        matrix[b][a] = f"{row['wins_b']}-{row['wins_a']}-{row['ties']}"
    with (HERE / "HEAD_TO_HEAD.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent", *agents])
        for name in agents:
            writer.writerow([name, *(matrix[name][other] for other in agents)])

    lines = ["# A06 R12 GPT Pro 交付方案两两对战排名", "",
             "13 个冻结主方案，78 组配对；每组使用同一批 50 个种子并交换座位，共 100 局。",
             "规则裁判为官方 Kaggriculture 1.32.7 Python 解释器，双方根据实时观察出招，不使用 Replay 动作。",
             "全部 7,800 局均完成 719 次状态转移，运行错误为零。", "",
             "按对战积分排名：胜 1 分、平 0.5 分、负 0 分。每个方案与另外 12 个各打 100 局，共 1,200 局。",
             "95% 区间对 50 个共同种子整体重采样，保留每个种子的全部对手和双座位结果。", "",
             "| 排名 | 方案 | 胜-平-负 / 1200 | 积分率 | 95% 种子重采样区间 | 平均现金差 | 排第一概率 |",
             "|---:|---|---:|---:|---:|---:|---:|" ]
    for item in ranking:
        ci = item["bootstrap_95_pct"]
        lines.append(f"| {item['rank']} | `{item['id']}` | {item['wins']}-{item['ties']}-{item['losses']} "
                     f"| {item['point_rate']:.2%} | {ci[0]:.2%}–{ci[1]:.2%} "
                     f"| {item['mean_margin']:+,.1f} | {item['bootstrap_first_probability']:.1%} |")
    lines.extend(["", "## 两两结果", "", "`HEAD_TO_HEAD.csv` 给出 13×13 胜-负-平矩阵；`RANKING.json` 包含每组现金差和两座位拆分。",
                  "", "## 核验与边界", "", "`POOL.json` 记录每个原始压缩包哈希、选用的主方案目录和提取后文件哈希。",
                  "`PROTOCOL.json` 冻结种子、配对、二进制哈希和官方裁判哈希。",
                  "压缩包文件名前缀数字是作者过去的公开对手面板成绩，不是本次两两对战成绩。",
                  "16 进程并发下的单步耗时会受机器争用影响；本地耗时不能直接当作 Kaggle 容器超时判定。", ""])
    (HERE / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"games": len(rows), "pairs": len(pairs), "top3":
                      [(item["id"], round(item["point_rate"], 4)) for item in ranking[:3]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

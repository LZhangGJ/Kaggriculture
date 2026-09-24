"""Audit all 3,000 games and compare each variant to the paired Cashflow panel."""

from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASHFLOW = ROOT / "experiments/r14_cashflow_vs_public_top10_20260924"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def score(row: dict) -> float:
    return row["local_win"] + 0.5 * row["tie"]


def percentile(values: list[float]) -> list[float]:
    values.sort()
    return [values[74], values[2924]]


def main() -> None:
    protocol = read(HERE / "PROTOCOL.json")
    cashflow_protocol = read(CASHFLOW / "PROTOCOL.json")
    candidates = [candidate["id"] for candidate in protocol["candidates"]]
    opponents = [opponent["ref"] for opponent in protocol["opponents"]]
    seeds = protocol["seeds"]
    if seeds != cashflow_protocol["seeds"] or opponents != [o["ref"] for o in cashflow_protocol["opponents"]]:
        raise ValueError("candidate panel no longer matches Cashflow opponents and seeds")
    games = rows(HERE / "games.jsonl")
    baseline = rows(CASHFLOW / "games.jsonl")
    expected = {(candidate, opponent, seed, seat) for candidate in candidates
                for opponent in opponents for seed in seeds for seat in (0, 1)}
    actual = {(r["local"], r["public"], r["seed"], r["local_seat"]) for r in games}
    if len(games) != len(actual) or actual != expected or len(games) != 3000:
        raise ValueError(f"incomplete/duplicate panel: {len(games)} rows, {len(actual)} keys")
    if any(r["error"] or r.get("steps") != 719 for r in games):
        raise ValueError("errors or incomplete games in candidate panel")
    base_expected = {(opponent, seed, seat) for opponent in opponents for seed in seeds for seat in (0, 1)}
    base_actual = {(r["public"], r["seed"], r["local_seat"]) for r in baseline}
    if len(baseline) != len(base_actual) or base_actual != base_expected or len(baseline) != 1000:
        raise ValueError("Cashflow baseline is not complete and paired")
    if any(r["error"] or r.get("steps") != 719 for r in baseline):
        raise ValueError("Cashflow baseline has errors/incomplete games")

    by_candidate = defaultdict(list)
    by_pair = defaultdict(list)
    by_seed = defaultdict(lambda: defaultdict(list))
    for row in games:
        by_candidate[row["local"]].append(row)
        by_pair[(row["local"], row["public"])].append(row)
        by_seed[row["local"]][row["seed"]].append(row)
    baseline_by_pair = defaultdict(list)
    baseline_by_seed = defaultdict(list)
    for row in baseline:
        baseline_by_pair[row["public"]].append(row)
        baseline_by_seed[row["seed"]].append(row)
    per_opponent = []
    for rank, opponent in enumerate(opponents, 1):
        base_group = baseline_by_pair[opponent]
        if len(base_group) != 100:
            raise ValueError(f"Cashflow not 100 games against {opponent}")
        for candidate in candidates:
            group = by_pair[(candidate, opponent)]
            if len(group) != 100:
                raise ValueError(f"{candidate} not 100 games against {opponent}")
            wins = sum(r["margin"] > 0 for r in group)
            ties = sum(r["margin"] == 0 for r in group)
            per_opponent.append({"frozen_rank": rank, "candidate": candidate, "public": opponent,
                                 "games": 100, "wins": wins, "ties": ties,
                                 "losses": 100 - wins - ties, "point_rate": (wins + 0.5 * ties) / 100,
                                 "cashflow_wins": sum(r["margin"] > 0 for r in base_group),
                                 "cashflow_ties": sum(r["margin"] == 0 for r in base_group),
                                 "seat0_wins": sum(r["margin"] > 0 for r in group if r["local_seat"] == 0),
                                 "seat1_wins": sum(r["margin"] > 0 for r in group if r["local_seat"] == 1),
                                 "mean_cash_margin": statistics.mean(r["margin"] for r in group)})

    baseline_score = sum(score(r) for r in baseline)
    baseline_mean_margin = statistics.mean(r["margin"] for r in baseline)
    selected_unique_refs = []
    seen_main_hashes = set()
    for opponent in protocol["opponents"]:
        main_hash = opponent["files"]["main.py"]
        if main_hash not in seen_main_hashes:
            seen_main_hashes.add(main_hash)
            selected_unique_refs.append(opponent["ref"])
    if len(selected_unique_refs) != 9 or opponents[5] in selected_unique_refs:
        raise ValueError("unexpected runnable-code duplicate structure")
    # The two last public entries have distinct files but identical 100-game
    # output vectors on this frozen panel, so also expose a panel-only dedup.
    selected_result_refs = [ref for index, ref in enumerate(opponents) if index not in (5, 9)]
    def vector(candidate: str, opponent: str):
        group = by_pair[(candidate, opponent)]
        return tuple((r["seed"], r["local_seat"], r["local_cash"], r["public_cash"], r["margin"])
                     for r in sorted(group, key=lambda r: (r["seed"], r["local_seat"])))
    for candidate in candidates:
        if vector(candidate, opponents[0]) != vector(candidate, opponents[5]) or \
           vector(candidate, opponents[8]) != vector(candidate, opponents[9]):
            raise ValueError(f"observed result-vector duplicate structure changed: {candidate}")
    summary = []
    rng = random.Random(2026100105)
    sampled_seeds = [[rng.choice(seeds) for _ in seeds] for _ in range(3000)]
    for candidate in candidates:
        group = by_candidate[candidate]
        if len(group) != 1000:
            raise ValueError(f"{candidate} not 1000 games")
        wins = sum(r["margin"] > 0 for r in group)
        ties = sum(r["margin"] == 0 for r in group)
        base_seed_scores = {seed: sum(score(r) for r in baseline_by_seed[seed]) for seed in seeds}
        candidate_seed_scores = {seed: sum(score(r) for r in by_seed[candidate][seed]) for seed in seeds}
        rates = [sum(candidate_seed_scores[s] for s in sample) / 1000 for sample in sampled_seeds]
        differences = [sum(candidate_seed_scores[s] - base_seed_scores[s] for s in sample) / 1000
                       for sample in sampled_seeds]
        summary.append({"candidate": candidate, "games": 1000, "wins": wins, "ties": ties,
                        "losses": 1000 - wins - ties,
                        "point_rate": (wins + 0.5 * ties) / 1000,
                        "bootstrap_95_point_rate": percentile(rates),
                        "paired_point_rate_delta_vs_cashflow": (wins + 0.5 * ties - baseline_score) / 1000,
                        "bootstrap_95_paired_delta": percentile(differences),
                        "mean_cash_margin": statistics.mean(r["margin"] for r in group),
                        "mean_margin_delta_vs_cashflow": statistics.mean(r["margin"] for r in group) - baseline_mean_margin,
                        "unique_main_point_rate": sum(score(r) for r in group if r["public"] in selected_unique_refs) / 900,
                        "unique_panel_result_point_rate": sum(score(r) for r in group if r["public"] in selected_result_refs) / 800,
                        "games_with_observed_action_over_1s": sum(max(r["max_local_action_s"], r["max_public_action_s"]) > 1 for r in group)})
    summary.sort(key=lambda item: (-item["point_rate"], -item["mean_cash_margin"], item["candidate"]))
    result = {"method": protocol["method"], "games": len(games), "candidates": len(candidates),
              "opponents": len(opponents), "baseline_cashflow": {"wins": sum(r["local_win"] for r in baseline),
              "ties": sum(r["tie"] for r in baseline), "point_rate": baseline_score / 1000,
              "mean_cash_margin": baseline_mean_margin}, "summary": summary,
              "per_opponent": per_opponent,
              "max_observed_action_seconds": max(max(r["max_local_action_s"], r["max_public_action_s"]) for r in games)}
    (HERE / "RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (HERE / "PER_OPPONENT.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["frozen_rank", "candidate", "public", "games", "wins", "ties", "losses",
                  "point_rate", "cashflow_wins", "cashflow_ties", "seat0_wins", "seat1_wins", "mean_cash_margin"]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_opponent)

    detail = {(r["candidate"], r["public"]): r for r in per_opponent}
    lines = ["# 三个 A06 改良版对冻结公开前十，各 100 局", "",
             "Rule R18、R14 Liquidity、R14 TL5 均使用原交付包的冻结 `main.py` 与依赖。",
             "公开对手与 Cashflow 轮相同，使用相同的 50 个种子、双座位；官方 Kaggriculture 1.32.7 Python 裁判，实时策略对战。",
             "全部 3,000 局完整走满 719 步且零运行错误。", "",
             "## 总体结果", "",
             "| 方案 | 胜-平-负 / 1000 | 积分率 | 95% 种子区间 | 较 Cashflow 差距 | 差距 95% 区间 | 平均终局现金差 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary:
        low, high = item["bootstrap_95_point_rate"]
        delta_low, delta_high = item["bootstrap_95_paired_delta"]
        lines.append(f"| `{item['candidate']}` | {item['wins']}-{item['ties']}-{item['losses']} "
                     f"| {item['point_rate']:.2%} | {low:.2%}–{high:.2%} "
                     f"| {item['paired_point_rate_delta_vs_cashflow']*100:+.1f} 个百分点 "
                     f"| {delta_low*100:+.1f}～{delta_high*100:+.1f} 个百分点 "
                     f"| {item['mean_cash_margin']:+,.1f} |")
    lines.append(f"| `r14_cashflow`（参照） | {result['baseline_cashflow']['wins']}-{result['baseline_cashflow']['ties']}-"
                 f"{1000-result['baseline_cashflow']['wins']-result['baseline_cashflow']['ties']} "
                 f"| {result['baseline_cashflow']['point_rate']:.2%} | 见上一轮报告 | — | — "
                 f"| {result['baseline_cashflow']['mean_cash_margin']:+,.1f} |")
    lines.extend(["", "## 逐对手胜场 / 各 100 局", "",
                  "| 冻结顺位 | 公开方案 | Rule R18 | R14 Liquidity | R14 TL5 | Cashflow 参照 |",
                  "|---:|---|---:|---:|---:|---:|"])
    for rank, opponent in enumerate(opponents, 1):
        cells = [detail[(candidate, opponent)] for candidate in candidates]
        lines.append(f"| {rank} | [{opponent}](https://www.kaggle.com/code/{opponent}) "
                     f"| {cells[0]['wins']} | {cells[1]['wins']} | {cells[2]['wins']} "
                     f"| {cells[0]['cashflow_wins']} |")
    lines.extend(["", "## 核验与边界", "",
                  "`PROTOCOL.json` 冻结候选与对手文件哈希、裁判哈希和种子；`games.jsonl` 保留全部逐局结果。",
                  "`RESULTS.json` 包含逐对手现金差、双座位成绩与配对种子重采样结果。",
                  "公开十方案是 2026-09-23/24 冻结资格面板中的前十，不代表实时天梯前十。",
                  "其中 Soil Remembers Rain 与 Demand-Preserving Sale Timing 的可运行 `main.py` 哈希相同；十条结果仍按用户要求分别呈现。",
                  "本地并发运行的单步耗时受机器争用影响，不能直接视为 Kaggle 容器超时结论。", ""])
    lines.insert(-1, "Farmer John 与 Pipe18 Six Layers 在本面板中的逐局双方终局现金也完全相同；只能说明这些测试盘面输出一致。")
    liquidity = next(item for item in summary if item["candidate"] == "r14_liquidity")
    lines.insert(-1, f"按 9 份不同 `main.py` 去重后，Liquidity 积分率 {liquidity['unique_main_point_rate']:.2%}；"
                 f"按本次 8 组不同结果向量去重后为 {liquidity['unique_panel_result_point_rate']:.2%}。")
    serial_path = HERE / "SERIAL_RECHECK.json"
    if serial_path.exists():
        serial = read(serial_path)
        if serial["games"] != serial["exact_cash_and_result_matches"]:
            raise ValueError("serial recheck contains mismatches")
        lines.insert(-1, f"单进程复跑 {serial['games']} 局，双方终局现金与正式并发评测逐局完全一致。")
    (HERE / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"games": len(games), "summary": [(r["candidate"], r["wins"], r["ties"],
                   round(r["point_rate"], 4)) for r in summary]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Cluster-aware analysis for the S3X A50 finite one-day branch audit."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import statistics as st


EXP = Path(__file__).resolve().parents[1]
KIND = {
    -1: "NONE",
    0: "WHEAT",
    1: "CARROT",
    2: "TOMATO",
    3: "STRAWBERRY",
    4: "MELON",
    9: "GOOSE",
    10: "COW",
    11: "SHEEP",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_payoff(choices: list[dict]) -> dict:
    return max(
        choices,
        key=lambda x: (
            x["cash"] > x["opponent_cash"],
            x["cash"] - x["opponent_cash"],
            x["cash"],
        ),
    )


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = st.fmean(xs), st.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    ordered = sorted(xs)
    def pick(q: float) -> float:
        return ordered[round(q * (len(ordered) - 1))]
    return {"p05": pick(0.05), "p25": pick(0.25), "p50": pick(0.50), "p75": pick(0.75), "p95": pick(0.95)}


def clustered_interval(rows: list[dict], field: str) -> dict:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(float(row[field]))
    clusters = [st.fmean(values) for values in by_seed.values()]
    mean = st.fmean(clusters)
    se = st.stdev(clusters) / math.sqrt(len(clusters)) if len(clusters) > 1 else None
    normal = [max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)] if se is not None else None
    radius = math.sqrt(math.log(40.0) / (2 * len(clusters)))
    return {
        "independent_seed_clusters": len(clusters),
        "mean": mean,
        "seed_cluster_se": se,
        "approximate_seed_cluster_95pct_interval": normal,
        "conservative_seed_hoeffding_95pct_bound": [max(0.0, mean - radius), min(1.0, mean + radius)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="receipts/s3x_branch_A50_v1")
    parser.add_argument("--out", default="receipts/s3x_branch_A50_v1_analysis")
    parser.add_argument("--report", default="reports/S3X_ONE_DAY_BRANCH_A50_REVIEW_ZH.md")
    args = parser.parse_args()
    source = EXP / args.input
    output = EXP / args.out
    report_path = EXP / args.report
    output.mkdir(exist_ok=False)
    meta = json.loads((source / "acceptance.json").read_text(encoding="utf8"))
    assert meta["status"] == "COMPLETE_FINITE_BRANCH_AUDIT_NOT_DEPLOYABLE_ORACLE"

    summaries = []
    input_hashes = {"acceptance.json": sha(source / "acceptance.json")}
    for path in sorted(source.glob("*.json.gz")):
        opponent = path.name.removesuffix(".json.gz")
        input_hashes[path.name] = sha(path)
        with gzip.open(path, "rt", encoding="utf8") as handle:
            raw = json.load(handle)
        games = []
        per_day: dict[int, list[dict]] = defaultdict(list)
        best_families = Counter()
        best_kinds = Counter()
        best_family_kind = Counter()
        all_family_effects: dict[str, list[float]] = defaultdict(list)
        score_delta, margin_delta = [], []
        direction_correct = direction_total = 0
        total_candidates = distinct_candidates = winning_candidates = 0
        for row in raw:
            game_flat = []
            keep_root_win = row["cash"] > row["opponent_cash"]
            for node in row["nodes"]:
                choices = node["choices"]
                keep = choices[0]
                keep_margin = keep["cash"] - keep["opponent_cash"]
                best = choose_payoff(choices)
                predicted = max(choices, key=lambda item: item["score"])
                best_margin = best["cash"] - best["opponent_cash"]
                predicted_margin = predicted["cash"] - predicted["opponent_cash"]
                unique = len({(item["cash"], item["opponent_cash"], item["suffix_hash_OFFLINE_ONLY"]) for item in choices})
                total_candidates += len(choices)
                distinct_candidates += unique
                winning_candidates += sum(item["cash"] > item["opponent_cash"] for item in choices)
                for item in choices[1:]:
                    actual_delta = (item["cash"] - item["opponent_cash"]) - keep_margin
                    predicted_delta = item["score"] - keep["score"]
                    all_family_effects[item["family"]].append(actual_delta)
                    if actual_delta != 0:
                        score_delta.append(predicted_delta)
                        margin_delta.append(actual_delta)
                        direction_correct += (predicted_delta > 0) == (actual_delta > 0)
                        direction_total += 1
                per_day[int(node["day"])].append({
                    "keep_win": keep["cash"] > keep["opponent_cash"],
                    "best_win": best["cash"] > best["opponent_cash"],
                    "predicted_win": predicted["cash"] > predicted["opponent_cash"],
                    "keep_margin": keep_margin,
                    "best_margin": best_margin,
                    "predicted_margin": predicted_margin,
                    "uplift": best_margin - keep_margin,
                    "best_family": best["family"],
                    "best_kind": KIND[int(best["kind"])],
                    "best_amount": int(best["amount"]),
                    "candidate_count": len(choices),
                    "distinct_count": unique,
                })
                game_flat.extend((int(node["day"]), item) for item in choices)
            best_day, best_one = max(
                game_flat,
                key=lambda pair: (
                    pair[1]["cash"] > pair[1]["opponent_cash"],
                    pair[1]["cash"] - pair[1]["opponent_cash"],
                    pair[1]["cash"],
                ),
            )
            best_margin = best_one["cash"] - best_one["opponent_cash"]
            keep_margin = row["cash"] - row["opponent_cash"]
            best_families[best_one["family"]] += 1
            best_kinds[KIND[int(best_one["kind"])]] += 1
            best_family_kind[f"{best_one['family']}:{KIND[int(best_one['kind'])]}:{int(best_one['amount'])}"] += 1
            games.append({
                "seed": int(row["seed"]),
                "seat": int(row["seat"]),
                "keep_win": keep_root_win,
                "finite_one_change_win": best_one["cash"] > best_one["opponent_cash"],
                "keep_margin": keep_margin,
                "finite_one_change_margin": best_margin,
                "margin_uplift": best_margin - keep_margin,
                "best_day": best_day,
                "best_family": best_one["family"],
                "best_kind": KIND[int(best_one["kind"])],
                "best_amount": int(best_one["amount"]),
            })

        keep_ci = clustered_interval(games, "keep_win")
        best_ci = clustered_interval(games, "finite_one_change_win")
        day_summary = {}
        for day, rows in sorted(per_day.items()):
            day_summary[str(day)] = {
                "nodes": len(rows),
                "keep_win_rate": st.fmean(item["keep_win"] for item in rows),
                "finite_payoff_best_win_rate": st.fmean(item["best_win"] for item in rows),
                "score_selected_win_rate": st.fmean(item["predicted_win"] for item in rows),
                "mean_margin_uplift": st.fmean(item["uplift"] for item in rows),
                "margin_uplift_quantiles": quantiles([item["uplift"] for item in rows]),
                "mean_candidates": st.fmean(item["candidate_count"] for item in rows),
                "mean_behaviorally_distinct": st.fmean(item["distinct_count"] for item in rows),
                "best_family_counts": dict(Counter(item["best_family"] for item in rows)),
            }
        baseline_losses = sum(not item["keep_win"] for item in games)
        recovered = sum((not item["keep_win"]) and item["finite_one_change_win"] for item in games)
        entry = {
            "opponent": opponent,
            "games": len(games),
            "baseline": keep_ci,
            "finite_one_change_offline_ceiling": best_ci,
            "baseline_losses": baseline_losses,
            "recovered_baseline_losses": recovered,
            "recovery_rate_among_baseline_losses": recovered / baseline_losses if baseline_losses else None,
            "mean_keep_cash": st.fmean(row["cash"] for row in raw),
            "mean_keep_opponent_cash": st.fmean(row["opponent_cash"] for row in raw),
            "mean_keep_margin": st.fmean(item["keep_margin"] for item in games),
            "mean_finite_one_change_margin": st.fmean(item["finite_one_change_margin"] for item in games),
            "mean_margin_uplift": st.fmean(item["margin_uplift"] for item in games),
            "margin_uplift_quantiles": quantiles([item["margin_uplift"] for item in games]),
            "score_margin_pearson": pearson(score_delta, margin_delta),
            "score_direction_accuracy": direction_correct / direction_total if direction_total else None,
            "all_candidate_win_fraction": winning_candidates / total_candidates,
            "behaviorally_distinct_fraction": distinct_candidates / total_candidates,
            "best_game_family_counts": dict(best_families),
            "best_game_kind_counts": dict(best_kinds),
            "best_game_family_kind_amount_counts": dict(best_family_kind),
            "candidate_family_margin_effect": {
                family: {"count": len(values), "mean": st.fmean(values), "quantiles": quantiles(values)}
                for family, values in sorted(all_family_effects.items())
            },
            "by_day": day_summary,
            "records": games,
        }
        summaries.append(entry)
        print(json.dumps({key: value for key, value in entry.items() if key not in ("records", "by_day", "candidate_family_margin_effect", "best_game_family_kind_amount_counts")}), flush=True)

    result = {
        "status": "COMPLETE_CLUSTERED_S3X_A50_ANALYSIS_NOT_DEPLOYABLE_ORACLE",
        "source_status": meta["status"],
        "input_hashes": input_hashes,
        "summary": summaries,
        "final_holdout_used": False,
        "full_goal_complete": False,
        "interpretation_guard": (
            "The finite-one-change result chooses the best day and candidate using the true suffix. "
            "It is an offline expressivity diagnostic, not a deployable policy or multistage oracle. "
            "Confidence intervals cluster the two seats by seed; candidate and day rows are descriptive only."
        ),
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf8")

    strong = [item for item in summaries if item["opponent"] not in ("pass", "ecobot_v7")]
    lines = [
        "# S3X：同局面一日投资分支 A50 审计",
        "",
        "## 结论边界",
        "",
        "本轮不是线上 Oracle，也不是多阶段最优搜索。每个候选只改变一个日初的投资安排，执行一天后恢复同一冻结基线；真实未来和冻结对手内部状态只用于离线续跑标签。",
        "",
        "## 核心结果",
        "",
        "| 对手 | 基线胜率 | 有限单次调整事后上限 | 基线败局被救回 | 平均分差改善 | 解析分数相关性 | 方向正确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        corr = item["score_margin_pearson"]
        corr_text = f"{corr:.3f}" if corr is not None else "n/a"
        direction_text = f"{item['score_direction_accuracy']:.1%}" if item["score_direction_accuracy"] is not None else "n/a"
        lines.append(
            f"| {item['opponent']} | {item['baseline']['mean']:.1%} | "
            f"{item['finite_one_change_offline_ceiling']['mean']:.1%} | "
            f"{item['recovered_baseline_losses']}/{item['baseline_losses']} | "
            f"{item['mean_margin_uplift']:+,.0f} | "
            f"{corr_text} | {direction_text} |"
        )
    lines.extend([
        "",
        "## 严格解释",
        "",
        "- 如果有限单次调整能大量救回基线败局，说明候选语言至少包含有价值的局部改法。",
        "- 如果解析分数与真实分差改善相关性低，说明当前选择依据不能识别这些改法；不能把事后上限冒充线上能力。",
        "- 若某些强对手连事后单次调整也很少救回，下一步应扩展候选或跨阶段组合，而不是继续调一个标量评分。",
        "- 所有区间按 50 个 seed 聚类，同 seed 的双座位不当作两个独立样本。它们不是天梯总体保证。",
        "",
        "## 工程验收",
        "",
        f"- 九组共 {sum(item['games'] for item in summaries)} 局，均使用 8 个日初节点。",
        "- KEEP 与既有 S3V-old 的现金和对手现金逐局完全一致。",
        "- 小样本 Pilot v2 已验证候选正序/倒序结果一致，无跨分支状态污染。",
        "- 本轮未使用最终保留集，也没有改变官方 1.32.7 规则。",
        "",
        "## 下一步门控",
        "",
        "先根据本报告区分候选覆盖与评分能力。只有候选覆盖足够而评分明显不足时，才开发仅使用公开信息和自有状态的后果估计器；否则先扩候选和跨阶段组合。",
        "",
        "## 状态",
        "",
        "S3X 为诊断轮，v7 全目标尚未完成。",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps({"analysis": str(output / "analysis.json"), "report": str(report_path), "strong_opponents": len(strong)}))


if __name__ == "__main__":
    main()

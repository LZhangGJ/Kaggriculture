"""Audit what can actually be selected at Day 0 without seed/opponent identity."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import statistics as st


EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "receipts/s3x_branch_A50_v1"
OUTPUT = EXP / "receipts/s3x_branch_A50_v1_day0_identifiability"
REPORT = EXP / "reports/S3X_DAY0_IDENTIFIABILITY_REVIEW_ZH.md"
ACTIVE = (
    "g001",
    "g003",
    "boatlee_v29",
    "kaito_v58",
    "lynn_v5",
    "yhay81_six_day",
    "yhay81_three_day",
    "ecobot_v7",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(choice: dict) -> str:
    # Full semantic edit signature; no result, score, seed, seat or opponent data.
    value = {
        "family": choice["family"],
        "kind": int(choice["kind"]),
        "amount": int(choice["amount"]),
        "edits": choice["edits"],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def summarize(rows: list[dict]) -> dict:
    by_opponent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_opponent[row["opponent"]].append(row)
    per_opponent = {}
    for opponent, values in sorted(by_opponent.items()):
        per_opponent[opponent] = {
            "games": len(values),
            "wins": sum(item["win"] for item in values),
            "win_rate": st.fmean(item["win"] for item in values),
            "mean_cash": st.fmean(item["cash"] for item in values),
            "mean_opponent_cash": st.fmean(item["opponent_cash"] for item in values),
            "mean_margin": st.fmean(item["margin"] for item in values),
        }
    active_values = [row for row in rows if row["opponent"] in ACTIVE]
    active_rates = [per_opponent[name]["win_rate"] for name in ACTIVE]
    return {
        "active_games": len(active_values),
        "active_wins": sum(item["win"] for item in active_values),
        "active_win_rate": st.fmean(item["win"] for item in active_values),
        "worst_active_opponent_win_rate": min(active_rates),
        "mean_active_margin": st.fmean(item["margin"] for item in active_values),
        "pass_mean_cash": per_opponent["pass"]["mean_cash"],
        "per_opponent": per_opponent,
    }


def main() -> None:
    OUTPUT.mkdir(exist_ok=False)
    meta = json.loads((SOURCE / "acceptance.json").read_text(encoding="utf8"))
    assert meta["status"] == "COMPLETE_FINITE_BRANCH_AUDIT_NOT_DEPLOYABLE_ORACLE"
    outcomes: dict[str, list[dict]] = defaultdict(list)
    semantic: dict[str, dict] = {}
    sets = []
    score_picks = set()
    input_hashes = {"acceptance.json": sha(SOURCE / "acceptance.json")}
    for path in sorted(SOURCE.glob("*.json.gz")):
        opponent = path.name.removesuffix(".json.gz")
        input_hashes[path.name] = sha(path)
        with gzip.open(path, "rt", encoding="utf8") as handle:
            games = json.load(handle)
        for game in games:
            day0 = next(node for node in game["nodes"] if int(node["day"]) == 0)
            current = set()
            for choice in day0["choices"]:
                sig = signature(choice)
                current.add(sig)
                semantic.setdefault(sig, json.loads(sig))
                outcomes[sig].append({
                    "opponent": opponent,
                    "seed": int(game["seed"]),
                    "seat": int(game["seat"]),
                    "cash": float(choice["cash"]),
                    "opponent_cash": float(choice["opponent_cash"]),
                    "margin": float(choice["cash"] - choice["opponent_cash"]),
                    "win": choice["cash"] > choice["opponent_cash"],
                })
            sets.append(current)
            score_picks.add(signature(max(day0["choices"], key=lambda item: item["score"])))
    common = set.intersection(*sets)
    union = set.union(*sets)
    assert common == union, "Day 0 public state should expose the same semantic candidate set"
    assert len(score_picks) == 1, "Current public score must make one identical Day 0 choice"
    expected_rows = 9 * 50 * 2
    assert all(len(outcomes[sig]) == expected_rows for sig in common)

    candidates = []
    for sig in sorted(common):
        item = {"signature": semantic[sig], **summarize(outcomes[sig])}
        candidates.append(item)
    robust = max(
        candidates,
        key=lambda item: (
            item["worst_active_opponent_win_rate"],
            item["active_win_rate"],
            item["pass_mean_cash"],
            item["mean_active_margin"],
        ),
    )
    best_average = max(
        candidates,
        key=lambda item: (item["active_win_rate"], item["mean_active_margin"], item["pass_mean_cash"]),
    )
    best_pass = max(candidates, key=lambda item: item["pass_mean_cash"])
    current_score = next(item for item in candidates if json.dumps(item["signature"], sort_keys=True, separators=(",", ":")) in score_picks)
    baseline = next(item for item in candidates if item["signature"]["family"] == "KEEP")
    identity_specific = {}
    for opponent in ACTIVE:
        identity_specific[opponent] = max(
            candidates,
            key=lambda item: (
                item["per_opponent"][opponent]["win_rate"],
                item["per_opponent"][opponent]["mean_margin"],
                item["pass_mean_cash"],
            ),
        )
    result = {
        "status": "COMPLETE_DAY0_IDENTIFIABILITY_AUDIT_NOT_A_POLICY",
        "candidate_count": len(candidates),
        "same_public_candidate_set_all_900_games": True,
        "same_current_score_pick_all_900_games": True,
        "baseline_keep": baseline,
        "current_score_pick": current_score,
        "best_identity_blind_worst_case": robust,
        "best_identity_blind_average": best_average,
        "best_pass_cash": best_pass,
        "identity_specific_best_DIAGNOSTIC_ONLY": identity_specific,
        "candidates": sorted(
            candidates,
            key=lambda item: (
                item["worst_active_opponent_win_rate"],
                item["active_win_rate"],
                item["pass_mean_cash"],
            ),
            reverse=True,
        ),
        "input_hashes": input_hashes,
        "final_holdout_used": False,
        "full_goal_complete": False,
        "interpretation_guard": (
            "At Day 0 all public observations and candidate sets are identical. A deployable identity-blind "
            "policy must make the same decision, so per-game or per-opponent hindsight choice is impossible."
        ),
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf8")

    def fmt(name: str, item: dict) -> str:
        sig = item["signature"]
        return (
            f"| {name} | {sig['family']} / {sig['kind']} / {sig['amount']} | "
            f"{item['active_win_rate']:.1%} | {item['worst_active_opponent_win_rate']:.1%} | "
            f"{item['pass_mean_cash']:,.0f} | {item['mean_active_margin']:+,.0f} |"
        )

    lines = [
        "# S3X：Day 0 候选可识别性审计",
        "",
        "## 为什么必须做",
        "",
        "Day 0 时不同 seed 和对手身份尚未从公开盘面出现。虽然事后可以为每局挑一个获胜候选，线上 Agent 实际只能对相同公开状态作出同一个选择。",
        "",
        f"900 局的 Day 0 候选集合完全一致，共 {len(candidates)} 个；现有解析评分在 900 局中也始终选择同一个候选。",
        "",
        "## 身份不可见时的固定 Day 0 结果",
        "",
        "| 选择法 | 候选 | 八对手平均胜率 | 最弱对手胜率 | PASS平均现金 | 八对手平均分差 |",
        "|---|---|---:|---:|---:|---:|",
        fmt("当前 KEEP", baseline),
        fmt("当前解析评分", current_score),
        fmt("最大化最弱对手", robust),
        fmt("最大化八对手平均", best_average),
        fmt("最大化 PASS 现金", best_pass),
        "",
        "## 结论",
        "",
        "- 每局事后 100% 左右的翻盘覆盖，只证明候选里存在好改法，不能证明线上能够识别。",
        "- Day 0 必须采用身份不可见的稳健开局；只有看到公开盘面变化后，动态选择器才有条件分流。",
        "- 若最稳健固定候选仍明显低于目标，下一步应把模拟后果估计用于 Day 1 之后的可观察分流，而不是用真实 seed 为 Day 0 贴标签。",
        "",
        "本审计未使用最终保留集，不是可提交策略，v7 总目标尚未完成。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps({
        "candidate_count": len(candidates),
        "baseline_active_win_rate": baseline["active_win_rate"],
        "robust_active_win_rate": robust["active_win_rate"],
        "robust_worst_opponent_win_rate": robust["worst_active_opponent_win_rate"],
        "best_average_win_rate": best_average["active_win_rate"],
        "best_pass_cash": best_pass["pass_mean_cash"],
        "report": str(REPORT),
    }))


if __name__ == "__main__":
    main()

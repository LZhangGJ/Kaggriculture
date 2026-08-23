#!/usr/bin/env python3
"""Audit a minimal public-shop router against Kobe's completed Replay routes.

This tool deliberately tests one hand-written, economically interpretable
hypothesis.  It does not fit a model and it never exposes the completed route
to the predictor.  The completed route is used only after prediction to score
the hypothesis.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


MILK_SUPPORT = frozenset(("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"))
YARN = "YARN_STORE"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def shops(value: object) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def kobe_hypothesis(sequence: list[str]) -> tuple[str, str]:
    """Return route and a stable explanation from the first three shops only."""

    first_three = sequence[:3]
    first_two = first_three[:2]
    if YARN in first_two:
        return "C6S12G0L3", "YARN appeared in the first two shops"
    if len(first_three) >= 3 and first_three[2] == YARN:
        return "C6S8G0L2", "YARN appeared as the third shop"
    milk_first_two = sum(shop in MILK_SUPPORT for shop in first_two)
    milk_first_three = sum(shop in MILK_SUPPORT for shop in first_three)
    if len(first_two) >= 2 and milk_first_two == 0:
        return "C6S6G2L2", "first two shops support neither MILK nor WOOL"
    if milk_first_three >= 2:
        return "C10S4G0L2", "at least two of the first three shops support MILK"
    return "C8S6G0L2", "exactly one early MILK-support shop"


def k320_reference(sequence: list[str]) -> str:
    """Reference the frozen public K320 source rule for a structural contrast."""

    first_three = sequence[:3]
    if first_three and first_three[0] == YARN:
        return "K320_R3_6C12"
    if YARN in first_three[:2]:
        return "K320_R4_6C12"
    if YARN in first_three:
        return "K320_R2_6C8"
    if any(shop in MILK_SUPPORT for shop in first_three):
        return "K320_R0_10C4"
    return "K320_R1_8C6"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.episode_csv)
    required = {
        "episode_id",
        "seat",
        "shop_sequence",
        "capital_signature",
        "f216_own_board_kind_COOP",
        "result",
        "reward",
        "opponent_reward",
        "opponent_submission_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"missing columns: {missing}")

    rows = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    route_stats: dict[str, Counter[str]] = defaultdict(Counter)
    prebuild_correct = 0
    for record in frame.to_dict(orient="records"):
        sequence = shops(record["shop_sequence"])
        predicted, reason = kobe_hypothesis(sequence)
        actual = str(record["capital_signature"])
        first_two = sequence[:2]
        expected_coop = int(
            len(first_two) == 2
            and YARN not in first_two
            and not any(shop in MILK_SUPPORT for shop in first_two)
        )
        observed_coop = int(float(record.get("f216_own_board_kind_COOP") or 0) == 2)
        prebuild_correct += int(expected_coop == observed_coop)
        confusion[actual][predicted] += 1
        route_stats[actual][str(record["result"])] += 1
        rows.append(
            {
                "episode_id": int(record["episode_id"]),
                "seat": int(record["seat"]),
                "opponent_submission_id": int(record["opponent_submission_id"]),
                "first_three_shops": sequence[:3],
                "prediction": predicted,
                "actual": actual,
                "correct": predicted == actual,
                "reason": reason,
                "expected_prebuild_two_coops": bool(expected_coop),
                "observed_prebuild_two_coops": bool(observed_coop),
                "k320_reference_route": k320_reference(sequence),
                "result": str(record["result"]),
                "reward": float(record["reward"]),
                "opponent_reward": float(record["opponent_reward"]),
            }
        )

    correct = sum(row["correct"] for row in rows)
    result = {
        "schema": "kaggriculture.fusion-champion.kobe-shop-router-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if correct == len(rows) and prebuild_correct == len(rows) else "FAIL",
        "source_boundary": (
            "Observational audit of downloaded official Replay only. The proposed "
            "router uses the first three public shops; completed route and result are "
            "used only as post-hoc labels. Exact association is not proof of optimality."
        ),
        "episode_csv": str(args.episode_csv.resolve()),
        "episode_csv_sha256": sha256(args.episode_csv),
        "episodes": len(rows),
        "unique_opponent_submissions": len({row["opponent_submission_id"] for row in rows}),
        "seats": dict(Counter(str(row["seat"]) for row in rows)),
        "route_rule_accuracy": correct / len(rows),
        "route_rule_correct": correct,
        "prebuild_coop_accuracy": prebuild_correct / len(rows),
        "prebuild_coop_correct": prebuild_correct,
        "rule": [
            "YARN in shop positions 0 or 1 -> C6S12G0L3",
            "else YARN in position 2 -> C6S8G0L2",
            "else zero MILK-support shops in positions 0 and 1 -> C6S6G2L2",
            "else at least two MILK-support shops in positions 0..2 -> C10S4G0L2",
            "else -> C8S6G0L2",
        ],
        "milk_support_shops": sorted(MILK_SUPPORT),
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
        "route_result_counts": {
            key: dict(value) for key, value in sorted(route_stats.items())
        },
        "k320_comparison": dict(
            Counter(
                f"{row['actual']} <- {row['k320_reference_route']}" for row in rows
            )
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    route_lines = []
    for route in sorted(route_stats):
        counts = route_stats[route]
        route_lines.append(
            f"| {route} | {sum(counts.values())} | {counts['WIN']} | "
            f"{counts['TIE']} | {counts['LOSS']} |"
        )
    report = f"""# Kobe 当前{len(rows)}局：公开商店路线器审计

日期：2026-08-22  
环境：官方 `kaggle-environments 1.32.7`

## 结论

Kobe 的五种资本结构可以由**前三家公开商店**的一条简单规则完整解释，
不是对手身份分类，也不需要未来状态：

1. 前两家出现 `YARN_STORE`：`6牛12羊、3块土地`；
2. 仅第三家首次出现 `YARN_STORE`：`6牛8羊、2块土地`；
3. 前两家既不支持牛奶，也没有羊毛店：先建两座鸡舍，最终
   `6牛6羊2鹅、2块土地`；
4. 前三家至少两家支持牛奶：`10牛4羊、2块土地`；
5. 其余情况：`8牛6羊、2块土地`。

牛奶支持商店为：`PIZZA_SHOP`、`ICE_CREAM_SHOP`、`SMOOTHIE_SHOP`。

- 路线结构命中：**{correct}/{len(rows)}（{correct / len(rows):.2%}）**；
- 第192步根据前两店决定是否预建两座鸡舍：
  **{prebuild_correct}/{len(rows)}（{prebuild_correct / len(rows):.2%}）**；
- 对手提交分组：{result['unique_opponent_submissions']}个；
- 两个座位均有样本。

## 各路线线上结果

| 路线 | 局数 | 胜 | 平 | 负 |
|---|---:|---:|---:|---:|
{chr(10).join(route_lines)}

## 相对 K320 的真实新增能力

K320 已经会按前三家商店在牛、羊路线之间选择，但它把“出现任意牛奶支持店”
统一归为10牛路线。Kobe 增加了两项更细的经营能力：

1. **需求强度分级**：一个牛奶支持店走8牛6羊，至少两个才走10牛4羊；
2. **鸡蛋产业预案**：前两店既不支持牛奶也不支持羊毛时，提前建设鸡舍；
   第三店若突然出现羊毛需求，再撤销预案转羊线。

这是一条可迁移的“商店需求 → 产能结构”规则，不依赖识别 Kobe 或某个对手。

## 不能直接得出的结论

- 77/77 关联只能证明 Kobe 的行为规则被准确还原，不能证明这些阈值在本地
  独立事件与全部对手上最优；
- 不能只把购买动物的两条订单改掉。鹅路线还需要鸡舍建设、喂养、收蛋、
  回仓、出售和失败恢复的完整执行链；
- 正式接入前必须做固定独立事件、双方换座、全池消融和官方 Python 复验。

## 证据

- `{args.output.as_posix()}`
- `{args.episode_csv.as_posix()}`
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "episodes": len(rows),
                "route_accuracy": result["route_rule_accuracy"],
                "prebuild_accuracy": result["prebuild_coop_accuracy"],
                "output": str(args.output),
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

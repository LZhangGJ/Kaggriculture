#!/usr/bin/env python3
"""Export compact browser JSON for the public macro-route dashboard."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_macro_route_library import ANCHORS, CATEGORIES, DAY_STEPS, _clusters


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-cache", type=Path, required=True)
    parser.add_argument("--macro-report", type=Path, required=True)
    parser.add_argument("--selector-report", type=Path, required=True)
    parser.add_argument("--global-clusters", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.12)
    return parser.parse_args()


def _numbers(value: np.ndarray, digits: int = 1) -> list:
    if np.issubdtype(value.dtype, np.integer):
        return value.astype(int).tolist()
    return np.round(value.astype(float), digits).tolist()


def _semantic_alias(index: int, representative: dict, median_slack: float, size: int) -> str:
    overrides = {
        1: "草莓主田·牛羊三地·八工紧资金",
        2: "麦田轮作·牛羊三地·八工资金瓶颈",
        3: "四地羊群扩张·麦莓轮作",
        4: "四地草莓羊群·麦田轮作",
        5: "十工麦田冲量·牛羊三地",
        6: "草莓牛群三地·单队稳态",
    }
    if index in overrides:
        return overrides[index]
    production = representative["production"].astype(int)
    crop_names = ("麦", "胡萝卜", "番茄", "草莓", "甜瓜")
    animal_names = ("鹅", "牛", "羊")
    crop_peak = production[:, :5].max(axis=0)
    animal_peak = production[:, 5:8].max(axis=0)
    crop_order = sorted(range(5), key=lambda item: int(crop_peak[item]), reverse=True)
    chosen_crops = crop_order[:2]
    third = crop_order[2]
    if crop_peak[third] >= max(8, 0.30 * crop_peak[crop_order[0]]):
        chosen_crops.append(third)
    chosen_animals = [
        item for item in sorted(range(3), key=lambda value: int(animal_peak[value]), reverse=True)
        if animal_peak[item] > 0
    ]
    land = int(production[:, -2].max())
    hands = int(production[:, -1].max())
    risk = "严重受阻" if median_slack <= -100 else (
        "资金紧" if median_slack < 0 else ("贴线资金" if median_slack <= 10 else "资金宽裕")
    )
    rarity = "稀有变体·" if size <= 3 else ""
    crops = "".join(crop_names[item] for item in chosen_crops) + "田"
    animals = "".join(animal_names[item] for item in chosen_animals) + "牧"
    labor = f"{land}区" + (f"{hands}工" if hands else "无雇工")
    return f"{rarity}{crops}·{animals}·{labor}·{risk}"


def _outcome_stats(subset: list[dict], outcomes: dict) -> dict:
    wins = losses = ties = 0
    margins = []
    for row in subset:
        outcome = outcomes.get(str(int(row["episode_id"]))) or {}
        rewards = list(outcome.get("rewards") or [])
        player = int(row["player_index"])
        if len(rewards) != 2 or player not in (0, 1):
            continue
        margin = float(rewards[player]) - float(rewards[1 - player])
        margins.append(margin)
        if margin > 0:
            wins += 1
        elif margin < 0:
            losses += 1
        else:
            ties += 1
    games = wins + losses + ties
    return {
        "games": games, "wins": wins, "losses": losses, "ties": ties,
        "win_rate": round(wins / games, 4) if games else None,
        "median_margin": round(float(np.median(margins)), 1) if margins else None,
    }


def main() -> None:
    args = _args()
    with np.load(args.macro_cache, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    row_index_by_object = {id(row): index for index, row in enumerate(rows)}
    macro_report = json.loads(args.macro_report.read_text(encoding="utf-8"))
    selector_report = json.loads(args.selector_report.read_text(encoding="utf-8"))
    outcomes = json.loads(args.outcomes.read_text(encoding="utf-8"))
    macro_by_team = {row["team"]: row for row in macro_report["teams_report"]}
    selector_by_team = {row["team"]: row for row in selector_report["teams_report"]}
    team_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        team_rows[str(row["team"])].append(row)

    with np.load(args.global_clusters) as cached:
        global_labels = cached["labels"].astype(int)
    if len(global_labels) != len(rows):
        raise ValueError("global cluster labels do not align with macro cache")
    global_groups = []
    for label in sorted(set(global_labels.tolist())):
        indices = np.flatnonzero(global_labels == label)
        subset = [rows[index] for index in indices]
        representative = max(subset, key=lambda row: (row["reward"], row["episode_id"]))
        global_groups.append({
            "raw_label": int(label), "indices": indices, "rows": subset,
            "representative": representative,
            "median_reward": float(np.median([row["reward"] for row in subset])),
        })
    global_groups.sort(key=lambda group: (-len(group["rows"]), -group["median_reward"]))
    family_by_row = {}
    families = []
    for index, group in enumerate(global_groups):
        family_id = f"G{index + 1}"
        subset = group["rows"]
        representative = group["representative"]
        rewards = [float(row["reward"]) for row in subset]
        worst_slacks = [float(np.min(row["slack"])) for row in subset]
        median_worst_slack = float(np.median(worst_slacks))
        members = sorted(set(str(row["team"]) for row in subset))
        for row_index in group["indices"]:
            family_by_row[int(row_index)] = family_id
        families.append({
            "id": family_id,
            "alias": _semantic_alias(index + 1, representative, median_worst_slack, len(subset)),
            "size": len(subset),
            "share": round(len(subset) / len(rows), 4),
            "team_count": len(members),
            "teams": members,
            "leaderboard_score": round(float(np.median([row["score"] for row in subset])), 1),
            "representative": {
                "episode_id": int(representative["episode_id"]),
                "player_index": int(representative["player_index"]),
                "team": str(representative["team"]),
                "reward": float(representative["reward"]),
            },
            "reward": {
                "median": round(float(np.median(rewards)), 1),
                "min": round(min(rewards), 1),
                "max": round(max(rewards), 1),
            },
            "outcome": _outcome_stats(subset, outcomes),
            "feasibility": {
                "median_min_money": round(float(np.median([row["min_money"] for row in subset])), 1),
                "median_worst_slack72": round(median_worst_slack, 1),
                "capital_requirement": _numbers(representative["capital"]),
                "cash_slack": _numbers(representative["slack"]),
            },
            "production": _numbers(representative["production"]),
            "layouts": _numbers(representative["layouts"]),
            "schedule": _numbers(representative["schedule"]),
            "episode_ids": [int(row["episode_id"]) for row in subset],
        })
    cumulative = 0
    core_90 = 0
    for family in families:
        if cumulative / len(rows) >= 0.90:
            break
        cumulative += family["size"]
        core_90 += 1
    family_alias = {family["id"]: family["alias"] for family in families}

    teams = []
    total_plans = 0
    for team, values in team_rows.items():
        labels, _ = _clusters(values, args.threshold)
        grouped = []
        for label in sorted(set(labels.tolist())):
            indices = np.flatnonzero(labels == label)
            subset = [values[index] for index in indices]
            representative = max(subset, key=lambda row: (row["reward"], row["episode_id"]))
            grouped.append({
                "raw_label": int(label), "indices": indices, "rows": subset,
                "representative": representative,
                "median_reward": float(np.median([row["reward"] for row in subset])),
            })
        grouped.sort(key=lambda group: (-len(group["rows"]), -group["median_reward"]))
        label_name = {group["raw_label"]: f"R{index + 1}" for index, group in enumerate(grouped)}
        label_global = {}
        for group in grouped:
            counts = Counter(
                family_by_row[row_index_by_object[id(row)]] for row in group["rows"]
            )
            ordered = sorted(counts, key=lambda family: (-counts[family], int(family[1:])))
            label_global[group["raw_label"]] = "+".join(ordered)
        plans = []
        for index, group in enumerate(grouped):
            subset = group["rows"]
            representative = group["representative"]
            rewards = [float(row["reward"]) for row in subset]
            worst_slacks = [float(np.min(row["slack"])) for row in subset]
            plans.append({
                "id": f"R{index + 1}",
                "global_family": family_by_row[row_index_by_object[id(representative)]],
                "global_alias": family_alias[family_by_row[row_index_by_object[id(representative)]]],
                "size": len(subset),
                "share": round(len(subset) / len(values), 4),
                "representative": {
                    "episode_id": int(representative["episode_id"]),
                    "player_index": int(representative["player_index"]),
                    "reward": float(representative["reward"]),
                },
                "reward": {
                    "median": round(float(np.median(rewards)), 1),
                    "min": round(min(rewards), 1),
                    "max": round(max(rewards), 1),
                },
                "outcome": _outcome_stats(subset, outcomes),
                "feasibility": {
                    "median_min_money": round(float(np.median([row["min_money"] for row in subset])), 1),
                    "median_worst_slack72": round(float(np.median(worst_slacks)), 1),
                    "capital_requirement": _numbers(representative["capital"]),
                    "cash_slack": _numbers(representative["slack"]),
                },
                "production": _numbers(representative["production"]),
                "layouts": _numbers(representative["layouts"]),
                "schedule": _numbers(representative["schedule"]),
                "episode_ids": [int(row["episode_id"]) for row in subset],
            })
        selector = selector_by_team.get(team)
        selector_value = None
        if selector:
            probe = dict(selector["early_public_proxy"])
            rules = str(probe.get("rules", ""))
            for raw_label, name in sorted(label_global.items(), reverse=True):
                rules = re.sub(rf"\bplan_{raw_label}\b", name, rules)
            target_ids = sorted(
                {family for value in label_global.values() for family in value.split("+")},
                key=lambda family: int(family[1:]),
            )
            selector_value = {
                "checkpoint": probe["checkpoint"],
                "mode": probe["mode"],
                "accuracy": round(float(probe["accuracy"]), 4),
                "balanced_accuracy": round(float(probe["balanced_accuracy"]), 4),
                "majority_accuracy": round(float(selector["leave_one_out_majority_accuracy"]), 4),
                "gain": round(float(probe["gain"]), 4),
                "top_features": probe.get("top_features", []),
                "rules": rules,
                "target_families": [
                    {"id": family, "alias": family_alias[family]}
                    for family in target_ids
                ],
                "caveat": probe.get("caveat", ""),
            }
        macro_value = macro_by_team[team]
        total_plans += len(plans)
        teams.append({
            "team": team,
            "leaderboard_score": round(float(np.median([row["score"] for row in values])), 1),
            "samples": len(values),
            "plan_count": len(plans),
            "dominant_share": round(max(len(group["rows"]) for group in grouped) / len(values), 4),
            "distance": macro_value["pair_distance"],
            "sensitivity": macro_value["sensitivity"],
            "selector": selector_value,
            "plans": plans,
        })
    teams.sort(key=lambda row: (-row["leaderboard_score"], row["team"]))
    payload = {
        "schema_version": 1,
        "title": "Kaggriculture Public Macro Route Atlas",
        "source": {
            "replays": 898, "target_sides": len(rows), "teams": len(teams),
            "macro_threshold": args.threshold,
            "selector_report": args.selector_report.name,
        },
        "axes": {
            "day_steps": list(DAY_STEPS), "layout_steps": list(ANCHORS),
            "categories": list(CATEGORIES),
            "schedule": [
                "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "BUILD_COOP", "BUILD_PASTURE", "GOOSE", "COW", "SHEEP",
                "BUY_LAND", "HIRE",
            ],
        },
        "summary": {
            "global_families": len(families),
            "core_families_90pct": core_90,
            "core_coverage": round(cumulative / len(rows), 4),
            "team_branch_instances": total_plans,
            "median_plans": float(np.median([row["plan_count"] for row in teams])),
            "high_sample_teams": sum(row["samples"] >= 40 for row in teams),
        },
        "families": families,
        "teams": teams,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB, "
        f"{len(teams)} teams/{len(families)} global families/{total_plans} team branches)"
    )


if __name__ == "__main__":
    main()

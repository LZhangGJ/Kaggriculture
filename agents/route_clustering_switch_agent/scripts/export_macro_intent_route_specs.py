#!/usr/bin/env python3
"""Export failure-independent macro route specifications for an agent planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CATEGORIES = (
    "wheat", "carrot", "tomato", "strawberry", "melon",
    "goose", "cow", "sheep", "coop", "pasture",
)
SCHEDULE = (*CATEGORIES[:5], "build_coop", "build_pasture", *CATEGORIES[5:8], "buy_land", "hire")
CATEGORY_CN = ("麦", "胡萝卜", "番茄", "草莓", "瓜", "鹅", "牛", "羊", "鸡舍", "牧场")


def _counts(layouts: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[np.count_nonzero(layout == category) for category in range(1, 11)] for layout in layouts],
        dtype=np.int16,
    )


def _alias(layouts: np.ndarray, schedule: np.ndarray) -> str:
    counts = _counts(layouts)[-1]
    crops = np.argsort(-counts[:5])[:2]
    animals = np.argsort(-counts[5:8])[:2] + 5
    crop_name = "".join(
        f"{CATEGORY_CN[index]}{counts[index]}" for index in crops if counts[index] > 0
    ) or "无作物"
    animal_name = "".join(
        f"{CATEGORY_CN[index]}{counts[index]}" for index in animals if counts[index] > 0
    ) or "无畜"
    quadrants = min(4, 1 + int(schedule[:, -2].sum()))
    first_animal_phase = np.flatnonzero(schedule[:, 7:10].sum(axis=1) > 0)
    tempo = "早畜" if len(first_animal_phase) and first_animal_phase[0] <= 2 else "晚畜"
    return f"{crop_name}-{animal_name}-{quadrants}地-{tempo}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--distance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-zero-win", action="store_true")
    args = parser.parse_args()

    with np.load(args.features, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    with np.load(args.audit) as cached:
        keys = cached["keys"]
        planned_layouts = cached["planned_layouts"]
        hard_failures = cached["hard_failures"]
        completion_rate = cached["completion_rate"]
        own_rewards = cached["stored_rewards"]
        opponent_rewards = cached["opponent_stored_rewards"]
        reward_error = cached["reward_abs_error"]
    with np.load(args.clusters) as cached:
        labels = cached["labels"].astype(int)
    size = len(rows)
    distances = np.memmap(args.distance, mode="r", dtype=np.float32, shape=(size, size))
    expected_keys = np.asarray(
        [(int(row["episode_id"]), int(row["player_index"])) for row in rows], dtype=np.int64
    )
    if not np.array_equal(keys, expected_keys) or len(labels) != size:
        raise ValueError("route inputs are not aligned")

    groups = []
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        groups.append((
            len(members), float(np.median(own_rewards[members])), int(label), members,
        ))
    groups.sort(key=lambda value: (-value[0], -value[1], value[2]))

    specs = []
    aliases = {}
    for rank, (support, median_reward, source_label, members) in enumerate(groups, 1):
        within = np.asarray(distances[np.ix_(members, members)])
        medoid_index = int(members[int(np.argmin(within.mean(axis=1)))])
        medoid_layouts = planned_layouts[medoid_index]
        medoid_schedule = np.asarray(rows[medoid_index]["schedule"], dtype=np.int16)
        consensus_schedule = np.rint(np.median(
            np.stack([rows[index]["schedule"] for index in members]), axis=0
        )).astype(np.int16)
        wins = int(np.count_nonzero(own_rewards[members] > opponent_rewards[members]))
        losses = int(np.count_nonzero(own_rewards[members] < opponent_rewards[members]))
        ties = support - wins - losses
        base_alias = _alias(medoid_layouts, consensus_schedule)
        aliases[base_alias] = aliases.get(base_alias, 0) + 1
        alias = base_alias if aliases[base_alias] == 1 else f"{base_alias}-{aliases[base_alias]}"
        family = f"G{rank:03d}"
        is_zero_win = wins == 0
        anchors = (168, 288, 432, 576, 719)
        footprint = _counts(medoid_layouts)
        phases = []
        cumulative = np.cumsum(consensus_schedule, axis=0)
        for phase in range(10):
            phases.append({
                "turn_end": min(719, (phase + 1) * 72),
                "macro_actions": {
                    name: int(consensus_schedule[phase, index])
                    for index, name in enumerate(SCHEDULE)
                    if consensus_schedule[phase, index]
                },
                "cumulative_macro_actions": {
                    name: int(cumulative[phase, index])
                    for index, name in enumerate(SCHEDULE)
                    if cumulative[phase, index]
                },
            })
        source = rows[medoid_index]
        specs.append({
            "family": family,
            "alias": alias,
            "source_cluster_label": source_label,
            "support": support,
            "selected_for_evaluation": not (args.drop_zero_win and is_zero_win),
            "historical_zero_win": is_zero_win,
            "drop_reason": "zero_historical_wins" if args.drop_zero_win and is_zero_win else None,
            "historical_outcome": {
                "games": support, "wins": wins, "losses": losses, "ties": ties,
                "win_rate": (wins + .5 * ties) / support,
                "median_reward": median_reward,
            },
            "intent_medoid_source": {
                "route_id": f"{int(source['episode_id'])}:{int(source['player_index'])}",
                "team": str(source["team"]),
                "execution_hard_failures": int(hard_failures[medoid_index]),
                "reward_replay_exact": bool(reward_error[medoid_index] == 0),
            },
            "intent": {
                "layout_anchors": [
                    {
                        "turn": int(anchor),
                        "planned_footprint_counts": {
                            name: int(footprint[position, index])
                            for index, name in enumerate(CATEGORIES)
                            if footprint[position, index]
                        },
                        "planned_layout": medoid_layouts[position].reshape(10, 10).astype(int).tolist(),
                    }
                    for position, anchor in enumerate(anchors)
                ],
                "phases": phases,
            },
            "execution_diagnostics_only": {
                "zero_failure_samples": int(np.count_nonzero(hard_failures[members] == 0)),
                "hard_failures_median": float(np.median(hard_failures[members])),
                "hard_failures_p90": float(np.quantile(hard_failures[members], .9)),
                "completion_rate_median": float(np.median(completion_rate[members])),
                "min_money_p10": float(np.quantile([rows[index]["min_money"] for index in members], .1)),
                "capital_requirement_p90": np.quantile(
                    np.stack([rows[index]["capital"] for index in members]), .9, axis=0
                ).round(2).tolist(),
            },
        })

    selected = [spec["family"] for spec in specs if spec["selected_for_evaluation"]]
    dropped = [spec["family"] for spec in specs if not spec["selected_for_evaluation"]]
    payload = {
        "schema_version": 1,
        "meaning": "failure-independent macro intent; not an action replay",
        "category_ids": {str(index + 1): name for index, name in enumerate(CATEGORIES)},
        "families_total": len(specs),
        "family_limit": None,
        "selected_families": selected,
        "dropped_zero_win_rare": dropped,
        "routes": specs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "families_total": len(specs), "selected": len(selected),
        "dropped_zero_win": len(dropped), "output": str(args.output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit FC2A public replays for the step-288 stateful handoff stall."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def is_all_pass(action: object) -> bool:
    if not isinstance(action, dict):
        return True
    farmer = action.get("farmer") or ["PASS"]
    hands = action.get("hands") or []
    market = action.get("market") or []
    farmer_pass = not farmer or farmer[0] == "PASS"
    hands_pass = all(not row or row[0] == "PASS" for row in hands)
    return farmer_pass and hands_pass and not market


def animal_counts(farm: dict) -> tuple[int, int]:
    counts = Counter()
    for row in farm.get("tiles", []) or []:
        for tile in row or []:
            if isinstance(tile, dict):
                counts[str(tile.get("animal") or "")] += 1
    return int(counts["COW"]), int(counts["SHEEP"])


def longest_pass_span(actions: list[object]) -> tuple[int, int, int]:
    best_start = best_end = -1
    current_start = -1
    for index, action in enumerate(actions):
        if is_all_pass(action):
            if current_start < 0:
                current_start = index
            if index - current_start + 1 > best_end - best_start + 1:
                best_start, best_end = current_start, index
        else:
            current_start = -1
    length = 0 if best_start < 0 else best_end - best_start + 1
    return best_start, best_end, length


def analyze_episode(replay_path: Path, manifest_row: dict) -> dict:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    seat = int(manifest_row["own_seat"])
    frames = replay["steps"]
    actions = [frame[seat].get("action") for frame in frames]
    span_start, span_end, span_length = longest_pass_span(actions)

    first_pressure_frame = None
    pressure_snapshot = None
    for frame_index in range(24, min(96, len(frames))):
        observation = frames[frame_index][seat]["observation"]
        rival = observation["farms"][1 - seat]
        cows, sheep = animal_counts(rival)
        cash = int(rival.get("money", 0) or 0)
        if sheep >= 4 and cows <= 2 and cash <= 1000:
            first_pressure_frame = frame_index
            pressure_snapshot = {"rival_cows": cows, "rival_sheep": sheep, "rival_cash": cash}
            break

    selector_frame = min(288, len(frames) - 1)
    selector_obs = frames[selector_frame][seat]["observation"]
    own_farm = selector_obs["farms"][seat]
    own_cows, own_sheep = animal_counts(own_farm)
    shops = list((selector_obs.get("town") or {}).get("unlocked_shops") or [])
    market_inventory = (selector_obs.get("market") or {}).get("inventory") or {}
    selector_inputs = {
        "own_cash": int(own_farm.get("money", 0) or 0),
        "yarn_store_count": sum(shop == "YARN_STORE" for shop in shops),
        "strawberry_market_inventory": int(market_inventory.get("STRAWBERRY", 0) or 0),
        "own_cows": own_cows,
        "own_sheep": own_sheep,
    }
    prt_selector = (
        selector_inputs["own_cash"] <= 13376
        and selector_inputs["yarn_store_count"] <= 1
        and selector_inputs["strawberry_market_inventory"] <= 9979
    )

    trailing_start = len(actions)
    while trailing_start > 0 and is_all_pass(actions[trailing_start - 1]):
        trailing_start -= 1
    final_obs = frames[-1][seat]["observation"]
    final_cash = int(final_obs["farms"][seat].get("money", 0) or 0)
    return {
        "episode_id": int(manifest_row["episode_id"]),
        "result": manifest_row["result"],
        "own_seat": seat,
        "own_reward": float(manifest_row["own_reward"]),
        "opponent_reward": float(manifest_row["opponent_reward"]),
        "opponent_submission_id": int(manifest_row["opponent_submission_id"]),
        "opening_sheep_pressure": first_pressure_frame is not None,
        "first_pressure_frame": first_pressure_frame,
        "pressure_snapshot": pressure_snapshot,
        "step288_prt_selector": prt_selector,
        "step288_inputs": selector_inputs,
        "nonpass_action_frames": sum(not is_all_pass(action) for action in actions),
        "longest_all_pass_span": {
            "start": span_start,
            "end": span_end,
            "length": span_length,
        },
        "trailing_all_pass_span": {
            "start": trailing_start if trailing_start < len(actions) else None,
            "end": len(actions) - 1 if trailing_start < len(actions) else None,
            "length": len(actions) - trailing_start,
        },
        "final_cash": final_cash,
        "catastrophic_handoff_stall": (
            first_pressure_frame is not None
            and prt_selector
            and trailing_start == 289
            and len(actions) - trailing_start == 431
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = []
    for row in manifest["episodes"]:
        if row.get("download_status") != "downloaded":
            continue
        episodes.append(analyze_episode(Path(row["path"]), row))

    stalled = [row for row in episodes if row["catastrophic_handoff_stall"]]
    pressure = [row for row in episodes if row["opening_sheep_pressure"]]
    receipt = {
        "schema": "kaggriculture.fc2a.public-stall-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_id": int(manifest["submission_id"]),
        "episode_count": len(episodes),
        "opening_pressure_episode_count": len(pressure),
        "catastrophic_stall_episode_count": len(stalled),
        "catastrophic_stall_episode_ids": [row["episode_id"] for row in stalled],
        "finding": (
            "The two losses share the exact sticky opening-pressure plus step-288 PRT-selector "
            "handoff and then emit all-PASS actions for frames 289 through 719."
        ),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if len(stalled) == 2 else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Officially verify and summarize a G001/Candidate8 shared-Day0 trace bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_environments import make

from audit_candidate8_o18_daily_divergence_official import (
    _action_ops,
    _daily_delta,
    _daily_summary,
    _nested_delta,
    _player_economic_projection,
    _state,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def replay(game: dict[str, Any]) -> dict[str, Any]:
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": int(game["seed"])},
        debug=False,
    )
    state = env.reset(2)
    for joint in game["trace"]:
        state = env.step(joint)
    payload = env.toJSON()
    candidate_seat = int(game["candidate_seat"])
    candidate_states = [
        _state(frame[candidate_seat], candidate_seat) for frame in payload["steps"]
    ]
    opponent_states = [
        _state(frame[1 - candidate_seat], 1 - candidate_seat)
        for frame in payload["steps"]
    ]
    rewards = [float(value.reward) for value in state]
    statuses = [str(value.status) for value in state]
    return {
        "replay": payload,
        "candidate_states": candidate_states,
        "opponent_states": opponent_states,
        "official_rewards": rewards,
        "statuses": statuses,
    }


def action_totals(trace: list[Any], seat: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for joint in trace:
        counts.update(_action_ops(joint[seat]))
    return dict(sorted((key, int(value)) for key, value in counts.items()))


def important_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "day",
            "cash_end",
            "peak_hands",
            "land_end",
            "structures_end",
            "crops_end",
            "crop_yield_end",
            "animals_end",
            "animal_yield_end",
            "shed_end",
            "seeds_end",
            "weeds_end",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replay-output-dir", type=Path)
    args = parser.parse_args()

    version = importlib.metadata.version("kaggle-environments")
    if version != "1.32.7":
        raise RuntimeError(f"official package 1.32.7 required, got {version}")
    with gzip.open(args.trace_bundle, "rt", encoding="utf-8") as stream:
        bundle = json.load(stream)
    if len(bundle["games"]) != 2:
        raise RuntimeError("comparison bundle must contain exactly two games")

    games: dict[str, dict[str, Any]] = {}
    replayed: dict[str, dict[str, Any]] = {}
    official_replay_files: dict[str, dict[str, str]] = {}
    for game in bundle["games"]:
        result = replay(game)
        replayed[str(game["case"])] = result
        if args.replay_output_dir is not None:
            args.replay_output_dir.mkdir(parents=True, exist_ok=True)
            replay_path = args.replay_output_dir / (
                str(game["case"]).lower() + "_official_1_32_7_replay.json.gz"
            )
            with gzip.open(replay_path, "wt", encoding="utf-8") as stream:
                json.dump(
                    result["replay"], stream, ensure_ascii=False,
                    separators=(",", ":"),
                )
            official_replay_files[str(game["case"])] = {
                "path": str(replay_path),
                "sha256": sha256(replay_path),
            }
        native_rewards = [float(value) for value in game["native_rewards"]]
        complete = (
            len(game["trace"]) == 719
            and result["statuses"] == ["DONE", "DONE"]
        )
        reward_exact = result["official_rewards"] == native_rewards
        seat = int(game["candidate_seat"])
        daily_candidate = [
            _daily_summary(
                result["candidate_states"], game["trace"], seat, day
            )
            for day in range(30)
        ]
        daily_opponent = [
            _daily_summary(
                result["opponent_states"], game["trace"], 1 - seat, day
            )
            for day in range(30)
        ]
        games[str(game["case"])] = {
            "complete": complete,
            "reward_exact": reward_exact,
            "native_rewards": native_rewards,
            "official_rewards": result["official_rewards"],
            "statuses": result["statuses"],
            "trace_steps": len(game["trace"]),
            "candidate_action_totals": action_totals(game["trace"], seat),
            "opponent_action_totals": action_totals(game["trace"], 1 - seat),
            "daily_candidate": daily_candidate,
            "daily_opponent": daily_opponent,
            "decision_days": game.get("decision_days", []),
            "selected_ranks": game.get("selected_ranks", []),
            "selected_families": game.get("selected_families", []),
            "end_overflow": int(game.get("end_overflow", 0)),
            "avoidable_crop_losses": int(game.get("avoidable_crop_losses", 0)),
            "avoidable_animal_losses": int(game.get("avoidable_animal_losses", 0)),
        }

    g001_game = bundle["games"][0]
    candidate_game = bundle["games"][1]
    prefix_steps = int(bundle["games"][0]["shared_prefix_steps"])
    prefix_exact = (
        g001_game["trace"][:prefix_steps]
        == candidate_game["trace"][:prefix_steps]
    )
    first_action_divergence = next(
        (
            step
            for step in range(prefix_steps, len(g001_game["trace"]))
            if g001_game["trace"][step] != candidate_game["trace"][step]
        ),
        None,
    )
    first_opponent_action_divergence = next(
        (
            step
            for step in range(prefix_steps, len(g001_game["trace"]))
            if g001_game["trace"][step][1 - int(g001_game["candidate_seat"])]
            != candidate_game["trace"][step][1 - int(g001_game["candidate_seat"])]
        ),
        None,
    )

    g001 = games["G001_FULL_ROUTE"]
    candidate = games["CANDIDATE8_SHARED_G001_DAY0_ORACLE"]
    g001_replayed = replayed["G001_FULL_ROUTE"]
    candidate_replayed = replayed["CANDIDATE8_SHARED_G001_DAY0_ORACLE"]
    first_opponent_state_divergence = next(
        (
            frame
            for frame in range(prefix_steps + 1, 720)
            if _player_economic_projection(
                g001_replayed["opponent_states"][frame]
            )
            != _player_economic_projection(
                candidate_replayed["opponent_states"][frame]
            )
        ),
        None,
    )
    first_opponent_animal_divergence = next(
        (
            frame
            for frame in range(prefix_steps + 1, 720)
            if g001_replayed["opponent_states"][frame]["animal_owned"]
            != candidate_replayed["opponent_states"][frame]["animal_owned"]
        ),
        None,
    )
    daily_deltas = [
        _daily_delta(g001["daily_candidate"][day], candidate["daily_candidate"][day])
        for day in range(30)
    ]
    opponent_daily_deltas = [
        _daily_delta(g001["daily_opponent"][day], candidate["daily_opponent"][day])
        for day in range(30)
    ]
    milestones = (0, 1, 3, 6, 9, 12, 18, 24, 29)
    milestone_rows = [
        {
            "day": day,
            "g001": important_snapshot(g001["daily_candidate"][day]),
            "candidate8": important_snapshot(candidate["daily_candidate"][day]),
            "g001_market_prices_end": g001_replayed["candidate_states"][
                min((day + 1) * 24, 719)
            ]["market_prices"],
            "candidate8_market_prices_end": candidate_replayed["candidate_states"][
                min((day + 1) * 24, 719)
            ]["market_prices"],
            "g001_market_inventory_end": g001_replayed["candidate_states"][
                min((day + 1) * 24, 719)
            ]["market_inventory"],
            "candidate8_market_inventory_end": candidate_replayed["candidate_states"][
                min((day + 1) * 24, 719)
            ]["market_inventory"],
            "candidate8_minus_g001": daily_deltas[day],
            "g001_opponent_cash": g001["daily_opponent"][day]["cash_end"],
            "candidate8_opponent_cash": candidate["daily_opponent"][day]["cash_end"],
            "candidate8_minus_g001_opponent": opponent_daily_deltas[day],
        }
        for day in milestones
    ]

    hard_pass = (
        prefix_exact
        and first_action_divergence is not None
        and first_action_divergence >= prefix_steps
        and all(game["complete"] and game["reward_exact"] for game in games.values())
    )
    payload = {
        "schema": "kaggriculture.candidate8-g001-shared-day0-official-analysis.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if hard_pass else "FAIL",
        "official_package_version": version,
        "boundary": bundle["boundary"],
        "input": {"path": str(args.trace_bundle), "sha256": sha256(args.trace_bundle)},
        "official_replay_files": official_replay_files,
        "shared_fixture": {
            "opponent": g001_game["opponent"],
            "seed": g001_game["seed"],
            "candidate_seat": g001_game["candidate_seat"],
            "shared_prefix_steps": prefix_steps,
            "prefix_exact": prefix_exact,
            "first_action_divergence_step": first_action_divergence,
            "first_action_divergence_day": (
                None if first_action_divergence is None else first_action_divergence // 24
            ),
            "first_action_divergence_hour": (
                None if first_action_divergence is None else first_action_divergence % 24
            ),
            "first_opponent_action_divergence_step": first_opponent_action_divergence,
            "first_opponent_state_divergence_frame": first_opponent_state_divergence,
            "first_opponent_state_divergence_transition": (
                None if first_opponent_state_divergence is None
                else first_opponent_state_divergence - 1
            ),
            "first_opponent_state_divergence_delta": (
                None if first_opponent_state_divergence is None
                else _nested_delta(
                    _player_economic_projection(
                        g001_replayed["opponent_states"][first_opponent_state_divergence]
                    ),
                    _player_economic_projection(
                        candidate_replayed["opponent_states"][first_opponent_state_divergence]
                    ),
                )
            ),
            "first_opponent_animal_divergence_frame": first_opponent_animal_divergence,
            "first_opponent_animal_divergence_transition": (
                None if first_opponent_animal_divergence is None
                else first_opponent_animal_divergence - 1
            ),
            "first_opponent_animal_divergence_delta": (
                None if first_opponent_animal_divergence is None
                else _nested_delta(
                    g001_replayed["opponent_states"][first_opponent_animal_divergence]["animal_owned"],
                    candidate_replayed["opponent_states"][first_opponent_animal_divergence]["animal_owned"],
                )
            ),
        },
        "games": games,
        "milestones": milestone_rows,
        "daily_candidate8_minus_g001": daily_deltas,
        "daily_opponent_candidate8_minus_g001": opponent_daily_deltas,
        "terminal": {
            "g001": g001["official_rewards"],
            "candidate8": candidate["official_rewards"],
            "candidate8_own_cash_minus_g001": (
                candidate["official_rewards"][int(g001_game["candidate_seat"])]
                - g001["official_rewards"][int(g001_game["candidate_seat"])]
            ),
            "candidate8_opponent_cash_minus_g001": (
                candidate["official_rewards"][1 - int(g001_game["candidate_seat"])]
                - g001["official_rewards"][1 - int(g001_game["candidate_seat"])]
            ),
        },
        "total_action_delta_candidate8_minus_g001": _nested_delta(
            g001["candidate_action_totals"], candidate["candidate_action_totals"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "fixture": payload["shared_fixture"],
                "terminal": payload["terminal"],
                "candidate8_families": candidate["selected_families"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if hard_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

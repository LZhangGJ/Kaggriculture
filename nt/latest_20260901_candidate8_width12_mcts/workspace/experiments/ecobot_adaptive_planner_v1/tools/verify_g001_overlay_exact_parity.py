#!/usr/bin/env python3
"""Verify that the disabled adaptive overlay is action-identical to G001."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_G001 = (
    ROOT
    / "research"
    / "team_mate"
    / "Kaggriculture_main_512631c"
    / "agents"
    / "route_clustering_switch_agent"
    / "main.py"
)
DEFAULT_WRAPPER = (
    ROOT
    / "experiments"
    / "ecobot_adaptive_planner_v1"
    / "agents"
    / "g001_overlay_v0"
    / "main.py"
)
DEFAULT_PASSIVE = (
    ROOT / "experiments" / "gold_rule_agents" / "opponents" / "passive" / "main.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _candidate_trace(candidate: Path, opponent: Path, seed: int, seat: int) -> dict[str, Any]:
    agents = [str(candidate), str(opponent)]
    if seat == 1:
        agents.reverse()
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": seed},
        debug=False,
    )
    env.run(agents)
    final = env.steps[-1]
    return {
        "frames": len(env.steps),
        "statuses": [str(value.status) for value in final],
        "rewards": [float(value.reward) for value in final],
        "candidate_actions": [_safe(frame[seat].action) for frame in env.steps],
        "opponent_actions": [_safe(frame[1 - seat].action) for frame in env.steps],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g001", type=Path, default=DEFAULT_G001)
    parser.add_argument("--wrapper", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--passive", type=Path, default=DEFAULT_PASSIVE)
    parser.add_argument("--seed-start", type=int, default=1_941_001)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    version = importlib.metadata.version("kaggle-environments")
    if version != "1.32.7":
        raise RuntimeError(f"official package 1.32.7 required, got {version}")
    paths = [args.g001.resolve(), args.wrapper.resolve(), args.passive.resolve()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    opponents = [("passive", paths[2]), ("g001", paths[0])]
    rows: list[dict[str, Any]] = []
    for opponent_name, opponent_path in opponents:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for seat in (0, 1):
                direct = _candidate_trace(paths[0], opponent_path, seed, seat)
                wrapped = _candidate_trace(paths[1], opponent_path, seed, seat)
                candidate_actions_exact = (
                    direct["candidate_actions"] == wrapped["candidate_actions"]
                )
                opponent_actions_exact = (
                    direct["opponent_actions"] == wrapped["opponent_actions"]
                )
                rewards_exact = direct["rewards"] == wrapped["rewards"]
                complete = (
                    direct["frames"] == 720
                    and wrapped["frames"] == 720
                    and direct["statuses"] == ["DONE", "DONE"]
                    and wrapped["statuses"] == ["DONE", "DONE"]
                    and all(math.isfinite(value) for value in direct["rewards"])
                    and all(math.isfinite(value) for value in wrapped["rewards"])
                )
                first_candidate_mismatch = next(
                    (
                        index
                        for index, (left, right) in enumerate(
                            zip(direct["candidate_actions"], wrapped["candidate_actions"])
                        )
                        if left != right
                    ),
                    None,
                )
                rows.append({
                    "opponent": opponent_name,
                    "seed": seed,
                    "candidate_seat": seat,
                    "complete": complete,
                    "candidate_actions_exact": candidate_actions_exact,
                    "opponent_actions_exact": opponent_actions_exact,
                    "rewards_exact": rewards_exact,
                    "first_candidate_mismatch": first_candidate_mismatch,
                    "direct_rewards": direct["rewards"],
                    "wrapped_rewards": wrapped["rewards"],
                })
                print(
                    f"parity {len(rows)}/{len(opponents) * args.seeds * 2}: "
                    f"{opponent_name} seed={seed} seat={seat} "
                    f"candidate={candidate_actions_exact} reward={rewards_exact}",
                    flush=True,
                )

    passed = all(
        row["complete"]
        and row["candidate_actions_exact"]
        and row["opponent_actions_exact"]
        and row["rewards_exact"]
        for row in rows
    )
    payload = {
        "schema": "kaggriculture.ecobot.g001-overlay-parity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "official_package_version": version,
        "purpose": "disabled general adaptive overlay must be action-identical to frozen G001",
        "identity_routing_allowed": False,
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "dual_seat": True,
        "opponents": [name for name, _ in opponents],
        "games": len(rows),
        "summary": {
            "complete_games": sum(int(row["complete"]) for row in rows),
            "candidate_action_exact_games": sum(
                int(row["candidate_actions_exact"]) for row in rows
            ),
            "opponent_action_exact_games": sum(
                int(row["opponent_actions_exact"]) for row in rows
            ),
            "reward_exact_games": sum(int(row["rewards_exact"]) for row in rows),
        },
        "inputs": {
            "g001": {"path": str(paths[0]), "sha256": _sha256(paths[0])},
            "wrapper": {"path": str(paths[1]), "sha256": _sha256(paths[1])},
            "passive": {"path": str(paths[2]), "sha256": _sha256(paths[2])},
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


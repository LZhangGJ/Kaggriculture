"""Official 1.32.7 directional screen for recently-run public agents.

This is deliberately a small CPU gate before an exact JAX transcription.  It
uses both seats and fresh official seeds, records every terminal status, and
does not infer strength from notebook titles or votes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[3]
RECENT = ROOT / "public_notebooks" / "recent_latest_20260825_scan_v1"
FC24B = ROOT / "submission" / "_staging_fc24b_20260823_v1" / "main.py"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    path: Path


CANDIDATES = (
    Candidate(
        "kaito_v48",
        "kaito_fast_route",
        RECENT
        / "kaitofukami__40-40-early-floor-39-46-top-10-v48-fast-routes"
        / "output"
        / "main.py",
    ),
    Candidate(
        "prvsiyan_soil_v139",
        "prvsiyan_moon_soil",
        RECENT
        / "prvsiyan__kaggriculture-frontier-the-soil-remembers-rain"
        / "output"
        / "main.py",
    ),
    Candidate(
        "prvsiyan_moon_v135",
        "prvsiyan_moon_soil",
        RECENT
        / "prvsiyan__kaggriculture-frontier-the-moon-counts-melons"
        / "output"
        / "main.py",
    ),
    Candidate(
        "adaptive_shop_guard",
        "kaito_fast_route",
        RECENT
        / "llccqq624__kaggriculture-adaptive-shop-guard"
        / "output"
        / "main.py",
    ),
    Candidate(
        "shops_remember_route",
        "kaito_fast_route",
        RECENT
        / "llccqq624__kaggriculture-the-shops-remember-the-route"
        / "output"
        / "main.py",
    ),
    Candidate(
        "premium_queue_split",
        "single_tape_d854",
        RECENT
        / "llccqq624__kaggriculture-premium-queue-split"
        / "output"
        / "main.py",
    ),
    Candidate(
        "salem_3000",
        "single_tape_d854",
        RECENT
        / "salemali7__kaggriculture-3000-socre"
        / "output"
        / "main.py",
    ),
    Candidate(
        "mugundh_v21_divergence",
        "prvsiyan_moon_soil",
        RECENT
        / "mugundhjb__v21-multi-route-agent-sell-divergence-guarder"
        / "output"
        / "main.py",
    ),
    Candidate(
        "tetsutani_read_town",
        "prvsiyan_moon_soil",
        RECENT
        / "tetsutani__read-the-town-build-the-farm-kaggriculture"
        / "output"
        / "submission_extract"
        / "main.py",
    ),
    Candidate(
        "lynn_farming_score_v2",
        "prvsiyan_moon_soil",
        RECENT
        / "lynnsakurai__farming-score-v2-a-better-approach"
        / "output"
        / "main.py",
    ),
    Candidate(
        "ecobot_v4",
        "ecobot_rules",
        RECENT
        / "premaananda108__economics-driven-rule-agent-ecobot-v4"
        / "output"
        / "main.py",
    ),
    Candidate(
        "steven_x594",
        "steven_x",
        RECENT
        / "stevenleehans__kaggriculture-x578-i-m-the-strongest"
        / "output"
        / "main.py",
    ),
    Candidate(
        "steven_e631",
        "steven_e",
        RECENT
        / "stevenleehans__kaggriculture-rank-238-oh-you-re-approaching-me"
        / "output"
        / "main.py",
    ),
    Candidate(
        "amey_deterministic",
        "single_tape_d854",
        RECENT
        / "ameythakur20__kaggriculture-deterministic-farm-planning-agent"
        / "output"
        / "main.py",
    ),
    Candidate(
        "akash_autonomous",
        "small_rule_agent",
        RECENT
        / "akashbabu17__kaggriculture-autonomous-multi-agent-farming"
        / "output"
        / "submission.py",
    ),
    Candidate(
        "koushik_starter",
        "starter",
        RECENT
        / "koushikkumardinda__kaggriculture-starter"
        / "output"
        / "submission.py",
    ),
)


def farm_summary(frame: object, seat: int) -> dict[str, int]:
    farms = list(frame.observation.get("farms", []) or [])
    if seat >= len(farms):
        return {"money": 0, "units": 0, "cows": 0, "sheep": 0, "geese": 0}
    farm = farms[seat]
    animals = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
    for row in farm.get("tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if isinstance(tile, dict) and tile.get("animal") in animals:
                animals[str(tile["animal"])] += 1
    return {
        "money": int(farm.get("money", 0) or 0),
        "units": len(farm.get("units", []) or []),
        "cows": animals["COW"],
        "sheep": animals["SHEEP"],
        "geese": animals["GOOSE"],
    }


def run_one(task: tuple[str, str, str, int, int]) -> dict[str, object]:
    name, family, path, seed, seat = task
    started = time.perf_counter()
    try:
        from kaggle_environments import make

        agents = [path, str(FC24B)]
        if seat == 1:
            agents.reverse()
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": 720, "seed": seed},
            debug=False,
        )
        env.run(agents)
        final = env.steps[-1]
        candidate_reward = float(final[seat].reward)
        opponent_reward = float(final[1 - seat].reward)
        statuses = [row.status for row in final]
        frames = len(env.steps)
        error = ""
        summary = farm_summary(final[seat], seat)
    except Exception as exc:  # Keep every failed candidate auditable.
        candidate_reward = 0.0
        opponent_reward = 0.0
        statuses = ["ERROR", "ERROR"]
        frames = 0
        error = f"{type(exc).__name__}: {exc}"
        summary = {"money": 0, "units": 0, "cows": 0, "sheep": 0, "geese": 0}
    return {
        "candidate": name,
        "family": family,
        "path": path,
        "seed": seed,
        "candidate_seat": seat,
        "frames": frames,
        "statuses": statuses,
        "candidate_reward": candidate_reward,
        "fc24b_reward": opponent_reward,
        "margin": candidate_reward - opponent_reward,
        "win": candidate_reward > opponent_reward,
        "tie": candidate_reward == opponent_reward,
        "candidate_farm": summary,
        "error": error,
        "seconds": time.perf_counter() - started,
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    completed = [
        row
        for row in rows
        if row["frames"] == 720 and row["statuses"] == ["DONE", "DONE"]
    ]
    margins = [float(row["margin"]) for row in completed]
    return {
        "games": len(rows),
        "completed_games": len(completed),
        "wins": sum(bool(row["win"]) for row in completed),
        "ties": sum(bool(row["tie"]) for row in completed),
        "win_rate": (
            sum(bool(row["win"]) for row in completed) / len(completed)
            if completed
            else 0.0
        ),
        "mean_margin": statistics.fmean(margins) if margins else 0.0,
        "mean_candidate_reward": (
            statistics.fmean(float(row["candidate_reward"]) for row in completed)
            if completed
            else 0.0
        ),
        "mean_fc24b_reward": (
            statistics.fmean(float(row["fc24b_reward"]) for row in completed)
            if completed
            else 0.0
        ),
        "seat_win_rate": {
            str(seat): (
                sum(bool(row["win"]) for row in completed if row["candidate_seat"] == seat)
                / max(1, sum(row["candidate_seat"] == seat for row in completed))
            )
            for seat in (0, 1)
        },
        "errors": sorted({str(row["error"]) for row in rows if row["error"]}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=1301001)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected_names = {value for value in args.candidates.split(",") if value}
    selected = [
        row for row in CANDIDATES if not selected_names or row.name in selected_names
    ]
    if selected_names - {row.name for row in selected}:
        raise ValueError(f"unknown candidates: {sorted(selected_names - {row.name for row in selected})}")
    missing = [str(row.path) for row in selected if not row.path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing candidate files: {missing}")
    tasks = [
        (row.name, row.family, str(row.path.resolve()), seed, seat)
        for row in selected
        for seed in range(args.seed_start, args.seed_start + args.seeds)
        for seat in (0, 1)
    ]
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 18)) as pool:
        rows = list(pool.map(run_one, tasks))
    aggregates = {
        row.name: {
            "family": row.family,
            "path": str(row.path.resolve()),
            **aggregate([result for result in rows if result["candidate"] == row.name]),
        }
        for row in selected
    }
    payload = {
        "schema": "kaggriculture.public-recent-20260825.official-vs-fc24b.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_environment": "kaggle-environments==1.32.7",
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "seat_swapped": True,
        "workers": min(max(1, args.workers), 18),
        "fc24b": str(FC24B.resolve()),
        "aggregates": aggregates,
        "rows": rows,
        "status": "PASS" if all(not value["errors"] for value in aggregates.values()) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": payload["status"], "aggregates": aggregates},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

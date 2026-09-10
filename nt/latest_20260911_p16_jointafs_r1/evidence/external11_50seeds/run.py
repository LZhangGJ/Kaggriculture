"""Evaluate P16 JointAFS R1 on the frozen 50-seed, 11-opponent live panel.

This intentionally reuses the exact seed/seat/opponent protocol from
takeover_repairs_four_versions_newseeds_public11_100_20260910.  The parent
T3R1 TakeoverMerged results are read from that completed experiment rather
than simulated again.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PACKAGE = (
    WORKSPACE
    / "gpt_review/gpt_code/Kaggriculture_P16_JointAFS_R1_20260910"
    / "Kaggriculture_P16_JointAFS_R1_20260910"
)
PARENT_EXPERIMENT = (
    WORKSPACE
    / "experiments/takeover_repairs_four_versions_newseeds_public11_100_20260910"
)
sys.path.insert(0, str(PACKAGE))
from arena import game  # noqa: E402


VERSION = "P16_JointAFS_R1"
PARENT_VERSION = "T3R1_TakeoverMerged_R1"
BINARY = PACKAGE / "policy/joint.so"
SEEDS = list(range(2610100000, 2610100050))
WORKERS = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def key(row: dict) -> tuple:
    return row["opponent"], row["seed"], row["opponent_seat"]


def run_job(job: tuple) -> dict:
    row = game(job)
    row["version"] = VERSION
    return row


def summarize(rows: list[dict]) -> dict:
    good = [row for row in rows if not row.get("runtime_error")]
    wins = sum(bool(row["win"]) for row in good)
    ties = sum(bool(row["tie"]) for row in good)
    return {
        "games": len(rows),
        "wins": wins,
        "losses": len(good) - wins - ties,
        "ties": ties,
        "errors": len(rows) - len(good),
        "win_rate": wins / len(good) if good else None,
        "mean_cash": statistics.mean(row["own_cash"] for row in good) if good else None,
        "median_cash": statistics.median(row["own_cash"] for row in good) if good else None,
        "mean_margin": statistics.mean(row["margin"] for row in good) if good else None,
        "median_margin": statistics.median(row["margin"] for row in good) if good else None,
        "max_latency_seconds": max((row["latency_max"] for row in good), default=None),
    }


def main() -> None:
    pool_file = PACKAGE / "POOL.json"
    pool = [entry["id"] for entry in json.loads(pool_file.read_text(encoding="utf-8"))]
    assert len(pool) == 11 and len(set(pool)) == 11
    parent_protocol = json.loads(
        (PARENT_EXPERIMENT / "PROTOCOL.json").read_text(encoding="utf-8")
    )
    assert parent_protocol["seed_start"] == SEEDS[0]
    assert parent_protocol["seed_end"] == SEEDS[-1]
    assert parent_protocol["opponents"] == pool
    assert parent_protocol["both_seats"] is True
    assert parent_protocol["official"] == "1.32.7"

    protocol = {
        "version": VERSION,
        "parent_version": PARENT_VERSION,
        "binary_sha256": sha256(BINARY),
        "opening": "none",
        "seed_start": SEEDS[0],
        "seed_end": SEEDS[-1],
        "seed_count": len(SEEDS),
        "both_seats": True,
        "opponents": pool,
        "games_per_opponent": 100,
        "total_games": len(pool) * len(SEEDS) * 2,
        "workers": WORKERS,
        "official": "1.32.7",
        "realtime_opponents": True,
        "config_sha256": sha256(PACKAGE / "policy/config.json"),
        "pool_sha256": sha256(pool_file),
        "engine_sha256": sha256(PACKAGE / "referee/official/kaggriculture.py"),
        "future_replay_access": False,
        "opponent_identity_input": False,
        "seed_input": False,
        "parent_results_reused_from": str(PARENT_EXPERIMENT),
    }
    protocol_file = HERE / "PROTOCOL.json"
    if protocol_file.exists():
        assert json.loads(protocol_file.read_text(encoding="utf-8")) == protocol
    else:
        save(protocol_file, protocol)

    final_file = HERE / "rows.json"
    partial_file = HERE / "rows.partial.json"
    if final_file.exists():
        rows = json.loads(final_file.read_text(encoding="utf-8"))
    elif partial_file.exists():
        rows = json.loads(partial_file.read_text(encoding="utf-8"))
    else:
        rows = []
    assert all(not row.get("runtime_error") for row in rows)
    done = {key(row) for row in rows}
    assert len(done) == len(rows)
    jobs = [
        (opponent, seed, seat, str(BINARY), "none", {}, 719, None)
        for seed in SEEDS
        for opponent in pool
        for seat in (0, 1)
        if (opponent, seed, seat) not in done
    ]

    started = time.perf_counter()
    if jobs:
        with cf.ProcessPoolExecutor(
            max_workers=WORKERS, mp_context=mp.get_context("spawn")
        ) as executor:
            for row in executor.map(run_job, jobs, chunksize=1):
                if row.get("runtime_error"):
                    save(HERE / "FAILED.json", row)
                    raise RuntimeError(row["runtime_error"])
                rows.append(row)
                if len(rows) % 50 == 0 or len(rows) == protocol["total_games"]:
                    save(partial_file, rows)
                    progress = {
                        "done": len(rows),
                        "total": protocol["total_games"],
                        "elapsed_seconds_this_run": time.perf_counter() - started,
                        "summary": summarize(rows),
                    }
                    save(HERE / "PROGRESS.json", progress)
                    print(json.dumps(progress, ensure_ascii=False), flush=True)

    rows.sort(key=key)
    assert len(rows) == protocol["total_games"]
    assert len({key(row) for row in rows}) == len(rows)
    assert all(row["steps"] == 719 and not row.get("runtime_error") for row in rows)
    save(final_file, rows)

    parent_all = json.loads((PARENT_EXPERIMENT / "rows.json").read_text(encoding="utf-8"))
    parent_rows = [row for row in parent_all if row["version"] == PARENT_VERSION]
    assert len(parent_rows) == len(rows)
    parent_by_key = {(row["opponent"], row["seed"], row["opponent_seat"]): row for row in parent_rows}
    assert set(parent_by_key) == {key(row) for row in rows}

    paired = []
    for row in rows:
        parent = parent_by_key[key(row)]
        paired.append(
            {
                "opponent": row["opponent"],
                "seed": row["seed"],
                "opponent_seat": row["opponent_seat"],
                "parent_win": bool(parent["win"]),
                "joint_win": bool(row["win"]),
                "parent_cash": parent["own_cash"],
                "joint_cash": row["own_cash"],
                "parent_margin": parent["margin"],
                "joint_margin": row["margin"],
                "margin_delta": row["margin"] - parent["margin"],
                "rescued": bool(row["win"] and not parent["win"]),
                "lost_parent_win": bool(parent["win"] and not row["win"]),
                "same_action_hash": parent["action_hash"] == row["action_hash"],
            }
        )
    save(HERE / "paired.json", paired)

    result = {
        "joint": summarize(rows),
        "parent": summarize(parent_rows),
        "paired": {
            "games": len(paired),
            "rescued": sum(row["rescued"] for row in paired),
            "lost_parent_wins": sum(row["lost_parent_win"] for row in paired),
            "margin_improved": sum(row["margin_delta"] > 0 for row in paired),
            "margin_worse": sum(row["margin_delta"] < 0 for row in paired),
            "same_actions": sum(row["same_action_hash"] for row in paired),
            "mean_margin_delta": statistics.mean(row["margin_delta"] for row in paired),
        },
        "by_opponent": {
            opponent: {
                "joint": summarize([row for row in rows if row["opponent"] == opponent]),
                "parent": summarize([row for row in parent_rows if row["opponent"] == opponent]),
                "rescued": sum(row["rescued"] for row in paired if row["opponent"] == opponent),
                "lost_parent_wins": sum(
                    row["lost_parent_win"] for row in paired if row["opponent"] == opponent
                ),
                "mean_margin_delta": statistics.mean(
                    row["margin_delta"] for row in paired if row["opponent"] == opponent
                ),
            }
            for opponent in pool
        },
        "seconds_this_run": time.perf_counter() - started,
    }
    save(HERE / "RESULTS.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

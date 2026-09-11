from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import kaggle_environments
from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[3]
NULL_AGENT_PATH = ROOT / "experiments" / "route_playbook_v1" / "opponents" / "null_agent" / "main.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_callable(path: Path, callable_name: str, tag: str) -> Callable[..., dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, callable_name)
    if not callable(value):
        raise TypeError(f"{path}::{callable_name} is not callable")
    return value


def run_game(
    agent_path: Path,
    seed: int,
    candidate_seat: int,
    opponent: str,
    nonce: int,
) -> dict[str, Any]:
    candidate = load_callable(agent_path, "agent", f"hp_candidate_{nonce}")
    if opponent == "null":
        other = load_callable(NULL_AGENT_PATH, "null_agent", f"hp_null_{nonce}")
    elif opponent == "mirror":
        other = load_callable(agent_path, "agent", f"hp_mirror_{nonce}")
    else:
        raise ValueError(opponent)
    agents = [candidate, other] if candidate_seat == 0 else [other, candidate]
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": int(seed)},
        debug=False,
    )
    started = time.perf_counter()
    env.run(agents)
    elapsed = time.perf_counter() - started
    final = env.steps[-1]
    rewards = [int(state.reward or 0) for state in final]
    statuses = [str(state.status) for state in final]
    return {
        "seed": int(seed),
        "candidate_seat": int(candidate_seat),
        "opponent": opponent,
        "steps": len(env.steps),
        "statuses": statuses,
        "rewards": rewards,
        "candidate_reward": rewards[candidate_seat],
        "opponent_reward": rewards[1 - candidate_seat],
        "candidate_margin": rewards[candidate_seat] - rewards[1 - candidate_seat],
        "elapsed_seconds": round(elapsed, 6),
        "done": statuses == ["DONE", "DONE"] and len(env.steps) == 720,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Official Python 1.32.7 smoke for the frozen high-potential agents."
    )
    parser.add_argument("--agents-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[98101, 98102])
    parser.add_argument(
        "--mirror-seed",
        type=int,
        default=98103,
        help="One additional self-play seed; set negative to skip.",
    )
    args = parser.parse_args()

    if kaggle_environments.__version__ != "1.32.7":
        raise RuntimeError(
            f"official smoke requires kaggle_environments 1.32.7, got {kaggle_environments.__version__}"
        )
    agents_root = args.agents_root.resolve()
    manifest = json.loads((agents_root / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    nonce = time.time_ns()
    for agent_entry in manifest["agents"]:
        slug = str(agent_entry["slug"])
        path = agents_root / slug / "main.py"
        if sha256(path) != agent_entry["main_sha256"]:
            raise RuntimeError(f"{slug}: frozen source hash no longer matches manifest")
        for seed in args.seeds:
            for seat in (0, 1):
                nonce += 1
                row = run_game(path, seed, seat, "null", nonce)
                row["slug"] = slug
                rows.append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
        if args.mirror_seed >= 0:
            for seat in (0, 1):
                nonce += 1
                row = run_game(path, args.mirror_seed, seat, "mirror", nonce)
                row["slug"] = slug
                rows.append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)

    summaries = []
    for agent_entry in manifest["agents"]:
        slug = str(agent_entry["slug"])
        selected = [row for row in rows if row["slug"] == slug]
        null_rewards = [row["candidate_reward"] for row in selected if row["opponent"] == "null"]
        summaries.append(
            {
                "slug": slug,
                "games": len(selected),
                "done_games": sum(bool(row["done"]) for row in selected),
                "null_games": len(null_rewards),
                "null_reward_min": min(null_rewards),
                "null_reward_mean": statistics.fmean(null_rewards),
                "null_reward_max": max(null_rewards),
                "mirror_games": sum(row["opponent"] == "mirror" for row in selected),
                "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in selected), 6),
                "status": "PASS" if all(row["done"] for row in selected) else "FAIL",
            }
        )

    result = {
        "schema": "kaggriculture.high_potential_official_smoke.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_package_version": kaggle_environments.__version__,
        "episode_steps": 720,
        "seeds": args.seeds,
        "mirror_seed": args.mirror_seed,
        "truth_boundary": (
            "Official 1.32.7 execution smoke and passive-opponent potential only; "
            "not a leaderboard estimate and not JAX parity evidence."
        ),
        "status": "PASS" if all(item["status"] == "PASS" for item in summaries) else "FAIL",
        "summaries": summaries,
        "rows": rows,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"result": result["status"], "summaries": summaries}, ensure_ascii=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

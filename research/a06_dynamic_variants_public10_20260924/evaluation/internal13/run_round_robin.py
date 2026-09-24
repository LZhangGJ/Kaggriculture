"""Official-rule, live-policy, two-seat round robin with resumable receipts."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import copy
from hashlib import sha256
import io
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERIFY = REPO / "dp/AFS_R2_DP_Fusion_R3_Experimental_Delivery_ver b/Kaggriculture_Fusion_R2_20260916/verification"
ENGINE = VERIFY / "referee/official/kaggriculture.py"
sys.path.insert(0, str(VERIFY))
from policy_host import LocalGame, Policy, load_engine  # noqa: E402
from referee.cpu_runtime import pass_agent  # noqa: E402


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verified_pool() -> list[dict]:
    pool = read(HERE / "POOL.json")
    if len(pool) != 13 or len({r["id"] for r in pool}) != len(pool):
        raise ValueError("expected 13 unique primary releases")
    for row in pool:
        folder = REPO / row["working"]
        for relative, expected in row["files"].items():
            path = folder / relative
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"frozen source changed: {path}")
    if digest(ENGINE) != "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e":
        raise ValueError("official referee changed")
    return pool


def play(job):
    a, b, seed, seat_a = job
    start = time.monotonic()
    result = {"a": a["id"], "b": b["id"] if b else "PASS", "seed": seed,
              "seat_a": seat_a, "error": None}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            policy_a = Policy(str(REPO / a["entry"]), "policy_a")
            policy_b = Policy(str(REPO / b["entry"]), "policy_b") if b else None
            env = LocalGame(seed, load_engine())
            max_a = max_b = 0.0
            while not env.done:
                obs_a = env.observation(seat_a)
                tick = time.monotonic()
                action_a = policy_a(obs_a, copy.deepcopy(env.configuration))
                max_a = max(max_a, time.monotonic() - tick)
                if policy_b:
                    obs_b = env.observation(1 - seat_a)
                    tick = time.monotonic()
                    action_b = policy_b(obs_b, copy.deepcopy(env.configuration))
                    max_b = max(max_b, time.monotonic() - tick)
                else:
                    action_b = pass_agent(env.observation(1 - seat_a))
                actions = [None, None]
                actions[seat_a], actions[1 - seat_a] = action_a, action_b
                env.advance(actions)
            if env.t != 719:
                raise ValueError(f"incomplete game: {env.t}")
            cash = [farm["money"] for farm in env.state[0].observation.farms]
            margin = cash[seat_a] - cash[1 - seat_a]
            result.update(steps=env.t, cash_a=cash[seat_a], cash_b=cash[1 - seat_a],
                          margin=margin, win_a=int(margin > 0), tie=int(margin == 0),
                          max_action_a_s=max_a, max_action_b_s=max_b)
    except Exception:
        result["error"] = traceback.format_exc()
    result["seconds"] = time.monotonic() - start
    return result


def run_jobs(jobs, output: Path, workers: int, label: str) -> list[dict]:
    previous = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line] if output.exists() else []
    def key(row):
        return row["a"], row["b"], row["seed"], row["seat_a"]
    if len({key(row) for row in previous}) != len(previous):
        raise ValueError("duplicate prior game")
    done = {key(row) for row in previous}
    pending = [job for job in jobs if (job[0]["id"], job[1]["id"] if job[1] else "PASS", job[2], job[3]) not in done]
    print(json.dumps({"phase": label, "done": len(previous), "remaining": len(pending),
                      "total": len(jobs)}, ensure_ascii=False), flush=True)
    started = time.monotonic()
    with output.open("a", encoding="utf-8") as sink:
        for offset in range(0, len(pending), 128):
            batch = pending[offset:offset + 128]
            with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                             max_tasks_per_child=1) as executor:
                tasks = [executor.submit(play, job) for job in batch]
                for future in futures.as_completed(tasks):
                    row = future.result()
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    previous.append(row)
                    if len(previous) % 25 == 0 or row["error"]:
                        print(json.dumps({"phase": label, "done": len(previous), "total": len(jobs),
                                          "errors": sum(bool(r["error"]) for r in previous),
                                          "elapsed_s": round(time.monotonic() - started, 1)},
                                         ensure_ascii=False), flush=True)
                        if row["error"]:
                            print(row["a"], row["b"], row["error"][-900:], flush=True)
    return previous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "run"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--pair-limit", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be 1..16")
    pool = verified_pool()
    if args.mode == "smoke":
        jobs = [(agent, None, 2026092409, 0) for agent in pool]
        rows = run_jobs(jobs, HERE / "smoke_games.jsonl", args.workers, "smoke")
        accepted = [r["a"] for r in rows if not r["error"] and r.get("steps") == 719]
        save(HERE / "SMOKE_SUMMARY.json", {"accepted": accepted,
             "errors": [r for r in rows if r["error"]], "total": len(rows)})
        if len(accepted) != len(pool):
            raise SystemExit(f"only {len(accepted)}/{len(pool)} agents passed full-game smoke")
        return
    smoke = read(HERE / "SMOKE_SUMMARY.json")
    if set(smoke["accepted"]) != {r["id"] for r in pool}:
        raise ValueError("smoke gate incomplete")
    seeds = random.Random(2026092901).sample(range(1_000_000_000, 2_000_000_000), 50)
    pairs = [(a, b) for i, a in enumerate(pool) for b in pool[i + 1:]]
    if args.pair_limit:
        pairs = pairs[:args.pair_limit]
    jobs = [(a, b, seed, seat) for a, b in pairs for seed in seeds for seat in (0, 1)]
    protocol = {
        "agents": [{"id": r["id"], "archive": r["archive"], "archive_sha256": r["archive_sha256"],
                    "main_sha256": r["files"]["main.py"], "native_sha256": r["files"]["policy/a06.so"],
                    "config_sha256": r["files"]["policy/config.json"]} for r in pool],
        "seeds": seeds,
        "pairs": [[a["id"], b["id"]] for a, b in pairs],
        "games": len(jobs),
        "games_per_pair": 100,
        "official_engine_sha256": digest(ENGINE),
        "method": "official 1.32.7 Python rules, live native policies, 50 common seeds x both seats per pair; no replay actions",
    }
    protocol_path = HERE / ("PROTOCOL.json" if not args.pair_limit else "PILOT_PROTOCOL.json")
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("frozen protocol differs")
    save(protocol_path, protocol)
    output = HERE / ("games.jsonl" if not args.pair_limit else "pilot_games.jsonl")
    rows = run_jobs(jobs, output, args.workers, "round_robin")
    summary = {"games": len(rows), "expected": len(jobs),
               "errors": sum(bool(r["error"]) for r in rows),
               "complete_719": sum(r.get("steps") == 719 for r in rows),
               "total_seconds": sum(r["seconds"] for r in rows)}
    save(HERE / ("RUN_SUMMARY.json" if not args.pair_limit else "PILOT_SUMMARY.json"), summary)
    if summary["errors"] or summary["complete_719"] != len(jobs):
        raise SystemExit("match panel contains errors or incomplete games")


if __name__ == "__main__":
    main()

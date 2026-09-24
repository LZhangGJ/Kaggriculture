"""Three frozen A06 variants versus the same public ten and seeds as Cashflow."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
from hashlib import sha256
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INTERNAL = ROOT / "experiments/a06_r12_gptpro_round_robin_20260924"
CASHFLOW = ROOT / "experiments/r14_cashflow_vs_public_top10_20260924"
ARCHIVES = ROOT / "gpt_review/gpt_code/a06_r12_update"
ENGINE = ROOT / "gpu_sim/reference/kaggle_environments_1_32_7/kaggriculture/kaggriculture.py"
IDS = ("rule_r18", "r14_liquidity", "r14_tl5")
sys.path.insert(0, str(ROOT / "experiments/local_dynamic4_vs_public_top5_20260923"))
from run_matchups import play  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify(folder: Path, files: dict[str, str]) -> None:
    for relative, expected in files.items():
        path = folder / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"frozen file missing or changed: {path}")


def freeze() -> dict:
    base = read(CASHFLOW / "PROTOCOL.json")
    if len(base["opponents"]) != 10 or len(base["seeds"]) != 50 or base["total_games"] != 1000:
        raise ValueError("Cashflow ten-opponent reference panel has changed")
    if digest(ENGINE) != base["engine_sha256"]:
        raise ValueError("official engine differs from Cashflow panel")
    for opponent in base["opponents"]:
        entry = Path(opponent["entry"])
        verify(entry.parent, opponent["files"])
    pool = {row["id"]: row for row in read(INTERNAL / "POOL.json")}
    candidates = []
    for ident in IDS:
        row = pool[ident]
        folder = INTERNAL / "agents" / ident
        verify(folder, row["files"])
        if digest(ARCHIVES / row["archive"]) != row["archive_sha256"]:
            raise ValueError(f"original delivery archive changed: {row['archive']}")
        candidates.append({"id": ident, "entry": str(folder / "main.py"),
                           "archive": row["archive"], "archive_sha256": row["archive_sha256"],
                           "files": row["files"]})
    return {"candidates": candidates, "opponents": base["opponents"], "seeds": base["seeds"],
            "cashflow_reference_protocol_sha256": digest(CASHFLOW / "PROTOCOL.json"),
            "engine_sha256": digest(ENGINE),
            "games_per_pair": 100, "total_games": 3000,
            "method": "Official Kaggriculture 1.32.7 Python referee, live policies, 50 Cashflow-reference seeds x both seats per candidate/opponent"}


def run_jobs(jobs: list[tuple], path: Path, workers: int) -> list[dict]:
    prior = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []
    def key(row):
        return row["local"], row["public"], row["seed"], row["local_seat"]
    done = {key(row) for row in prior}
    if len(done) != len(prior):
        raise ValueError("duplicate prior game")
    pending = [job for job in jobs if (job[0], job[2], job[4], job[5]) not in done]
    print(json.dumps({"done": len(prior), "remaining": len(pending), "total": len(jobs)}), flush=True)
    started = time.monotonic()
    with path.open("a", encoding="utf-8") as sink:
        for offset in range(0, len(pending), 128):
            batch = pending[offset:offset + 128]
            with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                             max_tasks_per_child=1) as executor:
                tasks = [executor.submit(play, job) for job in batch]
                for future in futures.as_completed(tasks):
                    row = future.result()
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    prior.append(row)
                    if len(prior) % 25 == 0 or row["error"]:
                        print(json.dumps({"done": len(prior), "total": len(jobs),
                                          "errors": sum(bool(r["error"]) for r in prior),
                                          "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
                        if row["error"]:
                            print(row["local"], row["public"], row["error"][-900:], flush=True)
    return prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "run"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        raise ValueError("workers must be 1..24")
    protocol = freeze()
    path = HERE / "PROTOCOL.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("frozen protocol differs")
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.mode == "smoke":
        rival = protocol["opponents"][0]
        jobs = [(row["id"], row["entry"], rival["ref"], rival["entry"], 2026092411, index % 2)
                for index, row in enumerate(protocol["candidates"])]
        rows = run_jobs(jobs, HERE / "smoke_games.jsonl", args.workers)
        summary = {"games": len(rows), "errors": sum(bool(r["error"]) for r in rows),
                   "complete_719": sum(r.get("steps") == 719 for r in rows)}
        (HERE / "SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if summary != {"games": 3, "errors": 0, "complete_719": 3}:
            raise SystemExit("smoke gate failed")
        return
    if read(HERE / "SMOKE_SUMMARY.json") != {"games": 3, "errors": 0, "complete_719": 3}:
        raise ValueError("smoke gate incomplete")
    jobs = [(local["id"], local["entry"], rival["ref"], rival["entry"], seed, seat)
            for local in protocol["candidates"] for rival in protocol["opponents"]
            for seed in protocol["seeds"] for seat in (0, 1)]
    rows = run_jobs(jobs, HERE / "games.jsonl", args.workers)
    summary = {"games": len(rows), "expected": len(jobs),
               "errors": sum(bool(r["error"]) for r in rows),
               "complete_719": sum(r.get("steps") == 719 for r in rows)}
    (HERE / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary != {"games": 3000, "expected": 3000, "errors": 0, "complete_719": 3000}:
        raise SystemExit("panel incomplete")


if __name__ == "__main__":
    main()

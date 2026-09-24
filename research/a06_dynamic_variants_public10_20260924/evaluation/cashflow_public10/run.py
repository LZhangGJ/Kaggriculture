"""Frozen, resumable 100-game-per-opponent live-policy panel."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
from hashlib import sha256
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROUND_ROBIN = ROOT / "experiments/a06_r12_gptpro_round_robin_20260924"
PUBLIC = ROOT / "experiments/a06_r12_public_top20_20260924"
PACKAGE = ROOT / "packages/A06_R12_r4_teammate_top10_original_cppsim_1327_20260924.zip"
ENGINE = ROOT / "gpu_sim/reference/kaggle_environments_1_32_7/kaggriculture/kaggriculture.py"
sys.path.insert(0, str(ROOT / "experiments/local_dynamic4_vs_public_top5_20260923"))
from run_matchups import play  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_files(folder: Path, files: dict[str, str]) -> None:
    for relative, expected in files.items():
        path = folder / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"frozen file missing or changed: {path}")


def freeze() -> dict:
    candidate = next(row for row in read(ROUND_ROBIN / "POOL.json") if row["id"] == "r14_cashflow")
    candidate_folder = ROUND_ROBIN / "agents/r14_cashflow"
    verify_files(candidate_folder, candidate["files"])
    if digest(ENGINE) != "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e":
        raise ValueError("official rule source changed")

    selected = read(PUBLIC / "QUALIFIED_TOP20.json")["selected_refs"][:10]
    pool = {row["ref"]: row for row in read(PUBLIC / "POOL_20260924.json")}
    with zipfile.ZipFile(PACKAGE) as archive:
        package_manifest = json.loads(archive.read("MANIFEST.json"))
    if [row["ref"] for row in package_manifest["opponents"]] != selected:
        raise ValueError("top-ten package no longer matches qualified panel")
    opponents = []
    for ref, packaged in zip(selected, package_manifest["opponents"]):
        row = pool[ref]
        folder = ROOT / row["working"]
        verify_files(folder, row["files"])
        if row["main_sha256"] != packaged["main_sha256"]:
            raise ValueError(f"package/pool main.py mismatch: {ref}")
        opponents.append({"id": row["id"], "ref": ref,
                          "entry": str(folder / "main.py"),
                          "version": packaged["version"],
                          "source_sha256": packaged["source_sha256"],
                          "files": row["files"]})

    seeds = random.Random(2026093003).sample(range(1_000_000_000, 2_000_000_000), 50)
    prior_seeds = set(read(ROUND_ROBIN / "PROTOCOL.json")["seeds"])
    for old in PUBLIC.glob("*_protocol.json"):
        prior_seeds.update(read(old).get("seeds", []))
    if prior_seeds.intersection(seeds):
        raise ValueError("new seeds overlap a prior A06/public evaluation")
    return {
        "candidate": {"id": "r14_cashflow", "entry": str(candidate_folder / "main.py"),
                      "archive": candidate["archive"], "archive_sha256": candidate["archive_sha256"],
                      "files": candidate["files"]},
        "opponents": opponents,
        "seeds": seeds,
        "games_per_opponent": 100,
        "total_games": 1000,
        "engine_sha256": digest(ENGINE),
        "top10_package_sha256": digest(PACKAGE),
        "method": "Official Kaggriculture 1.32.7 Python rules, live policy actions; 50 new common seeds x both seats per opponent",
    }


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
                for task in futures.as_completed(tasks):
                    row = task.result()
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    prior.append(row)
                    if len(prior) % 25 == 0 or row["error"]:
                        print(json.dumps({"done": len(prior), "total": len(jobs),
                                          "errors": sum(bool(r["error"]) for r in prior),
                                          "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
                        if row["error"]:
                            print(row["public"], row["error"][-900:], flush=True)
    return prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "run"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be 1..16")
    protocol = freeze()
    path = HERE / "PROTOCOL.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("frozen protocol differs")
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    local = protocol["candidate"]
    if args.mode == "smoke":
        jobs = [(local["id"], local["entry"], row["ref"], row["entry"], 2026092408, i % 2)
                for i, row in enumerate(protocol["opponents"])]
        rows = run_jobs(jobs, HERE / "smoke_games.jsonl", args.workers)
        summary = {"games": len(rows), "errors": sum(bool(r["error"]) for r in rows),
                   "complete_719": sum(r.get("steps") == 719 for r in rows)}
        (HERE / "SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if summary != {"games": 10, "errors": 0, "complete_719": 10}:
            raise SystemExit("smoke gate failed")
        return
    if read(HERE / "SMOKE_SUMMARY.json") != {"games": 10, "errors": 0, "complete_719": 10}:
        raise ValueError("smoke gate incomplete")
    jobs = [(local["id"], local["entry"], row["ref"], row["entry"], seed, seat)
            for row in protocol["opponents"] for seed in protocol["seeds"] for seat in (0, 1)]
    rows = run_jobs(jobs, HERE / "games.jsonl", args.workers)
    summary = {"games": len(rows), "expected": len(jobs),
               "errors": sum(bool(r["error"]) for r in rows),
               "complete_719": sum(r.get("steps") == 719 for r in rows)}
    (HERE / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary != {"games": 1000, "expected": 1000, "errors": 0, "complete_719": 1000}:
        raise SystemExit("panel incomplete")


if __name__ == "__main__":
    main()

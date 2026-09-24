#!/usr/bin/env python3
"""Evaluate the frozen 17-day student against downloaded Python agents."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib.util
import json
import os
import random
import runpy
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.collect_student_state_dagger_v3 import _load, _observations
from experiments.student_action_event_agent import StudentActionEventAgent, production
from experiments.train_student_action_event_rl_v3 import STUDENT_STEPS


ROOT = Path(__file__).resolve().parents[1]
SUBMITTED = None
SUBMITTED_METRICS = None


class _SubmittedEvalStudent(StudentActionEventAgent):
    def _validate_training_contract(self, checkpoint: dict) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        if (SUBMITTED_METRICS is None or
                checkpoint.get("shard_manifest_sha256") !=
                hashlib.sha256(self.manifest_path.read_bytes()).hexdigest() or
                any(manifest.get("dimensions", {}).get(key) != value for key, value in
                    checkpoint.get("model_dimensions", {}).items()) or
                SUBMITTED_METRICS.get("binary_sha256") !=
                hashlib.sha256(self.binary_path.read_bytes()).hexdigest()):
            raise ValueError("submitted-eval checkpoint/manifest/binary mismatch")


def _submitted_agent(seat: int):
    namespace, root = SUBMITTED
    modules = ("meta_agent.src.route_switch_features",
               "meta_agent.src.search_route_policy")
    previous = {name: sys.modules.get(name) for name in modules}
    try:
        for name in modules:
            path = root / (name.replace(".", "/") + ".py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return namespace["create_agent"](seat)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if str(root) in sys.path:
            sys.path.remove(str(root))


def _play(job: dict) -> dict:
    from fast_kaggriculture import Config, FastEnv

    started = time.perf_counter()
    dynamic = None
    opponent = None
    try:
        policy_seed = int.from_bytes(hashlib.blake2b(
            f'{job["seed"]}:{job["slug"]}:{job["seat"]}:{job["sample"]}'.encode(),
            digest_size=8).digest(), "little")
        random.seed(policy_seed)
        np.random.seed(policy_seed & 0xFFFFFFFF)
        torch.manual_seed(policy_seed)
        config = json.loads((ROOT / "policy/r1/config.json").read_text())
        if job.get("submission"):
            config["intraday"] = 0
        dynamic = (_SubmittedEvalStudent if job.get("submission") else StudentActionEventAgent)(
            config, binary_path=Path(job["binary"]),
            checkpoint_path=Path(job["checkpoint"]),
            manifest_path=Path(job["manifest"]), sample=job["sample"],
            student_steps=STUDENT_STEPS)
        replay = production.create_replay_agent(
            production.replay_deployment(),
            f'public_eval_{job["slug"]}_{job["seed"]}_{job["seat"]}')
        candidate = production.ReplayThenDynamicAgent(
            replay, dynamic, 288, handoff_land=None, handoff_floor=0,
            handoff_delay_days=0, selector=None)
        if job.get("submission"):
            opponent = _submitted_agent(job["seat"])
        else:
            parent = str(Path(job["path"]).resolve().parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            opponent = _load(Path(job["path"]),
                             f'public_eval_{os.getpid()}_{job["slug"]}_{job["seed"]}_{job["seat"]}').agent
        takes_config = len(inspect.signature(opponent).parameters) > 1
        env = FastEnv(Config(), int(job["seed"]))
        state = list(env.reset(int(job["seed"])))
        frames = 0
        while not env.done:
            actions = []
            for player, observation in enumerate(_observations(state)):
                if player == job["seat"]:
                    action = candidate(observation, {})
                else:
                    action = (opponent(observation, {}) if takes_config else
                              opponent(observation))
                actions.append(action)
            state = env.step(actions)
            frames += 1
        summary = dynamic.student_summary()
        if (frames != 719 or summary["successful_steps"] != list(STUDENT_STEPS)
                or summary["illegal"] or summary["fallbacks"]):
            raise RuntimeError(f'incomplete game or student fallback: {summary}')
        own = float(env.rewards[job["seat"]])
        rival = float(env.rewards[1 - job["seat"]])
        return {"status": "PASS", "slug": job["slug"], "seed": job["seed"],
                "seat": job["seat"], "own_cash": own, "rival_cash": rival,
                "margin": own - rival, "seconds": time.perf_counter() - started}
    except Exception as error:
        return {"status": "FAIL", "slug": job["slug"], "seed": job["seed"],
                "seat": job["seat"], "error": repr(error),
                "seconds": time.perf_counter() - started}
    finally:
        if dynamic is not None:
            dynamic.close()
        if opponent is not None and hasattr(opponent, "close"):
            opponent.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "work/agent-student-actor-owned-v3.so")
    parser.add_argument("--manifest", type=Path, default=(
        ROOT / "work/student-v1/action-event-v3-actor-owned-width3074-full/manifest.json"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--opponents", type=Path)
    source.add_argument("--submission", type=Path)
    parser.add_argument("--submission-arm-so", type=Path)
    parser.add_argument("--checkpoint-metrics", type=Path)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = (args.checkpoint, args.binary, args.manifest,
             args.submission or args.opponents)
    if (args.seeds < 1 or not 1 <= args.workers <= (32 if args.submission else 8) or
            args.seed_start < 0 or
            any(not path.is_file() for path in paths) or args.output.exists()):
        parser.error("invalid input or existing output")
    cgroup = Path("/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text()
                  .splitlines()[0].split(":", 2)[2])
    memory_max = cgroup / "memory.max"
    memory_limit_gb = 128 if args.submission else 64
    if (not memory_max.is_file() or memory_max.read_text().strip() == "max" or
            int(memory_max.read_text()) > memory_limit_gb * 1024**3):
        parser.error(f"run inside a memory-capped scope (MemoryMax <= {memory_limit_gb}G)")
    global SUBMITTED, SUBMITTED_METRICS
    if args.submission:
        if (not args.submission_arm_so or not args.submission_arm_so.is_file() or
                not args.checkpoint_metrics or not args.checkpoint_metrics.is_file()):
            parser.error("submitted evaluation requires parity build and checkpoint metrics")
        SUBMITTED_METRICS = json.loads(args.checkpoint_metrics.read_text())
        if (SUBMITTED_METRICS.get("status") != "PASS" or
                SUBMITTED_METRICS.get("student_intraday") != 0 or
                SUBMITTED_METRICS.get("checkpoint_out_sha256") !=
                hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()):
            parser.error("submitted evaluation checkpoint metrics mismatch")
        archive = runpy.run_path(str(args.submission.resolve()))
        root = Path(archive["_root"])
        shutil.copyfile(args.submission_arm_so,
                        root / "policy/r1/agent.so")
        sys.path.remove(str(root))
        SUBMITTED = (archive["_ns"], root)
        agents = [{"slug": "kaggle_submitted", "path": str(args.submission.resolve())}]
    else:
        agents = json.loads(args.opponents.read_text())["agents"]
    if (not agents or len({row["slug"] for row in agents}) != len(agents) or
            any(not Path(row["path"]).is_file() for row in agents)):
        parser.error("invalid opponent manifest")
    jobs = [{"slug": row["slug"], "path": row["path"], "seed": seed,
             "seat": seat, "sample": args.sample,
             "submission": bool(args.submission),
             "checkpoint": str(args.checkpoint.resolve()),
             "binary": str(args.binary.resolve()),
             "manifest": str(args.manifest.resolve())}
            for row in agents
            for seed in range(args.seed_start, args.seed_start + args.seeds)
            for seat in (0, 1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    if partial.exists() and not args.resume:
        parser.error("partial results exist; pass --resume")
    rows = [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
    expected = {(job["slug"], job["seed"], job["seat"]) for job in jobs}
    complete = {(row["slug"], row["seed"], row["seat"])
                for row in rows if row["status"] == "PASS"}
    if (len(complete) != len(rows) or not complete <= expected):
        parser.error("partial results contain failures, duplicates, or wrong jobs")
    pending = [job for job in jobs if
               (job["slug"], job["seed"], job["seat"]) not in complete]
    started = time.perf_counter()
    with partial.open("a") as sink, Pool(
            processes=min(args.workers, max(1, len(pending))),
            initializer=torch.set_num_threads, initargs=(1,),
            maxtasksperchild=4) as pool:
        for row in pool.imap_unordered(_play, pending, chunksize=1):
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if len(rows) % 16 == 0 or len(rows) == len(jobs):
                print(f'{len(rows)}/{len(jobs)} completed', flush=True)
    rows.sort(key=lambda row: (row["slug"], row["seed"], row["seat"]))
    report = {
        "scope": "frozen_student_fastenv_vs_python_public",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "opponent_manifest": str((args.submission or args.opponents).resolve()),
        "opponents": {row["slug"]: hashlib.sha256(
            Path(row["path"]).read_bytes()).hexdigest() for row in agents},
        "sample": args.sample, "seed_start": args.seed_start,
        "seeds": args.seeds, "games": len(rows),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "seconds": time.perf_counter() - started,
        "by_opponent": {
            row["slug"]: {
                "games": sum(x["slug"] == row["slug"] for x in rows),
                "passed": sum(x["slug"] == row["slug"] and x["status"] == "PASS"
                              for x in rows),
                "wins": sum(x["slug"] == row["slug"] and x["status"] == "PASS"
                            and x["margin"] > 0 for x in rows),
            } for row in agents},
        "cases": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_opponent"], indent=2), flush=True)


if __name__ == "__main__":
    main()

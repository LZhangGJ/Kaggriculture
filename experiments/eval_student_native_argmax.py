#!/usr/bin/env python3
"""Evaluate one frozen student with C++ argmax actions and C++ opponents."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent import main as production
from experiments.native_student_actor.check_fieldcraft_job_batch_parity import decode
from experiments.run_strong_ab import trajectory_frame
from fast_kaggriculture import Config, FastEnv
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


ROOT = Path(__file__).resolve().parents[1]
OPPONENTS = {
    "thomas_2945_cpp": (1, "thomas_2945_cpp/thomas_2945.assets.bin"),
    "metav4_2965": (2, "metav4_2965/metav4_2965.assets.bin"),
    "salemali7_2900": (4, "salemali7_2900/salemali7_2900.assets.bin"),
    "fieldcraft_2887": (5, "fieldcraft_2887/fieldcraft_2887.assets.bin"),
    "soil_current": (6, "metav4_2965/soil_current.assets.bin"),
    "replay_clean_g397": (3, None),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_trajectory(path: Path, seed: int, opponent: str, seat: int,
                    own_trace, rival_trace, terminal, label: str,
                    diagnostic: bool, branch_step: int = -1,
                    branch_future_seed: int = 0) -> None:
    if path.exists():
        raise FileExistsError(path)
    if len(own_trace) != 719 or len(rival_trace) != 719:
        raise RuntimeError("native action trace is incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        env = FastEnv(Config(), seed)
        state = list(env.reset(seed))
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as output:
            output.write(json.dumps({"type": "meta", "format": "kaggriculture-bc-v1",
                                     "label": label, "bot": opponent, "seed": seed,
                                     "seat": seat, "engine": "fast",
                                     "diagnostic": diagnostic}, separators=(",", ":")) + "\n")
            for own, rival in zip(own_trace, rival_trace):
                if env.step_count == branch_step and branch_future_seed:
                    env.reseed_future(branch_future_seed)
                actions = [None, None]
                actions[seat] = decode(list(own))
                actions[1 - seat] = decode(list(rival))
                output.write(json.dumps(trajectory_frame(state, actions),
                                        separators=(",", ":")) + "\n")
                state = env.step(actions)
            rewards = tuple(map(float, env.rewards))
            if (not env.done or env.step_count != 719 or
                    rewards[seat] != float(terminal["own_cash"]) or
                    rewards[1 - seat] != float(terminal["rival_cash"])):
                raise RuntimeError(f"native trace terminal mismatch: {seed}/{opponent}/{seat}")
            output.write(json.dumps({"type": "terminal", "rewards": rewards,
                                     "final": trajectory_frame(state, [None, None])},
                                    separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate(args):
    selected = tuple(name.strip() for name in args.opponents.split(","))
    if not selected or len(set(selected)) != len(selected) or set(selected) - OPPONENTS.keys():
        raise ValueError("unknown or repeated C++ opponent")
    if (args.seeds < 1 or args.threads < 1 or args.trajectory_workers < 1 or
            args.output.exists()):
        raise ValueError("invalid count or existing output")
    if (args.branch_step != -1 and
            (not args.sample or args.branch_step not in range(288, 673, 24))):
        raise ValueError("branch step requires sampled student-day evaluation")
    if args.branch_salt < 0 or args.branch_salt >= 1 << 64 or (
            args.branch_step == -1 and args.branch_salt):
        raise ValueError("invalid branch salt")
    if (args.branch_future_seed < 0 or args.branch_future_seed >= 1 << 64 or
            (args.branch_step == -1 and args.branch_future_seed)):
        raise ValueError("invalid branch future seed")
    if args.force_alternative and (args.branch_step == -1 or args.branch_salt):
        raise ValueError("forced alternative needs a branch day and no salt")
    if args.force_alternative and args.save_trajectories:
        raise ValueError("forced alternative is diagnostic, not BC trajectory data")
    module_path = next(args.module_dir.glob("_paused_plan*.so"))
    spec = importlib.util.spec_from_file_location("_paused_plan", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = NativeTeammateBundle(
        ROOT / "agent/teammate_base.py",
        ROOT / "agent/route_actions.json.zlib",
        ROOT / "agent/route_library.json")
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    probe = production.policy.Agent(config=config, binary_path=args.binary)
    try:
        settings = [float(probe.config[key]) for key in
                    production.policy._ORDER[:probe.settings_count]]
    finally:
        probe.close()
    if args.student_intraday is not None:
        settings[production.policy._ORDER.index("intraday")] = args.student_intraday
    deployment_routes = [bundle.index(name) for name in
                         ("G275", "G195", "G024", "G316", "G267")]
    jobs = [(seed, name, seat)
            for seed in range(args.seed_start, args.seed_start + args.seeds)
            for name in selected for seat in (0, 1)]
    assets = ROOT / "experiments/native_opponents"
    batch = module.JobBatch(
        str(args.binary), bundle.executor,
        [seed for seed, _, _ in jobs], [seat for _, _, seat in jobs],
        [OPPONENTS[name][0] for _, name, _ in jobs],
        [bundle.index("G397") if name == "replay_clean_g397" else -1
         for _, name, _ in jobs],
        list(range(len(jobs))), settings, deployment_routes,
        *(str(assets / OPPONENTS[name][1]) for name in (
            "thomas_2945_cpp", "metav4_2965", "salemali7_2900",
            "fieldcraft_2887", "soil_current")))
    started = time.perf_counter()
    trajectory_dir = args.trajectory_dir or args.output.with_name(
        args.output.stem + "-trajectories")
    capture = args.save_trajectories
    batch.run(args.threads, 2 << 20, capture)
    suffix_args = (str(args.weights), 0, args.threads, 2 << 20,
                   capture, not args.sample, capture)
    if args.branch_step == -1:
        batch.run_native_actor_suffix(*suffix_args)
    else:
        branch_args = (args.branch_step, args.branch_salt,
                       args.force_alternative)
        if args.branch_future_seed:
            branch_args += (args.branch_future_seed,)
        batch.run_native_actor_suffix(*suffix_args, *branch_args)
    rows = list(batch.summary()["cases"])
    if len(rows) != len(jobs) or any(
            not row["done"] or row["step"] != 719 or row["error"]
            for row in rows):
        raise RuntimeError("native argmax evaluation did not finish cleanly")
    by_opponent = {}
    for name in selected:
        subset = [row for row, (_, opponent, _) in zip(rows, jobs)
                  if opponent == name]
        margins = [float(row["own_cash"] - row["rival_cash"])
                   for row in subset]
        by_opponent[name] = {
            "games": len(subset),
            "wins": sum(margin > 0 for margin in margins),
            "draws": sum(margin == 0 for margin in margins),
            "win_rate": sum(margin > 0 for margin in margins) / len(subset),
            "mean_margin": sum(margins) / len(subset),
        }
    if capture:
        scores = args.output.with_suffix(".scores.json")
        if scores.exists():
            raise FileExistsError(scores)
        scores.parent.mkdir(parents=True, exist_ok=True)
        scores.write_text(json.dumps({
            "status": "games_complete_trajectories_pending",
            "weights_sha256": digest(args.weights),
            "job_module_sha256": digest(module_path), "seed_start": args.seed_start,
            "seeds": args.seeds, "sample": args.sample, "games": len(rows),
            "by_opponent": by_opponent,
        }, indent=2) + "\n")
    label = args.output.stem
    paths = ([trajectory_dir / label / name / f"{seed}-seat{seat}.jsonl.gz"
              for seed, name, seat in jobs] if capture else [None] * len(jobs))
    if capture:
        traces = batch.action_traces()
        if any(path.exists() for path in paths):
            raise FileExistsError("evaluation trajectory already exists")
        tasks = [(path, seed, name, seat, own, rival, row, label,
                  args.branch_step != -1, args.branch_step,
                  args.branch_future_seed)
                 for path, (seed, name, seat), row, own, rival in zip(
                     paths, jobs, rows, traces["own"], traces["rival"])]
        with ThreadPoolExecutor(max_workers=args.trajectory_workers) as workers:
            list(workers.map(lambda task: save_trajectory(*task), tasks))
    report = {
        "scope": ("cpp_sample_cpp_opponents_same_seed_both_seats"
                  if args.sample else
                  "cpp_argmax_cpp_opponents_same_seed_both_seats"),
        "checkpoint_weights": str(args.weights.resolve()),
        "weights_sha256": digest(args.weights),
        "job_module_sha256": digest(module_path),
        "seed_start": args.seed_start, "seeds": args.seeds,
        "student_intraday": args.student_intraday,
        "branch_step": args.branch_step, "branch_salt": args.branch_salt,
        "branch_future_seed": args.branch_future_seed,
        "force_alternative": args.force_alternative,
        "games": len(rows), "seconds": time.perf_counter() - started,
        "trajectory_format": "kaggriculture-bc-v1" if capture else None,
        "trajectory_dir": str(trajectory_dir) if capture else None,
        "cases": [
            {"seed": seed, "opponent": name, "seat": seat,
             "own_cash": float(row["own_cash"]),
             "rival_cash": float(row["rival_cash"]),
             "actor_hash": int(row["actor_hash"]),
             "prefix_action_hash": int(row["prefix_action_hash"]),
             "branch_prefix_action_hash": int(row.get("branch_prefix_action_hash", 0)),
             "forced_cell": int(row.get("forced_cell", -1)),
             "forced_from": int(row.get("forced_from", -1)),
             "forced_to": int(row.get("forced_to", -1))}
             | {"trajectory": str(path) if path is not None else None}
            for row, (seed, name, seat), path in zip(rows, jobs, paths)
        ],
        "by_opponent": by_opponent,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "work/agent-student-actor-owned-v3.so")
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--opponents", default=",".join(
        name for name in OPPONENTS if name != "replay_clean_g397"))
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seeds", default=128, type=int)
    parser.add_argument("--threads", default=64, type=int)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--student-intraday", type=int, choices=(0, 1))
    parser.add_argument("--branch-step", type=int, default=-1)
    parser.add_argument("--branch-salt", type=int, default=0)
    parser.add_argument("--branch-future-seed", type=int, default=0)
    parser.add_argument("--force-alternative", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--trajectory-workers", type=int, default=1)
    parser.add_argument("--save-trajectories", action="store_true",
                        help="slow BC/audit archive; default evaluation stores scores only")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2), flush=True)


if __name__ == "__main__":
    main()

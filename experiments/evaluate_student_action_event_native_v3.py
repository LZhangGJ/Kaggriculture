#!/usr/bin/env python3
"""Paired deterministic v3 actor evaluation against native C++ opponents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.collect_student_state_dagger_v3 import _observations
from experiments.student_action_event_agent import StudentActionEventAgent, production
from experiments.train_midgame_student_v1 import _sha256
from experiments.train_student_action_event_rl_v3 import (
    NATIVE_OPPONENTS,
    STUDENT_STEPS,
    _load_native,
    _native_artifacts,
    _worker_init,
)


ROOT = Path(__file__).resolve().parents[1]


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("opponent list is empty")
    return result


def _candidate(job: dict, checkpoint: str):
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    dynamic = StudentActionEventAgent(
        config,
        binary_path=Path(job["binary"]),
        checkpoint_path=Path(checkpoint),
        manifest_path=Path(job["manifest"]),
        sample=False,
        student_steps=STUDENT_STEPS,
        allow_unattested_steps=False,
    )
    route = production.create_replay_agent(
        production.replay_deployment(),
        f'native_v3_pair_{job["seed"]}_{job["seat"]}',
    )
    return production.ReplayThenDynamicAgent(
        route, dynamic, 288, handoff_land=None, handoff_floor=0,
        handoff_delay_days=0, selector=None)


def _play(job: dict, checkpoint: str) -> dict:
    from fast_kaggriculture import Config, FastEnv

    module = _load_native(job["native_module_name"], job["native_module_path"])
    opponent = module.Opponent(job["native_asset_path"])
    candidate = _candidate(job, checkpoint)
    env = FastEnv(Config(), int(job["seed"]))
    state = list(env.reset(int(job["seed"])))
    actions_seen = []
    frames = 0
    started = time.perf_counter()
    try:
        while not env.done:
            actions = []
            for player, observation in enumerate(_observations(state)):
                if player == int(job["seat"]):
                    action = candidate(observation, {})
                    actions_seen.append(json.dumps(
                        action, sort_keys=True, separators=(",", ":")))
                else:
                    action = opponent.action(env, player)
                actions.append(action)
            state = env.step(actions)
            frames += 1
        summary = candidate.dynamic.student_summary()
        if (frames != 719 or
                summary.get("successful_steps") != list(STUDENT_STEPS) or
                summary.get("fallbacks") != 0 or summary.get("illegal") != 0 or
                candidate.dynamic.student_failures):
            raise RuntimeError(f"invalid deterministic actor game: {summary}")
        own = float(env.rewards[int(job["seat"])])
        rival = float(env.rewards[1 - int(job["seat"])])
        return {
            "cash": own,
            "opponent_cash": rival,
            "margin": own - rival,
            "outcome": int((own > rival) - (own < rival)),
            "events": int(summary["events"]),
            "class_counts": summary["class_counts"],
            "action_sha256": hashlib.sha256(
                "\n".join(actions_seen).encode()).hexdigest(),
            "_actions": actions_seen,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        candidate.close()


def _evaluate_pair(job: dict) -> dict:
    started = time.perf_counter()
    try:
        parent = _play(job, job["parent"])
        child = _play(job, job["child"])
        parent_actions = parent.pop("_actions")
        child_actions = child.pop("_actions")
        if len(parent_actions) != 719 or len(child_actions) != 719:
            raise RuntimeError("candidate action trace is incomplete")
        return {
            "status": "PASS",
            "seed": job["seed"], "opponent": job["opponent"],
            "seat": job["seat"],
            "parent": parent, "child": child,
            "delta_margin": child["margin"] - parent["margin"],
            "action_differences": sum(
                left != right for left, right in zip(parent_actions, child_actions)),
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "status": "FAIL", "seed": job["seed"],
            "opponent": job["opponent"], "seat": job["seat"],
            "error": repr(error), "elapsed_seconds": time.perf_counter() - started,
        }


def _summary(rows: list[dict]) -> dict:
    parent_wins = sum(row["parent"]["outcome"] > 0 for row in rows)
    child_wins = sum(row["child"]["outcome"] > 0 for row in rows)
    rescued = sum(row["parent"]["outcome"] <= 0 < row["child"]["outcome"]
                  for row in rows)
    broken = sum(row["child"]["outcome"] <= 0 < row["parent"]["outcome"]
                 for row in rows)
    return {
        "pairs": len(rows),
        "parent_wins": parent_wins,
        "child_wins": child_wins,
        "delta_wins": child_wins - parent_wins,
        "rescued": rescued,
        "broken": broken,
        "parent_mean_margin": float(np.mean(
            [row["parent"]["margin"] for row in rows])),
        "child_mean_margin": float(np.mean(
            [row["child"]["margin"] for row in rows])),
        "mean_delta_margin": float(np.mean(
            [row["delta_margin"] for row in rows])),
        "changed_action_pairs": sum(row["action_differences"] > 0 for row in rows),
        "changed_action_frames": sum(row["action_differences"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--opponents", type=_csv,
                        default=("thomas_2945_cpp", "metav4_2965"))
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.seeds < 1 or args.workers < 1:
        parser.error("seeds and workers must be positive")
    unknown = set(args.opponents) - set(NATIVE_OPPONENTS)
    if unknown:
        parser.error(f"only parity-gated native C++ opponents are allowed: {sorted(unknown)}")
    for path in (args.parent, args.child, args.manifest, args.binary):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.output.exists():
        parser.error(f"refusing to overwrite: {args.output}")

    parent_sha = _sha256(args.parent)
    child_sha = _sha256(args.child)
    manifest_sha = _sha256(args.manifest)
    binary_sha = _sha256(args.binary)
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    child = torch.load(args.child, map_location="cpu", weights_only=False)
    if (parent.get("shard_manifest_sha256") != manifest_sha or
            child.get("shard_manifest_sha256") != manifest_sha or
            tuple(parent.get("training_state_steps", ())) != STUDENT_STEPS or
            tuple(child.get("training_state_steps", ())) != STUDENT_STEPS or
            child.get("rl", {}).get("parent_checkpoint_sha256") != parent_sha):
        raise RuntimeError("parent/child/manifest lineage does not close")

    artifacts = _native_artifacts(args.opponents)
    jobs = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for opponent in args.opponents:
            artifact = artifacts[opponent]
            for seat in (0, 1):
                jobs.append({
                    "seed": seed, "opponent": opponent, "seat": seat,
                    "parent": str(args.parent.resolve()),
                    "child": str(args.child.resolve()),
                    "manifest": str(args.manifest.resolve()),
                    "binary": str(args.binary.resolve()),
                    "native_module_name": artifact["module_name"],
                    "native_module_path": artifact["module_path"],
                    "native_asset_path": artifact["asset_path"],
                })

    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs)),
            initializer=_worker_init) as pool:
        futures = [pool.submit(_evaluate_pair, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({
                "event": "paired_eval", "status": row["status"],
                "seed": row["seed"], "opponent": row["opponent"],
                "seat": row["seat"], "delta_margin": row.get("delta_margin"),
                "seconds": row["elapsed_seconds"], "error": row.get("error"),
            }), flush=True)
    rows.sort(key=lambda row: (row["seed"], row["opponent"], row["seat"]))
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"native paired evaluation failed closed: {failures}")

    report = {
        "status": "PASS",
        "schema": "v3_actor_native_parent_child_paired_eval_v1",
        "opponent_backend": "native_cpp",
        "policy": "deterministic_argmax",
        "parent": str(args.parent.resolve()), "parent_sha256": parent_sha,
        "child": str(args.child.resolve()), "child_sha256": child_sha,
        "manifest_sha256": manifest_sha, "binary_sha256": binary_sha,
        "seed_start": args.seed_start, "seeds": args.seeds, "seats": [0, 1],
        "native_opponent_artifacts": {
            name: {key: value for key, value in artifact.items()
                   if key.endswith("_sha256")}
            for name, artifact in artifacts.items()
        },
        "overall": _summary(rows),
        "by_opponent": {
            opponent: _summary([row for row in rows
                                if row["opponent"] == opponent])
            for opponent in args.opponents
        },
        "by_seat": {
            str(seat): _summary([row for row in rows if row["seat"] == seat])
            for seat in (0, 1)
        },
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: report[key] for key in
                      ("status", "overall", "by_opponent", "wall_seconds")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

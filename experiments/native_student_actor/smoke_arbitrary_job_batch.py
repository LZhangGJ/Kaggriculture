#!/usr/bin/env python3
"""Parity gate for cache-free native rollout jobs and PPO tensor capture."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from experiments.student_action_event_agent import production
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


ROOT = Path(__file__).resolve().parents[2]
ROLLOUT = ROOT / "experiments/native_student_rollout"
MANIFEST = ROOT / "work/native-student-rollout/prefix-cache-v1/manifest.json"
WEIGHTS = ROOT / "work/student-v1/native-r3-parent.bin"
THOMAS = ROOT / "experiments/native_opponents/thomas_2945_cpp/thomas_2945.assets.bin"
META = ROOT / "experiments/native_opponents/metav4_2965/metav4_2965.assets.bin"


def settings(binary: Path) -> list[float]:
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    agent = production.policy.Agent(config=config, binary_path=binary)
    try:
        return [float(agent.config[name]) for name in
                production.policy._ORDER[:agent.settings_count]]
    finally:
        agent.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--module-dir", type=Path, default=ROLLOUT / "build")
    parser.add_argument("--no-action-traces", action="store_true")
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/arbitrary-job-cache-parity-b4.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    sys.path.insert(0, str(args.module_dir))
    native = importlib.import_module("_paused_plan")

    bundle = NativeTeammateBundle(
        ROOT / "agent/teammate_base.py",
        ROOT / "agent/route_actions.json.zlib",
        ROOT / "agent/route_library.json")
    replay = bundle.executor
    family_index = bundle.family_index
    families = ("G275", "G195", "G024", "G316", "G267")
    deployment_routes = [family_index[name] for name in families]
    policy_seed = 2026092301
    jobs = native.JobBatch(
        manifest["binary"], replay,
        [int(row["seed"]) for row in cases],
        [int(row["seat"]) for row in cases],
        [int(row["opponent_code"]) for row in cases],
        [-1] * len(cases),
        [policy_seed + index for index in range(len(cases))],
        settings(Path(manifest["binary"])), deployment_routes,
        str(THOMAS), str(META))
    oracle = native.PrefixBatch(
        manifest["binary"], [row["cache"] for row in cases],
        str(THOMAS), str(META), str(ROOT / "agent/replay_deployment.json"))
    started = time.perf_counter()
    jobs.run(args.threads, 2 << 20, True)
    job_prefix_seconds = time.perf_counter() - started
    oracle.run(args.threads, 2 << 20, True)
    job_prefix = jobs.summary()["cases"]
    oracle_prefix = oracle.summary()["cases"]
    job_actions = jobs.action_traces()
    oracle_actions = oracle.action_traces()
    prefix_action_equal = (
        job_actions["own"] == oracle_actions["own"] and
        job_actions["rival"] == oracle_actions["rival"])
    prefix_mismatches = []
    for session in range(len(cases)):
        for side in ("own", "rival"):
            for step, (got, want) in enumerate(zip(
                    job_actions[side][session], oracle_actions[side][session])):
                if got != want:
                    prefix_mismatches.append({
                        "session": session, "side": side, "step": step,
                        "job": got, "oracle": want,
                    })
                    break
    prefix_state_equal = all(
        got["packed_sha256"] == want["packed_sha256"] and
        got["context_sha256"] == want["context_sha256"]
        for got, want in zip(job_prefix, oracle_prefix))
    switch_steps = [row["opening_switch_step"] for row in job_prefix]
    if args.prefix_only or not prefix_action_equal or not prefix_state_equal:
        result = {
            "status": "PASS" if prefix_action_equal and prefix_state_equal else "FAIL",
            "scope": "cache-free-job-prefix-parity",
            "prefix_action_equal": prefix_action_equal,
            "prefix_state_equal": prefix_state_equal,
            "prefix_first_mismatches": prefix_mismatches,
            "switch_steps": switch_steps,
            "job_prefix_seconds": job_prefix_seconds,
            "job": jobs.summary(),
            "oracle": oracle.summary(),
        }
    else:
        jobs.clear_action_traces()
        oracle.clear_action_traces()
        suffix_args = (str(args.weights), 0, args.threads, 2 << 20, True)
        job_metrics = jobs.run_native_actor_suffix(
            *suffix_args, False, False) if args.no_action_traces else (
                jobs.run_native_actor_suffix(*suffix_args))
        oracle_metrics = oracle.run_native_actor_suffix(
            str(args.weights), policy_seed, args.threads, 2 << 20, True)
        arrays = {name: np.asarray(value) for name, value in jobs.ppo_arrays().items()}
        job_terminal = jobs.summary()["cases"]
        oracle_terminal = oracle.summary()["cases"]
        actor_equal = jobs.actor_traces() == oracle.actor_traces()
        suffix_action_equal = jobs.action_traces() == oracle.action_traces()
        traces_skipped = (args.no_action_traces and all(
            not trace for side in jobs.action_traces().values()
            for trace in side))
        terminal_equal = all(
            got["own_cash"] == want["own_cash"] and
            got["rival_cash"] == want["rival_cash"] and
            got["action_hash"] == want["action_hash"]
            for got, want in zip(job_terminal, oracle_terminal))
        expected_days = len(cases) * 17
        offsets = arrays["day_event_offsets"]
        ppo_valid = (
            arrays["context"].shape == (expected_days, 2233) and
            arrays["observation"].shape[0] == expected_days and
            arrays["token_continuous"].shape == (expected_days, 320, 24) and
            arrays["token_categories"].shape == (expected_days, 7, 320) and
            offsets.shape == (expected_days + 1,) and offsets[0] == 0 and
            offsets[-1] == arrays["event_action"].shape[0] and
            np.isfinite(arrays["context"]).all() and
            np.isfinite(arrays["observation"]).all() and
            np.isfinite(arrays["event_resources"]).all() and
            np.isfinite(arrays["old_logprob"]).all() and
            np.isfinite(arrays["old_entropy"]).all() and
            np.all(arrays["event_legal"][
                np.arange(len(arrays["event_action"])), arrays["event_action"]]))
        result = {
            "status": "PASS" if all((prefix_action_equal, prefix_state_equal,
                                      actor_equal,
                                      traces_skipped if args.no_action_traces
                                      else suffix_action_equal,
                                      terminal_equal, ppo_valid)) else "FAIL",
            "scope": "cache-free-job-full-rollout-and-ppo-arrays-parity",
            "prefix_action_equal": prefix_action_equal,
            "prefix_state_equal": prefix_state_equal,
            "actor_trace_equal": actor_equal,
            "suffix_action_equal": suffix_action_equal,
            "traces_skipped": traces_skipped,
            "terminal_equal": terminal_equal,
            "ppo_arrays_valid": bool(ppo_valid),
            "switch_steps": switch_steps,
            "job_prefix_seconds": job_prefix_seconds,
            "job_suffix_seconds": job_metrics["wall_seconds"],
            "oracle_suffix_seconds": oracle_metrics["wall_seconds"],
            "sessions": len(cases),
            "days": expected_days,
            "events": int(offsets[-1]),
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "job": jobs.summary(),
            "oracle": oracle.summary(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

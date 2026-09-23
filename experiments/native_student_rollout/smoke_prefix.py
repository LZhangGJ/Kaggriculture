#!/usr/bin/env python3
"""Replay frozen prefixes in C++ and hand the live handles to PlanBatch."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.student_action_event_agent import policy


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=(
        ROOT / "work/native-student-rollout/prefix-cache-v1/manifest.json"))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/native-prefix-smoke.json"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sys.path.insert(0, str(HERE / "build"))
    native = importlib.import_module("_paused_plan")
    prefix = native.PrefixBatch(
        manifest["binary"], [row["cache"] for row in manifest["cases"]],
        str(ROOT / "experiments/native_opponents/thomas_2945_cpp/thomas_2945.assets.bin"),
        str(ROOT / "experiments/native_opponents/metav4_2965/metav4_2965.assets.bin"),
        str(ROOT / "agent/replay_deployment.json"))
    started = time.perf_counter()
    prefix.run(args.threads)
    prefix_seconds = time.perf_counter() - started
    prefix_summary = prefix.summary()
    expected = manifest["cases"]
    actual = prefix_summary["cases"]
    hashes_match = all(
        got["packed_sha256"] == want["packed_sha256"] and
        got["context_sha256"] == want["context_sha256"] and
        got["validated_steps"] == 288 and not got["error"]
        for got, want in zip(actual, expected))
    contexts = np.asarray(prefix.contexts).copy()
    observation_pack_equal = all(
        np.array_equal(np.asarray(values), np.asarray(
            list(policy._pack(current)), dtype=np.float64))
        for current, values in zip(prefix.observations, prefix.packed))
    plan = native.PlanBatch(
        manifest["binary"], prefix.handles, prefix.packed, False, 2 << 20)
    plan_context_equal = np.array_equal(contexts, np.asarray(plan.contexts))
    traces = [[] for _ in expected]
    rounds = 0
    try:
        while True:
            event = plan.collect_ready(60_000, True)
            indices = np.asarray(event["session_indices"], dtype=np.int32)
            if indices.size == 0:
                if bool(np.asarray(event["terminal"]).all()):
                    break
                raise TimeoutError(plan.summary())
            suggested = np.asarray(event["suggested"], dtype=np.int32)
            stages = np.asarray(event["stages"], dtype=np.int32)
            cells = np.asarray(event["cells"], dtype=np.int32)
            masks = np.asarray(event["legal_masks"], dtype=np.int32)
            for row, index in enumerate(indices):
                choice = int(suggested[row])
                mask = int(masks[row])
                if not mask & (1 << choice):
                    raise AssertionError("Oracle suggestion outside legal mask")
                traces[int(index)].append((
                    int(stages[row]), int(cells[row]), choice, mask))
            plan.apply(indices, suggested)
            rounds += 1
        plan.join()
        plan_summary = plan.summary()
    finally:
        plan.close()
    result = {
        "status": "PASS" if (
            prefix_summary["complete"] and hashes_match and
            observation_pack_equal and plan_context_equal and plan_summary["states"] ==
            ["done"] * len(expected)) else "FAIL",
        "scope": "native-prefix-to-paused-plan",
        "manifest": str(args.manifest.resolve()),
        "prefix_seconds": prefix_seconds,
        "cases": len(expected),
        "hashes_match_python_baseline": hashes_match,
        "observation_pack_equal": observation_pack_equal,
        "plan_context_equal": plan_context_equal,
        "plan_rounds": rounds,
        "event_counts": [len(trace) for trace in traces],
        "prefix": prefix_summary,
        "plan": plan_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

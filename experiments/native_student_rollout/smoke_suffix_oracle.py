#!/usr/bin/env python3
"""Run the native prefix/paused-planner/native-suffix seam to terminal."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=(
        ROOT / "work/native-student-rollout/prefix-cache-v1/manifest.json"))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/native-suffix-oracle-smoke.json"))
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
    days = []
    for step in range(288, 673, 24):
        if prefix.current_step != step:
            raise AssertionError((prefix.current_step, step))
        plan = native.PlanBatch(
            manifest["binary"], prefix.handles, prefix.packed, False)
        counts = [0] * len(manifest["cases"])
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
                for index in indices:
                    counts[int(index)] += 1
                plan.apply(indices, suggested)
                rounds += 1
            plan.join()
            summary = plan.summary()
            if summary["states"] != ["done"] * len(counts):
                raise RuntimeError(summary)
        finally:
            plan.close()
        prefix.advance_to(step + 24, args.threads)
        days.append({"step": step, "rounds": rounds, "event_counts": counts})
    prefix.advance_to(719, args.threads)
    summary = prefix.summary()
    cases = summary["cases"]
    result = {
        "status": "PASS" if (
            prefix.current_step == 719 and
            all(row["done"] and not row["error"] and
                row["suffix_steps"] == 431 for row in cases)) else "FAIL",
        "scope": "native-prefix-paused-oracle-native-suffix",
        "wall_seconds": time.perf_counter() - started,
        "days": days,
        "terminal": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

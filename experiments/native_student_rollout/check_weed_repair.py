#!/usr/bin/env python3
"""Regression: expired replay wheat must not erase a student's chosen crop."""

import argparse
import importlib
import json
import sys
from pathlib import Path

from agent import main as production
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, default=ROOT / "experiments/native_student_rollout/build")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module_dir))
    native = importlib.import_module("_paused_plan")
    binary = args.binary.resolve()
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    probe = production.policy.Agent(config=config, binary_path=binary)
    try:
        settings = [float(probe.config[key]) for key in
                    production.policy._ORDER[:probe.settings_count]]
    finally:
        probe.close()
    settings[production.policy._ORDER.index("intraday")] = 0
    bundle = NativeTeammateBundle(
        ROOT / "agent/teammate_base.py",
        ROOT / "agent/route_actions.json.zlib",
        ROOT / "agent/route_library.json")
    routes = [bundle.index(name) for name in ("G275", "G195", "G024", "G316", "G267")]
    assets = ROOT / "experiments/native_opponents"
    seeds = [2901000002, 2901000003]
    batch = native.JobBatch(
        str(binary), bundle.executor, seeds, [0, 0], [1, 1], [-1, -1], [0, 1],
        settings, routes,
        *(str(assets / path) for path in (
            "thomas_2945_cpp/thomas_2945.assets.bin",
            "metav4_2965/metav4_2965.assets.bin",
            "salemali7_2900/salemali7_2900.assets.bin",
            "fieldcraft_2887/fieldcraft_2887.assets.bin",
            "metav4_2965/soil_current.assets.bin")))
    batch.run(2, 2 << 20, False)
    batch.plan_native_actor(str(args.weights.resolve()), 0, 2, 2 << 20, True)
    batch.advance_to(305, 2, 2 << 20, True)
    crops = [row["farms"][0]["tiles"][0][6]["crop"] for row in batch.observations]
    assert crops == ["TOMATO", "STRAWBERRY"], crops
    print("PASS: expired cell-6 wheat became the two NN-selected crops")


if __name__ == "__main__":
    main()

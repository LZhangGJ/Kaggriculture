#!/usr/bin/env python3
"""Check a forced G275→DSM JobBatch prefix against the existing route executor."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from agent import main as production
from experiments.native_student_actor.smoke_replay_clean_job_batch import encode
from fast_kaggriculture import Config, FastEnv
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_dsm_prefix_native import state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3300771000)
    parser.add_argument("--route", choices=("D011", "D018"), default="D011")
    parser.add_argument("--opponent", default="G397")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_paused_plan", args.module)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    base = ROOT / "work/dsm-route-research-120/export/combined"
    bundle = NativeTeammateBundle(ROOT / "agent/teammate_base.py",
                                  base / "route-actions.json.zlib",
                                  base / "route-library.json")
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    config.update(max_land=4, intraday=0)
    probe = production.policy.Agent(config=config, binary_path=args.binary.resolve())
    try:
        settings = [float(probe.config[key]) for key in
                    production.policy._ORDER[:probe.settings_count]]
    finally:
        probe.close()
    paths = ROOT / "experiments/native_opponents"
    batch = native.JobBatch(
        str(args.binary.resolve()), bundle.executor, [args.seed, args.seed], [0, 1],
        [3, 3], [bundle.index(args.opponent)] * 2, [0, 1], settings,
        [bundle.index(name) for name in ("G275", "G195", "G024", "G316", "G267")],
        str(paths / "thomas_2945_cpp/thomas_2945.assets.bin"),
        str(paths / "metav4_2965/metav4_2965.assets.bin"), "", "", "",
        bundle.index(args.route), 144)
    batch.run(2, 2 << 20, True)
    traces = batch.action_traces()
    summaries = batch.summary()["cases"]
    assert all(row["error"] == "" and row["opening_switch_step"] == 144
               for row in summaries), summaries
    for seat in (0, 1):
        source = bundle.play("G275" if seat == 0 else args.opponent,
                             args.opponent if seat == 0 else "G275", args.seed,
                             switch_step=144,
                             switch_target=args.route, seat=seat,
                             capture_trace=True, stop_after_steps=288)
        assert len(source["trace"]) == 288
        env = FastEnv(Config(), args.seed)
        for step, pair in enumerate(source["trace"]):
            for side, player in (("own", seat), ("rival", 1 - seat)):
                actual = traces[side][seat][step]
                expected = encode(pair[player])
                assert actual == expected, (seat, step, side, actual, expected)
            env.step_raw(pair)
        expected_state = state(env.observation(seat), seat)
        actual_state = state(batch.observations[seat], seat)
        assert actual_state == expected_state, (seat, actual_state, expected_state)
        assert actual_state["land"] == 4, (seat, actual_state)
    print(f"PASS: {args.route} G275→DSM step144, {args.seed}, both seats, "
          "576 exact prefix action pairs and four-land handoff")


if __name__ == "__main__":
    main()

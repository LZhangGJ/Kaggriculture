#!/usr/bin/env python3
"""One-case check that Kaggle's observation encoder matches native RL input."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    from agent import main as production
    from experiments.student_action_event_agent import create_agent
    from meta_agent.src.native_teammate_executor import NativeTeammateBundle

    module_path = next((ROOT / "build/v306-parity-native").glob("_paused_plan*.so"))
    spec = importlib.util.spec_from_file_location("_paused_plan", module_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    bridge = ROOT / "work/agent-student-actor-owned-v3.so"
    weights = ROOT / "work/student-v1/native-economic-v1-v306-1536g.bin"
    bundle = NativeTeammateBundle(
        ROOT / "agent/teammate_base.py", ROOT / "agent/route_actions.json.zlib",
        ROOT / "agent/route_library.json")
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    probe = production.policy.Agent(config=config, binary_path=bridge)
    settings = [float(probe.config[name]) for name in
                production.policy._ORDER[:probe.settings_count]]
    probe.close()
    settings[production.policy._ORDER.index("intraday")] = 0
    assets = ROOT / "experiments/native_opponents"
    opponents = [str(assets / name) for name in (
        "thomas_2945_cpp/thomas_2945.assets.bin",
        "metav4_2965/metav4_2965.assets.bin",
        "salemali7_2900/salemali7_2900.assets.bin",
        "fieldcraft_2887/fieldcraft_2887.assets.bin",
        "metav4_2965/soil_current.assets.bin")]
    routes = [bundle.index(name) for name in
              ("G275", "G195", "G024", "G316", "G267")]
    batch = native.JobBatch(str(bridge), bundle.executor, [3000097008], [0],
                            [1], [-1], [0], settings, routes, *opponents)
    batch.run(1, 2 << 20, False)
    observation = batch.observations[0]
    batch.run_native_actor_suffix(str(weights), 0, 1, 2 << 20,
                                  True, False, False)
    arrays = batch.ppo_arrays()
    os.environ["STUDENT_CHECKPOINT"] = str(
        ROOT / "work/student-v1/v3-ppo-native-job-economic-v1-v306-1536g.pt")
    os.environ["STUDENT_R1_BINARY"] = str(bridge)
    os.environ["STUDENT_V3_MANIFEST"] = str(
        ROOT / "models/student-v45/manifest.json")
    student = create_agent(0).dynamic
    state = student._state_tensors(observation, [0.0] * 2233)
    continuous = state[3][0].numpy()
    categories = np.stack([value[0].numpy() for value in state[4]])
    assert int(state[-1][0]) == int(arrays["token_count"][0])
    np.testing.assert_array_equal(continuous, arrays["token_continuous"][0])
    np.testing.assert_array_equal(categories, arrays["token_categories"][0])
    np.testing.assert_allclose(state[1][0].numpy(), arrays["observation"][0],
                               rtol=0, atol=3e-7)
    print("v306 public observation, token and economic input parity: PASS")


if __name__ == "__main__":
    main()

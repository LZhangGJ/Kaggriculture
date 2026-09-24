#!/usr/bin/env python3
"""Check an isolated four-land student bridge on a real DSM step-288 state."""

import argparse
import ctypes
import json
from pathlib import Path

import orjson

from experiments.audit_teacher_batch import r1_module
from experiments.test_student_v3_semantics import ReleaseCallback, SlotCallback, bind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--execute-day", action="store_true")
    args = parser.parse_args()
    if args.execute_day and not args.checkpoint:
        parser.error("--execute-day requires --checkpoint")
    replay = orjson.loads(args.replay.read_bytes())
    if len(replay["steps"]) != 720:
        raise ValueError("incomplete replay")
    r1 = r1_module()
    config = json.loads((Path(__file__).parents[1] / "policy/r1/config.json").read_text())
    config.update(max_land=4, intraday=0)
    if args.checkpoint:
        if not args.manifest:
            parser.error("--checkpoint requires --manifest")
        from experiments.student_action_event_agent import StudentActionEventAgent
        agent = StudentActionEventAgent(
            config, binary_path=args.binary, checkpoint_path=args.checkpoint,
            manifest_path=args.manifest, student_steps=(288,))
    else:
        agent = r1.Agent(config=config, binary_path=args.binary)
    try:
        bind(agent.lib)
        for step in range(288):
            observation = replay["steps"][step][args.seat]["observation"]
            action = replay["steps"][step + 1][args.seat]["action"] or {}
            agent.observe_external(observation, action)
        observation = replay["steps"][288][args.seat]["observation"]
        farm = observation["farms"][args.seat]
        assert len(farm["unlocked_quadrants"]) == 4
        if args.checkpoint:
            result = agent._install_student_plan(observation)
            events = result["events"]
            fourth = [event for event in events if event["cell"] // 10 >= 5 and
                      event["cell"] % 10 >= 5]
            summary = {
                "status": "PASS", "step": 288, "events": len(events),
                "fourth_events": len(fourth),
                "fourth_placements": sum(event["selected_class"] >= 3 for event in fourth),
                "stop_cell": next((event["cell"] for event in events
                                   if event["selected_class"] == 0), None),
            }
            if args.execute_day:
                import fast_kaggriculture as fk
                from scripts.audit_macro_execution_quality import _make_config
                from experiments.audit_student_execution_v3 import landed

                env = fk.FastEnv(_make_config(fk, replay), int(replay["info"]["seed"]))
                for step in range(288):
                    env.step_raw([replay["steps"][step + 1][seat]["action"] or {}
                                  for seat in range(2)])
                before = env.observation(args.seat)
                actual_farm = before["farms"][args.seat]
                assert (actual_farm["money"] == farm["money"] and
                        actual_farm["tiles"] == farm["tiles"] and
                        actual_farm["unlocked_quadrants"] == farm["unlocked_quadrants"])
                for step in range(288, 312):
                    own = r1.Agent.__call__(agent, env.observation(args.seat))
                    rival = replay["steps"][step + 1][1 - args.seat]["action"] or {}
                    env.step_raw([own, rival] if args.seat == 0 else [rival, own])
                after = env.observation(args.seat)
                chosen = [event for event in fourth if event["selected_class"] >= 3]
                summary["fourth_landed"] = sum(landed(
                    after, event["cell"],
                    event["selected_class"] - 3 if event["selected_class"] < 8
                    else event["selected_class"] + 1, 12) for event in chosen)
                summary["frozen_rival_prefix"] = True
            print(json.dumps(summary))
            return
        packed = r1._pack(observation)
        if agent.lib.td_activate_external(agent.handle, packed, len(packed)):
            raise RuntimeError(agent.lib.td_debug(agent.handle).decode())
        events = {"release": 0, "placement": 0, "fourth": 0}

        @ReleaseCallback
        def release(_user, cell, _suggested, mask, _resources, width):
            assert width == 347 and mask & 0b110
            events["release"] += 1
            return 1 if mask & 0b010 else 2

        @SlotCallback
        def placement(_user, cell, _suggested, mask, _resources, width):
            assert width == 347 and mask & 0b010 and 0 <= cell < 100
            events["placement"] += 1
            events["fourth"] += cell // 10 >= 5 and cell % 10 >= 5
            return 1  # NONE: inspect feasibility without changing the planting policy.

        count = agent.lib.td_student_plan_v3_callback_observation(
            agent.handle, packed, len(packed), release, placement, None, None, 0)
        if count < 0:
            raise RuntimeError(agent.lib.td_debug(agent.handle).decode())
        assert count == events["release"] + events["placement"]
        assert events["fourth"] > 0
        print(json.dumps({"status": "PASS", "step": 288, "events": events}))
    finally:
        agent.close()


if __name__ == "__main__":
    main()

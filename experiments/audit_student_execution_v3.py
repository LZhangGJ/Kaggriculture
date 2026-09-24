#!/usr/bin/env python3
"""Read-only native audit of student choices, jobs, orders, and next-day landings."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.eval_student_native_argmax import OPPONENTS, ROOT
from experiments import build_midgame_action_events_v3 as contract
from experiments.student_economic_features_v1 import _unpack
from experiments.train_midgame_student_v1 import _canonical_observation


ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
         "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP")


def atoms(encoded):
    units, market = map(int, encoded[:2])
    if len(encoded) != 2 + 3 * (units + market):
        raise RuntimeError("malformed native action trace")
    rows = [tuple(map(int, encoded[i:i + 3]))
            for i in range(2, len(encoded), 3)]
    return rows[:units], rows[units:]


def tile(observation, cell):
    seat = int(observation["player"])
    return observation["farms"][seat]["tiles"][cell // 10][cell % 10]


def landed(observation, cell, kind, day):
    value = tile(observation, cell)
    if not isinstance(value, dict):
        return False
    if kind < 5:
        return (value.get("crop") == ITEMS[kind] and
                int(value.get("planted_day", -1)) == day)
    return (value.get("animal") == ITEMS[kind] and
            int(value.get("placed_day", -1)) == day)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, default=ROOT / "experiments/native_student_rollout/build")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--opponents", default="thomas_2945_cpp,metav4_2965,fieldcraft_2887")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--policy-seed-offset", type=int, default=0)
    parser.add_argument("--teacher-handoff", action="store_true",
                        help="compare R1 and student on the same step288 state")
    parser.add_argument("--capture-rollout", action="store_true",
                        help="save native PPO arrays without updating the model")
    parser.add_argument("--greedy", action="store_true",
                        help="use argmax for a captured diagnostic rollout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(args.opponents.split(","))
    if (args.seeds < 1 or args.threads < 1 or args.output.exists() or
            not selected or set(selected) - OPPONENTS.keys() or
            (args.greedy and not args.capture_rollout) or
            (args.capture_rollout and
             (args.teacher_handoff or args.output.suffix != ".npz"))):
        parser.error("invalid seeds, opponents, threads, or existing output")

    import importlib.util
    from agent import main as production
    from meta_agent.src.native_teammate_executor import NativeTeammateBundle

    module_path = next(args.module_dir.glob("_paused_plan*.so"))
    spec = importlib.util.spec_from_file_location("_paused_plan", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = NativeTeammateBundle(
        ROOT / "agent/teammate_base.py", ROOT / "agent/route_actions.json.zlib",
        ROOT / "agent/route_library.json")
    probe = production.policy.Agent(
        config=json.loads((ROOT / "policy/r1/config.json").read_text()),
        binary_path=args.binary)
    try:
        settings = [float(probe.config[key]) for key in
                    production.policy._ORDER[:probe.settings_count]]
    finally:
        probe.close()
    settings[production.policy._ORDER.index("intraday")] = 0
    routes = [bundle.index(name) for name in
              ("G275", "G195", "G024", "G316", "G267")]
    cases = [(seed, opponent, seat)
             for seed in range(args.seed_start, args.seed_start + args.seeds)
             for opponent in selected for seat in (0, 1)]
    assets = ROOT / "experiments/native_opponents"
    batch_args = (
        str(args.binary.resolve()), bundle.executor,
        [seed for seed, _, _ in cases], [seat for _, _, seat in cases],
        [OPPONENTS[name][0] for _, name, _ in cases], [-1] * len(cases),
        list(range(args.policy_seed_offset,
                   args.policy_seed_offset + len(cases))), settings, routes,
        *(str(assets / OPPONENTS[name][1]) for name in
          ("thomas_2945_cpp", "metav4_2965", "salemali7_2900",
           "fieldcraft_2887", "soil_current")))
    try:
        batch = module.JobBatch(*batch_args)
    except TypeError as error:
        if "incompatible constructor arguments" not in str(error):
            raise
        batch = module.JobBatch(*batch_args[:-1])
    lib = ctypes.CDLL(str(args.binary.resolve()))
    lib.td_student_candidate_job_actions_observation.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_candidate_job_actions_observation.restype = ctypes.c_int
    lib.td_student_live_targets.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_live_targets.restype = ctypes.c_int
    lib.td_student_live_plan_meta.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_live_plan_meta.restype = ctypes.c_int
    if args.teacher_handoff:
        contract.bind(lib)
        lib.td_clone.argtypes = [ctypes.c_void_p]
        lib.td_clone.restype = ctypes.c_void_p
        lib.td_delete.argtypes = [ctypes.c_void_p]
        lib.td_delete.restype = None
    batch.run(args.threads, 2 << 20, False)
    if batch.current_step != 288:
        raise RuntimeError("native handoff did not reach step288")
    if args.capture_rollout:
        batch.run_native_actor_suffix(str(args.weights.resolve()), 0,
                                      args.threads, 2 << 20, True,
                                      *([True] if args.greedy else []))
        if batch.current_step != 719:
            raise RuntimeError("native rollout did not reach terminal")
        arrays = batch.ppo_arrays()
        if not (np.array_equal(arrays["seed"], [seed for seed, _, _ in cases]) and
                np.array_equal(arrays["seat"], [seat for _, _, seat in cases]) and
                np.array_equal(arrays["opponent"],
                               [OPPONENTS[name][0] for _, name, _ in cases])):
            raise RuntimeError("native rollout identity drift")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **arrays)
        print(json.dumps({"scope": "read_only_native_rollout",
                          "games": len(cases), "days": len(arrays["day_step"]),
                          "events": len(arrays["event_action"]),
                          "output": str(args.output)}))
        return
    rows = []
    teacher_rows = []
    for step in range(288, 312 if args.teacher_handoff else 673, 24):
        before = list(batch.observations)
        traces_before = list(batch.actor_traces())
        action_counts = [len(part) for part in batch.action_traces()["own"]]
        if args.teacher_handoff:
            for (seed, opponent, seat), state, handle, observation in zip(
                    cases, batch.packed, batch.handles, before):
                packed_state = np.asarray(state, dtype=np.float64)
                actor_packed = np.asarray(production.policy._pack(
                    _canonical_observation(observation)), dtype=np.float64)
                shops = _unpack(packed_state)[3]
                if not len(shops):
                    raise RuntimeError("shop counterfactual requires an open shop")

                def teacher_plan(raw):
                    values = (ctypes.c_double * len(raw))(*raw)
                    teacher_handle = lib.td_clone(ctypes.c_void_p(handle))
                    canonical_handle = lib.td_clone(ctypes.c_void_p(handle))
                    if not teacher_handle or not canonical_handle:
                        if teacher_handle:
                            lib.td_delete(teacher_handle)
                        if canonical_handle:
                            lib.td_delete(canonical_handle)
                        raise RuntimeError("R1 handle clone failed")
                    try:
                        teacher_jobs = contract.actor_owned_teacher_jobs(
                            SimpleNamespace(lib=lib, handle=teacher_handle), values)
                        (events, projected, resources_exact, fixed_point,
                         stop_exact, controlled_exact) = contract.plan_from_teacher(
                            SimpleNamespace(lib=lib, handle=canonical_handle),
                            values, teacher_jobs)
                        if not all((resources_exact, fixed_point, stop_exact,
                                    controlled_exact)):
                            raise RuntimeError("R1 teacher action-event contract failed")
                        return {
                            "placements": contract.placement_map(projected),
                            "events": [(event["stage"], event["cell"],
                                        event["label"], event["mask"])
                                       for event in events]}
                    finally:
                        lib.td_delete(teacher_handle)
                        lib.td_delete(canonical_handle)

                changed = packed_state.copy()
                changed[-len(shops)] = 7 if shops[0] == 4 else 4
                teacher_rows.append({
                    "seed": seed, "opponent": opponent, "seat": seat,
                    "packed_integer_sha256": hashlib.sha256(
                        np.rint(packed_state).astype("<i4").tobytes()).hexdigest(),
                    "actor_packed_integer_sha256": hashlib.sha256(
                        np.rint(actor_packed).astype("<i4").tobytes()).hexdigest(),
                    "shop_before": int(shops[0]),
                    "shop_after": int(changed[-len(shops)]),
                    **teacher_plan(packed_state),
                    "counterfactual": teacher_plan(changed)})
        batch.plan_native_actor(str(args.weights.resolve()), 0,
                                args.threads, 2 << 20, True)
        traces_after = list(batch.actor_traces())
        packed = list(batch.packed)
        handles = list(batch.handles)
        plans = []
        for index, (state, handle) in enumerate(zip(packed, handles)):
            raw = (ctypes.c_int32 * 400)()
            values = np.asarray(state, dtype=np.float64)
            pointer = values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if lib.td_student_candidate_job_actions_observation(
                    ctypes.c_void_p(handle), -1, pointer, len(values),
                    raw, len(raw)) != 100:
                raise RuntimeError(f"job export failed at {step}, case {index}")
            jobs = np.asarray(raw, dtype=np.int32).reshape(100, 4).copy()
            targets = (ctypes.c_int32 * 100)()
            meta = (ctypes.c_int32 * 5)()
            if (lib.td_student_live_targets(ctypes.c_void_p(handle), targets, 100) != 100 or
                    lib.td_student_live_plan_meta(ctypes.c_void_p(handle), meta, 5) != 5):
                raise RuntimeError(f"target export failed at {step}, case {index}")
            choices = [(int(event[3]),
                        int(event[6]) - 3 if int(event[6]) <= 7 else int(event[6]) + 1)
                       for event in traces_after[index][len(traces_before[index]):]
                       if int(event[2]) == 1 and 3 <= int(event[6]) <= 10]
            if args.teacher_handoff:
                student_events = [(int(event[2]), int(event[3]), int(event[6]),
                                   int(event[5]))
                                  for event in traces_after[index][len(traces_before[index]):]]
                teacher = teacher_rows[index]
                matched = next((at for at, (expected, actual) in enumerate(
                    zip(teacher["events"], student_events))
                    if expected != actual), min(len(teacher["events"]), len(student_events)))
                teacher["student_placements"] = contract.placement_map(jobs)
                teacher["student_events"] = student_events
                teacher["common_event_prefix"] = matched
                teacher["full_event_match"] = teacher["events"] == student_events
                teacher["placement_match"] = (
                    teacher["placements"] == teacher["student_placements"])
            plans.append((jobs, choices, tuple(targets), tuple(meta)))
        batch.advance_to(step + 24, args.threads, 2 << 20, True)
        after = list(batch.observations)
        actions = list(batch.action_traces()["own"])
        for index, (seed, opponent, seat) in enumerate(cases):
            jobs, choices, targets, plan_meta = plans[index]
            counts = Counter()
            purchases = Counter()
            sales = Counter()
            placement_atoms = Counter()
            for encoded in actions[index][action_counts[index]:]:
                units, market = atoms(encoded)
                counts["route_moves"] += sum(op in (1, 2, 3, 4) for op, _, _ in units)
                counts["emitted_plant"] += sum(op == 8 for op, _, _ in units)
                counts["emitted_place_animal"] += sum(
                    op == 7 and item >= 9 for op, item, _ in units)
                counts["inventory_return"] += sum(
                    op == 7 and item < 9 for op, item, _ in units)
                placement_atoms.update(f"{op}:{ITEMS[item]}:{quantity}"
                                       for op, item, quantity in units
                                       if op in (7, 8))
                for op, item, quantity in market:
                    if op == 18:
                        counts["hire"] += quantity
                    elif op in (20, 21, 22):
                        purchases[f"{op}:{ITEMS[item]}"] += quantity
                    elif op == 23:
                        sales[ITEMS[item]] += quantity
            misses = []
            selected_by_kind = Counter()
            landed_by_kind = Counter()
            for cell, kind in choices:
                counts["nn_selected"] += 1
                selected_by_kind[ITEMS[kind]] += 1
                job_kind = int(jobs[cell, 1 if kind < 5 else 2])
                if job_kind == kind:
                    counts["job_present"] += 1
                else:
                    counts["job_missing"] += 1
                if landed(after[index], cell, kind, step // 24):
                    counts["landed"] += 1
                    landed_by_kind[ITEMS[kind]] += 1
                else:
                    misses.append({"cell": cell, "kind": ITEMS[kind],
                                   "job_kind": job_kind,
                                   "target_kind": int(targets[cell]),
                                   "before_tile": tile(before[index], cell)})
            selected = set(choices)
            unselected = []
            for cell in range(100):
                value = tile(after[index], cell)
                if not isinstance(value, dict):
                    continue
                item = value.get("crop", value.get("animal"))
                if item not in ITEMS:
                    continue
                kind = ITEMS.index(item)
                if landed(after[index], cell, kind, step // 24) and (cell, kind) not in selected:
                    unselected.append({"cell": cell, "kind": item,
                                       "job_kind": int(jobs[cell, 1 if kind < 5 else 2])})
            counts["not_landed"] = len(misses)
            counts["unselected_landed"] = len(unselected)
            if unselected:
                raise RuntimeError(f"hidden placement at {seed}/{opponent}/{seat}/{step}: {unselected}")
            if (counts["emitted_plant"] != sum(landed_by_kind[crop] for crop in ITEMS[:5]) or
                    counts["emitted_place_animal"] !=
                    sum(landed_by_kind[animal] for animal in ITEMS[9:])):
                raise RuntimeError(f"emitted placement did not match next-day board at {seed}/{step}")
            for crop in ITEMS[:5]:
                seed_used = (int(before[index]["private"]["seeds"][crop]) +
                             purchases[f"20:{crop}"] -
                             int(after[index]["private"]["seeds"][crop]))
                if seed_used != landed_by_kind[crop]:
                    raise RuntimeError(f"seed conservation failed at {seed}/{step}/{crop}")
            farm_before = before[index]["farms"][seat]
            farm_after = after[index]["farms"][seat]
            rows.append({"seed": seed, "opponent": opponent, "seat": seat,
                         "step": step, "cash_before": farm_before["money"],
                         "cash_after": farm_after["money"],
                         "hands_before": len(farm_before["hands"]),
                         "hands_after": len(farm_after["hands"]),
                         "seeds_before": dict(before[index]["private"]["seeds"]),
                         "seeds_after": dict(after[index]["private"]["seeds"]),
                         "selected_by_kind": dict(selected_by_kind),
                         "landed_by_kind": dict(landed_by_kind),
                         "plan_meta": list(plan_meta),
                         "counts": dict(counts), "purchases": dict(purchases),
                         "sales": dict(sales), "placement_atoms": dict(placement_atoms),
                         "misses": misses,
                         "unselected": unselected})
    totals = Counter()
    for row in rows:
        totals.update(row["counts"])
    report = {"scope": "read_only_student_execution", "weights": str(args.weights),
              "binary": str(args.binary), "student_intraday": 0,
              "cases": len(cases), "days": len(rows), "totals": dict(totals),
              "rows": rows}
    if args.teacher_handoff:
        report["teacher_handoff"] = teacher_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("scope", "cases", "days", "totals")}))


if __name__ == "__main__":
    main()

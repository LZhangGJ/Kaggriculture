#!/usr/bin/env python3
"""Audit whether the nine-way slot actor can represent the executed R1 plan."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.audit_teacher_batch import r1_module
from experiments.build_midgame_live_slots_v2 import _bind_live, _load_jobs
from experiments.train_midgame_autofill_v1 import _bind_slots
from scripts.extract_midgame_bc import observation


ROOT = Path(__file__).resolve().parents[1]
FINITE = {"WHEAT", "CARROT", "MELON"}


def bind(lib):
    _bind_slots(lib)
    _bind_live(lib)
    lib.td_student_live_targets.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_live_targets.restype = ctypes.c_int
    lib.td_student_live_release.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_live_release.restype = ctypes.c_int
    lib.td_student_live_plan_meta.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_student_live_plan_meta.restype = ctypes.c_int
    lib.td_student_candidate_plan_audit.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
    lib.td_student_candidate_plan_audit.restype = ctypes.c_int


def slot_snapshot(lib, handle, candidate, width):
    if candidate is None:
        count = lib.td_student_live_slot_count(handle)
        meta_fn = lambda out: lib.td_student_live_slot_meta(handle, out, len(out))
        resource_fn = lambda out: lib.td_student_live_slot_resources(handle, out, len(out))
    else:
        count = lib.td_student_candidate_slot_count(handle, candidate)
        meta_fn = lambda out: lib.td_student_candidate_slot_meta(
            handle, candidate, out, len(out))
        resource_fn = lambda out: lib.td_student_candidate_slot_resources(
            handle, candidate, out, len(out))
    if count < 0:
        raise RuntimeError("slot count export failed")
    meta = (ctypes.c_int32 * (4 * count))()
    resources = (ctypes.c_double * (width * count))()
    if meta_fn(meta) != count or resource_fn(resources) != count:
        raise RuntimeError("slot snapshot export failed")
    rows = [tuple(map(int, meta[4 * i:4 * i + 4])) for i in range(count)]
    values = np.asarray(resources, dtype=np.float64).reshape(count, width)
    return rows, values


def plan_snapshot(lib, handle, candidate):
    targets = (ctypes.c_int32 * 100)()
    release = (ctypes.c_int32 * 100)()
    meta = (ctypes.c_int32 * 5)()
    if candidate is None:
        calls = (
            lib.td_student_live_targets(handle, targets, len(targets)),
            lib.td_student_live_release(handle, release, len(release)),
            lib.td_student_live_plan_meta(handle, meta, len(meta)),
        )
        if calls != (100, 100, 5):
            raise RuntimeError("live plan audit export failed")
    elif lib.td_student_candidate_plan_audit(
            handle, candidate, targets, release, meta):
        raise RuntimeError("candidate plan audit export failed")
    return tuple(targets), tuple(release), tuple(meta)


def resource_family(index):
    if index < 5:
        return "meta"
    if index < 17:
        return "stock"
    return ("flow" if (index - 17) % 11 < 9 else
            "labor" if (index - 17) % 11 == 9 else "fixed")


def tile_at(observation_value, seat, cell):
    farm = observation_value["farms"][seat]
    tile = farm["tiles"][cell // 10][cell % 10]
    return tile if isinstance(tile, dict) else {}


def tile_category(observation_value, seat, cell):
    farm = observation_value["farms"][seat]
    tile = farm["tiles"][cell // 10][cell % 10]
    if isinstance(tile, str):
        return tile.lower()
    if not isinstance(tile, dict):
        return "empty"
    if tile.get("animal"):
        return "animal"
    if tile.get("kind") == "PLANT":
        crop = tile.get("crop", "unknown")
        return (f"finite_{crop.lower()}" if crop in FINITE else
                f"ongoing_{str(crop).lower()}")
    return str(tile.get("kind", "empty")).lower()


def kind_bit(kind):
    if kind == -1:
        return 0
    if 0 <= kind < 5:
        return kind + 1
    if 9 <= kind < 12:
        return kind - 3
    return -1


def audit_game(job):
    module = r1_module()
    agent = module.Agent(binary_path=Path(job["binary"]))
    lib = agent.lib
    bind(lib)
    names = lib.td_student_slot_resource_names().decode().splitlines()
    width = len(names)
    counts = Counter()
    maxima = Counter()
    examples = []
    steps = []
    terminal = 0
    try:
        with gzip.open(job["trajectory"], "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("type") == "terminal":
                    terminal += 1
                    continue
                if row.get("type") != "step":
                    continue
                step = int(row["step"])
                steps.append(step)
                seat = int(job["seat"])
                current = observation(row, seat)
                expected = row["actions"][seat]
                if step < 288:
                    agent.observe_external(current, expected)
                    continue
                base = None
                if step % 24 == 0:
                    packed = module._pack(current)
                    if agent.external:
                        if lib.td_activate_external(agent.handle, packed, len(packed)):
                            raise RuntimeError(lib.td_debug(agent.handle).decode())
                        agent.external = False
                    count = lib.td_student_prepare_prefix_observation(
                        agent.handle, packed, len(packed), None, None, 0, None, 0)
                    if count < 0:
                        raise RuntimeError(lib.td_debug(agent.handle).decode())
                    base = (*slot_snapshot(lib, agent.handle, 0, width),
                            *plan_snapshot(lib, agent.handle, 0))
                actual = agent(current)
                if actual != expected:
                    raise RuntimeError(f"action parity mismatch at step {step}")
                counts["action_frames"] += 1
                if base is None:
                    continue
                base_slots, base_resources, base_targets, base_release, base_meta = base
                winner_slots, winner_resources = slot_snapshot(lib, agent.handle, None, width)
                winner_targets, winner_release, winner_meta = plan_snapshot(
                    lib, agent.handle, None)
                counts["states"] += 1
                counts[f"planned_land_base_{base_meta[0]}"] += 1
                counts[f"planned_land_winner_{winner_meta[0]}"] += 1
                if base_meta[0] != winner_meta[0]:
                    counts["planned_land_diff_states"] += 1

                release_diff = False
                for cell in range(100):
                    tile = tile_at(current, seat, cell)
                    crop = tile.get("crop")
                    if tile.get("kind") != "PLANT" or crop not in FINITE:
                        continue
                    base_today = base_release[cell] == step // 24
                    winner_today = winner_release[cell] == step // 24
                    counts[f"finite_{crop}_eligible"] += 1
                    counts[f"finite_{crop}_base_release_today"] += base_today
                    counts[f"finite_{crop}_winner_release_today"] += winner_today
                    if base_today != winner_today:
                        release_diff = True
                        counts[f"finite_{crop}_release_diff"] += 1
                        counts[f"finite_{crop}_{'winner_only' if winner_today else 'base_only'}"] += 1
                counts["finite_release_diff_states"] += release_diff

                base_cells = {slot[0] for slot in base_slots}
                winner_cells = {slot[0] for slot in winner_slots}
                for cell in winner_cells - base_cells:
                    counts[f"winner_only_tile_{tile_category(current, seat, cell)}"] += 1
                for cell in base_cells - winner_cells:
                    counts[f"base_only_tile_{tile_category(current, seat, cell)}"] += 1
                counts["winner_only_cells"] += len(winner_cells - base_cells)
                counts["base_only_cells"] += len(base_cells - winner_cells)

                intents = [slot for slot in winner_slots if slot[1] >= 0]
                deleted = rewritten = admitted = missing = 0
                projection_illegal = 0
                for cell, kind, mask, _terminal in intents:
                    final = winner_targets[cell]
                    if final == kind:
                        admitted += 1
                    elif final == -1:
                        deleted += 1
                    elif final == -2:
                        missing += 1
                    else:
                        rewritten += 1
                    bit = kind_bit(final)
                    projection_illegal += bit < 0 or not (mask & (1 << bit))
                counts["intent_slots"] += len(intents)
                counts["intent_admitted"] += admitted
                counts["intent_deleted"] += deleted
                counts["intent_rewritten"] += rewritten
                counts["intent_missing_target"] += missing
                counts["intent_final_projection_illegal"] += projection_illegal
                counts["intent_final_projection_legal"] += len(intents) - projection_illegal
                post_changed = deleted + rewritten + missing > 0
                counts["post_greedy_changed_states"] += post_changed
                counts["preview_removed_counter"] += winner_meta[4]
                counts["preview_changed_counter_states"] += winner_meta[4] > 0

                same_meta = winner_slots == base_slots
                counts["same_slot_meta_states"] += same_meta
                if same_meta and winner_resources.shape == base_resources.shape:
                    difference = np.abs(winner_resources - base_resources)
                    exact = not np.any(difference)
                    counts["same_meta_resource_exact_states"] += exact
                    counts["same_meta_resource_diff_states"] += not exact
                    for index in range(width):
                        family = resource_family(index)
                        column = difference[:, index]
                        counts[f"resource_{family}_nonzero"] += int(np.count_nonzero(column))
                        maxima[f"resource_{family}_max_abs"] = max(
                            maxima[f"resource_{family}_max_abs"],
                            float(column.max(initial=0.0)))
                if (release_diff or base_meta[0] != winner_meta[0] or post_changed) and len(examples) < 4:
                    examples.append({
                        "seed": job["seed"], "opponent": job["opponent"],
                        "seat": seat, "step": step,
                        "base_cells": [slot[0] for slot in base_slots],
                        "winner_cells": [slot[0] for slot in winner_slots],
                        "planned_land": [base_meta[0], winner_meta[0]],
                        "release_diff": release_diff,
                        "intent": {"count": len(intents), "admitted": admitted,
                                   "deleted": deleted, "rewritten": rewritten,
                                   "missing": missing},
                        "preview_removed_counter": winner_meta[4],
                    })
        if steps != list(range(719)) or terminal != 1:
            raise RuntimeError("trajectory completeness failure")
        counts["games"] = 1
        return dict(counts), dict(maxima), examples
    finally:
        agent.close()


def worker_init():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/bc/midgame-v1")
    parser.add_argument("--policy-manifest", default="policy-deployment-bd49e2a13e44.json")
    parser.add_argument("--binding", default="corpus-policy-binding-bd49e2a13e44.json")
    parser.add_argument("--manifest", action="append")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--max-games", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(160, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs, source = _load_jobs(args)
    for job in jobs:
        job["binary"] = str(args.binary.resolve())
    totals, maxima, examples = Counter(), Counter(), []
    worker_init()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
        futures = [pool.submit(audit_game, job) for job in jobs]
        for future in as_completed(futures):
            counts, maximum, rows = future.result()
            totals.update(counts)
            for key, value in maximum.items():
                maxima[key] = max(maxima[key], value)
            if len(examples) < 24:
                examples.extend(rows[:24 - len(examples)])
    states = totals["states"]
    intents = totals["intent_slots"]
    result = {
        "status": "PASS", "audit": "student-action-contract-v1",
        "binary": str(args.binary.resolve()), "source": source,
        "counts": dict(sorted(totals.items())), "maxima": dict(sorted(maxima.items())),
        "rates": {
            "planned_land_diff_state": totals["planned_land_diff_states"] / states,
            "finite_release_diff_state": totals["finite_release_diff_states"] / states,
            "post_greedy_changed_state": totals["post_greedy_changed_states"] / states,
            "intent_admitted": totals["intent_admitted"] / intents if intents else 1.0,
            "intent_deleted": totals["intent_deleted"] / intents if intents else 0.0,
            "intent_rewritten": totals["intent_rewritten"] / intents if intents else 0.0,
        },
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      "games": totals["games"], "states": states,
                      **result["rates"], **dict(maxima)}))


if __name__ == "__main__":
    main()

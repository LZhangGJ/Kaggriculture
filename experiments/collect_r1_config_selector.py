#!/usr/bin/env python3
"""Collect public-observation features and paired one-day config suffixes."""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from run_r1_candidate_oracle import run
from run_strong_ab import BOTS

CLASSES = ("default", "competitive_sale", "crop_succession")


def choose(proposals, include_base=False):
    classes = CLASSES + (("base_plan",) if include_base else ())
    bases = [row for row in proposals if not row.get("diagnostic")]
    selected = {"default": max(bases, key=lambda row: row["score"])} if bases else {}
    if include_base:
        base_plan = [row for row in bases if row.get("id") == 0]
        if len(base_plan) == 1:
            selected["base_plan"] = base_plan[0]
    for row in proposals:
        name = row.get("diagnostic") or ""
        if name in CLASSES and name not in selected:
            selected[name] = row
    if set(selected) != set(classes):
        raise ValueError(f"missing selector classes: {sorted(set(classes) - set(selected))}")
    if any(len(row.get("features", ())) != 356 for row in selected.values()):
        raise ValueError("selector candidates require 356 features")
    return selected


def self_check():
    rows = [
        {"id": 0, "diagnostic": "", "index": 0, "score": 1.0, "features": [0.0] * 356},
        {"id": 1, "diagnostic": "", "index": 2, "score": 2.0, "features": [0.5] * 356},
        {"id": 101, "diagnostic": "competitive_sale", "index": 4, "features": [1.0] * 356},
        {"id": 100, "diagnostic": "crop_succession", "index": 3, "features": [2.0] * 356},
    ]
    selected = choose(rows)
    matrix = np.asarray([selected[name]["features"] for name in CLASSES], dtype=np.float32)
    assert matrix.shape == (3, 356) and [selected[name]["index"] for name in CLASSES] == [2, 4, 3]
    extended = choose(rows, include_base=True)
    classes = CLASSES + ("base_plan",)
    matrix = np.asarray([extended[name]["features"] for name in classes], dtype=np.float32)
    assert matrix.shape == (4, 356) and [extended[name]["index"] for name in classes] == [2, 4, 3, 0]
    print(json.dumps({"status": "PASS", "classes": classes, "shape": matrix.shape}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2609900000)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--days", default="11,15")
    parser.add_argument("--opponents", default=",".join(BOTS))
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--include-base", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if not args.output or args.output.exists():
        parser.error("--output must be a new .npz path")
    bots = [name for name in args.opponents.split(",") if name]
    days = [value if value == "handoff" else int(value)
            for value in args.days.split(",") if value]
    if (args.seeds < 1 or args.workers < 1 or not days or len(bots) != len(set(bots)) or
            any(name not in BOTS for name in bots)):
        parser.error("invalid seeds/workers/days/opponents")
    groups = [(bot, seed, seat, day) for bot in bots
              for seed in range(args.start, args.start + args.seeds)
              for seat in (0, 1) for day in days]
    context = mp.get_context("spawn")
    discovery = [(*group, None, True) for group in groups]
    started = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        discovered = list(pool.map(run, discovery, chunksize=1))
    discovery_seconds = time.perf_counter() - started
    print(json.dumps({"stage": "discovery", "games": len(discovered),
                      "seconds": discovery_seconds}), flush=True)
    classes = CLASSES + (("base_plan",) if args.include_base else ())
    selected = [choose(row["proposals"], args.include_base) if not row["error"] else None
                for row in discovered]
    variant = lambda name: "auto" if name == "default" else "" if name == "base_plan" else name
    tasks = [(*group, "auto" if name == "default" else selected[index][name]["index"])
             for index, group in enumerate(groups) if selected[index] is not None for name in classes]
    started = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        suffixes = list(pool.map(run, tasks, chunksize=1))
    suffix_seconds = time.perf_counter() - started
    print(json.dumps({"stage": "suffix", "games": len(suffixes),
                      "seconds": suffix_seconds}), flush=True)
    outcomes = {(row["bot"], row["seed"], row["seat"], row["target"], row["variant"]): row
                for row in suffixes if not row["error"]}
    keep = []
    for index, group in enumerate(groups):
        if selected[index] is None:
            continue
        rows = [outcomes.get((*group, variant(name))) for name in classes]
        if all(rows) and len({row["day"] for row in rows}) == 1:
            keep.append((group, selected[index], rows))
    errors = sum(bool(row["error"]) for row in discovered + suffixes)
    if errors or len(keep) != len(groups):
        raise RuntimeError(f"incomplete collection: errors={errors}, groups={len(keep)}/{len(groups)}")
    features = np.asarray([[entry[name]["features"] for name in classes]
                           for _, entry, _ in keep], dtype=np.float32)
    scores = np.asarray([[entry[name]["score"] for name in classes]
                         for _, entry, _ in keep], dtype=np.float64)
    horizon_scores = np.asarray([[entry[name]["scores_horizon"] for name in classes]
                                 for _, entry, _ in keep], dtype=np.float64)
    margins = np.asarray([[row["margin"] for row in rows] for _, _, rows in keep], dtype=np.float32)
    labels = np.asarray([max(range(len(classes)), key=lambda i: (margins[j, i] > 0, margins[j, i]))
                         for j in range(len(keep))], dtype=np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features, scores=scores,
                        horizon_scores=horizon_scores, margins=margins, labels=labels,
                        classes=np.asarray(classes), bots=np.asarray([g[0] for g, _, _ in keep]),
                        seeds=np.asarray([g[1] for g, _, _ in keep], dtype=np.int64),
                        seats=np.asarray([g[2] for g, _, _ in keep], dtype=np.int8),
                        days=np.asarray([rows[0]["day"] for _, _, rows in keep], dtype=np.int8),
                        targets=np.asarray([str(g[3]) for g, _, _ in keep]),
                        default_base_indices=np.asarray([entry["default"]["index"]
                                                         for _, entry, _ in keep], dtype=np.int8),
                        auto_debug=np.asarray([json.dumps(rows[0]["auto_debug"], separators=(",", ":"))
                                               for _, _, rows in keep]))
    handoff_days = [rows[0]["day"] for group, _, rows in keep if group[3] == "handoff"]
    handoff_distribution = {str(day): handoff_days.count(day) for day in sorted(set(handoff_days))}
    joint_switches = [rows[0]["auto_debug"].get("joint_switches", 0) for _, _, rows in keep]
    summary = {"groups": len(keep), "requested_groups": len(groups), "suffix_games": len(suffixes),
               "errors": errors, "feature_shape": list(features.shape), "classes": classes,
               "discovery_seconds": discovery_seconds, "suffix_seconds": suffix_seconds,
               "handoff_day_distribution": handoff_distribution,
               "auto_joint_switch_groups": sum(value > 0 for value in joint_switches)}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

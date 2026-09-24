#!/usr/bin/env python3
"""Paired diagnostic of four-land prefixes with an unchanged three-land student."""

import argparse
import importlib.util
import json
from pathlib import Path

from agent import main as production
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


ROOT = Path(__file__).resolve().parents[1]
OPPONENTS = {"thomas": 1, "meta": 2, "fieldcraft": 5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=3300790000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--max-land", type=int, choices=(3, 4), default=4)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_paused_plan", args.module)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    base = ROOT / "work/dsm-route-research-120/export/combined"
    bundle = NativeTeammateBundle(ROOT / "agent/teammate_base.py",
                                  base / "route-actions.json.zlib",
                                  base / "route-library.json")
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    config.update(max_land=args.max_land, intraday=0)
    probe = production.policy.Agent(config=config, binary_path=args.binary.resolve())
    try:
        settings = [float(probe.config[key]) for key in
                    production.policy._ORDER[:probe.settings_count]]
    finally:
        probe.close()
    asset = ROOT / "experiments/native_opponents"
    assets = [str(asset / name) for name in (
        "thomas_2945_cpp/thomas_2945.assets.bin",
        "metav4_2965/metav4_2965.assets.bin",
        "salemali7_2900/salemali7_2900.assets.bin",
        "fieldcraft_2887/fieldcraft_2887.assets.bin",
        "metav4_2965/soil_current.assets.bin")]
    deployment = [bundle.index(name) for name in
                  ("G275", "G195", "G024", "G316", "G267")]
    jobs = [(seed, seat, name, code) for seed in range(
        args.seed_start, args.seed_start + args.seeds)
        for name, code in OPPONENTS.items() for seat in (0, 1)]
    rows = []
    for arm in (("baseline",) if args.baseline_only else
                ("baseline", "D011", "D018")):
        route = -1 if arm == "baseline" else bundle.index(arm)
        batch_args = [
            str(args.binary.resolve()), bundle.executor,
            [seed for seed, _, _, _ in jobs], [seat for _, seat, _, _ in jobs],
            [code for _, _, _, code in jobs], [-1] * len(jobs),
            list(range(len(jobs))), settings, deployment, *assets]
        if args.max_land == 4:
            batch_args.extend((route, -1 if route < 0 else 144))
        batch = native.JobBatch(*batch_args)
        batch.run(args.threads, 2 << 20, False)
        prefix = batch.summary()["cases"]
        if any(row["error"] for row in prefix):
            raise RuntimeError([row["error"] for row in prefix if row["error"]][:3])
        land = [len(obs["farms"][seat]["unlocked_quadrants"])
                for obs, (_, seat, _, _) in zip(batch.observations, jobs)]
        if args.prefix_only:
            for (seed, seat, opponent, _), plots, result in zip(jobs, land, prefix):
                rows.append({"arm": arm, "seed": seed, "seat": seat,
                             "opponent": opponent, "land288": plots,
                             "fourth_purchase_state_step": result["fourth_purchase_state_step"],
                             "prefix_action_hash": result["prefix_action_hash"]})
            print(f"{arm}: four land {sum(value == 4 for value in land)}/{len(land)}, "
                  f"purchase state steps {sorted(set(row['fourth_purchase_state_step'] for row in prefix))}",
                  flush=True)
            continue
        suffix = batch.run_native_actor_suffix(
            str(args.weights.resolve()), 0, args.threads, 2 << 20,
            False, args.greedy, False)
        terminal = batch.summary()["cases"]
        if any(row["error"] or row["step"] != 719 for row in terminal):
            raise RuntimeError([(row["seed"], row["seat"], row["error"], row["step"])
                                for row in terminal if row["error"] or row["step"] != 719][:3])
        for (seed, seat, opponent, _), plots, result in zip(jobs, land, terminal):
            rows.append({"arm": arm, "seed": seed, "seat": seat,
                         "opponent": opponent, "land288": plots,
                         "own_cash": result["own_cash"],
                         "rival_cash": result["rival_cash"],
                         "margin": result["own_cash"] - result["rival_cash"],
                         "win": result["own_cash"] > result["rival_cash"],
                         "prefix_action_hash": result["prefix_action_hash"],
                         "actor_days": result["actor_days"]})
        print(f"{arm}: {sum(row['win'] for row in rows if row['arm'] == arm)}/{len(jobs)} "
              f"wins, four land {sum(value == 4 for value in land)}/{len(land)}, "
              f"suffix {suffix['wall_seconds']:.1f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed_start": args.seed_start,
                                       "seeds": args.seeds, "greedy": args.greedy,
                                       "max_land": args.max_land,
                                       "prefix_only": args.prefix_only,
                                       "rows": rows}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

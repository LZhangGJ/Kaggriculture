#!/usr/bin/env python3
"""FastEnv exploratory league for isolated public opponents; not formal validation."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import importlib.util
import inspect
import json
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
from experiments.run_strong_ab import observations, trajectory_frame


def load(path: str, name: str):
    path_obj = Path(path).resolve()
    import sys
    sys.path.insert(0, str(path_obj.parent))
    spec = importlib.util.spec_from_file_location(name, path_obj)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quadrant_animals(farm: dict) -> dict[str, dict[str, int]]:
    half = len(farm["tiles"]) // 2
    result = {name: Counter() for name in ("NW", "NE", "SW", "SE")}
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("animal"):
                quadrant = ("N" if y < half else "S") + ("W" if x < half else "E")
                result[quadrant][str(tile["animal"])] += 1
    return {name: dict(counts) for name, counts in result.items()}


def play(task):
    (candidate_path, slug, opponent_path, source_family, opponent_source, seed, seat,
     trajectory_dir, policy_source, policy_manifest_canonical_sha256,
     policy_manifest_file_sha256, diagnostic) = task
    from fast_kaggriculture import Config, FastEnv

    identity = f"{os.getpid()}_{slug}_{seed}_{seat}"
    candidate_module = load(candidate_path, f"candidate_{identity}")
    candidate = candidate_module.create_agent() if hasattr(candidate_module, "create_agent") else candidate_module.agent
    opponent = load(opponent_path, f"opponent_{identity}").agent
    candidate_config = len(inspect.signature(candidate).parameters) > 1
    opponent_config = len(inspect.signature(opponent).parameters) > 1
    env = FastEnv(Config(), seed)
    state = list(env.reset(seed))
    configuration = {}
    milestones = []
    previous_land = 1
    started = time.perf_counter()
    trajectory = trajectory_path = temporary_path = None
    if trajectory_dir:
        trajectory_path = Path(trajectory_dir) / "candidate" / slug / f"{seed}-seat{seat}.jsonl.gz"
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        if trajectory_path.exists():
            raise FileExistsError(f"refusing to overwrite trajectory: {trajectory_path}")
        temporary_path = trajectory_path.with_name(f".{trajectory_path.name}.tmp-{os.getpid()}")
        trajectory = gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=6)
        trajectory.write(json.dumps({"type": "meta", "format": "kaggriculture-bc-v1",
            "label": "candidate", "bot": slug, "seed": seed, "seat": seat, "engine": "fast",
            "future_seed": None, "config_override": None, "handoff_selector": "0",
            "policy_source": policy_source, "diagnostic": diagnostic,
            "policy_manifest_sha256": policy_manifest_canonical_sha256,
            "policy_manifest_canonical_sha256": policy_manifest_canonical_sha256,
            "policy_manifest_file_sha256": policy_manifest_file_sha256,
            "opponent_source": opponent_source, "source_family": source_family},
            separators=(",", ":")) + "\n")
    try:
        while not env.done:
            current = observations(state, "fast")
            actions = []
            for player, observation in enumerate(current):
                policy = candidate if player == seat else opponent
                with_config = candidate_config if player == seat else opponent_config
                actions.append(policy(observation, configuration) if with_config else policy(observation))
            if trajectory:
                trajectory.write(json.dumps(trajectory_frame(current, actions), separators=(",", ":")) + "\n")
            state = env.step(actions)
            rival_farm = state[1 - seat]["farms"][1 - seat]
            land = len(rival_farm["unlocked_quadrants"])
            if land != previous_land:
                rival_observation = state[1 - seat]
                milestones.append({
                    "step": int(rival_observation.get("step", int(rival_observation.get("day", 0)) * 24 + int(rival_observation.get("hour", 0)))), "land": land,
                    "unlocked": list(rival_farm["unlocked_quadrants"]),
                    "animals": quadrant_animals(rival_farm),
                })
                previous_land = land
        own, rival = float(env.rewards[seat]), float(env.rewards[1 - seat])
        if trajectory:
            trajectory.write(json.dumps({"type": "terminal",
                "rewards": [own, rival] if seat == 0 else [rival, own],
                "final": trajectory_frame(observations(state, "fast"), [None, None])},
                separators=(",", ":")) + "\n")
        rival_farm = state[1 - seat]["farms"][1 - seat]
        debug = getattr(candidate, "debug", None)
        diagnostics = debug() if callable(debug) else {}
        return {
            "opponent": slug, "seed": seed, "seat": seat, "cash": own,
            "opponent_cash": rival, "margin": own - rival, "error": None,
            "wall_seconds": time.perf_counter() - started,
            "opponent_final_unlocked": list(rival_farm["unlocked_quadrants"]),
            "opponent_final_animals": quadrant_animals(rival_farm),
            "opponent_land_milestones": milestones,
            "policy_debug": {key: diagnostics.get(key) for key in
                             ("search_choices", "search_calls", "search_scenarios")},
            "trajectory": str(trajectory_path) if trajectory_path else None,
        }
    except Exception as exc:
        if trajectory:
            trajectory.write(json.dumps({"type": "error", "error": repr(exc)},
                                        separators=(",", ":")) + "\n")
        return {"opponent": slug, "seed": seed, "seat": seat, "error": repr(exc),
                "wall_seconds": time.perf_counter() - started,
                "trajectory": str(trajectory_path) if trajectory_path else None}
    finally:
        if trajectory:
            trajectory.close()
            os.replace(temporary_path, trajectory_path)
        close = getattr(candidate, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass


def summarize(rows):
    result = {}
    for slug in sorted({row["opponent"] for row in rows}):
        selected = [row for row in rows if row["opponent"] == slug]
        valid = [row for row in selected if not row["error"]]
        by_seed = defaultdict(list)
        for row in valid:
            by_seed[row["seed"]].append(row["margin"] > 0)
        units = [sum(values) / len(values) for values in by_seed.values() if len(values) == 2]
        layouts = Counter(json.dumps(row["opponent_final_animals"], sort_keys=True) for row in valid)
        result[slug] = {
            "games": len(valid), "errors": len(selected) - len(valid),
            "wins": sum(row["margin"] > 0 for row in valid),
            "win_rate": sum(row["margin"] > 0 for row in valid) / len(valid) if valid else None,
            "mean_margin": sum(row["margin"] for row in valid) / len(valid) if valid else None,
            "paired_seed_score": sum(units) / len(units) if units else None,
            "mean_wall_seconds": sum(row["wall_seconds"] for row in valid) / len(valid) if valid else None,
            "final_animal_layouts": dict(layouts),
        }
    return result


def self_check():
    farm = {"tiles": [[None] * 10 for _ in range(10)]}
    farm["tiles"][6][2] = {"animal": "SHEEP"}
    farm["tiles"][7][8] = {"animal": "COW"}
    assert quadrant_animals(farm) == {"NW": {}, "NE": {}, "SW": {"SHEEP": 1}, "SE": {"COW": 1}}
    assert manifest_digest({"b": 2, "a": 1}) == manifest_digest({"a": 1, "b": 2})
    print("PASS")


def manifest_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory-dir")
    parser.add_argument("--policy-source")
    parser.add_argument("--policy-manifest")
    parser.add_argument("--opponent-source", default="isolated-new-public")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--seats", choices=("0", "1", "both"), default="both")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    policy_source = args.policy_source
    policy_manifest_canonical_sha256 = None
    policy_manifest_file_sha256 = None
    if args.policy_manifest:
        manifest_bytes = Path(args.policy_manifest).read_bytes()
        policy_manifest_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        document = json.loads(manifest_bytes)
        policy_manifest_canonical_sha256 = document.pop("canonical_sha256", None)
        semantics = document.pop("digest_semantics", None)
        expected_semantics = "canonical JSON payload excluding canonical_sha256 and digest_semantics"
        if semantics != expected_semantics:
            parser.error("--policy-manifest has unknown digest_semantics")
        actual = manifest_digest(document)
        if not policy_manifest_canonical_sha256 or actual != policy_manifest_canonical_sha256:
            parser.error("--policy-manifest has an invalid canonical_sha256")
        policy_source = f"deployment-{actual[:12]}"
    if args.trajectory_dir and not policy_source:
        parser.error("--policy-source is required with --trajectory-dir")
    manifest = json.loads(Path(args.manifest).read_text())
    opponents = {row["slug"]: row for row in manifest["agents"]}
    seats = (0, 1) if args.seats == "both" else (int(args.seats),)
    tasks = [(args.candidate, slug, row["path"], row.get("source_family", slug),
              args.opponent_source, seed, seat,
              args.trajectory_dir, policy_source, policy_manifest_canonical_sha256,
              policy_manifest_file_sha256, args.diagnostic)
             for slug, row in opponents.items()
             for seed in range(args.start, args.start + args.seeds) for seat in seats]
    rows = []
    started = time.perf_counter()
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)), mp_context=context,
                                max_tasks_per_child=1) as pool:
        for row in pool.map(play, tasks):
            rows.append(row)
    rows.sort(key=lambda row: (row["opponent"], row["seed"], row["seat"]))
    report = {"engine": "FastEnv", "exploratory_only": True,
              "policy_source": policy_source,
              "policy_manifest": str(Path(args.policy_manifest).resolve()) if args.policy_manifest else None,
              "policy_manifest_sha256": policy_manifest_canonical_sha256,
              "policy_manifest_canonical_sha256": policy_manifest_canonical_sha256,
              "policy_manifest_file_sha256": policy_manifest_file_sha256,
              "diagnostic": args.diagnostic,
              "trajectory_format": "kaggriculture-bc-v1" if args.trajectory_dir else None,
              "trajectory_dir": str(Path(args.trajectory_dir).resolve()) if args.trajectory_dir else None,
              "policy_sha256": hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
              "opponent_source": args.opponent_source,
              "seed_range": [args.start, args.start + args.seeds],
              "both_seats": args.seats == "both", "seats": list(seats),
              "wall_seconds": time.perf_counter() - started,
              "summary": summarize(rows), "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

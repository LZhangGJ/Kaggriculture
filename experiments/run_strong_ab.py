#!/usr/bin/env python3
"""Official-engine same-seed, both-seat A/B against the seven strong bots."""

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import importlib.util
import inspect
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

ROOT = Path(__file__).resolve().parents[1]
BOTS = {name: str(ROOT / "opponents" / name / "main.py") for name in (
    "thomas_2945", "melon_2749", "demand_preserving", "ahmed_v47", "pipe8",
    "herd_safe_2700", "salemali7_2900")}


def entry(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "main.py"
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"agent entry does not exist: {path}")
    return path


def load(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child_environment(config_override, handoff_selector):
    if config_override is not None:
        os.environ["R1_CONFIG_OVERRIDES"] = config_override
    else:
        os.environ.pop("R1_CONFIG_OVERRIDES", None)
    if handoff_selector != "0":
        os.environ["REPLAY_HANDOFF_SELECTOR"] = handoff_selector
    else:
        os.environ.pop("REPLAY_HANDOFF_SELECTOR", None)


def observations(state, engine):
    result = []
    for player in (0, 1):
        raw = state[player] if engine == "fast" else state[player].observation
        observation = json.loads(json.dumps(raw))
        if observation.get("step") is None:
            observation["step"] = observation.get("day", 0) * 24 + observation.get("hour", 0)
        observation["player"] = player
        result.append(observation)
    return result


def trajectory_frame(current, actions):
    public = dict(current[0])
    public.pop("private", None)
    public.pop("player", None)
    return {"type": "step", "step": public["step"], "public": public,
            "private": [value.get("private", {}) for value in current], "actions": actions}


def play(task):
    label, policy_path, config_override, handoff_selector, bot, bot_path, seed, seat, engine, trajectory_dir, *future = task
    if len(future) > 1 or future and engine != "fast":
        raise ValueError("future reseeding accepts one seed and requires FastEnv")
    future_seed = future[0] if future else None
    child_environment(config_override, handoff_selector)
    policy_module = load(policy_path, f"policy_{label}_{seed}_{seat}")
    policy = policy_module.create_agent() if hasattr(policy_module, "create_agent") else policy_module.agent
    opponent = load(bot_path, f"opponent_{bot}_{seed}_{seat}").agent
    opponent_takes_configuration = len(inspect.signature(opponent).parameters) > 1
    start = time.perf_counter()
    decision_seconds = 0.0
    handoff_hash = None
    prefix_digest = hashlib.sha256()
    trajectory_path = Path(trajectory_dir) / label / bot / f"{seed}-seat{seat}.jsonl.gz"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    if trajectory_path.exists():
        raise FileExistsError(f"refusing to overwrite trajectory: {trajectory_path}")
    temporary_path = trajectory_path.with_name(f".{trajectory_path.name}.tmp-{os.getpid()}")
    trajectory = gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=6)
    trajectory.write(json.dumps({"type": "meta", "format": "kaggriculture-bc-v1",
        "label": label, "bot": bot, "seed": seed, "seat": seat, "engine": engine,
        "future_seed": future_seed, "config_override": config_override,
        "handoff_selector": handoff_selector}, separators=(",", ":")) + "\n")
    if engine == "fast":
        from fast_kaggriculture import Config, FastEnv
        env = FastEnv(Config(), seed)
        state = list(env.reset(seed))
        configuration = {}
    else:
        from kaggle_environments import make
        env = make("kaggriculture", configuration={"seed": seed}, debug=True)
        state = env.reset()
        configuration = env.configuration
    try:
        while not env.done:
            if future_seed is not None and handoff_hash is None:
                prefix_digest.update(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())
            current = observations(state, engine)
            actions = []
            for player, observation in enumerate(current):
                if player == seat:
                    if future_seed is not None and handoff_hash is None:
                        ready = getattr(policy, "ready", None)
                        if ready is None:
                            raise ValueError("future reseeding requires a warm-handoff policy")
                        if ready(observation):
                            handoff_hash = prefix_digest.hexdigest()
                            env.reseed_future(future_seed)
                    decision_start = time.perf_counter()
                    actions.append(policy(observation, configuration))
                    decision_seconds += time.perf_counter() - decision_start
                else:
                    actions.append(opponent(observation, configuration) if opponent_takes_configuration else
                                   opponent(observation))
            if future_seed is not None and handoff_hash is None:
                prefix_digest.update(json.dumps(actions, sort_keys=True, separators=(",", ":")).encode())
            trajectory.write(json.dumps(trajectory_frame(current, actions), separators=(",", ":")) + "\n")
            state = env.step(actions)
        if engine == "fast":
            own, rival = map(float, (env.rewards[seat], env.rewards[1 - seat]))
        else:
            farms = json.loads(json.dumps(state[0].observation))["farms"]
            own, rival = farms[seat]["money"], farms[1 - seat]["money"]
        trajectory.write(json.dumps({"type": "terminal", "rewards": [own, rival] if seat == 0 else [rival, own],
            "final": trajectory_frame(observations(state, engine), [None, None])}, separators=(",", ":")) + "\n")
        return {"label": label, "bot": bot, "seed": seed, "seat": seat,
                "future_seed": future_seed, "handoff_hash": handoff_hash,
                "future_shops": list(state[seat]["town"]["unlocked_shops"]) if engine == "fast" else None,
                "cash": own, "opponent_cash": rival, "margin": own - rival, "error": None,
                "trajectory": str(trajectory_path),
                "wall_seconds": time.perf_counter() - start, "decision_seconds": decision_seconds}
    except Exception as exc:
        trajectory.write(json.dumps({"type": "error", "error": repr(exc)}, separators=(",", ":")) + "\n")
        return {"label": label, "bot": bot, "seed": seed, "seat": seat,
                "future_seed": future_seed, "handoff_hash": handoff_hash,
                "trajectory": str(trajectory_path), "error": repr(exc)}
    finally:
        trajectory.close()
        os.replace(temporary_path, trajectory_path)
        close = getattr(policy, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass


def summarize(rows, bots=BOTS):
    summary = {}
    for label in ("baseline", "candidate"):
        summary[label] = {}
        for bot in bots:
            selected = [r for r in rows if r["label"] == label and r["bot"] == bot]
            valid = [r for r in selected if not r["error"]]
            complete = len(valid) == len(selected)
            summary[label][bot] = {
                "games": len(valid), "wins": sum(r["cash"] > r["opponent_cash"] for r in valid),
                "win_rate": (sum(r["cash"] > r["opponent_cash"] for r in valid) / len(valid)
                             if valid and complete else None),
                "mean_margin": (sum(r["margin"] for r in valid) / len(valid)
                                if valid and complete else None),
                "errors": len(selected) - len(valid),
                "complete": complete,
            }
    lookup = {(r["label"], r["bot"], r["seed"], r["seat"]): r for r in rows if not r["error"]}
    paired = []
    for bot in bots:
        keys = sorted((r["seed"], r["seat"]) for r in rows if r["label"] == "baseline" and r["bot"] == bot)
        pairs = [(lookup.get(("baseline", bot, *key)), lookup.get(("candidate", bot, *key))) for key in keys]
        expected = len(pairs)
        pairs = [(base, candidate) for base, candidate in pairs if base and candidate]
        paired.append({
            "bot": bot, "pairs": len(pairs), "expected_pairs": expected,
            "dropped_pairs": expected - len(pairs),
            "mean_margin_delta": sum(candidate["margin"] - base["margin"] for base, candidate in pairs) / len(pairs) if pairs else None,
            "win_delta": sum((candidate["cash"] > candidate["opponent_cash"]) -
                             (base["cash"] > base["opponent_cash"]) for base, candidate in pairs),
        })
    return summary, paired


def self_check():
    rows = [
        {"label": label, "bot": bot, "seed": 1, "seat": seat, "cash": 2 + (label == "candidate"),
         "opponent_cash": 2, "margin": label == "candidate", "error": None}
        for label in ("baseline", "candidate") for bot in BOTS for seat in (0, 1)
    ]
    summary, paired = summarize(rows)
    assert len(rows) == 28 and len(paired) == 7
    assert all(summary["candidate"][bot]["win_rate"] == 1 for bot in BOTS)
    child_environment('{"delay_sale":1}', "/tmp/selector.json")
    assert os.environ["R1_CONFIG_OVERRIDES"] == '{"delay_sale":1}'
    assert os.environ["REPLAY_HANDOFF_SELECTOR"] == "/tmp/selector.json"
    child_environment(None, "0")
    assert "R1_CONFIG_OVERRIDES" not in os.environ and "REPLAY_HANDOFF_SELECTOR" not in os.environ
    frame = trajectory_frame([
        {"step": 3, "player": 0, "private": {"shed": {"WHEAT": 1}}, "farms": []},
        {"step": 3, "player": 1, "private": {"shed": {"WHEAT": 2}}, "farms": []},
    ], [{"farmer": ["PASS"]}, {"farmer": ["PASS"]}])
    assert "private" not in frame["public"] and frame["private"][1]["shed"]["WHEAT"] == 2
    print(json.dumps({"status": "PASS", "bots": list(BOTS)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=entry)
    parser.add_argument("--candidate", type=entry)
    parser.add_argument("--baseline-r1-config")
    parser.add_argument("--candidate-r1-config")
    parser.add_argument("--baseline-handoff-selector", default="0")
    parser.add_argument("--candidate-handoff-selector", default="0")
    parser.add_argument("--opponents", default=",".join(BOTS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--start", type=int, default=2609400000)
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument("--engine", choices=("official", "fast"), default="official")
    parser.add_argument("--trajectory-dir", type=Path,
                        help="compressed per-game BC traces (default: <output stem>-trajectories)")
    parser.add_argument("--single", action="store_true",
                        help="evaluate only --candidate; skip baseline and paired deltas")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.candidate or not args.output or (not args.single and not args.baseline):
        parser.error("--candidate and --output are required; --baseline is required without --single")
    if args.seeds < 1 or args.workers < 1:
        parser.error("--seeds and --workers must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    names = [name for name in args.opponents.split(",") if name]
    if not names or len(names) != len(set(names)) or any(name not in BOTS for name in names):
        parser.error("--opponents must be unique names from the configured strong bots")
    bots = {name: BOTS[name] for name in names}
    trajectory_dir = (args.trajectory_dir or
                      args.output.with_name(f"{args.output.stem}-trajectories")).resolve()

    policies = ({"candidate": args.candidate} if args.single else
                {"baseline": args.baseline, "candidate": args.candidate})
    overrides = {
        "baseline": args.baseline_r1_config,
        "candidate": args.candidate_r1_config,
    }
    selectors = {
        "baseline": args.baseline_handoff_selector,
        "candidate": args.candidate_handoff_selector,
    }
    overrides = {label: overrides[label] for label in policies}
    selectors = {label: selectors[label] for label in policies}
    for value in overrides.values():
        if value is not None:
            try:
                if not isinstance(json.loads(value), dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                parser.error("R1 config overrides must be JSON objects")
    for label, value in selectors.items():
        if value != "0":
            path = Path(value).resolve()
            if not path.is_file():
                parser.error(f"--{label}-handoff-selector must be an existing file or 0")
            selectors[label] = str(path)
    tasks = [(label, str(path), overrides[label], selectors[label], bot, bot_path, seed, seat,
              args.engine, str(trajectory_dir))
             for label, path in policies.items() for bot, bot_path in bots.items()
             for seed in range(args.start, args.start + args.seeds) for seat in (0, 1)]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                max_tasks_per_child=1) as pool:
        rows = list(pool.map(play, tasks))
    summary, paired = summarize(rows, bots)
    if args.single:
        summary, paired = {"candidate": summary["candidate"]}, []
    result = {
        "engine": ("fast_kaggriculture:FastEnv" if args.engine == "fast" else
                   "kaggle_environments:kaggriculture"),
        "seed_range": [args.start, args.start + args.seeds],
        "both_seats": True, "process_isolation": "spawn; one game per child",
        "policies": {label: str(path) for label, path in policies.items()},
        "policy_entry_sha256": {
            label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for label, path in policies.items()
        },
        "r1_config_overrides": overrides,
        "handoff_selectors": selectors,
        "trajectory_format": "kaggriculture-bc-v1",
        "trajectory_dir": str(trajectory_dir),
        "opponents": bots,
        "opponent_entry_sha256": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in bots.items()
        },
        "summary": summary, "paired": paired, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "games": len(rows), "paired": paired}))
    errors = sum(bool(row["error"]) for row in rows)
    if errors:
        raise SystemExit(f"A/B incomplete: {errors} game(s) failed; result must not be accepted")


if __name__ == "__main__":
    main()

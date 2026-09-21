#!/usr/bin/env python3
"""Official-engine same-seed, both-seat A/B against the seven strong bots."""

import argparse
import concurrent.futures as cf
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


def play(task):
    label, policy_path, config_override, handoff_selector, bot, bot_path, seed, seat, engine, *future = task
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
            actions = []
            for player in (0, 1):
                raw = state[player] if engine == "fast" else state[player].observation
                observation = json.loads(json.dumps(raw))
                if observation.get("step") is None:
                    observation["step"] = observation.get("day", 0) * 24 + observation.get("hour", 0)
                observation["player"] = player
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
            state = env.step(actions)
        if engine == "fast":
            own, rival = map(float, (env.rewards[seat], env.rewards[1 - seat]))
        else:
            farms = json.loads(json.dumps(state[0].observation))["farms"]
            own, rival = farms[seat]["money"], farms[1 - seat]["money"]
        return {"label": label, "bot": bot, "seed": seed, "seat": seat,
                "future_seed": future_seed, "handoff_hash": handoff_hash,
                "future_shops": list(state[seat]["town"]["unlocked_shops"]) if engine == "fast" else None,
                "cash": own, "opponent_cash": rival, "margin": own - rival, "error": None,
                "wall_seconds": time.perf_counter() - start, "decision_seconds": decision_seconds}
    except Exception as exc:
        return {"label": label, "bot": bot, "seed": seed, "seat": seat,
                "future_seed": future_seed, "handoff_hash": handoff_hash, "error": repr(exc)}
    finally:
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
            summary[label][bot] = {
                "games": len(valid), "wins": sum(r["cash"] > r["opponent_cash"] for r in valid),
                "win_rate": sum(r["cash"] > r["opponent_cash"] for r in valid) / len(valid) if valid else None,
                "mean_margin": sum(r["margin"] for r in valid) / len(valid) if valid else None,
                "errors": len(selected) - len(valid),
            }
    lookup = {(r["label"], r["bot"], r["seed"], r["seat"]): r for r in rows if not r["error"]}
    paired = []
    for bot in bots:
        keys = sorted((r["seed"], r["seat"]) for r in rows if r["label"] == "baseline" and r["bot"] == bot)
        pairs = [(lookup.get(("baseline", bot, *key)), lookup.get(("candidate", bot, *key))) for key in keys]
        pairs = [(base, candidate) for base, candidate in pairs if base and candidate]
        paired.append({
            "bot": bot, "pairs": len(pairs),
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
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.baseline or not args.candidate or not args.output:
        parser.error("--baseline, --candidate and --output are required")
    if args.seeds < 1 or args.workers < 1:
        parser.error("--seeds and --workers must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    names = [name for name in args.opponents.split(",") if name]
    if not names or len(names) != len(set(names)) or any(name not in BOTS for name in names):
        parser.error("--opponents must be unique names from the configured strong bots")
    bots = {name: BOTS[name] for name in names}

    policies = {"baseline": args.baseline, "candidate": args.candidate}
    overrides = {
        "baseline": args.baseline_r1_config,
        "candidate": args.candidate_r1_config,
    }
    selectors = {
        "baseline": args.baseline_handoff_selector,
        "candidate": args.candidate_handoff_selector,
    }
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
              args.engine)
             for label, path in policies.items() for bot, bot_path in bots.items()
             for seed in range(args.start, args.start + args.seeds) for seat in (0, 1)]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                max_tasks_per_child=1) as pool:
        rows = list(pool.map(play, tasks))
    summary, paired = summarize(rows, bots)
    result = {
        "engine": ("fast_kaggriculture:FastEnv" if args.engine == "fast" else
                   "kaggle_environments:kaggriculture"),
        "seed_range": [args.start, args.start + args.seeds],
        "both_seats": True, "process_isolation": "spawn; one game per child",
        "policies": {label: str(path) for label, path in policies.items()},
        "r1_config_overrides": overrides,
        "handoff_selectors": selectors,
        "opponents": bots, "summary": summary, "paired": paired, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "games": len(rows), "paired": paired}))


if __name__ == "__main__":
    main()

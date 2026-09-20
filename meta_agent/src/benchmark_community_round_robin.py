"""Parallel official-environment round robin for audited community agents."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEEDS = (1103, 1701, 2207, 2903, 3301, 4001, 4703, 5309)


def _selector_source(
    *,
    manage_sells: bool = False,
    lead_sells: bool = False,
    lead_turns: int = 5,
    lead_batch: int = 20,
    lead_max_distance: int = 8,
    repair_weeds: bool = False,
    reanchor_at_checkpoints: bool = False,
) -> str:
    library = PROJECT_ROOT / "meta_agent" / "artifacts" / "route-library-land-v2-full.json.gz"
    return f'''from meta_agent.src.replay_trie_agent import ReplayTrieAgent
_POLICY = ReplayTrieAgent(
    {str(library)!r},
    land_weight=4.0,
    manage_sells={manage_sells!r},
    lead_sells={lead_sells!r},
    lead_turns={lead_turns!r},
    lead_batch={lead_batch!r},
    lead_max_distance={lead_max_distance!r},
    repair_weeds={repair_weeds!r},
    reanchor_at_checkpoints={reanchor_at_checkpoints!r},
)
def agent(observation, configuration=None):
    return _POLICY(observation, configuration)
'''


def _sharded_selector_source(
    *, repair_weeds: bool = False, reanchor_at_checkpoints: bool = False
) -> str:
    library = PROJECT_ROOT / "meta_agent" / "artifacts" / "route-runtime-v6"
    return f'''from meta_agent.src.replay_trie_agent import ReplayTrieAgent
_POLICY = ReplayTrieAgent(
    {str(library)!r},
    land_weight=4.0,
    manage_sells=True,
    lead_sells=True,
    lead_turns=5,
    lead_batch=20,
    lead_max_distance=8,
    repair_weeds={repair_weeds!r},
    reanchor_at_checkpoints={reanchor_at_checkpoints!r},
)
def agent(observation, configuration=None):
    return _POLICY(observation, configuration)
'''


def _community_module():
    path = PROJECT_ROOT / "kaggrl" / "community_agents.py"
    spec = importlib.util.spec_from_file_location("meta_community_sources", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LoadedAgent:
    def __init__(self, source: str, name: str) -> None:
        namespace: dict[str, Any] = {"__name__": name}
        exec(compile(source, name, "exec"), namespace)
        self.namespace = namespace
        self.policy = namespace.get("_POLICY")
        self.agent = namespace["agent"]
        parameters = inspect.signature(self.agent).parameters.values()
        self.takes_configuration = any(
            parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            or parameter.name in {"config", "configuration"}
            for parameter in parameters
        ) or len(inspect.signature(self.agent).parameters) >= 2

    def __call__(self, observation: Any, configuration: Any = None):
        if self.takes_configuration:
            return self.agent(observation, configuration)
        return self.agent(observation)


def _game(task: tuple[str, str, str, str, int, int]) -> dict[str, Any]:
    left_name, left_source, right_name, right_source, seed, left_seat = task
    from kaggle_environments import make

    left = LoadedAgent(left_source, f"{left_name}_{seed}_{left_seat}")
    right = LoadedAgent(right_source, f"{right_name}_{seed}_{left_seat}")
    agents = [left, right] if left_seat == 0 else [right, left]
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    steps = env.run(agents)
    rewards = [float(row.get("reward", 0.0) or 0.0) for row in steps[-1]]
    own = rewards[left_seat]
    other = rewards[1 - left_seat]
    result = {
        "left": left_name,
        "right": right_name,
        "seed": seed,
        "left_seat": left_seat,
        "left_reward": own,
        "right_reward": other,
        "margin": own - other,
        "left_win": own > other,
        "draw": own == other,
        "statuses": [str(row.get("status")) for row in steps[-1]],
    }
    for label, loaded in (("left", left), ("right", right)):
        repair = getattr(loaded.policy, "weed_repair", None)
        if repair is not None:
            result[f"{label}_weed_repair"] = dict(repair.telemetry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents",
        nargs="+",
        default=("salem_3000", "kaito_v25", "frontier_soil", "andrews_2883", "meta_v1"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--anchor",
        default=None,
        help="Run only pairs containing this agent instead of a full round robin.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "meta_agent" / "artifacts" / "community-round-robin.json",
    )
    args = parser.parse_args()
    if args.anchor is not None and args.anchor not in args.agents:
        parser.error(f"--anchor {args.anchor!r} is not present in --agents")
    community = _community_module()
    sources = {
        name: (
            (PROJECT_ROOT / "meta_agent" / "submission_v1" / "main.py").read_text(encoding="utf-8")
            if name == "meta_v1"
            else (PROJECT_ROOT / "meta_agent" / "submission_v2" / "main.py").read_text(encoding="utf-8")
            if name == "meta_v2"
            else (PROJECT_ROOT / "meta_agent" / "submission_v4" / "main.py").read_text(encoding="utf-8")
            if name == "meta_v4"
            else _selector_source()
            if name == "meta_selector"
            else _selector_source(manage_sells=True)
            if name == "meta_selector_market"
            else _selector_source(manage_sells=True, lead_sells=True)
            if name == "meta_selector_lead"
            else _selector_source(manage_sells=True, lead_sells=True, repair_weeds=True)
            if name == "meta_selector_lead_repair"
            else _selector_source(
                manage_sells=True,
                lead_sells=True,
                repair_weeds=True,
                reanchor_at_checkpoints=True,
            )
            if name == "meta_selector_lead_repair_reanchor"
            else _sharded_selector_source()
            if name == "meta_selector_sharded"
            else _sharded_selector_source(repair_weeds=True)
            if name == "meta_selector_sharded_repair"
            else _sharded_selector_source(repair_weeds=True)
            if name == "meta_v7"
            else community.community_agent_source(name)
        )
        for name in args.agents
    }
    pairings = [
        pair
        for pair in itertools.combinations(args.agents, 2)
        if args.anchor is None or args.anchor in pair
    ]
    tasks = [
        (left, sources[left], right, sources[right], seed, seat)
        for left, right in pairings
        for seed in args.seeds
        for seat in (0, 1)
    ]
    context = mp.get_context("spawn")
    with context.Pool(processes=min(max(1, args.workers), len(tasks))) as pool:
        games = list(pool.map(_game, tasks))
    pairs = []
    for left, right in pairings:
        rows = [row for row in games if row["left"] == left and row["right"] == right]
        pairs.append(
            {
                "left": left,
                "right": right,
                "games": len(rows),
                "left_wins": sum(int(row["left_win"]) for row in rows),
                "right_wins": sum(int(not row["left_win"] and not row["draw"]) for row in rows),
                "draws": sum(int(row["draw"]) for row in rows),
                "mean_margin_left": sum(row["margin"] for row in rows) / max(1, len(rows)),
            }
        )
    table = {name: {"wins": 0, "losses": 0, "draws": 0, "margin": 0.0, "games": 0} for name in args.agents}
    for row in games:
        left, right = row["left"], row["right"]
        table[left]["games"] += 1
        table[right]["games"] += 1
        table[left]["margin"] += row["margin"]
        table[right]["margin"] -= row["margin"]
        if row["draw"]:
            table[left]["draws"] += 1
            table[right]["draws"] += 1
        elif row["left_win"]:
            table[left]["wins"] += 1
            table[right]["losses"] += 1
        else:
            table[right]["wins"] += 1
            table[left]["losses"] += 1
    standings = [
        {
            "agent": name,
            **values,
            "win_rate": values["wins"] / max(1, values["games"]),
            "mean_margin": values["margin"] / max(1, values["games"]),
        }
        for name, values in table.items()
    ]
    standings.sort(key=lambda row: (-row["win_rate"], -row["mean_margin"]))
    result = {"seeds": args.seeds, "standings": standings, "pairs": pairs, "games": games}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"standings": standings, "pairs": pairs}, ensure_ascii=False))


if __name__ == "__main__":
    main()

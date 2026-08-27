#!/usr/bin/env python3
"""Screen fixed mature routes against a real submission in the official simulator.

This is deliberately separate from the native all-pairs search.  It is the
authoritative gate for opponent-specific counter routes and can also reveal
which route the searched opponent switches to in response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping


AGENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DEPS = Path(__file__).resolve().parents[4] / ".ke"
sys.path.insert(0, str(AGENT_ROOT / "src"))
if WORKSPACE_DEPS.exists():
    sys.path.insert(0, str(WORKSPACE_DEPS))

from meta_agent.src.teammate_expanded_routes import (  # noqa: E402
    TeammateExpandedRouteAgent,
    load_action_tapes,
)


_WORKER_SOURCE: str | None = None
_WORKER_TAPES: Mapping[str, Any] | None = None
_WORKER_ROUTE_BY_FAMILY: dict[str, str] | None = None
_WORKER_OPPONENT: Path | None = None


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    if separator:
        return tuple(range(int(start), int(stop)))
    return tuple(int(part) for part in value.split(","))


def _opponent_state(namespace: Mapping[str, Any]) -> dict[str, Any]:
    controller = namespace.get("_S_CONTROLLER") or namespace.get("_CONTROLLER")
    if controller is None:
        return {}
    return {
        "opening": str(getattr(controller, "opening", "")),
        "final_route": str(getattr(controller, "current", "")),
        "switched": bool(getattr(controller, "switched", False)),
    }


def _reward(environment: Any, seat: int) -> float:
    return float(_get(environment.state[seat], "reward", 0.0) or 0.0)


def _init_worker(
    source_path: str,
    actions_path: str,
    metadata_path: str,
    opponent_path: str,
    workspace_deps: str,
) -> None:
    global _WORKER_SOURCE, _WORKER_TAPES, _WORKER_ROUTE_BY_FAMILY, _WORKER_OPPONENT
    if workspace_deps and workspace_deps not in sys.path:
        sys.path.insert(0, workspace_deps)
    source = Path(source_path)
    actions = Path(actions_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    _WORKER_SOURCE = source.read_text(encoding="utf-8")
    _WORKER_TAPES = load_action_tapes(actions)
    _WORKER_ROUTE_BY_FAMILY = {
        str(entry["family"]): str(entry["route_id"])
        for entry in metadata["opponent_routes"]
    }
    _WORKER_OPPONENT = Path(opponent_path)


def _run_scenario(task: tuple[str, int, int]) -> dict[str, Any]:
    from kaggle_environments import make

    family, seed, seat = task
    assert _WORKER_SOURCE is not None
    assert _WORKER_TAPES is not None
    assert _WORKER_ROUTE_BY_FAMILY is not None
    assert _WORKER_OPPONENT is not None
    started = time.perf_counter()
    try:
        carrier = TeammateExpandedRouteAgent(
            _WORKER_SOURCE,
            _WORKER_TAPES,
            f"official_counter_{family}_{seed}_{seat}",
        )
        carrier.select(_WORKER_ROUTE_BY_FAMILY[family])
        opponent_namespace = runpy.run_path(str(_WORKER_OPPONENT))
        opponent = opponent_namespace.get("agent")
        if not callable(opponent):
            raise ValueError(f"submission has no callable agent: {_WORKER_OPPONENT}")
        agents = [opponent, opponent]
        agents[seat] = carrier
        environment = make(
            "kaggriculture",
            configuration={"episodeSteps": 720},
            info={"seed": int(seed)},
            debug=True,
        )
        environment.run(agents)
        own = _reward(environment, seat)
        other = _reward(environment, 1 - seat)
        return {
            "family": family,
            "seed": int(seed),
            "seat": int(seat),
            "reward": own,
            "opponent_reward": other,
            "margin": own - other,
            "win": own > other,
            "completed": len(environment.steps) == 720,
            "opponent_route": _opponent_state(opponent_namespace),
            "error": None,
            "seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "family": family,
            "seed": int(seed),
            "seat": int(seat),
            "completed": False,
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--opponent-agent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=_ints, default=(17,))
    parser.add_argument("--one-seat", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--workspace-deps", type=Path)
    parser.add_argument(
        "--families",
        nargs="*",
        help="Optional family subset.  By default every metadata route is screened.",
    )
    args = parser.parse_args()

    from kaggle_environments import make

    started = time.perf_counter()
    source_path = args.source.resolve()
    actions_path = args.actions.resolve()
    metadata_path = args.metadata.resolve()
    opponent_path = args.opponent_agent.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = list(metadata["opponent_routes"])
    route_by_family = {
        str(entry["family"]): str(entry["route_id"]) for entry in entries
    }
    selected = list(args.families or route_by_family)
    missing = [family for family in selected if family not in route_by_family]
    if missing:
        raise KeyError(f"unknown route family: {missing[0]}")
    seats = (0,) if args.one_seat else (0, 1)
    workspace_deps = (
        args.workspace_deps.resolve()
        if args.workspace_deps else WORKSPACE_DEPS.resolve()
    )
    tasks = [
        (family, seed, seat)
        for family in selected
        for seed in args.seeds
        for seat in seats
    ]
    scenario_count = len(selected) * len(args.seeds) * len(seats)
    scenarios_by_family = {family: [] for family in selected}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(
            str(source_path), str(actions_path), str(metadata_path),
            str(opponent_path), str(workspace_deps),
        ),
    ) as pool:
        for completed_games, scenario in enumerate(
            pool.map(_run_scenario, tasks, chunksize=1), 1
        ):
            scenarios_by_family[str(scenario["family"])].append(scenario)
            if completed_games == scenario_count or completed_games % max(
                1, scenario_count // 20
            ) == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"games={completed_games}/{scenario_count} "
                    f"elapsed={elapsed:.1f}s rate={completed_games / elapsed:.3f}/s",
                    flush=True,
                )

    rows = []
    for family_index, family in enumerate(selected, 1):
        scenarios = scenarios_by_family[family]
        valid = [value for value in scenarios if value["error"] is None]
        margins = [value["margin"] for value in valid]
        rewards = [value["reward"] for value in valid]
        row = {
            "family": family,
            "route_id": route_by_family[family],
            "mean_margin": statistics.fmean(margins) if margins else float("-inf"),
            "minimum_margin": min(margins) if margins else float("-inf"),
            "win_rate": statistics.fmean(float(value["win"]) for value in valid) if valid else 0.0,
            "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
            "minimum_reward": min(rewards) if rewards else 0.0,
            "completed": sum(bool(value["completed"]) for value in valid),
            "scenario_count": len(scenarios),
            "valid_scenarios": len(valid),
            "scenarios": scenarios,
        }
        rows.append(row)
        elapsed = time.perf_counter() - started
        print(
            f"{family_index}/{len(selected)} {family} "
            f"margin={row['mean_margin']:.1f} win={row['win_rate']:.3f} "
            f"reward={row['mean_reward']:.1f} "
            f"games={completed_games}/{scenario_count} elapsed={elapsed:.1f}s",
            flush=True,
        )

    rows.sort(
        key=lambda row: (
            -float(row["mean_margin"]),
            -float(row["win_rate"]),
            -float(row["minimum_margin"]),
            str(row["family"]),
        )
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    payload = {
        "schema": "official-fixed-route-counter-search-v1",
        "engine": "kaggle-environments official local simulator",
        "source": str(source_path),
        "actions": str(actions_path),
        "metadata": str(metadata_path),
        "opponent_agent": str(opponent_path),
        "opponent_sha256": hashlib.sha256(opponent_path.read_bytes()).hexdigest(),
        "seeds": list(args.seeds),
        "seats": list(seats),
        "workers": args.workers,
        "candidate_count": len(rows),
        "games": scenario_count,
        "valid_games": sum(row["valid_scenarios"] for row in rows),
        "completed_games": sum(row["completed"] for row in rows),
        "errors": [
            scenario
            for row in rows
            for scenario in row["scenarios"]
            if scenario["error"] is not None
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "ranking": rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "games": scenario_count,
        "elapsed_seconds": payload["elapsed_seconds"],
        "top10": [
            {key: row[key] for key in (
                "rank", "family", "mean_margin", "minimum_margin",
                "win_rate", "mean_reward",
            )}
            for row in rows[:10]
        ],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

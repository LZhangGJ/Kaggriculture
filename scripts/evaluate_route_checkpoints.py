"""Paired official-engine evaluation of route-selector checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence


for _variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "1"

from kaggle_environments import make

from kaggrl import CommunityAgent, community_agent_specs
from meta_agent.src.equilibrium_selector import EquilibriumOpeningSelector
from meta_agent.src.replay_trie_agent import ReplayTrieAgent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = {
    "best": ROOT / "recurrent-route-best.pt",
}
STRONG_OPPONENTS = (
    "public_b85", "schedule_93311715", "frontier_soil",
    "adaptive_farming", "kaito_v35",
)
DEFAULT_SEEDS = tuple(range(20260824, 20260840))
_POLICIES: dict[str, tuple[ReplayTrieAgent, EquilibriumOpeningSelector]] = {}


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected integers")
    return result


def _csv_names(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    available = {spec.name for spec in community_agent_specs()}
    unknown = sorted(set(result) - available)
    if not result or unknown:
        raise argparse.ArgumentTypeError(f"unknown or empty opponents: {unknown}")
    return result


def _checkpoint_map(value: str | None) -> dict[str, Path]:
    if not value:
        result = DEFAULT_CHECKPOINTS
    else:
        result = {}
        for raw in value.split(","):
            label, separator, path = raw.partition("=")
            if not separator or not label.strip() or not path.strip():
                raise argparse.ArgumentTypeError("checkpoints must be label=path pairs")
            result[label.strip()] = Path(path.strip()).resolve()
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise argparse.ArgumentTypeError(f"missing checkpoints: {missing}")
    return dict(result)


def _salt(opponent: str, seed: int, seat: int) -> int:
    # Opponent identity is deliberately excluded.  A live submission does not
    # know the opponent's notebook name, and the same seed/seat should draw the
    # same opening mixture member against every benchmark opponent.
    del opponent
    raw = f"route-checkpoint-ab-v2:{seed}:{seat}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _policy(label: str, path: str, salt: int) -> tuple[ReplayTrieAgent, EquilibriumOpeningSelector]:
    cached = _POLICIES.get(label)
    if cached is None:
        selector = EquilibriumOpeningSelector(
            str(ROOT / "route-runtime-v8"),
            str(ROOT / "branch-descriptors-v8.pkl"),
            tree_q_model_path=None,
            recurrent_model_path=path,
            residual_weight=1.0,
            enable_trading=False,
            epsilon=0.0,
            top_k=16,
            prior_weight=0.0,
            seed=0,
            mixture_path=str(ROOT / "opening-mixture-v8.json"),
            mixture_salt=salt,
        )
        policy = ReplayTrieAgent(
            ROOT / "route-runtime-v8",
            selector_override=selector,
            manage_sells=True,
            lead_sells=True,
            lead_turns=5,
            lead_batch=20,
            lead_max_distance=8,
            repair_weeds=True,
            weed_replay_steps=8,
        )
        cached = policy, selector
        _POLICIES[label] = cached
    policy, selector = cached
    selector.fixed_mixture_salt = salt
    selector.mixture_salt = salt
    return policy, selector


def _reward(state: Any) -> float:
    value = getattr(state, "reward", None)
    if value is None:
        raise RuntimeError(f"missing reward: {getattr(state, 'status', None)}")
    return float(value)


def play(task: tuple[str, str, str, int, int]) -> dict[str, Any]:
    label, checkpoint, opponent_name, seed, seat = task
    salt = _salt(opponent_name, seed, seat)
    submission, selector = _policy(label, checkpoint, salt)
    opponent = CommunityAgent(opponent_name)
    agents = [submission, opponent] if seat == 0 else [opponent, submission]
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    started = time.perf_counter()
    environment.run(agents)
    elapsed = time.perf_counter() - started
    states = list(environment.state)
    statuses = [str(getattr(state, "status", "")) for state in states]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"bad statuses {label}/{opponent_name}/{seed}/{seat}: {statuses}")
    rewards = [_reward(state) for state in states]
    own, other = rewards[seat], rewards[1 - seat]
    margin = own - other
    return {
        "checkpoint": label,
        "opponent": opponent_name,
        "seed": seed,
        "submission_seat": seat,
        "opening_salt": salt,
        "opening_route_id": selector.opening_route_id,
        "submission_reward": own,
        "opponent_reward": other,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
        "elapsed_seconds": elapsed,
    }


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    result: dict[str, Any] = {
        "games": len(rows), "wins": wins, "draws": draws,
        "losses": len(rows) - wins - draws,
        "score_rate": (wins + 0.5 * draws) / len(rows),
        "mean_margin": sum(row["margin"] for row in rows) / len(rows),
    }
    for field, output in (("opponent", "versus"), ("submission_seat", "by_seat")):
        grouped = {}
        for value in sorted({row[field] for row in rows}, key=str):
            subset = [row for row in rows if row[field] == value]
            local_wins = sum(row["result"] == "win" for row in subset)
            local_draws = sum(row["result"] == "draw" for row in subset)
            grouped[str(value)] = {
                "games": len(subset), "wins": local_wins, "draws": local_draws,
                "losses": len(subset) - local_wins - local_draws,
                "score_rate": (local_wins + 0.5 * local_draws) / len(subset),
                "mean_margin": sum(row["margin"] for row in subset) / len(subset),
            }
        result[output] = grouped
    return result


def _paired(games: Sequence[dict[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    base = labels[0]
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for game in games:
        key = game["opponent"], game["seed"], game["submission_seat"]
        grouped.setdefault(key, {})[game["checkpoint"]] = game
    result = {}
    for label in labels[1:]:
        deltas = []
        loss_to_win = win_to_loss = route_mismatches = 0
        for pair in grouped.values():
            before, after = pair[base], pair[label]
            deltas.append(after["margin"] - before["margin"])
            loss_to_win += int(before["result"] == "loss" and after["result"] == "win")
            win_to_loss += int(before["result"] == "win" and after["result"] == "loss")
            route_mismatches += int(before["opening_route_id"] != after["opening_route_id"])
        ordered = sorted(deltas)
        result[label] = {
            "pairs": len(deltas), "route_mismatches": route_mismatches,
            "mean_margin_delta": sum(deltas) / len(deltas),
            "median_margin_delta": ordered[len(ordered) // 2],
            "improved": sum(value > 0 for value in deltas),
            "tied": sum(value == 0 for value in deltas),
            "worsened": sum(value < 0 for value in deltas),
            "loss_to_win": loss_to_win, "win_to_loss": win_to_loss,
        }
    return result


def main_cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--opponents", type=_csv_names, default=STRONG_OPPONENTS)
    parser.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--seats", type=_csv_ints, default=(0, 1))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation/route-checkpoint-selection-official-16seeds.json"),
    )
    args = parser.parse_args(argv)
    checkpoints = _checkpoint_map(args.checkpoints)
    tasks = [
        (label, str(path), opponent, seed, seat)
        for opponent in args.opponents
        for seed in args.seeds
        for seat in args.seats
        for label, path in checkpoints.items()
    ]
    worker_count = min(args.workers or (os.cpu_count() or 1), len(tasks))
    games = []
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(worker_count) as pool:
        for game in pool.imap_unordered(play, tasks, chunksize=1):
            games.append(game)
            if len(games) % 40 == 0 or len(games) == len(tasks):
                print(f"[{len(games):04d}/{len(tasks):04d}]", flush=True)
    labels = list(checkpoints)
    report = {
        "environment": "kaggle_environments.make('kaggriculture')",
        "kaggle_environments_version": version("kaggle-environments"),
        "configuration": {"marketParams": {}, "official_defaults": True},
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "opponents": list(args.opponents), "seeds": list(args.seeds),
        "seats": list(args.seats), "workers": worker_count,
        "wall_seconds": time.perf_counter() - started,
        "summary": {
            label: _summary([game for game in games if game["checkpoint"] == label])
            for label in labels
        },
        "paired_vs_base": _paired(games, labels),
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "paired": report["paired_vs_base"]}, indent=2))
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main_cli()

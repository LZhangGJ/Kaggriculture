"""Evaluate route checkpoints against an archived historical submission."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import tarfile
import tempfile
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

from meta_agent.src.equilibrium_selector import EquilibriumOpeningSelector
from meta_agent.src.replay_trie_agent import ReplayTrieAgent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    ROOT / "data/kaggle_own_submissions/dzjiann/55646877/submission.tar.gz"
)
DEFAULT_CHECKPOINTS = {
    "v8": ROOT / "recurrent-route-v8.pt",
    "best": ROOT / "recurrent-route-best.pt",
}
DEFAULT_SEEDS = tuple(range(20261301, 20261365))
_CURRENT: dict[str, tuple[ReplayTrieAgent, EquilibriumOpeningSelector]] = {}
_HISTORICAL: dict[str, Any] = {}


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected integers")
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


def _salt(seed: int, seat: int) -> int:
    payload = f"route-vs-own-submission-v1:{seed}:{seat}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _current_policy(
    label: str, checkpoint: str, salt: int
) -> tuple[ReplayTrieAgent, EquilibriumOpeningSelector]:
    cached = _CURRENT.get(label)
    if cached is None:
        selector = EquilibriumOpeningSelector(
            str(ROOT / "route-runtime-v8"),
            str(ROOT / "branch-descriptors-v8.pkl"),
            tree_q_model_path=None,
            recurrent_model_path=checkpoint,
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
        _CURRENT[label] = cached
    policy, selector = cached
    selector.fixed_mixture_salt = salt
    selector.mixture_salt = salt
    return policy, selector


def _historical_policy(extracted_root: str, submission_id: int) -> Any:
    cached = _HISTORICAL.get(extracted_root)
    if cached is not None:
        return cached
    root = Path(extracted_root)
    if (root / "route-runtime-v6/manifest.pkl").is_file():
        package_root = root / "meta_agent"
        package_name = f"historical_submission_{submission_id}_meta_agent"
        if package_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                package_name,
                package_root / "__init__.py",
                submodule_search_locations=[str(package_root)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load historical meta_agent package")
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)
        replay_module = importlib.import_module(
            f"{package_name}.src.replay_trie_agent"
        )
        cached = replay_module.ReplayTrieAgent(
            root / "route-runtime-v6",
            land_weight=4.0,
            manage_sells=True,
            lead_sells=True,
            lead_turns=5,
            lead_batch=20,
            lead_max_distance=8,
        )
    else:
        module_name = f"historical_submission_{submission_id}_main"
        spec = importlib.util.spec_from_file_location(module_name, root / "main.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load historical standalone main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        cached = module.agent
    _HISTORICAL[extracted_root] = cached
    return cached


def _reward(state: Any) -> float:
    value = getattr(state, "reward", None)
    if value is None:
        raise RuntimeError(f"missing reward: {getattr(state, 'status', None)}")
    return float(value)


def play(task: tuple[str, str, str, int, int, int]) -> dict[str, Any]:
    label, checkpoint, extracted_root, submission_id, seed, seat = task
    salt = _salt(seed, seat)
    current, selector = _current_policy(label, checkpoint, salt)
    historical = _historical_policy(extracted_root, submission_id)
    agents = [current, historical] if seat == 0 else [historical, current]
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    started = time.perf_counter()
    environment.run(agents)
    states = list(environment.state)
    statuses = [str(getattr(state, "status", "")) for state in states]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"bad statuses {label}/{seed}/{seat}: {statuses}")
    rewards = [_reward(state) for state in states]
    own, other = rewards[seat], rewards[1 - seat]
    margin = own - other
    return {
        "checkpoint": label,
        "opponent_submission_id": submission_id,
        "seed": seed,
        "submission_seat": seat,
        "opening_salt": salt,
        "opening_route_id": selector.opening_route_id,
        "submission_reward": own,
        "opponent_reward": other,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
        "elapsed_seconds": time.perf_counter() - started,
    }


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    result: dict[str, Any] = {
        "games": len(rows),
        "wins": wins,
        "draws": draws,
        "losses": len(rows) - wins - draws,
        "score_rate": (wins + 0.5 * draws) / len(rows),
        "mean_margin": sum(row["margin"] for row in rows) / len(rows),
    }
    result["by_seat"] = {}
    for seat in (0, 1):
        subset = [row for row in rows if row["submission_seat"] == seat]
        local_wins = sum(row["result"] == "win" for row in subset)
        local_draws = sum(row["result"] == "draw" for row in subset)
        result["by_seat"][str(seat)] = {
            "games": len(subset),
            "wins": local_wins,
            "draws": local_draws,
            "losses": len(subset) - local_wins - local_draws,
            "score_rate": (local_wins + 0.5 * local_draws) / len(subset),
            "mean_margin": sum(row["margin"] for row in subset) / len(subset),
        }
    return result


def _paired(games: Sequence[dict[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    base = labels[0]
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for game in games:
        key = game["seed"], game["submission_seat"]
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
            route_mismatches += int(
                before["opening_route_id"] != after["opening_route_id"]
            )
        result[label] = {
            "pairs": len(deltas),
            "route_mismatches": route_mismatches,
            "mean_margin_delta": sum(deltas) / len(deltas),
            "improved": sum(value > 0 for value in deltas),
            "tied": sum(value == 0 for value in deltas),
            "worsened": sum(value < 0 for value in deltas),
            "loss_to_win": loss_to_win,
            "win_to_loss": win_to_loss,
        }
    return result


def _extract_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            member_path = Path(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member.issym()
                or member.islnk()
            ):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")
    standalone = destination / "main.py"
    packaged = (
        destination / "meta_agent/src/replay_trie_agent.py",
        destination / "route-runtime-v6/manifest.pkl",
    )
    if not standalone.is_file() or not (
        all(path.is_file() for path in packaged)
        or not any(path.exists() for path in packaged)
    ):
        raise RuntimeError("historical archive is incomplete")


def main_cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--opponent-id", type=int, default=55646877)
    parser.add_argument("--opponent-submitted-by", default="dzjiann")
    parser.add_argument("--opponent-public-score", type=float, default=2077.6)
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--seats", type=_csv_ints, default=(0, 1))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/route-vs-dzjiann-55646877-official-64seeds.json"),
    )
    args = parser.parse_args(argv)
    archive = args.archive.resolve()
    if not archive.is_file():
        parser.error(f"missing archive: {archive}")
    checkpoints = _checkpoint_map(args.checkpoints)
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="kaggriculture-own-submission-") as temporary:
        extracted_root = Path(temporary)
        _extract_archive(archive, extracted_root)
        tasks = [
            (label, str(path), str(extracted_root), args.opponent_id, seed, seat)
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
                if len(games) % 32 == 0 or len(games) == len(tasks):
                    print(f"[{len(games):04d}/{len(tasks):04d}]", flush=True)
    labels = list(checkpoints)
    report = {
        "environment": "kaggle_environments.make('kaggriculture')",
        "kaggle_environments_version": version("kaggle-environments"),
        "configuration": {"marketParams": {}, "official_defaults": True},
        "historical_submission": {
            "id": args.opponent_id,
            "submitted_by": args.opponent_submitted_by,
            "public_score_at_evaluation": args.opponent_public_score,
            "archive": str(archive),
            "archive_sha256": archive_sha256,
        },
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "seeds": list(args.seeds),
        "seats": list(args.seats),
        "workers": worker_count,
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
    print(json.dumps({
        "summary": report["summary"],
        "paired_vs_base": report["paired_vs_base"],
    }, indent=2))
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main_cli()

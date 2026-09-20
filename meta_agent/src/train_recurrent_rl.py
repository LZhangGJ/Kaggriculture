"""Train recurrent route PPO checkpoints on the verified C++ environment."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

# Multiprocessing ``spawn`` imports this module before running the pool
# initializer.  These limits therefore must be set here, before NumPy/Torch are
# imported, otherwise every rollout worker permanently creates a 192-thread
# BLAS/OpenMP pool (180 workers became more than 46,000 runtime threads).
for _thread_env in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np

from .recurrent_meta import (
    RecurrentTreeQSelector,
    SymmetricHistoryModel,
    market_town_vector,
    public_farm_vector,
)
from .replay_trie_agent import ReplayTrieAgent


ROOT = Path(__file__).resolve().parents[2]
FAST_PYTHON = ROOT / "fast_kaggriculture/python"
RUNTIME = ROOT / "route-runtime-v8"
DESCRIPTORS = ROOT / "branch-descriptors-v8.pkl"
TREE_Q = ROOT / "meta_agent/runs/tree-q-v1/tree-q-latest.pkl"
BC_CHECKPOINT = ROOT / "meta_agent/runs/recurrent-meta-v1/recurrent-meta-latest.pt"
V3_OPENING_MIXTURE = ROOT / "opening-mixture-v8.json"
V3_HISTORY_INITIAL = (
    ROOT
    / "meta_agent/runs/continuous-trade-ppo-v7-controlled-warmup"
    / "continuous-trade-policy.pt"
)
DEFAULT_RUN = ROOT / "training_runs/route-ppo-v9-hard5"
OPPONENTS = (
    # The official 16-seed, dual-seat screen retired rule_based, kaito_v25,
    # mega_ensemble_3000, and andrews_2883 because they no longer provide a
    # useful decision boundary for v8.  Keep only the five hard opponents.
    "public_b85", "schedule_93311715", "frontier_soil", "adaptive_farming",
    "kaito_v35",
)

_COMMUNITY: dict[str, Any] = {}
_POLICY_CACHE: dict[tuple[str, str], tuple[Any, Any, tuple[str, int, int]]] = {}


class ConfigDict(dict):
    def __getattr__(self, key):
        return self[key]


CONFIG = ConfigDict(
    episodeSteps=720, turnsPerDay=24, shedCapacity=100, boardSize=10,
    startingMoney=3000, maxMarketOrders=10, farmHandCostMult=1,
)


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if str(FAST_PYTHON) not in sys.path:
        sys.path.insert(0, str(FAST_PYTHON))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _community_agent(name: str):
    if name not in _COMMUNITY:
        from kaggrl.community_agents import community_agent_source
        from .benchmark_community_round_robin import LoadedAgent
        _COMMUNITY[name] = LoadedAgent(
            community_agent_source(name), f"cpp_torch_rl_{name}_{os.getpid()}"
        )
    return _COMMUNITY[name]


def _policy(
    checkpoint: str,
    seed: int,
    epsilon: float,
    algorithm: str = "mc_q",
    temperature: float = 1.0,
    uniform_mix: float = 0.0,
    cache_slot: str = "learner",
    runtime: str = str(RUNTIME),
    descriptors: str = str(DESCRIPTORS),
    top_k: int = 8,
):
    scratch_ppo = algorithm == "scratch_ppo"
    gru_ppo = algorithm == "gru_ppo"
    rollout_checkpoint = (
        str(Path(checkpoint).with_suffix(".pkl")) if gru_ppo else checkpoint
    )
    stat = os.stat(rollout_checkpoint)
    signature = (str(checkpoint), int(stat.st_mtime_ns), int(stat.st_size))
    cache_key = (algorithm, cache_slot, runtime, descriptors, int(top_k))
    cached = _POLICY_CACHE.get(cache_key)
    if cached is not None:
        policy, selector, previous_signature = cached
        if previous_signature != signature:
            if gru_ppo:
                # The NumPy GRU owns recurrent state and immutable array views;
                # recreate it atomically when a new PPO snapshot is published.
                cached = None
            else:
                import torch
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                selector.model.load_state_dict(payload["model"])
                selector.model.eval()
                selector.loaded = True
        if cached is not None:
            policy.reset()
            if gru_ppo:
                selector.rng.seed(int(seed))
            else:
                selector.selector.rng.seed(int(seed))
                selector.selector.epsilon = 0.0 if scratch_ppo else float(epsilon)
                selector.temperature = max(1e-4, float(temperature))
                selector.uniform_mix = max(0.0, min(1.0, float(uniform_mix)))
            _POLICY_CACHE[cache_key] = (policy, selector, signature)
            return policy, selector

    if gru_ppo:
        from .route_gru import ControlledReanchorGRUSelector

        selector = ControlledReanchorGRUSelector(
            runtime,
            descriptors,
            route_model_path=rollout_checkpoint,
            opening_mixture_path=V3_OPENING_MIXTURE,
            tree_q_model_path=TREE_Q,
            top_k=top_k,
            prior_weight=0.05,
            temperature=temperature,
            uniform_mix=uniform_mix,
            stochastic=True,
            seed=seed,
        )
        policy = ReplayTrieAgent(
            runtime,
            selector_override=selector,
            manage_sells=True,
            lead_sells=True,
            lead_turns=5,
            lead_batch=20,
            lead_max_distance=8,
            repair_weeds=True,
        )
        _POLICY_CACHE[cache_key] = (policy, selector, signature)
        return policy, selector

    selector = RecurrentTreeQSelector(
        runtime, descriptors,
        tree_q_model_path=None if scratch_ppo else str(TREE_Q),
        recurrent_model_path=checkpoint, residual_weight=1.0,
        enable_trading=False, epsilon=0.0 if scratch_ppo else epsilon,
        top_k=top_k, prior_weight=0.0 if scratch_ppo else 0.05, seed=seed,
        stochastic_softmax=scratch_ppo, temperature=temperature,
        uniform_mix=uniform_mix,
    )
    policy = ReplayTrieAgent(
        str(RUNTIME), selector_override=selector, manage_sells=True,
        lead_sells=True, lead_turns=5, lead_batch=20, lead_max_distance=8,
        repair_weeds=True,
    )
    _POLICY_CACHE[cache_key] = (policy, selector, signature)
    return policy, selector


def _game(task: tuple[Any, ...]) -> dict[str, Any]:
    (game_id, seed, seat, opponent_name, checkpoint, snapshot, epsilon,
     algorithm, temperature, uniform_mix, runtime, descriptors, top_k) = task
    started = time.perf_counter()
    try:
        from fast_kaggriculture import Config as FastConfig, FastEnv

        learner, learner_selector = _policy(
            checkpoint, seed * 17 + seat, epsilon, algorithm, temperature,
            uniform_mix, "learner", runtime, descriptors, top_k
        )
        selectors = [(learner_selector, seat, opponent_name)]
        if opponent_name == "selfplay":
            opponent, opponent_selector = _policy(
                snapshot, seed * 31 + 1 - seat, max(0.02, epsilon * 0.5),
                algorithm, temperature, uniform_mix, "selfplay_opponent",
                runtime, descriptors, top_k,
            )
            selectors.append((opponent_selector, 1 - seat, "selfplay"))
        else:
            opponent = _community_agent(opponent_name)
        agents = [learner, opponent] if seat == 0 else [opponent, learner]
        env = FastEnv(FastConfig(), int(seed))
        observations = list(env.reset(int(seed)))
        policy_seconds = opponent_seconds = environment_seconds = 0.0
        while not env.done:
            step = int(env.step_count)
            observations[0]["step"] = step
            observations[1]["step"] = step
            if opponent_name == "selfplay":
                public = tuple(public_farm_vector(farm) for farm in observations[0]["farms"])
                market = market_town_vector(observations[0])
                for observation in observations:
                    observation["_meta_public_vectors"] = public
                    observation["_meta_market_vector"] = market
            actions = [None, None]
            begin = time.perf_counter()
            actions[seat] = learner(observations[seat], CONFIG)
            policy_seconds += time.perf_counter() - begin
            begin = time.perf_counter()
            actions[1 - seat] = opponent(observations[1 - seat], CONFIG)
            opponent_seconds += time.perf_counter() - begin
            begin = time.perf_counter()
            observations = list(env.step(actions))
            environment_seconds += time.perf_counter() - begin
        rewards = [float(value) for value in env.rewards]
        traces = []
        for selector, player, label in selectors:
            trace = selector.trace_arrays(top_k=top_k)
            trace.update(
                opponent=label,
                win=float(rewards[player] > rewards[1 - player]),
                margin=rewards[player] - rewards[1 - player],
            )
            traces.append(trace)
        return {
            "valid": True, "game_id": game_id, "opponent": opponent_name,
            "win": float(rewards[seat] > rewards[1 - seat]),
            "margin": rewards[seat] - rewards[1 - seat], "traces": traces,
            "elapsed": time.perf_counter() - started,
            "policy_seconds": policy_seconds, "opponent_seconds": opponent_seconds,
            "environment_seconds": environment_seconds,
        }
    except Exception as error:
        return {
            "valid": False, "game_id": game_id, "opponent": opponent_name,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=12),
            "elapsed": time.perf_counter() - started,
        }


def _read_stats(path: Path) -> dict[str, dict[str, int]]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        name: {"games": 0, "wins": 0, "draws": 0, "losses": 0}
        for name in (*OPPONENTS, "selfplay")
    }


def _schedule(
    games: int,
    selfplay_fraction: float,
    uniform_opponent_fraction: float,
    stats: dict,
    rng: random.Random,
) -> list[str]:
    selfplay = max(0, min(games, round(games * selfplay_fraction)))
    community = games - selfplay
    uniform = round(community * uniform_opponent_fraction)
    result = [OPPONENTS[index % len(OPPONENTS)] for index in range(uniform)]
    weights = []
    for name in OPPONENTS:
        row = stats.get(name, {})
        count, wins = row.get("games", 0), row.get("wins", 0)
        draws = row.get("draws", 0)
        own_rate = (wins + 0.5 * draws + 1) / (count + 2)
        weights.append(max(0.05, 1 - own_rate + 1 / math.sqrt(count + 2)))
    result.extend(rng.choices(OPPONENTS, weights=weights, k=community - uniform))
    result.extend(["selfplay"] * selfplay)
    rng.shuffle(result)
    return result


def _tasks(args, iteration: int, stats: dict, checkpoint: Path, snapshot: Path):
    rng = random.Random(args.seed + iteration * 1_000_003)
    schedule = _schedule(
        args.games, args.selfplay_fraction, args.uniform_opponent_fraction, stats, rng
    )
    # Pair the same opponent and seed across both seats.  The former code only
    # alternated seats in a shuffled opponent list, so nominal pairs frequently
    # faced different policies and retained substantial shop/seat variance.
    counts = defaultdict(int)
    for opponent in schedule:
        counts[opponent] += 1
    result = []
    game_index = 0
    pair_index = 0
    for opponent in sorted(counts):
        count = counts[opponent]
        for _ in range(count // 2):
            seed = (
                args.seed + iteration * 10_000_019 + pair_index * 97 + rng.randrange(97)
            )
            pair_index += 1
            for seat in (0, 1):
                result.append((iteration * args.games + game_index, seed, seat, opponent,
                               str(checkpoint), str(snapshot), args.epsilon,
                               args.algorithm, args.temperature, args.exploration_mix,
                               str(args.runtime), str(args.descriptors), args.top_k))
                game_index += 1
        if count % 2:
            seed = (
                args.seed + iteration * 10_000_019 + pair_index * 97 + rng.randrange(97)
            )
            pair_index += 1
            result.append((iteration * args.games + game_index, seed, rng.randrange(2), opponent,
                           str(checkpoint), str(snapshot), args.epsilon,
                           args.algorithm, args.temperature, args.exploration_mix,
                           str(args.runtime), str(args.descriptors), args.top_k))
            game_index += 1
    assert len(result) == args.games
    rng.shuffle(result)
    return result


def _save_replay(results: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    traces = [trace for row in results if row.get("valid") for trace in row["traces"]]
    if not traces:
        first = next((row for row in results if not row.get("valid")), {})
        raise RuntimeError(f"rollout produced no valid traces: {first.get('error', 'unknown error')}")
    keys = (
        "self_public", "opponent_public", "private_plan", "market_town",
        "checkpoints", "selected", "route_features", "base_scores",
        "candidate_counts", "old_logprobs", "old_values", "policy_entropies",
    )
    payload = {key: np.stack([trace[key] for trace in traces]) for key in keys}
    payload.update(
        opponents=np.asarray([trace["opponent"] for trace in traces]),
        wins=np.asarray([trace["win"] for trace in traces], dtype=np.float16),
        margins=np.asarray([trace["margin"] for trace in traces], dtype=np.float32),
    )
    np.savez(path, **payload)
    valid = [row for row in results if row.get("valid")]
    by_opponent = defaultdict(
        lambda: {"games": 0, "wins": 0, "draws": 0, "losses": 0}
    )
    for row in valid:
        current = by_opponent[row["opponent"]]
        current["games"] += 1
        margin = float(row["margin"])
        current["wins"] += int(margin > 0)
        current["draws"] += int(margin == 0)
        current["losses"] += int(margin < 0)
    scores = [float(row["margin"] > 0) + 0.5 * float(row["margin"] == 0) for row in valid]
    return {
        "games": len(results), "valid_games": len(valid), "errors": len(results) - len(valid),
        "training_sides": len(traces), "win_rate": float(np.mean([row["win"] for row in valid])),
        "score_rate": float(np.mean(scores)),
        "draw_rate": float(np.mean([row["margin"] == 0 for row in valid])),
        "mean_margin": float(np.mean([row["margin"] for row in valid])),
        "games_per_second": len(valid) / max(1e-9, max(row["elapsed"] for row in valid)),
        "mean_game_seconds": float(np.mean([row["elapsed"] for row in valid])),
        "mean_policy_seconds": float(np.mean([row["policy_seconds"] for row in valid])),
        "mean_opponent_seconds": float(np.mean([row["opponent_seconds"] for row in valid])),
        "mean_environment_seconds": float(np.mean([row["environment_seconds"] for row in valid])),
        "by_opponent": dict(by_opponent), "path": str(path),
    }


def _batch_indices(opponents: np.ndarray, size: int, stats: dict, rng) -> np.ndarray:
    groups = {name: np.flatnonzero(opponents == name) for name in np.unique(opponents)}
    community = [name for name in groups if name != "selfplay"]
    result = []
    self_count = size // 4 if "selfplay" in groups else 0
    for index in range(size - self_count):
        if index % 2 == 0:
            name = community[index % len(community)]
        else:
            weights = np.asarray([
                1 - (stats.get(name, {}).get("wins", 0) + 1)
                / (stats.get(name, {}).get("games", 0) + 2)
                for name in community
            ])
            name = rng.choice(community, p=weights / weights.sum())
        result.append(int(rng.choice(groups[name])))
    if self_count:
        result.extend(int(value) for value in rng.choice(groups["selfplay"], self_count, replace=True))
    rng.shuffle(result)
    return np.asarray(result)


def _train(args, paths: list[Path], checkpoint: Path, stats: dict) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(args.train_threads)
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401 - registers the NPU backend with torch
    arrays = [np.load(path) for path in paths]
    keys = ("self_public", "opponent_public", "private_plan", "market_town",
            "checkpoints", "selected", "route_features", "base_scores", "wins", "margins")
    data = {key: np.concatenate([row[key] for row in arrays]) for key in keys}
    opponents = np.concatenate([row["opponents"] for row in arrays])
    model = SymmetricHistoryModel().to(device)
    source = checkpoint if checkpoint.is_file() else BC_CHECKPOINT
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=False)["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed + len(paths) * 101)
    steps = max(1, math.ceil(len(opponents) / args.batch_size)) * args.epochs
    totals = defaultdict(float)
    started = time.perf_counter()
    model.train()
    for _ in range(steps):
        indices = _batch_indices(opponents, args.batch_size, stats, rng)
        def tensor(key, dtype=np.float32):
            return torch.from_numpy(np.asarray(data[key][indices], dtype=dtype)).to(
                device, non_blocking=True
            )
        own, opponent = tensor("self_public"), tensor("opponent_public")
        private, market = tensor("private_plan"), tensor("market_town")
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "npu",
        ):
            context = model.forward_sequence(own, opponent, private, market)
            checkpoints = tensor("checkpoints", np.int64).long()
            decision_context = torch.gather(
                context, 1, checkpoints.unsqueeze(-1).expand(-1, -1, context.shape[-1])
            )
            scores = tensor("base_scores") + model.score_routes(
                decision_context, tensor("route_features")
            )
            selected = tensor("selected", np.int64).long()
            chosen = torch.gather(scores, -1, selected.unsqueeze(-1)).squeeze(-1)
            wins, margins = tensor("wins"), tensor("margins")
            target = (0.8 * wins + 0.2 * (0.5 + 0.5 * torch.tanh(margins / 20_000)))
            target = target.unsqueeze(-1).expand_as(chosen)
            q_loss = F.binary_cross_entropy_with_logits(chosen.float(), target.float())
            value_loss = F.binary_cross_entropy_with_logits(
                model.value_head(context[:, -1]).squeeze(-1).float(), target[:, 0].float()
            )
            next_target = torch.cat([own[:, 1:], opponent[:, 1:], market[:, 1:]], dim=-1)
            auxiliary_loss = F.smooth_l1_loss(
                model.next_public_head(context[:, :-1]).float(), next_target.float()
            )
            loss = q_loss + 0.2 * value_loss + 0.05 * auxiliary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for name, value in (("loss", loss), ("q_loss", q_loss),
                            ("value_loss", value_loss), ("aux_loss", auxiliary_loss)):
            totals[name] += float(value.detach())
    if device.type == "npu":
        torch.npu.synchronize(device)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({"model": state, "algorithm": "opponent_stratified_mc_q"}, checkpoint)
    return {"samples": len(opponents), "gradient_steps": steps,
            "seconds": time.perf_counter() - started,
            "device": str(device), "precision": "bf16" if device.type == "npu" else "fp32",
            **{key: value / steps for key, value in totals.items()}}


def _train_scratch_ppo(args, replay: Path, checkpoint: Path) -> dict[str, float]:
    """On-policy PPO over sparse route-branch decisions; no Tree-Q or BC targets."""
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(args.train_threads)
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401

    row = np.load(replay)
    keys = (
        "self_public", "opponent_public", "private_plan", "market_town",
        "checkpoints", "selected", "route_features", "candidate_counts",
        "old_logprobs", "old_values", "wins", "margins",
    )
    data = {key: row[key] for key in keys}
    episodes = len(data["wins"])
    decisions = data["selected"].shape[1]
    # Kaggle ranks the two agents by the final result.  Optimizing the cash
    # surplus against already-weak opponents steals gradient from close games,
    # so route PPO uses the exact zero-sum objective: win +1, draw 0, loss -1.
    outcome = np.sign(np.asarray(data["margins"], dtype=np.float32))
    returns = outcome[:, None]
    returns = np.broadcast_to(returns, (episodes, decisions)).copy()
    old_values_array = np.asarray(data["old_values"], dtype=np.float32)
    actionable = np.asarray(data["candidate_counts"]) > 1
    # Standard explained variance is 1 - Var(target - prediction) / Var(target).
    # It must also be measured on the decisions where the baseline is consumed;
    # most stored checkpoints have only one candidate and never enter policy loss.
    active_returns = returns[actionable]
    active_old_values = old_values_array[actionable]
    return_variance = float(np.var(active_returns))
    explained_variance = (
        1.0 - float(np.var(active_returns - active_old_values)) / return_variance
        if return_variance > 1e-8 else 0.0
    )
    # A negatively-explained critic is noisier than a zero/constant baseline.
    # Keep training and reporting the value head, but do not inject its error
    # into PPO advantages until it actually explains terminal outcomes.
    value_baseline_used = explained_variance > 0.0
    advantages = returns - old_values_array if value_baseline_used else returns.copy()
    actionable_advantages = advantages[actionable]
    advantages = (
        advantages - float(actionable_advantages.mean())
    ) / max(1e-6, float(actionable_advantages.std()))

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SymmetricHistoryModel().to(device)
    model.load_state_dict(payload["model"])
    value_parameters = list(model.value_head.parameters())
    value_ids = {id(parameter) for parameter in value_parameters}
    policy_parameters = [parameter for parameter in model.parameters() if id(parameter) not in value_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": policy_parameters, "lr": args.lr},
            {"params": value_parameters, "lr": args.value_lr},
        ],
        weight_decay=args.weight_decay,
    )
    if payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
        # load_state_dict also restores stale param-group learning rates.  CLI
        # changes on a resumed run must remain authoritative.
        optimizer.param_groups[0]["lr"] = args.lr
        optimizer.param_groups[1]["lr"] = args.value_lr
        for state in optimizer.state.values():
            for name, value in list(state.items()):
                if torch.is_tensor(value):
                    state[name] = value.to(device)

    feature_keys = {
        "self_public", "opponent_public", "private_plan", "market_town", "route_features"
    }
    integer_keys = {"checkpoints", "selected", "candidate_counts"}
    device_data = {}
    for key in (
        "self_public", "opponent_public", "private_plan", "market_town",
        "checkpoints", "selected", "route_features", "candidate_counts",
        "old_logprobs", "old_values",
    ):
        tensor = torch.from_numpy(np.asarray(data[key]))
        if key in integer_keys:
            tensor = tensor.to(device=device, dtype=torch.long)
        elif key in feature_keys and device.type == "npu":
            tensor = tensor.to(device=device, dtype=torch.bfloat16)
        else:
            tensor = tensor.to(device=device, dtype=torch.float32)
        device_data[key] = tensor
    device_returns = torch.from_numpy(returns).to(device=device, dtype=torch.float32)
    device_advantages = torch.from_numpy(advantages).to(device=device, dtype=torch.float32)

    rng = np.random.default_rng(args.seed + int(replay.stem.split("-")[-1]) * 104729)
    totals = defaultdict(float)
    gradient_steps = 0
    epochs_completed = 0
    started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        permutation = rng.permutation(episodes)
        epoch_kl = []
        for offset in range(0, episodes, args.batch_size):
            indices = permutation[offset : offset + args.batch_size]
            batch_indices = torch.as_tensor(indices, device=device, dtype=torch.long)
            def batch(key):
                return device_data[key].index_select(0, batch_indices)
            own = batch("self_public")
            opponent = batch("opponent_public")
            private = batch("private_plan")
            market = batch("market_town")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "npu",
            ):
                context = model.forward_sequence(own, opponent, private, market)
                checkpoints = batch("checkpoints")
                decision_context = torch.gather(
                    context, 1,
                    checkpoints.unsqueeze(-1).expand(-1, -1, context.shape[-1]),
                )
                logits = model.score_routes(
                    decision_context, batch("route_features")
                ).float() / args.temperature
                counts = batch("candidate_counts")
                active = counts > 1
                candidate_index = torch.arange(logits.shape[-1], device=device)
                logits = logits.masked_fill(
                    candidate_index.view(1, 1, -1) >= counts.unsqueeze(-1), -1e9
                )
                probabilities = torch.softmax(logits, dim=-1)
                if args.exploration_mix > 0.0:
                    valid_candidates = (
                        candidate_index.view(1, 1, -1) < counts.unsqueeze(-1)
                    ).float()
                    uniform = valid_candidates / counts.unsqueeze(-1).float()
                    probabilities = (
                        (1.0 - args.exploration_mix) * probabilities
                        + args.exploration_mix * uniform
                    )
                log_probs = torch.log(probabilities.clamp_min(1e-12))
                selected = batch("selected")
                new_logprob = torch.gather(
                    log_probs, -1, selected.unsqueeze(-1)
                ).squeeze(-1)
                decision_entropy = -(probabilities * log_probs).sum(dim=-1)
                entropy = decision_entropy[active].mean()
                normalized_entropy = (
                    decision_entropy[active]
                    / torch.log(counts[active].float()).clamp_min(1e-6)
                ).mean()
                entropy_scale = torch.clamp(
                    args.target_normalized_entropy
                    / normalized_entropy.detach().clamp_min(1e-3),
                    min=1.0,
                    max=args.max_entropy_coef / max(args.entropy_coef, 1e-8),
                )
                effective_entropy_coef = args.entropy_coef * entropy_scale
                old_logprob = batch("old_logprobs")
                advantage = device_advantages.index_select(0, batch_indices)
                ratio = torch.exp(new_logprob - old_logprob)
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
                ) * advantage
                policy_loss = -torch.minimum(unclipped, clipped)[active].mean()

                values = model.value_head(decision_context).squeeze(-1).float()
                old_values = batch("old_values")
                targets = device_returns.index_select(0, batch_indices)
                clipped_values = old_values + torch.clamp(
                    values - old_values, -args.value_clip, args.value_clip
                )
                value_error = 0.5 * torch.maximum(
                    (values - targets).square(), (clipped_values - targets).square()
                )
                # The critic is an actor baseline at route branch points.  Do
                # not let the many deterministic (one-candidate) checkpoints
                # dominate its regression objective.
                value_loss = value_error[active].mean()
                next_target = torch.cat(
                    [own[:, 1:], opponent[:, 1:], market[:, 1:]], dim=-1
                )
                auxiliary_loss = F.smooth_l1_loss(
                    model.next_public_head(context[:, :-1]).float(), next_target.float()
                )
                loss = (
                    policy_loss + args.value_coef * value_loss
                    - effective_entropy_coef * entropy
                    + args.aux_coef * auxiliary_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = (old_logprob - new_logprob)[active].mean()
                clip_fraction = (
                    ((ratio - 1.0).abs() > args.clip_ratio).float()[active].mean()
                )
            epoch_kl.append(float(approx_kl.detach()))
            for name, value in (
                ("loss", loss), ("policy_loss", policy_loss),
                ("value_loss", value_loss), ("aux_loss", auxiliary_loss),
                ("entropy", entropy), ("normalized_entropy", normalized_entropy),
                ("effective_entropy_coef", effective_entropy_coef),
                ("approx_kl", approx_kl),
                ("clip_fraction", clip_fraction), ("gradient_norm", gradient_norm),
            ):
                totals[name] += float(value.detach())
            gradient_steps += 1
        epochs_completed += 1
        if epoch_kl and float(np.mean(epoch_kl)) > args.target_kl:
            break

    if device.type == "npu":
        torch.npu.synchronize(device)

    def cpu_tree(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_tree(item) for item in value]
        return value

    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(
        {
            "model": state,
            "optimizer": cpu_tree(optimizer.state_dict()),
            "algorithm": "scratch_route_ppo",
            "temperature": args.temperature,
            "exploration_mix": args.exploration_mix,
        },
        checkpoint,
    )
    return {
        "samples": episodes,
        "decisions": episodes * decisions,
        "actionable_decisions": int(actionable.sum()),
        "gradient_steps": gradient_steps,
        "epochs_completed": epochs_completed,
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "precision": "bf16" if device.type == "npu" else "fp32",
        "mean_return": float(returns.mean()),
        "explained_variance_before": explained_variance,
        "value_baseline_used": value_baseline_used,
        **{key: value / max(1, gradient_steps) for key, value in totals.items()},
    }


def _train_gru_ppo(args, replay: Path, checkpoint: Path) -> dict[str, float]:
    """Sequence PPO for V3: the true GRUs remain inside the gradient graph."""
    import torch
    import torch.nn.functional as F

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    from .route_gru import (
        OPPONENT_CLASSES,
        build_route_gru_model,
        export_route_gru_numpy,
    )

    torch.set_num_threads(args.train_threads)
    device = torch.device(args.device)
    row = np.load(replay, allow_pickle=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    Model = build_route_gru_model()
    model = Model().to(device)
    model.load_state_dict(payload["model"])
    if device.type == "npu":
        # Ascend DynamicGRUV2 accepts FP16 rather than FP32 on this server.
        model.public_gru.half()
        model.private_gru.half()
        model.market_gru.half()

    gru_parameters = (
        [
            *model.public_gru.parameters(),
            *model.private_gru.parameters(),
            *model.market_gru.parameters(),
        ]
        if device.type == "npu"
        else []
    )
    history_parameters = [
        *model.public_input.parameters(),
        *model.private_input.parameters(),
        *model.market_input.parameters(),
        *model.fusion.parameters(),
        *model.opponent_probe.parameters(),
        *model.next_public_head.parameters(),
    ]
    if not gru_parameters:
        history_parameters.extend(
            [
                *model.public_gru.parameters(),
                *model.private_gru.parameters(),
                *model.market_gru.parameters(),
            ]
        )
    actor_parameters = [
        *model.branch_encoder.parameters(), *model.route_head.parameters()
    ]
    critic_parameters = list(model.critic.parameters())
    ordinary_parameters = [
        *history_parameters, *actor_parameters, *critic_parameters
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": history_parameters, "lr": args.history_lr},
            {"params": actor_parameters, "lr": args.lr},
            {"params": critic_parameters, "lr": args.value_lr},
        ],
        weight_decay=args.weight_decay,
    )
    gru_master = None
    if gru_parameters:
        from .fp16_master import FP16MasterAdamW

        gru_master = FP16MasterAdamW(
            torch, gru_parameters, lr=args.history_lr, weight_decay=args.weight_decay
        )
    optimizer_layout = "npu_fp16_master" if gru_parameters else "unified_fp32"
    if (
        payload.get("optimizer")
        and payload.get("optimizer_layout", "unified_fp32") == optimizer_layout
    ):
        optimizer.load_state_dict(payload["optimizer"])
        if gru_master is not None and payload.get("gru_optimizer"):
            gru_master.load_state_dict(payload["gru_optimizer"])
        for group, learning_rate in zip(
            optimizer.param_groups,
            (args.history_lr, args.lr, args.value_lr),
            strict=True,
        ):
            group["lr"] = learning_rate
        for state in optimizer.state.values():
            for name, value in list(state.items()):
                if torch.is_tensor(value):
                    state[name] = value.to(device)

    episodes = len(row["margins"])
    outcome = np.sign(np.asarray(row["margins"], dtype=np.float32))
    decisions = row["selected"].shape[1]
    returns = np.broadcast_to(outcome[:, None], (episodes, decisions)).copy()
    counts_np = np.asarray(row["candidate_counts"], dtype=np.int64)
    actionable_np = counts_np > 1
    old_values_np = np.asarray(row["old_values"], dtype=np.float32)
    variance = float(np.var(returns[actionable_np]))
    explained_variance = (
        1.0
        - float(np.var(returns[actionable_np] - old_values_np[actionable_np]))
        / variance
        if variance > 1e-8
        else 0.0
    )
    use_value_baseline = explained_variance > 0.0
    advantages = returns - old_values_np if use_value_baseline else returns.copy()
    active_advantages = advantages[actionable_np]
    advantages = (advantages - float(active_advantages.mean())) / max(
        1e-6, float(active_advantages.std())
    )
    opponent_to_id = {name: index for index, name in enumerate(OPPONENT_CLASSES)}
    opponent_ids = np.asarray(
        [opponent_to_id.get(str(name), -1) for name in row["opponents"]],
        dtype=np.int64,
    )

    rng = np.random.default_rng(
        args.seed + int(payload.get("iteration", 0)) * 104729
    )
    totals = defaultdict(float)
    gradient_steps = 0
    epochs_completed = 0
    early_stopped = False
    started = time.perf_counter()
    model.train()

    def batch_tensor(name: str, indices: np.ndarray, dtype=torch.float32):
        value = torch.from_numpy(np.asarray(row[name][indices]))
        return value.to(device=device, dtype=dtype)

    for _ in range(args.epochs):
        permutation = rng.permutation(episodes)
        epoch_kl = []
        for offset in range(0, episodes, args.sequence_batch_size):
            indices = permutation[offset : offset + args.sequence_batch_size]
            if not len(indices):
                continue
            own = batch_tensor("self_public", indices)
            opponent = batch_tensor("opponent_public", indices)
            private = batch_tensor("private_plan", indices)
            market = batch_tensor("market_town", indices)
            checkpoints = batch_tensor("checkpoints", indices, torch.long)
            candidate_features = batch_tensor("route_features", indices)
            base_scores = batch_tensor("base_scores", indices)
            counts = batch_tensor("candidate_counts", indices, torch.long)
            selected = batch_tensor("selected", indices, torch.long)
            old_logprob = batch_tensor("old_logprobs", indices)
            old_values = batch_tensor("old_values", indices)
            target = torch.from_numpy(returns[indices]).to(device)
            advantage = torch.from_numpy(advantages[indices]).to(device)
            labels = torch.from_numpy(opponent_ids[indices]).to(device)

            context = model.encode_history(own, opponent, private, market)
            decision_context = torch.gather(
                context,
                1,
                checkpoints.unsqueeze(-1).expand(-1, -1, context.shape[-1]),
            )
            logits = (
                base_scores
                + model.score_routes(decision_context, candidate_features)
            ).float() / args.temperature
            candidate_index = torch.arange(logits.shape[-1], device=device)
            valid = candidate_index.view(1, 1, -1) < counts.unsqueeze(-1)
            logits = logits.masked_fill(~valid, -1e9)
            probabilities = torch.softmax(logits, dim=-1)
            if args.exploration_mix > 0.0:
                uniform = valid.float() / counts.unsqueeze(-1).float()
                probabilities = (
                    (1.0 - args.exploration_mix) * probabilities
                    + args.exploration_mix * uniform
                )
            log_probabilities = torch.log(probabilities.clamp_min(1e-12))
            new_logprob = torch.gather(
                log_probabilities, -1, selected.unsqueeze(-1)
            ).squeeze(-1)
            active = counts > 1
            entropy_by_decision = -(
                probabilities * log_probabilities
            ).sum(dim=-1)
            entropy = entropy_by_decision[active].mean()
            normalized_entropy = (
                entropy_by_decision[active]
                / torch.log(counts[active].float()).clamp_min(1e-6)
            ).mean()
            ratio = torch.exp((new_logprob - old_logprob).clamp(-20.0, 20.0))
            unclipped = ratio * advantage
            clipped = torch.clamp(
                ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
            ) * advantage
            policy_loss = -torch.minimum(unclipped, clipped)[active].mean()

            # A noisy value head must not rewrite the causal history encoder.
            values = model.value(decision_context.detach()).float()
            clipped_values = old_values + torch.clamp(
                values - old_values, -args.value_clip, args.value_clip
            )
            value_loss = 0.5 * torch.maximum(
                (values - target).square(), (clipped_values - target).square()
            )[active].mean()

            label_grid = labels.unsqueeze(1).expand_as(counts)
            probe_mask = active & (label_grid >= 0)
            probe_loss = (
                F.cross_entropy(
                    model.opponent_probe(decision_context)[probe_mask].float(),
                    label_grid[probe_mask],
                )
                if probe_mask.any()
                else torch.zeros((), device=device)
            )
            # Eight-turn stride keeps the auxiliary causal but inexpensive.
            dynamics_context = context[:, :-1:8]
            dynamics_target = torch.cat(
                [own[:, 1::8], opponent[:, 1::8], market[:, 1::8]], dim=-1
            )[:, : dynamics_context.shape[1]]
            dynamics_loss = F.smooth_l1_loss(
                model.next_public_head(dynamics_context).float(),
                dynamics_target.float(),
            )
            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy
                + args.opponent_aux_coef * probe_loss
                + args.aux_coef * dynamics_loss
            )
            optimizer.zero_grad(set_to_none=True)
            if gru_master is not None:
                gru_master.zero_grad()
            loss.backward()
            if gru_master is not None:
                gru_master.copy_unscaled_grads(1.0)
            gradient_parameters = [
                *ordinary_parameters,
                *(gru_master.master_parameters if gru_master is not None else []),
            ]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                gradient_parameters, args.max_grad_norm
            )
            optimizer.step()
            if gru_master is not None:
                gru_master.step()
            with torch.no_grad():
                approx_kl = (old_logprob - new_logprob)[active].mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > args.clip_ratio
                ).float()[active].mean()
            epoch_kl.append(float(approx_kl.detach()))
            for name, value in (
                ("loss", loss),
                ("policy_loss", policy_loss),
                ("value_loss", value_loss),
                ("opponent_probe_loss", probe_loss),
                ("dynamics_loss", dynamics_loss),
                ("entropy", entropy),
                ("normalized_entropy", normalized_entropy),
                ("approx_kl", approx_kl),
                ("clip_fraction", clip_fraction),
                ("gradient_norm", gradient_norm),
            ):
                totals[name] += float(value.detach())
            gradient_steps += 1
        epochs_completed += 1
        if args.target_kl > 0.0 and epoch_kl and np.mean(epoch_kl) > args.target_kl:
            early_stopped = True
            break

    def cpu_tree(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_tree(item) for item in value]
        return value

    model_cpu = model.cpu()
    torch.save(
        {
            "architecture": model.architecture,
            "model": model_cpu.state_dict(),
            "optimizer": cpu_tree(optimizer.state_dict()),
            "gru_optimizer": (
                gru_master.state_dict() if gru_master is not None else None
            ),
            "algorithm": "route_gru_v3_ppo",
            "iteration": int(payload.get("iteration", 0)) + 1,
            "optimizer_layout": optimizer_layout,
        },
        checkpoint,
    )
    export_route_gru_numpy(checkpoint, checkpoint.with_suffix(".pkl"))
    if device.type == "npu":
        torch.npu.synchronize(device)
    return {
        "samples": episodes,
        "decisions": episodes * decisions,
        "actionable_decisions": int(actionable_np.sum()),
        "gradient_steps": gradient_steps,
        "epochs_completed": epochs_completed,
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "precision": "fp16-gru-fp32-heads" if device.type == "npu" else "fp32",
        "mean_return": float(returns.mean()),
        "explained_variance_before": explained_variance,
        "value_baseline_used": use_value_baseline,
        "early_stopped": early_stopped,
        **{key: value / max(1, gradient_steps) for key, value in totals.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--descriptors", type=Path, default=DESCRIPTORS)
    parser.add_argument(
        "--history-initial-checkpoint", type=Path, default=V3_HISTORY_INITIAL
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--algorithm", choices=("mc_q", "scratch_ppo", "gru_ppo"), default="mc_q"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--selfplay-fraction", type=float, default=0.25)
    parser.add_argument("--uniform-opponent-fraction", type=float, default=0.40)
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--replay-window", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--history-lr", type=float, default=3e-5)
    parser.add_argument("--value-lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--exploration-mix", type=float, default=0.0)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--target-normalized-entropy", type=float, default=0.65)
    parser.add_argument("--max-entropy-coef", type=float, default=0.12)
    parser.add_argument("--aux-coef", type=float, default=0.02)
    parser.add_argument("--opponent-aux-coef", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument("--train-threads", type=int, default=1)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--keep-replays", action="store_true")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run_dir / "recurrent-route-latest.pt"
    if not checkpoint.is_file():
        if args.algorithm == "gru_ppo":
            import torch
            from .route_gru import build_route_gru_model, export_route_gru_numpy

            torch.manual_seed(args.seed)
            Model = build_route_gru_model()
            initial_model = Model().cpu()
            # Reuse the already trained causal history representation when its
            # tensor shapes match.  Route heads remain zero-initialized.
            initialized_from = None
            if args.history_initial_checkpoint.is_file():
                source = torch.load(
                    args.history_initial_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                ).get("model", {})
                target = initial_model.state_dict()
                copied = {
                    key: value
                    for key, value in source.items()
                    if key in target and tuple(value.shape) == tuple(target[key].shape)
                    and key.startswith(
                        (
                            "public_input.", "private_input.", "market_input.",
                            "public_gru.", "private_gru.", "market_gru.", "fusion.",
                        )
                    )
                }
                target.update(copied)
                initial_model.load_state_dict(target)
                initialized_from = str(args.history_initial_checkpoint)
            torch.save(
                {
                    "architecture": initial_model.architecture,
                    "model": initial_model.state_dict(),
                    "algorithm": "route_gru_v3_ppo",
                    "iteration": 0,
                    "initialized_from": initialized_from,
                },
                checkpoint,
            )
            export_route_gru_numpy(checkpoint, checkpoint.with_suffix(".pkl"))
        elif args.algorithm == "scratch_ppo":
            import torch
            torch.manual_seed(args.seed)
            initial_model = SymmetricHistoryModel().cpu()
            torch.save(
                {
                    "model": initial_model.state_dict(),
                    "algorithm": "scratch_route_ppo",
                    "temperature": args.temperature,
                },
                checkpoint,
            )
        else:
            shutil.copy2(BC_CHECKPOINT, checkpoint)
    stats_path = args.run_dir / "opponent-stats.json"
    history_path = args.run_dir / "history.jsonl"
    start_iteration = 0
    if history_path.is_file():
        with history_path.open(encoding="utf-8") as handle:
            start_iteration = sum(1 for line in handle if line.strip())
    pool_context = mp.get_context("spawn")
    with pool_context.Pool(min(args.workers, args.games), initializer=_worker_init) as pool:
        for iteration in range(start_iteration, start_iteration + args.iterations):
            stats = _read_stats(stats_path)
            snapshot = args.run_dir / f"snapshot-{iteration:03d}.pt"
            shutil.copy2(checkpoint, snapshot)
            if args.algorithm == "gru_ppo":
                shutil.copy2(
                    checkpoint.with_suffix(".pkl"), snapshot.with_suffix(".pkl")
                )
            tasks = _tasks(args, iteration, stats, checkpoint, snapshot)
            wall_started = time.perf_counter()
            results = []
            for result in pool.imap_unordered(_game, tasks, chunksize=1):
                results.append(result)
                if len(results) % max(1, args.progress_every) == 0 or len(results) == len(tasks):
                    print(json.dumps({
                        "event": "rollout_progress", "iteration": iteration,
                        "completed": len(results), "total": len(tasks),
                        "valid": sum(int(row.get("valid", False)) for row in results),
                        "last_game_seconds": round(float(result.get("elapsed", 0.0)), 3),
                        "wall_seconds": round(time.perf_counter() - wall_started, 3),
                    }), flush=True)
            rollout_wall = time.perf_counter() - wall_started
            errors = [row for row in results if not row.get("valid")]
            if errors:
                print(json.dumps({"event": "errors", "examples": errors[:2]}), flush=True)
            replay = args.run_dir / f"replay-{iteration:03d}.npz"
            rollout = _save_replay(results, replay)
            rollout["wall_seconds"] = rollout_wall
            rollout["wall_games_per_second"] = rollout["valid_games"] / rollout_wall
            for name, row in rollout["by_opponent"].items():
                current = stats.setdefault(
                    name, {"games": 0, "wins": 0, "draws": 0, "losses": 0}
                )
                current["games"] += row["games"]
                current["wins"] += row["wins"]
                current["draws"] = current.get("draws", 0) + row.get("draws", 0)
                current["losses"] = current.get("losses", 0) + row.get("losses", 0)
            stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
            if args.algorithm == "gru_ppo":
                training = _train_gru_ppo(args, replay, checkpoint)
            elif args.algorithm == "scratch_ppo":
                training = _train_scratch_ppo(args, replay, checkpoint)
            else:
                paths = sorted(args.run_dir.glob("replay-*.npz"))[-args.replay_window :]
                training = _train(args, paths, checkpoint, stats)
            row = {"event": "iteration", "iteration": iteration,
                   "rollout": rollout, "training": training}
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if not args.keep_replays:
                replay.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

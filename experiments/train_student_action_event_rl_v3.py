#!/usr/bin/env python3
"""Minimal on-policy PPO smoke for the post-step-288 v3 per-cell actor.

The opening remains the frozen replay policy.  At every day boundary from
step 288 through 672 the stochastic actor owns RELEASE and placement choices
through the exact native v3 mask.  FastEnv supplies only the terminal result;
the update uses clipped PPO with a paired-seed leave-one-out baseline and a
win-first, bounded margin-shaped reward.

Formal rollout defaults to the parity-gated C++ Thomas and MetaV4 opponents;
the clean native replay shell is opt-in through ``--opponents``.  The old
Python-opponent path requires an explicit audit-only flag.  This script never
writes the production agent or R1 binary, and every rollout is saved before an
optimizer step is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
# These must be fixed before NumPy/Torch initialize their thread pools.  Each
# rollout already owns a process; inheriting the host-wide defaults multiplies
# 56 workers into thousands of runnable threads.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np
import torch

from experiments.collect_student_state_dagger_v3 import _load, _observations
from experiments.student_action_event_agent import (
    EVENT_CLASSES,
    ROOT,
    StudentActionEventAgent,
    production,
)
from experiments.train_midgame_autofill_v3 import build_model
from experiments.train_midgame_student_v1 import _sha256


FORMAL_OPPONENTS = (
    "thomas_2945", "melon_2749", "demand_preserving", "ahmed_v47",
    "pipe8", "herd_safe_2700", "salemali7_2900",
)
NATIVE_OPPONENTS = {
    "thomas_2945_cpp": {
        "module": "thomas_2945_cpp_native",
        "directory": ROOT / "experiments/native_opponents/thomas_2945_cpp",
        "asset": "thomas_2945.assets.bin",
    },
    "metav4_2965": {
        "module": "metav4_2965_native",
        "directory": ROOT / "experiments/native_opponents/metav4_2965",
        "asset": "metav4_2965.assets.bin",
    },
    "salemali7_2900": {
        "module": "salemali7_2900_native",
        "directory": ROOT / "experiments/native_opponents/salemali7_2900",
        "asset": "salemali7_2900.assets.bin",
    },
    "fieldcraft_2887": {
        "module": "fieldcraft_2887_native",
        "directory": ROOT / "experiments/native_opponents/fieldcraft_2887",
        "asset": "fieldcraft_2887.assets.bin",
    },
}
NATIVE_POOL = (*NATIVE_OPPONENTS, "replay_clean")
NATIVE_JOB_OPPONENT_CODES = {
    "thomas_2945_cpp": 1, "metav4_2965": 2, "replay_clean": 3,
    "salemali7_2900": 4, "fieldcraft_2887": 5,
}
NATIVE_JOB_REPLAY_FAMILIES = ("G397",)
STUDENT_STEPS = tuple(range(288, 673, 24))
ROLLOUT_SCHEMA = "v3_action_event_on_policy_rollout_v2_day_bundle"
NATIVE_ROLLOUT_FORMAT = "native_job_batch_npz_v1"
NATIVE_ROLLOUT_METADATA = "__rollout_metadata_json__"
PPO_ALGORITHM = (
    "on_policy_clipped_ppo_day_bundle_paired_seed_loo_unscaled_advantage")
_NATIVE_MODULE_CACHE = {}
_REPLAY_POOL_CACHE = None


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("opponent list is empty")
    return result


def _native_artifacts(opponents: tuple[str, ...]) -> dict[str, dict]:
    artifacts = {}
    for name in opponents:
        if name == "replay_clean":
            from meta_agent.src.teammate_expanded_routes import load_action_tapes
            actions = ROOT / "agent/route_actions.json.zlib"
            metadata = ROOT / "agent/route_library.json"
            fast_builds = sorted((ROOT / "fast_kaggriculture/python/fast_kaggriculture").glob(
                "_fast_kaggriculture*.so"))
            if len(fast_builds) != 1:
                raise RuntimeError("replay_clean needs exactly one FastEnv build")
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            action_tapes = load_action_tapes(actions)
            all_entries = payload["opponent_routes"]
            family_to_bundle = {
                str(entry["family"]): index
                for index, entry in enumerate(all_entries)
            }
            if len(all_entries) != 245 or len(family_to_bundle) != 245:
                raise RuntimeError("replay bundle must contain 245 unique families")
            clean_entries = [entry for entry in all_entries
                             if entry["route_id"] in action_tapes and
                             int(entry.get("source_execution_hard_failures") or 0) <= 0]
            if len(clean_entries) != 68:
                raise RuntimeError("replay_clean needs exactly 68 clean routes")
            clean_by_family = {
                str(entry["family"]): (index, entry)
                for index, entry in enumerate(clean_entries)
            }
            if any(family not in clean_by_family
                   for family in NATIVE_JOB_REPLAY_FAMILIES):
                raise RuntimeError("native JobBatch replay allowlist is not clean")
            entries = [clean_by_family[family] for family in
                       NATIVE_JOB_REPLAY_FAMILIES]
            artifacts[name] = {
                "kind": "replay",
                "module_sha256": _sha256(fast_builds[0]),
                "asset_sha256": _sha256(actions),
                "metadata_sha256": _sha256(metadata),
                "pool_routes": len(entries),
                "entries": [{
                    "clean_index": clean_index,
                    "route_id": entry["route_id"],
                    "family": entry["family"],
                    "bundle_index": family_to_bundle[str(entry["family"])],
                } for clean_index, entry in entries],
            }
            continue
        spec = NATIVE_OPPONENTS[name]
        builds = sorted((spec["directory"] / "build").glob(
            f'{spec["module"]}*.so'))
        if len(builds) != 1:
            raise RuntimeError(
                f"{name} needs exactly one native build, found {len(builds)}")
        asset = spec["directory"] / spec["asset"]
        if not asset.is_file():
            raise FileNotFoundError(asset)
        artifacts[name] = {
            "kind": "public_cpp",
            "module_name": spec["module"],
            "module_path": str(builds[0].resolve()),
            "module_sha256": _sha256(builds[0]),
            "asset_path": str(asset.resolve()),
            "asset_sha256": _sha256(asset),
        }
    return artifacts


def _load_native(name: str, path: str):
    cached = _NATIVE_MODULE_CACHE.get(path)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _NATIVE_MODULE_CACHE[path] = module
    return module


def _load_replay_pool():
    global _REPLAY_POOL_CACHE
    if _REPLAY_POOL_CACHE is None:
        from experiments.native_opponents.replay_pool.smoke import build_pool
        _REPLAY_POOL_CACHE = build_pool(
            ROOT / "agent/route_actions.json.zlib",
            ROOT / "agent/route_library.json", 0)
    return _REPLAY_POOL_CACHE


def _worker_init() -> None:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def _validate_native_job_routes(arrays: dict[str, np.ndarray],
                                game_count: int,
                                jobs: list[dict] | None = None) -> None:
    route = np.asarray(arrays.get("route"))
    opponent = np.asarray(arrays.get("opponent"))
    if (route.shape != (game_count,) or opponent.shape != (game_count,) or
            not np.issubdtype(route.dtype, np.integer) or
            not np.issubdtype(opponent.dtype, np.integer) or
            np.any(~np.isin(opponent, tuple(NATIVE_JOB_OPPONENT_CODES.values()))) or
            np.any((opponent != 3) & (route != -1)) or
            np.any((opponent == 3) & ((route < 0) | (route >= 245)))):
        raise RuntimeError("native JobBatch opponent/route identity is invalid")
    if jobs is not None:
        expected_opponents = np.asarray(
            [NATIVE_JOB_OPPONENT_CODES[job["opponent"]] for job in jobs],
            dtype=opponent.dtype)
        expected_routes = np.asarray([
            int(job["native_bundle_route_index"])
            if job["opponent"] == "replay_clean" else -1
            for job in jobs
        ], dtype=route.dtype)
        if (not np.array_equal(opponent, expected_opponents) or
                not np.array_equal(route, expected_routes)):
            raise RuntimeError("native JobBatch opponent/route identity drift")


class RolloutActionEventAgent(StudentActionEventAgent):
    """Capture the exact normalized tensors consumed by the sampled policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rl_days: list[dict] = []

    def _install_student_plan(self, observation: dict) -> dict:
        captured_state: dict = {}
        captured_events: list[dict] = []
        original_initial = self.model.initial_hidden
        original_step = self.model.step

        def initial_hidden(context, packed, length, continuous, categories, count):
            captured_state.update({
                "context": _numpy(context),
                "observation": _numpy(packed),
                "observation_length": _numpy(length),
                "token_continuous": _numpy(continuous),
                "token_categories": [_numpy(value) for value in categories],
                "token_count": _numpy(count),
            })
            return original_initial(
                context, packed, length, continuous, categories, count)

        def step(hidden, resources, cells, stages, previous, legal):
            captured_events.append({
                "resources": _numpy(resources),
                "cell": int(cells.item()),
                "stage": int(stages.item()),
                "previous": int(previous.item()),
                "legal": _numpy(legal),
            })
            return original_step(hidden, resources, cells, stages, previous, legal)

        self.model.initial_hidden = initial_hidden
        self.model.step = step
        try:
            record = super()._install_student_plan(observation)
        finally:
            self.model.initial_hidden = original_initial
            self.model.step = original_step

        if len(captured_events) != len(record["events"]):
            raise RuntimeError("rollout tensor/event count mismatch")
        previous = len(EVENT_CLASSES)
        for native, sampled in zip(captured_events, record["events"]):
            if (native["cell"] != sampled["cell"] or
                    native["stage"] != sampled["stage"] or
                    native["previous"] != previous or
                    not bool(native["legal"][0, sampled["selected_class"]]) or
                    sampled["legal_mask"] != sum(
                        (1 << index) for index, value in
                        enumerate(native["legal"][0]) if value)):
                raise RuntimeError("rollout capture disagrees with native event")
            native.update({
                "action": int(sampled["selected_class"]),
                "old_logprob": float(sampled["log_prob"]),
                "old_entropy": float(sampled["entropy"]),
                "legal_mask": int(sampled["legal_mask"]),
            })
            previous = native["action"]
        self.rl_days.append({
            "step": production.policy.observed_step(observation),
            "state": captured_state,
            "events": captured_events,
        })
        return record


def _create_agent(job: dict):
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    dynamic = RolloutActionEventAgent(
        config,
        binary_path=Path(job["binary"]),
        checkpoint_path=Path(job["checkpoint"]),
        manifest_path=Path(job["manifest"]),
        sample=True,
        temperature=float(job["temperature"]),
        student_steps=STUDENT_STEPS,
        allow_unattested_steps=False,
    )
    replay = production.replay_deployment()
    route = production.create_replay_agent(
        replay, f'v3_ppo_replay_{job["seed"]}_{job["seat"]}')
    return production.ReplayThenDynamicAgent(
        route, dynamic, 288, handoff_land=None, handoff_floor=0,
        handoff_delay_days=0, selector=None)


def _collect_one(job: dict) -> dict:
    from fast_kaggriculture import Config, FastEnv

    started = time.perf_counter()
    candidate = None
    try:
        # Sampling is reproducible for a fixed rollout job but the seed is
        # audit-only and is never passed through the actor input.
        torch.manual_seed(int(job["policy_seed"]))
        np.random.seed(int(job["policy_seed"]) & 0xFFFFFFFF)
        if job.get("native_kind") == "replay":
            from fast_kaggriculture import NativeReplayOpponent
            executor, entries = _load_replay_pool()
            entry = entries[int(job["native_route_index"])]
            if (entry["route_id"] != job["native_route_id"] or
                    entry["family"] != job["native_route_family"]):
                raise RuntimeError("native replay route metadata drift")
            opponent = NativeReplayOpponent(
                executor, int(job["native_route_index"]), True, 0)
            takes_configuration = False
        elif job["opponent_backend"] == "native_cpp":
            module = _load_native(
                job["native_module_name"], job["native_module_path"])
            opponent = module.Opponent(job["native_asset_path"])
            takes_configuration = False
        else:
            path = ROOT / "opponents" / job["opponent"] / "main.py"
            opponent = _load(
                path,
                f'v3_ppo_{job["opponent"]}_{job["seed"]}_{job["seat"]}').agent
            takes_configuration = len(inspect.signature(opponent).parameters) > 1
        candidate = _create_agent(job)
        env = FastEnv(Config(), int(job["seed"]))
        native_rollout = job["opponent_backend"] == "native_cpp"
        if native_rollout:
            env.reset_raw(int(job["seed"]))
        else:
            state = list(env.reset(int(job["seed"])))
        frames = 0
        while not env.done:
            if native_rollout:
                # FastEnv already returns a fresh observation with the acting
                # player set.  Building both observations and JSON-roundtripping
                # them cost more than the native simulator/opponent combined.
                observation = env.observation(int(job["seat"]))
                if observation.get("step") is None:
                    observation["step"] = env.step_count
                candidate_action = candidate(observation, {})
                if job.get("native_kind") == "replay":
                    opponent_action = opponent.action(
                        env, 1 - int(job["seat"]), env.step_count)
                else:
                    opponent_action = opponent.action(env, 1 - int(job["seat"]))
                actions = [None, None]
                actions[int(job["seat"])] = candidate_action
                actions[1 - int(job["seat"])] = opponent_action
                env.step_raw(actions)
                frames += 1
                continue
            actions = []
            for player, observation in enumerate(_observations(state)):
                if player == int(job["seat"]):
                    action = candidate(observation, {})
                elif job.get("native_kind") == "replay":
                    action = opponent.action(env, player, env.step_count)
                elif job["opponent_backend"] == "native_cpp":
                    action = opponent.action(env, player)
                else:
                    action = (opponent(observation, {}) if takes_configuration
                              else opponent(observation))
                actions.append(action)
            state = env.step(actions)
            frames += 1
        dynamic = candidate.dynamic
        summary = dynamic.student_summary()
        successful = [int(day["step"]) for day in dynamic.rl_days]
        if (frames != 719 or successful != list(STUDENT_STEPS) or
                summary.get("successful_steps") != list(STUDENT_STEPS) or
                summary.get("fallbacks") != 0 or summary.get("illegal") != 0 or
                dynamic.student_failures):
            raise RuntimeError(f"invalid stochastic actor rollout: {summary}")
        own = float(env.rewards[int(job["seat"])])
        rival = float(env.rewards[1 - int(job["seat"])])
        margin = own - rival
        outcome = float((margin > 0) - (margin < 0))
        reward = outcome + float(job["margin_weight"]) * math.tanh(
            margin / float(job["margin_scale"]))
        return {
            "status": "PASS",
            "seed": int(job["seed"]), "opponent": job["opponent"],
            "opponent_backend": job["opponent_backend"],
            "opponent_variant": job.get("opponent_variant", job["opponent"]),
            "opponent_module_sha256": job.get("native_module_sha256"),
            "opponent_asset_sha256": job.get("native_asset_sha256"),
            "seat": int(job["seat"]), "policy_seed": int(job["policy_seed"]),
            "policy_sha256": job["checkpoint_sha256"],
            "binary_sha256": job["binary_sha256"],
            "manifest_sha256": job["manifest_sha256"],
            "frames": frames, "own_cash": own, "opponent_cash": rival,
            "margin": margin, "outcome": outcome, "reward": reward,
            "illegal": int(summary.get("illegal", 0)),
            "fallbacks": int(summary.get("fallbacks", 0)),
            "days": dynamic.rl_days,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "status": "FAIL", "seed": int(job["seed"]),
            "opponent": job["opponent"], "seat": int(job["seat"]),
            "opponent_backend": job["opponent_backend"],
            "error": repr(error), "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        if candidate is not None:
            candidate.close()


def _collect_native_job_batch(
        args, jobs: list[dict]) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    """Run the accepted replay opening, v3 actor, and suffix entirely in C++."""
    if set(args.opponents) - set(NATIVE_JOB_OPPONENT_CODES):
        raise ValueError("native JobBatch opponent is unsupported")
    from experiments.native_student_actor.native_job_batch import ppo_games
    from meta_agent.src.native_teammate_executor import NativeTeammateBundle

    build = ROOT / "experiments/native_student_rollout/build"
    modules = list(build.glob("_paused_plan*.so"))
    if len(modules) != 1:
        raise RuntimeError(f"native JobBatch needs one extension, found {modules}")
    if _sha256(modules[0]) != jobs[0]["native_job_module_sha256"]:
        raise RuntimeError("native JobBatch module changed after input validation")
    sys.path.insert(0, str(build))
    native = importlib.import_module("_paused_plan")
    bundle = NativeTeammateBundle(
        ROOT / "agent/teammate_base.py",
        ROOT / "agent/route_actions.json.zlib",
        ROOT / "agent/route_library.json")
    deployment_routes = [bundle.index(name) for name in
                         ("G275", "G195", "G024", "G316", "G267")]
    config = json.loads((ROOT / "policy/r1/config.json").read_text())
    probe = production.policy.Agent(config=config, binary_path=args.binary)
    try:
        settings = [float(probe.config[name]) for name in
                    production.policy._ORDER[:probe.settings_count]]
    finally:
        probe.close()
    routes = []
    for job in jobs:
        if job["opponent"] != "replay_clean":
            routes.append(-1)
            continue
        route = bundle.index(job["native_route_family"])
        if (route != int(job["native_bundle_route_index"]) or
                bundle.route_ids[route] != job["native_route_id"]):
            raise RuntimeError("native replay bundle route metadata drift")
        routes.append(route)
    thomas = (ROOT / "experiments/native_opponents/thomas_2945_cpp/"
              "thomas_2945.assets.bin")
    meta = (ROOT / "experiments/native_opponents/metav4_2965/"
            "metav4_2965.assets.bin")
    salemali = (ROOT / "experiments/native_opponents/salemali7_2900/"
                "salemali7_2900.assets.bin")
    fieldcraft = (ROOT / "experiments/native_opponents/fieldcraft_2887/"
                  "fieldcraft_2887.assets.bin")
    batch = native.JobBatch(
        str(args.binary), bundle.executor,
        [int(job["seed"]) for job in jobs],
        [int(job["seat"]) for job in jobs],
        [NATIVE_JOB_OPPONENT_CODES[job["opponent"]] for job in jobs],
        routes,
        [int(job["policy_seed"]) for job in jobs], settings,
        deployment_routes, str(thomas), str(meta), str(salemali),
        str(fieldcraft))
    started = time.perf_counter()
    batch.run(args.native_job_threads, 2 << 20, False)
    prefix_seconds = time.perf_counter() - started
    suffix = batch.run_native_actor_suffix(
        str(args.native_weights), 0, args.native_job_threads, 2 << 20, True)
    arrays = {name: np.asarray(value)
              for name, value in batch.ppo_arrays().items()}
    _validate_native_job_routes(arrays, len(jobs), jobs)
    results = ppo_games(
        arrays, margin_weight=args.margin_weight, margin_scale=args.margin_scale,
        materialize_events=False)
    total_seconds = time.perf_counter() - started
    if len(results) != len(jobs):
        raise RuntimeError("native JobBatch result count mismatch")
    for result, job in zip(results, jobs):
        if (result["seed"] != job["seed"] or result["seat"] != job["seat"] or
                result["opponent"] != job["opponent"] or
                result["route"] != (-1 if job["opponent"] != "replay_clean"
                                    else job["native_bundle_route_index"])):
            raise RuntimeError("native JobBatch result identity drift")
        result.update({
            "opponent_backend": "native_cpp",
            "opponent_variant": job.get("opponent_variant", job["opponent"]),
            "opponent_module_sha256": job.get("native_module_sha256"),
            "opponent_asset_sha256": job.get("native_asset_sha256"),
            "policy_sha256": job["checkpoint_sha256"],
            "binary_sha256": job["binary_sha256"],
            "manifest_sha256": job["manifest_sha256"],
            "elapsed_seconds": total_seconds / len(jobs),
        })
    summary = batch.summary()["cases"]
    if not all(row["done"] and row["step"] == 719 and not row["error"]
               for row in summary):
        raise RuntimeError("native JobBatch did not reach every terminal")
    metrics = {
        "engine": "pure_cpp_job_batch",
        "games": len(results),
        "events": int(np.asarray(arrays["event_action"]).size),
        "prefix_seconds": prefix_seconds,
        "actor_plan_seconds": float(suffix["plan_seconds"]),
        "environment_seconds": float(suffix["environment_seconds"]),
        "suffix_seconds": float(suffix["wall_seconds"]),
        "wall_seconds": total_seconds,
        "games_per_second": len(results) / total_seconds,
        "threads": args.native_job_threads,
        "module_sha256": jobs[0]["native_job_module_sha256"],
    }
    print(json.dumps({"event": "native_job_rollout", **metrics}), flush=True)
    return results, metrics, arrays


def _batch_terms(model, games: list[dict], device: torch.device,
                 temperature: float, *, safe_hidden_index_copy: bool | None = None):
    """Re-evaluate variable-length day sequences in one padded time loop."""
    if safe_hidden_index_copy is None:
        # torch-npu index_copy mutates its input storage; clone preserves the
        # initial hidden used by tanh backward. CPU does not need this copy.
        safe_hidden_index_copy = device.type == "npu" and torch.is_grad_enabled()
    days = [(game_index, day) for game_index, game in enumerate(games)
            for day in game["days"]]
    if not days:
        raise RuntimeError("empty PPO trajectory batch")
    native = "native_index" in days[0][1]
    if native:
        values = days[0][1]["native_arrays"]
        day_ids = np.asarray([day["native_index"] for _game, day in days],
                             dtype=np.int64)
        offsets = values["day_event_offsets"]

    def state(name, dtype):
        rows = (values[name][day_ids] if native else np.concatenate(
            [day["state"][name] for _game, day in days], axis=0))
        return torch.from_numpy(np.ascontiguousarray(rows)).to(
                device=device, dtype=dtype)

    categories = [torch.from_numpy(np.ascontiguousarray(
        values["token_categories"][day_ids, index] if native else
        np.concatenate([
            day["state"]["token_categories"][index] for _game, day in days],
            axis=0))).to(device=device, dtype=torch.long)
                  for index in range(7)]
    hidden = model.initial_hidden(
        state("context", torch.float32),
        state("observation", torch.float32),
        state("observation_length", torch.float32),
        state("token_continuous", torch.float32), categories,
        state("token_count", torch.long))
    previous = torch.full(
        (len(days),), len(EVENT_CLASSES), device=device, dtype=torch.long)
    lengths = (offsets[day_ids + 1] - offsets[day_ids] if native else
               np.asarray([len(day["events"]) for _game, day in days]))
    logprobs, old_logprobs, entropies = [], [], []
    actionable, event_games, event_days = [], [], []
    for event_index in range(int(lengths.max())):
        active_np = np.flatnonzero(event_index < lengths)
        active = torch.as_tensor(active_np, device=device, dtype=torch.long)
        event_ids = offsets[day_ids[active_np]] + event_index if native else None
        rows = ([] if native else
                [days[index][1]["events"][event_index] for index in active_np])
        resources_np = (values["event_resources"][event_ids] if native else
                        np.concatenate([row["resources"] for row in rows], axis=0))
        resources = torch.from_numpy(np.ascontiguousarray(resources_np)).to(
                device=device, dtype=torch.float32)
        cells = torch.as_tensor(
            values["event_cell"][event_ids] if native else
            [row["cell"] for row in rows], device=device, dtype=torch.long)
        stages = torch.as_tensor(
            values["event_stage"][event_ids] if native else
            [row["stage"] for row in rows], device=device, dtype=torch.long)
        legal_np = (values["event_legal"][event_ids] if native else
                    np.concatenate([row["legal"] for row in rows], axis=0))
        legal = torch.from_numpy(np.ascontiguousarray(legal_np)).to(
                device=device, dtype=torch.bool)
        actions = torch.as_tensor(
            values["event_action"][event_ids] if native else
            [row["action"] for row in rows], device=device, dtype=torch.long)
        logits, next_hidden = model.step(
            hidden.index_select(0, active), resources, cells, stages,
            previous.index_select(0, active), legal)
        distribution = torch.distributions.Categorical(
            logits=logits.float() / temperature)
        logprobs.append(distribution.log_prob(actions))
        old_logprobs.append(torch.as_tensor(
            values["old_logprob"][event_ids] if native else
            [row["old_logprob"] for row in rows],
            device=device, dtype=torch.float32))
        entropies.append(distribution.entropy())
        actionable.extend(
            (np.count_nonzero(legal_np, axis=1) > 1).tolist() if native else
            (int(row["legal_mask"]).bit_count() > 1 for row in rows))
        event_games.extend(days[index][0] for index in active_np)
        event_days.extend(map(int, active_np))
        hidden = (hidden.clone() if safe_hidden_index_copy else hidden).index_copy(
            0, active, next_hidden)
        previous = previous.index_copy(0, active, actions)
    return (
        torch.cat(logprobs), torch.cat(old_logprobs), torch.cat(entropies),
        torch.tensor(actionable, device=device, dtype=torch.bool),
        torch.tensor(event_games, device=device, dtype=torch.long),
        torch.tensor(event_days, device=device, dtype=torch.long),
        torch.tensor([game for game, _day in days],
                     device=device, dtype=torch.long),
    )


def _approx_kl(new_logprob, old_logprob, active):
    logratio = (new_logprob - old_logprob)[active]
    return (torch.expm1(logratio) - logratio).clamp_min(0).mean()


def _stratified_loo_advantages(results: list[dict], rewards: np.ndarray):
    """Remove exogenous opponent/seat difficulty without entering the policy."""
    if len(results) != len(rewards) or not len(rewards):
        raise ValueError("advantage inputs must be non-empty and aligned")
    global_loo = (rewards.copy() if len(rewards) == 1 else
                  rewards - (rewards.sum() - rewards) / (len(rewards) - 1))
    advantages = global_loo.copy()
    groups = defaultdict(list)
    for index, row in enumerate(results):
        groups[(row["opponent"], int(row["seat"]))].append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        values = rewards[indices]
        advantages[indices] = (
            values - (values.sum() - values) / (len(values) - 1))
    return advantages


def _paired_seed_loo_advantages(results: list[dict], rewards: np.ndarray):
    """Remove shared environment-seed noise using only other trajectories."""
    advantages = _stratified_loo_advantages(results, rewards)
    groups = defaultdict(list)
    for index, row in enumerate(results):
        groups[int(row["seed"])].append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        values = rewards[indices]
        advantages[indices] = (
            values - (values.sum() - values) / (len(values) - 1))
    return advantages


def _native_session_rewards(arrays: dict[str, np.ndarray],
                            margin_weight: float,
                            margin_scale: float) -> np.ndarray:
    own = np.asarray(arrays["own_cash"], dtype=np.float64)
    rival = np.asarray(arrays["rival_cash"], dtype=np.float64)
    if (own.ndim != 1 or rival.shape != own.shape or not len(own) or
            not np.all(np.isfinite(own)) or not np.all(np.isfinite(rival)) or
            not math.isfinite(margin_weight) or
            not math.isfinite(margin_scale) or margin_scale <= 0):
        raise RuntimeError("invalid native terminal reward arrays")
    margin = own - rival
    return np.sign(margin) + margin_weight * np.tanh(margin / margin_scale)


def _attach_native_day_advantages(
        results: list[dict], arrays: dict[str, np.ndarray],
        day_advantages: np.ndarray) -> None:
    """Join native array-order day values to sorted game dictionaries."""
    from experiments.native_student_actor.native_job_batch import OPPONENT_NAMES

    sessions = np.asarray(arrays["day_session_index"], dtype=np.int64)
    steps = np.asarray(arrays["day_step"], dtype=np.int64)
    values = np.asarray(day_advantages, dtype=np.float64)
    session_count = len(np.asarray(arrays["seed"]))
    if (sessions.shape != steps.shape or values.shape != steps.shape or
            np.any((sessions < 0) | (sessions >= session_count)) or
            not np.all(np.isfinite(values))):
        raise RuntimeError("invalid native day-aligned advantage arrays")
    identity_to_session = {}
    for session in range(session_count):
        opponent_code = int(arrays["opponent"][session])
        if opponent_code not in OPPONENT_NAMES:
            raise RuntimeError("unknown native opponent code in day join")
        identity = (
            int(arrays["seed"][session]), OPPONENT_NAMES[opponent_code],
            int(arrays["seat"][session]), int(arrays["route"][session]),
            int(arrays["policy_seed"][session]),
        )
        if identity in identity_to_session:
            raise RuntimeError("duplicate native identity in day join")
        identity_to_session[identity] = session
    claimed_days = set()
    for game in results:
        identity = (
            int(game["seed"]), game["opponent"], int(game["seat"]),
            int(game["route"]), int(game["policy_seed"]),
        )
        session = identity_to_session.pop(identity, None)
        if session is None:
            raise RuntimeError("native result/array identity mismatch in day join")
        day_indices = np.flatnonzero(sessions == session)
        by_step = {int(steps[index]): int(index) for index in day_indices}
        if len(by_step) != len(day_indices) or len(game["days"]) != len(by_step):
            raise RuntimeError("native result/array day coverage mismatch")
        for day in game["days"]:
            index = by_step.get(int(day["step"]))
            if index is None or index in claimed_days:
                raise RuntimeError("native result/array day identity mismatch")
            day["day_state_advantage"] = float(values[index])
            claimed_days.add(index)
    if identity_to_session or len(claimed_days) != len(steps):
        raise RuntimeError("unclaimed native session/day in advantage join")


def _day_state_crossfit_advantages(
        results: list[dict], arrays: dict[str, np.ndarray],
        checkpoint: dict, args) -> tuple[dict, np.ndarray]:
    """Build and attach the opt-in, training-only day-state baseline."""
    from experiments.day_state_control_variate import (
        actionable_day_counts,
        crossfit_global_hgb,
        extract_day_state_features,
    )

    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict):
        raise RuntimeError("behavior checkpoint has no normalization contract")
    started = time.perf_counter()
    features = extract_day_state_features(
        arrays["observation"], arrays["observation_length"],
        arrays["day_step"], normalization)
    extraction_seconds = time.perf_counter() - started
    counts = actionable_day_counts(
        len(features), arrays["event_day_index"], arrays["event_legal_mask"])
    native_rewards = _native_session_rewards(
        arrays, args.margin_weight, args.margin_scale)
    result = crossfit_global_hgb(
        features, arrays["day_session_index"], arrays["day_step"],
        arrays["seed"], native_rewards, actionable_counts=counts,
        n_splits=6, max_workers=args.day_state_critic_workers)
    _attach_native_day_advantages(results, arrays, result.advantages)

    # The native arrays retain JobBatch order while ``results`` is sorted.
    # Validate the terminal target through the same identity join used above.
    reward_by_identity = {}
    from experiments.native_student_actor.native_job_batch import OPPONENT_NAMES
    for session, reward in enumerate(native_rewards):
        identity = (
            int(arrays["seed"][session]),
            OPPONENT_NAMES[int(arrays["opponent"][session])],
            int(arrays["seat"][session]), int(arrays["route"][session]),
            int(arrays["policy_seed"][session]),
        )
        reward_by_identity[identity] = float(reward)
    for game in results:
        identity = (
            int(game["seed"]), game["opponent"], int(game["seat"]),
            int(game["route"]), int(game["policy_seed"]),
        )
        if (identity not in reward_by_identity or
                abs(float(game["reward"]) - reward_by_identity[identity]) > 1e-7):
            raise RuntimeError("native result/array reward mismatch")
    valid_advantages = result.advantages[counts > 0]
    if not len(valid_advantages) or not np.all(np.isfinite(valid_advantages)):
        raise RuntimeError("day-state baseline produced no finite policy signal")
    metrics = {
        **result.metrics,
        "extraction_seconds": extraction_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    return metrics, valid_advantages


def _day_bundle_objective(new_logprob, old_logprob, entropy_values,
                          actionable, event_days, day_games,
                          game_advantages, clip_ratio, *,
                          day_advantages=None):
    """Build exact autoregressive day ratios and an equal-day PPO loss."""
    day_count = len(day_games)
    active_days = event_days[actionable]
    event_logratio = (new_logprob - old_logprob)[actionable]
    day_logratio = torch.zeros(
        day_count, device=new_logprob.device, dtype=new_logprob.dtype)
    day_logratio.index_add_(0, active_days, event_logratio)
    day_actionable = torch.zeros_like(day_logratio)
    day_actionable.index_add_(0, active_days, torch.ones_like(event_logratio))
    day_entropy = torch.zeros_like(day_logratio)
    day_entropy.index_add_(0, active_days, entropy_values[actionable])
    day_entropy = day_entropy / day_actionable.clamp_min(1)
    valid_days = day_actionable > 0
    if not bool(valid_days.any()):
        raise RuntimeError("PPO batch has no actionable day")

    # A day is the macro action. Its log probability is the sum of its slot
    # log probabilities, and the terminal advantage applies once to that
    # complete action. Dividing by slot count would optimize a different,
    # action-length-weighted objective instead of expected terminal reward.
    if day_advantages is None:
        advantage = game_advantages.index_select(0, day_games)
    else:
        if (day_advantages.ndim != 1 or len(day_advantages) != day_count or
                day_advantages.device != new_logprob.device):
            raise ValueError("day-aligned advantage shape/device mismatch")
        advantage = day_advantages
    ratio = torch.exp(day_logratio)
    unclipped = ratio * advantage
    clipped = torch.clamp(
        ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    return (-torch.minimum(unclipped, clipped)[valid_days].mean(),
            day_entropy[valid_days].mean(), {
        "event_logratio": event_logratio,
        "day_logratio": day_logratio,
        "day_actionable": day_actionable,
        "valid_days": valid_days,
        "day_ratio": ratio,
        "day_advantage": advantage,
    })


def _full_policy_drift(model, games, device, temperature, batch_games,
                       clip_ratio):
    """Exact post-update drift on the complete behavior rollout."""
    totals = {
        name: torch.zeros((), device=device, dtype=torch.float32)
        for name in (
            "event_logratio", "event_approx_kl", "event_clipped",
            "day_logratio", "day_bundle_approx_kl", "day_clipped",
            "event_count", "day_count", "total_day_count",
        )
    }
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(games), batch_games):
            batch = games[offset:offset + batch_games]
            (new, old, entropy_values, actionable, _event_games,
             event_days, day_games) = _batch_terms(
                 model, batch, device, temperature)
            dummy_advantages = torch.zeros(
                len(batch), device=device, dtype=torch.float32)
            _loss, _entropy, bundle = _day_bundle_objective(
                new, old, entropy_values, actionable, event_days, day_games,
                dummy_advantages, clip_ratio)
            event_logratio = bundle["event_logratio"].float()
            valid_days = bundle["valid_days"]
            day_logratio = bundle["day_logratio"][valid_days].float()
            totals["event_logratio"] += event_logratio.sum()
            totals["event_approx_kl"] += (
                torch.expm1(event_logratio) - event_logratio
            ).clamp_min(0).sum()
            totals["event_clipped"] += (
                (torch.exp(event_logratio) - 1).abs() > clip_ratio).sum()
            totals["day_logratio"] += day_logratio.sum()
            totals["day_bundle_approx_kl"] += (
                torch.expm1(day_logratio) - day_logratio
            ).clamp_min(0).sum()
            totals["day_clipped"] += (
                (torch.exp(day_logratio) - 1).abs() > clip_ratio).sum()
            totals["event_count"] += len(event_logratio)
            totals["day_count"] += len(day_logratio)
            totals["total_day_count"] += len(day_games)
    model.train(was_training)
    values = {name: float(value.cpu()) for name, value in totals.items()}
    if not values["event_count"] or not values["day_count"]:
        raise RuntimeError("empty full-rollout policy drift audit")
    result = {
        "event_forward_kl": -values["event_logratio"] / values["event_count"],
        "event_approx_kl": values["event_approx_kl"] / values["event_count"],
        "event_clip_fraction": values["event_clipped"] / values["event_count"],
        "day_forward_kl": -values["day_logratio"] / values["day_count"],
        "day_chain_approx_kl": values["event_approx_kl"] / values["day_count"],
        "day_bundle_approx_kl": (
            values["day_bundle_approx_kl"] / values["day_count"]),
        "day_clip_fraction": values["day_clipped"] / values["day_count"],
        "actionable_events": int(values["event_count"]),
        "actionable_days": int(values["day_count"]),
        "total_days": int(values["total_day_count"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"non-finite full-rollout policy drift: {result}")
    return result


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {name: _cpu_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _optimizer_state_for_resume(payload: dict):
    state = payload.get("rl_optimizer")
    if "rl" in payload and state is None:
        raise RuntimeError("RL checkpoint is missing optimizer state")
    return state


def _save_native_rollout(path: Path, rollout_payload: dict,
                         arrays: dict[str, np.ndarray]) -> None:
    """Persist native PPO arrays without expanding the event dictionaries."""
    if NATIVE_ROLLOUT_METADATA in arrays:
        raise RuntimeError("native rollout array name collides with metadata")
    stored = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject or not array.flags.c_contiguous:
            raise RuntimeError(f"native rollout array is not contiguous: {name}")
        stored[name] = array
    games = rollout_payload.get("games")
    if not isinstance(games, list) or not games:
        raise RuntimeError("native rollout has no game metadata")
    metadata = {name: value for name, value in rollout_payload.items()
                if name != "games"}
    metadata.update({
        "storage_format": NATIVE_ROLLOUT_FORMAT,
        "game_metadata": [
            {name: value for name, value in game.items() if name != "days"}
            for game in games
        ],
    })
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    with path.open("wb") as target:
        np.savez(target, **stored, **{
            NATIVE_ROLLOUT_METADATA: np.frombuffer(encoded, dtype=np.uint8),
        })


def _load_native_rollout(path: Path) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (len(set(archive.files)) != len(archive.files) or
                    NATIVE_ROLLOUT_METADATA not in archive.files):
                raise RuntimeError("native rollout metadata is missing or duplicated")
            encoded = archive[NATIVE_ROLLOUT_METADATA]
            if encoded.dtype != np.uint8 or encoded.ndim != 1:
                raise RuntimeError("native rollout metadata encoding is invalid")
            metadata = json.loads(encoded.tobytes().decode("utf-8"))
            arrays = {name: archive[name] for name in archive.files
                      if name != NATIVE_ROLLOUT_METADATA}
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            UnicodeDecodeError) as error:
        raise RuntimeError(f"invalid native rollout archive: {error}") from error
    if (not isinstance(metadata, dict) or
            metadata.pop("storage_format", None) != NATIVE_ROLLOUT_FORMAT):
        raise RuntimeError("native rollout storage format mismatch")
    game_metadata = metadata.pop("game_metadata", None)
    if (not isinstance(game_metadata, list) or not game_metadata or
            any(not isinstance(game, dict) for game in game_metadata)):
        raise RuntimeError("native rollout game metadata is invalid")
    return metadata, arrays, game_metadata


def _restore_native_games(arrays: dict[str, np.ndarray], game_metadata: list[dict],
                          margin_weight: float, margin_scale: float) -> list[dict]:
    from experiments.native_student_actor.native_job_batch import ppo_games

    games = ppo_games(
        arrays, margin_weight=margin_weight, margin_scale=margin_scale,
        materialize_events=False)
    _validate_native_job_routes(arrays, len(games))
    if len(games) != len(game_metadata):
        raise RuntimeError("native rollout game metadata count mismatch")
    identity_fields = ("seed", "opponent", "seat", "route", "policy_seed")
    by_identity = {}
    for game in games:
        identity = tuple(game[name] for name in identity_fields)
        if identity in by_identity:
            raise RuntimeError("duplicate native rollout array identity")
        by_identity[identity] = game
    restored = []
    for saved in game_metadata:
        try:
            identity = tuple(saved[name] for name in identity_fields)
        except KeyError as error:
            raise RuntimeError("native rollout game identity is incomplete") from error
        game = by_identity.pop(identity, None)
        if game is None:
            raise RuntimeError("native rollout arrays/game identity mismatch")
        if ("days" in saved or any(
                name not in saved or saved[name] != value
                for name, value in game.items()
                if name not in ("days", "opponent_variant"))):
            raise RuntimeError("native rollout arrays/game metadata mismatch")
        game.update(saved)
        restored.append(game)
    if by_identity:
        raise RuntimeError("native rollout arrays contain unclaimed games")
    return restored


def _jobs(args, fingerprints: dict, native_artifacts: dict) -> list[dict]:
    seed_block = 2 * len(args.opponents)
    if args.games % seed_block:
        raise ValueError(
            "paired-seed baseline requires complete opponent-by-seat blocks")
    jobs = []
    for index in range(args.games):
        pair = index // 2
        opponent = args.opponents[pair % len(args.opponents)]
        seed = args.seed_start + pair // len(args.opponents)
        seat = index % 2
        digest = hashlib.sha256(
            (f'{args.policy_seed}:{fingerprints["checkpoint_sha256"]}:'
             f'{index}:{seed}:{opponent}:{seat}').encode()).digest()
        policy_seed = int.from_bytes(digest[:8], "little") % (2**31 - 1)
        job = {
            "seed": seed, "opponent": opponent, "seat": seat,
            "opponent_backend": args.opponent_backend,
            "policy_seed": policy_seed,
            "checkpoint": str(args.checkpoint.resolve()),
            "manifest": str(args.manifest.resolve()),
            "binary": str(args.binary.resolve()),
            "temperature": args.temperature,
            "margin_weight": args.margin_weight,
            "margin_scale": args.margin_scale,
            **fingerprints,
        }
        if args.opponent_backend == "native_cpp":
            artifact = native_artifacts[opponent]
            if artifact["kind"] == "replay":
                route_digest = hashlib.sha256(
                    f'replay_clean:{seed}'.encode()).digest()
                route_index = int.from_bytes(route_digest[:8], "little") % artifact["pool_routes"]
                entry = artifact["entries"][route_index]
                job.update({
                    "native_kind": "replay",
                    "native_route_index": entry["clean_index"],
                    "native_bundle_route_index": entry["bundle_index"],
                    "native_route_id": entry["route_id"],
                    "native_route_family": entry["family"],
                    "native_module_sha256": artifact["module_sha256"],
                    "native_asset_sha256": artifact["asset_sha256"],
                    "opponent_variant": (
                        f'replay:{entry["family"]}:{entry["route_id"]}'),
                })
            else:
                job.update({
                    "native_kind": "public_cpp",
                    "native_module_name": artifact["module_name"],
                    "native_module_path": artifact["module_path"],
                    "native_module_sha256": artifact["module_sha256"],
                    "native_asset_path": artifact["asset_path"],
                    "native_asset_sha256": artifact["asset_sha256"],
                    "opponent_variant": opponent,
                })
        jobs.append(job)
    identities = [(row["seed"], row["opponent"], row["seat"])
                  for row in jobs]
    policy_seeds = [row["policy_seed"] for row in jobs]
    if (len(set(identities)) != len(jobs) or
            len(set(policy_seeds)) != len(jobs)):
        raise RuntimeError("duplicate environment identity or policy RNG seed")
    expected_cells = {
        (opponent, seat) for opponent in args.opponents for seat in (0, 1)
    }
    seed_cells = defaultdict(set)
    for row in jobs:
        seed_cells[row["seed"]].add((row["opponent"], row["seat"]))
    if any(cells != expected_cells for cells in seed_cells.values()):
        raise RuntimeError("incomplete paired-seed opponent-by-seat block")
    return jobs


def _validate_inputs(args) -> dict:
    if getattr(args, "day_state_crossfit_baseline", False):
        if not getattr(args, "native_job_rollout", False):
            raise ValueError(
                "day-state cross-fit baseline requires native JobBatch rollout")
        if getattr(args, "day_state_critic_workers", 0) <= 0:
            raise ValueError("day-state critic workers must be positive")
        if not math.isclose(args.margin_weight, 0.1, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "day-state baseline reward support is frozen to margin_weight=0.1")
    if (getattr(args, "native_job_rollout", False) and
            args.rollout_output.suffix != ".npz"):
        raise ValueError("native JobBatch rollout output must use .npz")
    for path in (args.checkpoint, args.manifest, args.binary):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output, args.metrics_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_rollout:
        if not args.rollout_output.is_file():
            raise FileNotFoundError(args.rollout_output)
    elif args.rollout_output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.rollout_output}")
    args.rollout_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (tuple(checkpoint.get("event_classes", ())) != EVENT_CLASSES or
            tuple(checkpoint.get("training_state_steps", ())) != STUDENT_STEPS or
            checkpoint.get("model_dimensions") != {
                "causal_context": 2233,
                "packed_observation": 3074,
                "event_resources": 347,
            }):
        raise ValueError("checkpoint is not the full-day v3 actor")
    fingerprints = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "manifest_sha256": _sha256(args.manifest),
        "binary_sha256": _sha256(args.binary),
    }
    if checkpoint.get("shard_manifest_sha256") != fingerprints["manifest_sha256"]:
        raise ValueError("checkpoint/manifest fingerprint mismatch")
    if getattr(args, "native_job_rollout", False):
        if not args.native_weights.is_file():
            raise FileNotFoundError(args.native_weights)
        if args.temperature != 1.0:
            raise ValueError("native JobBatch sampling currently requires temperature=1")
        if args.games > 2048:
            raise ValueError("native JobBatch supports at most 2048 games")
        if set(args.opponents) - set(NATIVE_JOB_OPPONENT_CODES):
            raise ValueError("native JobBatch opponent is unsupported")
        frozen = args.native_weights.read_bytes()
        if (len(frozen) < 236 or frozen[:8] != b"KAGSV3A\0" or
                frozen[108:140].hex() != fingerprints["checkpoint_sha256"]):
            raise ValueError("native actor weights/checkpoint fingerprint mismatch")
        fingerprints["native_weights_sha256"] = _sha256(args.native_weights)
        native_job_modules = list((
            ROOT / "experiments/native_student_rollout/build").glob(
                "_paused_plan*.so"))
        if len(native_job_modules) != 1:
            raise RuntimeError(
                f"native JobBatch needs one extension, found {native_job_modules}")
        fast_modules = list((
            ROOT / "fast_kaggriculture/python/fast_kaggriculture").glob(
                "_fast_kaggriculture*.so"))
        if len(fast_modules) != 1:
            raise RuntimeError(
                f"native JobBatch needs one FastEnv extension, found {fast_modules}")
        fingerprints["native_job_module_sha256"] = _sha256(
            native_job_modules[0])
        fingerprints["native_job_inputs_sha256"] = {
            "fast_env": _sha256(fast_modules[0]),
            "r1_config": _sha256(ROOT / "policy/r1/config.json"),
            "teammate_source": _sha256(ROOT / "agent/teammate_base.py"),
            "route_actions": _sha256(ROOT / "agent/route_actions.json.zlib"),
            "route_library": _sha256(ROOT / "agent/route_library.json"),
        }
    native_artifacts = (_native_artifacts(args.opponents)
                        if args.opponent_backend == "native_cpp" else {})
    return {"checkpoint": checkpoint, "fingerprints": fingerprints,
            "native_artifacts": native_artifacts}


def _validate_resume_metadata(rollout_payload: dict, args, fingerprints: dict,
                              native_artifacts: dict) -> None:
    if (rollout_payload.get("schema") != ROLLOUT_SCHEMA or
            rollout_payload.get("action_unit") != "day_bundle" or
            rollout_payload.get("policy_version") != fingerprints["checkpoint_sha256"] or
            rollout_payload.get("binary_sha256") != fingerprints["binary_sha256"] or
            rollout_payload.get("manifest_sha256") != fingerprints["manifest_sha256"] or
            rollout_payload.get("opponent_backend") != args.opponent_backend or
            tuple(rollout_payload.get("student_steps", ())) != STUDENT_STEPS or
            rollout_payload.get("reward") != {
                "formula": "sign(margin)+weight*tanh(margin/scale)",
                "margin_weight": args.margin_weight,
                "margin_scale": args.margin_scale,
            }):
        raise RuntimeError("rollout does not match the requested policy contract")
    if (("temperature" in rollout_payload and
         rollout_payload["temperature"] != args.temperature) or
            ("temperature" not in rollout_payload and args.temperature != 1.0)):
        raise RuntimeError("rollout temperature does not match the behavior policy")
    identities = lambda artifacts: {
        name: {key: value for key, value in artifact.items()
               if key.endswith("_sha256") or
               key in ("pool_routes", "entries")}
        for name, artifact in artifacts.items()
    }
    saved_artifacts = rollout_payload.get("native_opponent_artifacts")
    if (not isinstance(saved_artifacts, dict) or
            identities(saved_artifacts) != identities(native_artifacts)):
        raise RuntimeError("rollout native opponent artifact hash mismatch")
    native_job = getattr(args, "native_job_rollout", False)
    saved_engine = rollout_payload.get("rollout_engine", "python_workers")
    expected_engine = "pure_cpp_job_batch" if native_job else "python_workers"
    if saved_engine != expected_engine:
        raise RuntimeError("rollout engine does not match the requested backend")
    if native_job and rollout_payload.get(
            "native_weights_sha256") != fingerprints["native_weights_sha256"]:
        raise RuntimeError("rollout native actor weight hash mismatch")
    if native_job and rollout_payload.get(
            "native_job_module_sha256") != fingerprints[
                "native_job_module_sha256"]:
        raise RuntimeError("rollout native JobBatch module hash mismatch")
    if native_job and rollout_payload.get(
            "native_job_inputs_sha256") != fingerprints[
                "native_job_inputs_sha256"]:
        raise RuntimeError("rollout native JobBatch input hash mismatch")


def train(args) -> dict:
    validated = _validate_inputs(args)
    payload, fingerprints = validated["checkpoint"], validated["fingerprints"]
    jobs = _jobs(args, fingerprints, validated["native_artifacts"])
    started = time.perf_counter()
    rollout_payload = None
    native_job_metrics = None
    native_rollout_arrays = None
    rollout_save_seconds = 0.0
    rollout_save_wait_seconds = 0.0
    save_pool = None
    save_future = None
    npu_init_pool = None
    npu_init_future = None
    npu_init_seconds = npu_init_wait_seconds = 0.0
    critic_pool = None
    critic_future = None
    critic_wait_seconds = 0.0
    if args.resume_rollout:
        if args.native_job_rollout:
            rollout_payload, native_rollout_arrays, game_metadata = (
                _load_native_rollout(args.rollout_output))
        else:
            rollout_payload = torch.load(
                args.rollout_output, map_location="cpu", weights_only=False)
        _validate_resume_metadata(
            rollout_payload, args, fingerprints, validated["native_artifacts"])
        if args.native_job_rollout:
            results = _restore_native_games(
                native_rollout_arrays, game_metadata,
                args.margin_weight, args.margin_scale)
            rollout_payload["games"] = results
        else:
            results = list(rollout_payload.get("games", ()))
        expected = sorted((row["seed"], row["opponent"], row["seat"],
                           (row["native_bundle_route_index"]
                            if row["opponent"] == "replay_clean" else -1),
                           row["policy_seed"],
                           row.get("opponent_variant", row["opponent"]),
                           row.get("native_module_sha256"),
                           row.get("native_asset_sha256")) for row in jobs)
        actual = sorted((row["seed"], row["opponent"], row["seat"],
                         row["route"],
                         row["policy_seed"],
                         row.get("opponent_variant", row["opponent"]),
                         row.get("opponent_module_sha256"),
                         row.get("opponent_asset_sha256")) for row in results)
        if actual != expected:
            raise RuntimeError("rollout games do not match the requested jobs")
        rollout_seconds = float(rollout_payload.get(
            "rollout_wall_seconds",
            max((row.get("elapsed_seconds", 0.0) for row in results), default=0.0)))
    else:
        if args.native_job_rollout:
            if args.device.startswith("npu"):
                # Hide one-time NPU runtime initialization behind the C++ rollout.
                def initialize_npu():
                    init_started = time.perf_counter()
                    import torch_npu  # noqa: F401
                    if not torch.npu.is_available():
                        raise RuntimeError("requested NPU is unavailable")
                    torch.npu.set_device(args.device)
                    return time.perf_counter() - init_started

                npu_init_pool = ThreadPoolExecutor(max_workers=1)
                npu_init_future = npu_init_pool.submit(initialize_npu)
            results, native_job_metrics, native_rollout_arrays = (
                _collect_native_job_batch(args, jobs))
            rollout_seconds = native_job_metrics["wall_seconds"]
        else:
            results = []
            with ProcessPoolExecutor(
                    max_workers=min(args.workers, len(jobs)),
                    initializer=_worker_init) as pool:
                futures = [pool.submit(_collect_one, job) for job in jobs]
                for future in as_completed(futures):
                    row = future.result()
                    results.append(row)
                    print(json.dumps({
                        "event": "rollout", "status": row["status"],
                        "seed": row["seed"], "opponent": row["opponent"],
                        "seat": row["seat"], "margin": row.get("margin"),
                        "seconds": row["elapsed_seconds"],
                        "error": row.get("error"),
                    }), flush=True)
            rollout_seconds = time.perf_counter() - started
        results.sort(key=lambda row: (row["seed"], row["opponent"], row["seat"]))
    failures = [row for row in results if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"on-policy rollout failed closed: {failures}")
    if any(row["policy_sha256"] != fingerprints["checkpoint_sha256"]
           for row in results):
        raise RuntimeError("mixed policy versions in PPO rollout")
    if rollout_payload is None:
        rollout_payload = {
            "schema": ROLLOUT_SCHEMA,
            "action_unit": "day_bundle",
            "rollout_engine": ("pure_cpp_job_batch" if args.native_job_rollout
                               else "python_workers"),
            "opponent_backend": args.opponent_backend,
            "native_opponent_artifacts": validated["native_artifacts"],
            "policy_version": fingerprints["checkpoint_sha256"],
            "binary_sha256": fingerprints["binary_sha256"],
            "manifest_sha256": fingerprints["manifest_sha256"],
            "student_steps": STUDENT_STEPS,
            "temperature": args.temperature,
            "reward": {
                "formula": "sign(margin)+weight*tanh(margin/scale)",
                "margin_weight": args.margin_weight,
                "margin_scale": args.margin_scale,
            },
            "games": results,
            "rollout_wall_seconds": rollout_seconds,
            "reuse_contract": (
                "eligible only before any optimizer step from this policy; "
                "archive is audit-only after the update"),
        }
        if args.native_job_rollout:
            rollout_payload["native_weights_sha256"] = fingerprints[
                "native_weights_sha256"]
            rollout_payload["native_job_module_sha256"] = fingerprints[
                "native_job_module_sha256"]
            rollout_payload["native_job_inputs_sha256"] = fingerprints[
                "native_job_inputs_sha256"]
            rollout_payload["native_job_metrics"] = native_job_metrics
        temporary_rollout = args.rollout_output.with_suffix(
            args.rollout_output.suffix + ".tmp")
        if args.native_job_rollout:
            if native_rollout_arrays is None:
                raise RuntimeError("native rollout arrays were not retained")

            def save_rollout():
                save_started = time.perf_counter()
                _save_native_rollout(
                    temporary_rollout, rollout_payload, native_rollout_arrays)
                os.replace(temporary_rollout, args.rollout_output)
                return time.perf_counter() - save_started

            save_pool = ThreadPoolExecutor(max_workers=1)
            save_future = save_pool.submit(save_rollout)
        else:
            save_started = time.perf_counter()
            torch.save(rollout_payload, temporary_rollout)
            os.replace(temporary_rollout, args.rollout_output)
            rollout_save_seconds = time.perf_counter() - save_started

    rewards = np.asarray([row["reward"] for row in results], dtype=np.float32)
    if not np.any(np.abs(rewards) > 1e-8):
        raise RuntimeError("PPO rollout has zero terminal policy signal")
    advantages = _paired_seed_loo_advantages(results, rewards)
    if not np.any(np.abs(advantages) > 1e-8):
        raise RuntimeError("PPO rollout has zero leave-one-out advantage")
    if args.day_state_crossfit_baseline:
        if native_rollout_arrays is None:
            raise RuntimeError("day-state baseline lost native rollout arrays")
        critic_pool = ThreadPoolExecutor(max_workers=1)
        critic_future = critic_pool.submit(
            _day_state_crossfit_advantages,
            results, native_rollout_arrays, payload, args)

    torch.set_num_threads(args.train_threads)
    model = build_model(
        payload["model_dimensions"]["causal_context"],
        payload["model_dimensions"]["packed_observation"],
        payload["model_dimensions"]["event_resources"],
        payload["model_scale"])
    model.load_state_dict(payload["model"])
    model.train()

    # The behavior policy ran on CPU, so semantic replay parity is checked on
    # CPU.  A second gate records the small backend numerical drift before NPU
    # optimization; conflating the two made valid rollouts fail at 3e-4.
    max_logprob_error = 0.0
    logprob_abs_error_sum = 0.0
    replay_kl_sum = 0.0
    total_events = actionable_events = 0
    cpu_replay_started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(results), args.replay_batch_games):
            (new, old, _entropy, actionable, _event_games,
             _event_days, _day_games) = _batch_terms(
                 model, results[offset:offset + args.replay_batch_games],
                 torch.device("cpu"), args.temperature)
            delta = new - old
            max_logprob_error = max(
                max_logprob_error, float(delta.abs().max()))
            total_events += len(new)
            active_delta = delta[actionable].float()
            actionable_events += len(active_delta)
            logprob_abs_error_sum += float(
                active_delta.abs().double().sum())
            replay_kl_sum += float((
                torch.expm1(active_delta) - active_delta
            ).clamp_min(0).double().sum())
    if not actionable_events:
        raise RuntimeError("PPO rollout has no multi-action event")
    mean_logprob_error = logprob_abs_error_sum / actionable_events
    replay_approx_kl = replay_kl_sum / actionable_events
    if args.native_job_rollout:
        if max_logprob_error > args.native_logprob_max_tolerance:
            print(json.dumps({
                "event": "native_logprob_max_warning",
                "max_abs_error": max_logprob_error,
                "warning_threshold": args.native_logprob_max_tolerance,
            }), flush=True)
        gate_failed = (
            not all(map(math.isfinite, (
                max_logprob_error, mean_logprob_error, replay_approx_kl))) or
            mean_logprob_error > args.native_logprob_mean_tolerance or
            replay_approx_kl > args.native_replay_kl_tolerance)
    else:
        gate_failed = max_logprob_error > args.logprob_tolerance
    if gate_failed:
        raise RuntimeError(
            "captured CPU old-policy logprob replay mismatch: "
            f"max={max_logprob_error}, mean={mean_logprob_error}, "
            f"kl={replay_approx_kl}")
    cpu_replay_seconds = time.perf_counter() - cpu_replay_started
    if save_future is not None:
        wait_started = time.perf_counter()
        rollout_save_seconds = save_future.result()
        rollout_save_wait_seconds = time.perf_counter() - wait_started
        save_pool.shutdown()
    if args.device.startswith("npu"):
        if npu_init_future is not None:
            wait_started = time.perf_counter()
            npu_init_seconds = npu_init_future.result()
            npu_init_wait_seconds = time.perf_counter() - wait_started
            npu_init_pool.shutdown()
        else:
            import torch_npu  # noqa: F401
            if not torch.npu.is_available():
                raise RuntimeError("requested NPU is unavailable")
        torch.npu.set_device(args.device)
    elif args.device != "cpu":
        raise ValueError("device must be cpu or npu[:index]")
    device = torch.device(args.device)
    device_replay_started = time.perf_counter()
    model.to(device)
    device_logprob_error = 0.0
    device_replay_kl_sum = 0.0
    device_replay_shift_sum = 0.0
    device_replay_events = 0
    with torch.no_grad():
        for offset in range(0, len(results), args.replay_batch_games):
            (new, old, _entropy, actionable, _event_games,
             _event_days, _day_games) = _batch_terms(
                 model, results[offset:offset + args.replay_batch_games],
                 device, args.temperature)
            device_logprob_error = max(
                device_logprob_error, float((new - old).abs().max().cpu()))
            count = int(actionable.sum().cpu())
            device_replay_kl_sum += float(
                _approx_kl(new, old, actionable).cpu()) * count
            device_replay_shift_sum += float(
                (old - new)[actionable].sum().cpu())
            device_replay_events += count
    device_replay_approx_kl = device_replay_kl_sum / device_replay_events
    device_replay_mean_shift = device_replay_shift_sum / device_replay_events
    if (device_logprob_error > args.device_logprob_tolerance or
            device_replay_approx_kl > args.device_replay_kl_tolerance):
        raise RuntimeError(
            "device old-policy drift: "
            f"max={device_logprob_error}, kl={device_replay_approx_kl}")
    device_replay_seconds = time.perf_counter() - device_replay_started

    day_state_baseline_metrics = None
    policy_advantage_values = advantages
    if critic_future is not None:
        wait_started = time.perf_counter()
        (day_state_baseline_metrics,
         policy_advantage_values) = critic_future.result()
        critic_wait_seconds = time.perf_counter() - wait_started
        critic_pool.shutdown()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    optimizer_state = _optimizer_state_for_resume(payload)
    optimizer_state_restored = optimizer_state is not None
    optimizer_moments_reset = []
    if optimizer_state_restored:
        optimizer.load_state_dict(optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["weight_decay"] = args.weight_decay
        if not optimizer.state:
            raise RuntimeError("checkpoint RL optimizer state is empty")
        if (device.type == "npu" and
                not payload.get("rl", {}).get("npu_hidden_index_copy_safe", False)):
            for name, parameter in model.named_parameters():
                if name.startswith((
                        "context.", "observation.", "observation_length.",
                        "token_embeddings.", "token.", "begin.")):
                    if optimizer.state.pop(parameter, None) is not None:
                        optimizer_moments_reset.append(name)
    optimizer_step_before = max(
        (float(state.get("step", 0)) for state in optimizer.state.values()),
        default=0.0)
    initial = {name: value.detach().cpu().clone()
               for name, value in model.state_dict().items()}
    rng = np.random.default_rng(args.policy_seed)
    totals = defaultdict(float)
    gradient_norms = []
    epoch_drift = []
    drift_audit_seconds = 0.0
    gradient_steps = epochs_completed = 0
    early_stopped = False
    update_started = time.perf_counter()
    for epoch in range(args.epochs):
        permutation = rng.permutation(len(results))
        for offset in range(0, len(results), args.batch_games):
            indices = permutation[offset:offset + args.batch_games]
            if not len(indices):
                continue
            games = [results[int(index)] for index in indices]
            (new, old, entropy_values, active, _event_games,
             event_days, day_games) = _batch_terms(
                 model, games, device, args.temperature)
            batch_advantages = torch.tensor(
                [float(advantages[int(index)]) for index in indices],
                device=device, dtype=torch.float32)
            batch_day_advantages = None
            if args.day_state_crossfit_baseline:
                batch_day_advantages = torch.tensor([
                    float(day["day_state_advantage"])
                    for game in games for day in game["days"]
                ], device=device, dtype=torch.float32)
            policy_loss, entropy, _bundle = _day_bundle_objective(
                new, old, entropy_values, active, event_days, day_games,
                batch_advantages, args.clip_ratio,
                day_advantages=batch_day_advantages)
            loss = policy_loss - args.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            optimizer.step()
            for name, value in (
                    ("loss", loss), ("policy_loss", policy_loss),
                    ("entropy", entropy), ("gradient_norm", gradient_norm)):
                totals[name] += float(value.detach().cpu())
            gradient_steps += 1
        epochs_completed += 1
        audit_started = time.perf_counter()
        drift = _full_policy_drift(
            model, results, device, args.temperature, args.replay_batch_games,
            args.clip_ratio)
        drift_audit_seconds += time.perf_counter() - audit_started
        epoch_drift.append({"epoch": epoch + 1, **drift})
        print(json.dumps({
            "event": "ppo_epoch", "epoch": epoch + 1,
            **drift,
        }), flush=True)
        if drift["day_chain_approx_kl"] > args.target_kl:
            early_stopped = True
            break
    if device.type == "npu":
        torch.npu.synchronize(device)
    parameter_delta_l2 = math.sqrt(sum(
        float((value.detach().cpu() - initial[name]).square().sum())
        for name, value in model.state_dict().items()))
    if (not gradient_steps or not math.isfinite(parameter_delta_l2) or
            parameter_delta_l2 <= 0):
        raise RuntimeError("PPO did not produce a finite parameter update")
    gradient_norms = np.asarray(gradient_norms, dtype=np.float64)

    output_payload = dict(payload)
    update_seconds = time.perf_counter() - update_started
    rollout_sha256 = _sha256(args.rollout_output)
    output_payload["model"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()}
    output_payload["rl_optimizer"] = _cpu_tree(optimizer.state_dict())
    output_payload["rl"] = {
        "algorithm": (PPO_ALGORITHM + "_day_state_crossfit_hgb"
                      if args.day_state_crossfit_baseline else PPO_ALGORITHM),
        "action_unit": "day_bundle",
        "deployment_eligible": False,
        "scope": args.scope,
        "parent_checkpoint_sha256": fingerprints["checkpoint_sha256"],
        "rollout": str(args.rollout_output.resolve()),
        "rollout_sha256": rollout_sha256,
        "student_steps": list(STUDENT_STEPS),
        "games": len(results), "events": total_events,
        "actionable_events": actionable_events,
        "epochs_completed": epochs_completed,
        "gradient_steps": gradient_steps,
        "temperature": args.temperature,
        "clip_ratio": args.clip_ratio,
        "terminal_reward": rollout_payload["reward"],
        "baseline": (
            "whole-environment-seed six-fold cross-fitted global HGB over "
            "peer return and frozen 163D pre-day state"
            if args.day_state_crossfit_baseline else
            "mean terminal reward of other policy-RNG-independent games with "
            "the same environment seed; opponent/seat-stratified leave-one-out "
            "fallback when a seed has no peer"),
        "advantage_normalization": (
            "none; terminal advantage applies once per actionable day bundle"),
        "trajectory_reuse": "discarded_from_policy_gradient_after_this_update",
        "optimizer_state_restored": optimizer_state_restored,
        "npu_hidden_index_copy_safe": device.type == "npu",
        "optimizer_moments_reset": optimizer_moments_reset,
        "optimizer_step_before": optimizer_step_before,
        "training_batch_games": args.batch_games,
        "replay_batch_games": args.replay_batch_games,
        "rollout_engine": rollout_payload.get("rollout_engine", "python_workers"),
        "native_weights_sha256": rollout_payload.get("native_weights_sha256"),
        "native_job_module_sha256": rollout_payload.get(
            "native_job_module_sha256"),
        "native_job_inputs_sha256": rollout_payload.get(
            "native_job_inputs_sha256"),
    }
    if day_state_baseline_metrics is not None:
        output_payload["rl"]["day_state_control_variate"] = (
            day_state_baseline_metrics)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output_payload, temporary_output)
    os.replace(temporary_output, args.output)

    by_opponent = {}
    for opponent in sorted(set(row["opponent"] for row in results)):
        rows = [row for row in results if row["opponent"] == opponent]
        by_opponent[opponent] = {
            "games": len(rows),
            "ratio": len(rows) / len(results),
            "wins": sum(row["outcome"] > 0 for row in rows),
            "draws": sum(row["outcome"] == 0 for row in rows),
            "mean_margin": float(np.mean([row["margin"] for row in rows])),
            "mean_reward": float(np.mean([row["reward"] for row in rows])),
        }
    by_seat = {}
    for seat in sorted(set(row["seat"] for row in results)):
        rows = [row for row in results if row["seat"] == seat]
        by_seat[str(seat)] = {
            "games": len(rows),
            "wins": sum(row["outcome"] > 0 for row in rows),
            "draws": sum(row["outcome"] == 0 for row in rows),
            "mean_margin": float(np.mean([row["margin"] for row in rows])),
            "mean_reward": float(np.mean([row["reward"] for row in rows])),
        }
    by_opponent_variant = {}
    for variant in sorted(set(row["opponent_variant"] for row in results)):
        rows = [row for row in results if row["opponent_variant"] == variant]
        by_opponent_variant[variant] = {
            "games": len(rows), "ratio": len(rows) / len(results),
            "wins": sum(row["outcome"] > 0 for row in rows),
            "mean_margin": float(np.mean([row["margin"] for row in rows])),
            "mean_reward": float(np.mean([row["reward"] for row in rows])),
        }
    total_wall_seconds = time.perf_counter() - started
    metrics = {
        "status": "PASS",
        "deployment_eligible": False,
        "scope": args.scope,
        "rollout_engine": rollout_payload.get("rollout_engine", "python_workers"),
        "opponent_backend": args.opponent_backend,
        "native_opponent_artifacts": {
            name: {key: value for key, value in artifact.items()
                   if key.endswith("_sha256") or key == "pool_routes"}
            for name, artifact in validated["native_artifacts"].items()
        },
        "algorithm": (PPO_ALGORITHM + "_day_state_crossfit_hgb"
                      if args.day_state_crossfit_baseline else PPO_ALGORITHM),
        "action_unit": "day_bundle",
        "checkpoint_in": str(args.checkpoint.resolve()),
        "checkpoint_in_sha256": fingerprints["checkpoint_sha256"],
        "checkpoint_out": str(args.output.resolve()),
        "checkpoint_out_sha256": _sha256(args.output),
        "rollout": str(args.rollout_output.resolve()),
        "rollout_sha256": rollout_sha256,
        "binary_sha256": fingerprints["binary_sha256"],
        "native_weights_sha256": rollout_payload.get("native_weights_sha256"),
        "native_job_module_sha256": rollout_payload.get(
            "native_job_module_sha256"),
        "native_job_inputs_sha256": rollout_payload.get(
            "native_job_inputs_sha256"),
        "native_job_metrics": rollout_payload.get("native_job_metrics"),
        "games": len(results), "events": total_events,
        "actionable_events": actionable_events,
        "wins": int(np.sum(np.asarray([row["outcome"] for row in results]) > 0)),
        "draws": int(np.sum(np.asarray([row["outcome"] for row in results]) == 0)),
        "mean_margin": float(np.mean([row["margin"] for row in results])),
        "mean_reward": float(rewards.mean()),
        "reward_min": float(rewards.min()), "reward_max": float(rewards.max()),
        "advantage_mean": float(policy_advantage_values.mean()),
        "advantage_std": float(policy_advantage_values.std()),
        "advantage_min": float(policy_advantage_values.min()),
        "advantage_max": float(policy_advantage_values.max()),
        "advantage_baseline": (
            "whole_seed_6fold_global_hgb_pre_day_state"
            if args.day_state_crossfit_baseline else
            "same_environment_seed_leave_one_out"),
        "game_advantage_mean": float(advantages.mean()),
        "game_advantage_std": float(advantages.std()),
        "day_state_control_variate": day_state_baseline_metrics,
        "by_opponent": by_opponent,
        "by_opponent_variant": by_opponent_variant,
        "by_seat": by_seat,
        "unique_environment_seeds": len(set(row["seed"] for row in results)),
        "unique_policy_seeds": len(set(row["policy_seed"] for row in results)),
        "environment_seed_min": min(row["seed"] for row in results),
        "environment_seed_max": max(row["seed"] for row in results),
        "illegal": sum(row["illegal"] for row in results),
        "fallbacks": sum(row["fallbacks"] for row in results),
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "entropy_coef": args.entropy_coef,
            "clip_ratio": args.clip_ratio,
            "target_kl": args.target_kl,
            "training_batch_games": args.batch_games,
            "replay_batch_games": args.replay_batch_games,
            "day_state_crossfit_baseline": args.day_state_crossfit_baseline,
            "day_state_critic_workers": (
                args.day_state_critic_workers
                if args.day_state_crossfit_baseline else None),
        },
        "old_logprob_replay_max_abs_error": max_logprob_error,
        "old_logprob_replay_mean_abs_error": mean_logprob_error,
        "old_logprob_replay_approx_kl": replay_approx_kl,
        "old_logprob_replay_gate": {
            "max_abs_policy": "warn" if args.native_job_rollout else "hard",
            "max_abs_tolerance": (args.native_logprob_max_tolerance
                                  if args.native_job_rollout
                                  else args.logprob_tolerance),
            "mean_abs_tolerance": (args.native_logprob_mean_tolerance
                                   if args.native_job_rollout else None),
            "approx_kl_tolerance": (args.native_replay_kl_tolerance
                                    if args.native_job_rollout else None),
        },
        "device_logprob_replay_max_abs_error": device_logprob_error,
        "device_logprob_replay_approx_kl": device_replay_approx_kl,
        "device_logprob_replay_mean_shift": device_replay_mean_shift,
        "parameter_delta_l2": parameter_delta_l2,
        "optimizer_state_restored": optimizer_state_restored,
        "npu_hidden_index_copy_safe": device.type == "npu",
        "optimizer_moments_reset": optimizer_moments_reset,
        "optimizer_step_before": optimizer_step_before,
        "ppo": {
            "epochs_requested": args.epochs,
            "epochs_completed": epochs_completed,
            "gradient_steps": gradient_steps,
            "early_stopped": early_stopped,
            "kl_gate": (
                "soft full-rollout post-epoch early-stop on chain-rule day "
                "approximate KL; not a hard checkpoint limit"),
            "kl_hard_limit": False,
            "gradient_norm_clipped_fraction": float(np.mean(
                gradient_norms > args.max_grad_norm)),
            "gradient_norm_p50": float(np.quantile(gradient_norms, 0.50)),
            "gradient_norm_p95": float(np.quantile(gradient_norms, 0.95)),
            "gradient_norm_max": float(gradient_norms.max()),
            "epoch_drift": epoch_drift,
            **epoch_drift[-1],
            **{name: value / gradient_steps for name, value in totals.items()},
        },
        "timing": {
            "rollout_seconds": rollout_seconds,
            "rollout_save_seconds": rollout_save_seconds,
            "rollout_save_wait_seconds": rollout_save_wait_seconds,
            "cpu_replay_seconds": cpu_replay_seconds,
            "device_replay_seconds": device_replay_seconds,
            "npu_init_seconds": npu_init_seconds,
            "npu_init_wait_seconds": npu_init_wait_seconds,
            "day_state_baseline_seconds": (
                day_state_baseline_metrics["total_seconds"]
                if day_state_baseline_metrics is not None else 0.0),
            "day_state_baseline_wait_seconds": critic_wait_seconds,
            "rollout_games_per_second": len(results) / rollout_seconds,
            "rollout_events_per_second": total_events / rollout_seconds,
            "rollout_and_update_seconds": total_wall_seconds,
            "total_wall_seconds": total_wall_seconds,
            "update_seconds": update_seconds,
            "drift_audit_seconds": drift_audit_seconds,
        },
    }
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=(
        ROOT / "work/student-v1/action-event-v3-actor-owned-dagger-r3-scale3-e20.pt"))
    parser.add_argument("--manifest", type=Path, default=(
        ROOT / "work/student-v1/action-event-v3-actor-owned-width3074-full/manifest.json"))
    parser.add_argument("--binary", type=Path, default=(
        ROOT / "work/agent-student-actor-owned-v3.so"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--opponents", type=_csv,
                        help=("comma-separated native pool; defaults to "
                              "Thomas+MetaV4 (clean replay is opt-in)"))
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    parser.add_argument("--seed-start", type=int, default=2630000000)
    parser.add_argument("--policy-seed", type=int, default=20260923)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.1)
    parser.add_argument("--margin-scale", type=float, default=10000.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-games", type=int, default=2)
    parser.add_argument(
        "--replay-batch-games", type=int,
        help=("independent batch size for replay gates and full-rollout drift "
              "audit; defaults to --batch-games"))
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--target-kl", type=float, default=0.02,
        help=("soft full-rollout post-epoch early-stop threshold; with one "
              "epoch this is diagnostic only"))
    parser.add_argument("--logprob-tolerance", type=float, default=2e-5)
    parser.add_argument(
        "--native-job-rollout", action="store_true",
        help="run replay opening, actor, and suffix in the experiment C++ JobBatch")
    parser.add_argument("--native-weights", type=Path)
    parser.add_argument("--native-job-threads", type=int,
                        default=min(128, os.cpu_count() or 1))
    parser.add_argument("--native-logprob-max-tolerance", type=float,
                        default=2.5e-4)
    parser.add_argument("--native-logprob-mean-tolerance", type=float,
                        default=5e-7)
    parser.add_argument("--native-replay-kl-tolerance", type=float,
                        default=1e-8)
    parser.add_argument("--device-logprob-tolerance", type=float, default=1e-2)
    parser.add_argument("--device-replay-kl-tolerance", type=float, default=1e-5)
    parser.add_argument("--resume-rollout", action="store_true",
                        help="resume a saved on-policy rollout before any update")
    parser.add_argument(
        "--day-state-crossfit-baseline", action="store_true",
        help=("opt in to the training-only whole-seed 6-fold global-HGB "
              "pre-day control variate; native JobBatch and margin-weight=.1 "
              "are required"))
    parser.add_argument("--day-state-critic-workers", type=int, default=6)
    parser.add_argument("--train-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--python-opponent-pipeline-audit", action="store_true",
        help="required acknowledgement; this backend is forbidden for formal RL")
    args = parser.parse_args()
    if args.replay_batch_games is None:
        args.replay_batch_games = args.batch_games
    if args.python_opponent_pipeline_audit:
        if args.native_job_rollout:
            parser.error("native JobBatch is incompatible with Python-opponent audit")
        args.opponent_backend = "python_audit"
        args.scope = "python_opponent_pipeline_audit_only"
        args.opponents = args.opponents or FORMAL_OPPONENTS
        unknown = set(args.opponents) - set(FORMAL_OPPONENTS)
    else:
        args.opponent_backend = "native_cpp"
        args.scope = ("native_cpp_job_batch" if args.native_job_rollout
                      else "native_cpp_mixed_pool")
        args.opponents = args.opponents or (
            ("thomas_2945_cpp", "metav4_2965")
            if args.native_job_rollout else NATIVE_POOL)
        unknown = set(args.opponents) - set(NATIVE_POOL)
    if unknown:
        parser.error(f"unsupported {args.opponent_backend} opponents: {sorted(unknown)}")
    if (args.games < 1 or args.workers < 1 or args.epochs < 1 or
            args.batch_games < 1 or args.replay_batch_games < 1 or
            args.train_threads < 1 or
            not math.isfinite(args.temperature) or args.temperature <= 0 or
            not 0 <= args.margin_weight < 1 or args.margin_scale <= 0 or
            not 0 < args.clip_ratio < 1 or args.logprob_tolerance <= 0 or
            args.native_job_threads < 1 or
            args.native_logprob_max_tolerance <= 0 or
            args.native_logprob_mean_tolerance <= 0 or
            args.native_replay_kl_tolerance <= 0 or
            not math.isfinite(args.target_kl) or args.target_kl <= 0 or
            args.device_logprob_tolerance <= 0 or
            args.device_replay_kl_tolerance <= 0):
        parser.error("invalid rollout/PPO hyperparameter")
    if args.native_job_rollout and args.native_weights is None:
        parser.error("--native-weights is required with --native-job-rollout")
    train(args)


if __name__ == "__main__":
    main()

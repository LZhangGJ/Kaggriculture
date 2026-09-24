#!/usr/bin/env python3
"""Generate paired real-suffix labels for every normal R1 candidate at step 288.

This is an offline diagnostic, not a policy gate.  Each arm deterministically
replays the complete warm prefix in FastEnv, installs one normal proposal for
day 12, reseeds only the future, and then resumes the unchanged closed-loop R1
through terminal.  Future replicas stay nested under their original checkpoint.

The diagnostic teacher binary must expose the feature-only v2 ABI.  That ABI
enumerates every deduplicated normal proposal without short-score top-K or outer
candidates and exports the exact proposal_key used by C++.
"""

import argparse
import concurrent.futures as cf
import ctypes
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import statistics
import struct
import time
from pathlib import Path

from run_strong_ab import BOTS, entry, load, observations


ROOT = Path(__file__).resolve().parents[1]
HANDOFF_STEP = 288
KEY_HASH = "sha256_prefix_128"


def json_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_fingerprints(policy_path, binary, opponents):
    return {
        "policy_entry": str(policy_path), "policy_entry_sha256": file_hash(policy_path),
        "r1_python_sha256": file_hash(ROOT / "policy/r1/agent.py"),
        "binary": str(binary), "binary_sha256": file_hash(binary),
        "config": str(ROOT / "policy/r1/config.json"),
        "config_sha256": file_hash(ROOT / "policy/r1/config.json"),
        "deployment_sha256": file_hash(ROOT / "agent/replay_deployment.json"),
        "opponent_entry_sha256": {name: file_hash(BOTS[name]) for name in opponents},
    }


def child_environment(binary):
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["R1_BINARY_PATH"] = str(Path(binary).resolve())
    os.environ["REPLAY_HANDOFF_STEP"] = str(HANDOFF_STEP)
    os.environ["REPLAY_HANDOFF_LAND"] = ""  # exact step, no earlier state trigger
    os.environ["REPLAY_HANDOFF_MIN_STEP"] = "0"
    os.environ["REPLAY_HANDOFF_LAND_DELAY"] = "0"
    os.environ["REPLAY_CAPITAL_SELL_FIRST"] = "1"
    for name in ("R1_CONFIG_OVERRIDES", "REPLAY_HANDOFF_SELECTOR",
                 "REPLAY_FORCED_OPENING", "REPLAY_DISABLE_DEPLOYMENT"):
        os.environ.pop(name, None)


def bind_teacher(agent):
    lib = agent.lib
    required = (
        "td_teacher_abi_version", "td_teacher_contract_json",
        "td_teacher_prepare_features_observation", "td_teacher_candidate_key",
        "td_candidate_id",
    )
    missing = [name for name in required if not hasattr(lib, name)]
    if missing:
        raise RuntimeError(f"teacher feature ABI missing: {missing}")
    lib.td_teacher_abi_version.argtypes = []
    lib.td_teacher_abi_version.restype = ctypes.c_int
    lib.td_teacher_contract_json.argtypes = []
    lib.td_teacher_contract_json.restype = ctypes.c_char_p
    lib.td_teacher_prepare_features_observation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.td_teacher_prepare_features_observation.restype = ctypes.c_int
    lib.td_teacher_candidate_key.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.td_teacher_candidate_key.restype = ctypes.c_int
    lib.td_candidate_id.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.td_candidate_id.restype = ctypes.c_int
    version = lib.td_teacher_abi_version()
    contract = json.loads(lib.td_teacher_contract_json().decode())
    if version < 2 or contract.get("state_scope") != "first_handoff_step288_only":
        raise RuntimeError(f"incompatible teacher ABI: version={version}, contract={contract}")
    return lib, version, contract


def candidate_key(lib, handle, index):
    size = lib.td_teacher_candidate_key(handle, index, None, 0)
    if size <= 0:
        raise RuntimeError(f"candidate {index} has invalid key size {size}")
    values = (ctypes.c_int32 * size)()
    actual = lib.td_teacher_candidate_key(handle, index, values, size)
    if actual != size:
        raise RuntimeError(f"candidate {index} key export returned {actual}, expected {size}")
    encoded = struct.pack(f"<{size}i", *values)
    return hashlib.sha256(encoded).hexdigest()[:32], size


def prepare_normals(main, dynamic, own):
    lib, version, contract = bind_teacher(dynamic)
    packed = main.policy._pack(own)
    count = lib.td_teacher_prepare_features_observation(
        dynamic.handle, packed, len(packed))
    if count <= 0:
        raise RuntimeError(dynamic.lib.td_debug(dynamic.handle).decode())
    rows = []
    for index in range(count):
        key, key_size = candidate_key(lib, dynamic.handle, index)
        rows.append({"prepared_index": index,
                     "candidate_id": int(lib.td_candidate_id(dynamic.handle, index)),
                     "candidate_key_sha256_128": key,
                     "candidate_key_int32_count": key_size})
    if len({row["candidate_key_sha256_128"] for row in rows}) != len(rows):
        raise RuntimeError("normal proposal keys are not unique")
    anchors = [row for row in rows if row["candidate_id"] == 0]
    if len(anchors) != 1:
        raise RuntimeError(f"expected one canonical id0 anchor, found {len(anchors)}")
    candidate_set_hash = json_hash(sorted(
        (row["candidate_id"], row["candidate_key_sha256_128"]) for row in rows))
    return rows, anchors[0]["candidate_key_sha256_128"], candidate_set_hash, version, contract


def run(task):
    mode, policy_path, binary, bot, seed, seat, *arm = task
    future_seed = candidate_hash = expected_set_hash = None
    if mode == "arm":
        candidate_hash, future_seed, expected_set_hash = arm
    child_environment(binary)
    policy = None
    started = time.perf_counter()
    try:
        suffix = f"{mode}_{bot}_{seed}_{seat}_{candidate_hash or 'discover'}_{future_seed}"
        main = load(policy_path, f"suffix_policy_{suffix}")
        policy = main.create_agent(seat)
        if not getattr(policy, "dynamic", None) or policy.dynamic.external is not False:
            raise RuntimeError("candidate suffix labels require the warm replay-to-R1 policy")
        if float(policy.dynamic.config.get("scenario", 0)) == 0:
            raise RuntimeError("closed-loop R1 is disabled by scenario=0")
        rival = load(BOTS[bot], f"suffix_rival_{suffix}").agent
        with_config = len(inspect.signature(rival).parameters) > 1
        from fast_kaggriculture import Config, FastEnv
        env = FastEnv(Config(), seed)
        state = list(env.reset(seed))
        prefix_digest = hashlib.sha256()
        installed = closed_loop_resumed = False
        while not env.done:
            current = observations(state, "fast")
            step = current[0]["step"]
            own = current[seat]
            prefix_digest.update(json.dumps(
                state, sort_keys=True, separators=(",", ":")).encode())
            if step < HANDOFF_STEP and policy.ready(own):
                raise RuntimeError(f"warm handoff triggered early at step {step}")
            if step == HANDOFF_STEP:
                if not policy.ready(own):
                    raise RuntimeError("warm handoff did not trigger at step 288")
                handoff_hash = json_hash(own)
                rows, anchor_hash, set_hash, abi, contract = prepare_normals(
                    main, policy.dynamic, own)
                common = {
                    "bot": bot, "seed": seed, "seat": seat,
                    "handoff_step": step,
                    "handoff_observation_hash": handoff_hash,
                    "prefix_transcript_hash": prefix_digest.hexdigest(),
                    "candidate_set_hash": set_hash,
                    "canonical_anchor_key": anchor_hash,
                    "normal_candidates": rows,
                    "effective_config_sha256": json_hash(policy.dynamic.config),
                    "teacher_abi_version": abi,
                    "teacher_contract": contract,
                }
                if mode == "discover":
                    return {**common, "error": None,
                            "wall_seconds": time.perf_counter() - started}
                if set_hash != expected_set_hash:
                    raise RuntimeError(
                        f"candidate set changed: {set_hash} != {expected_set_hash}")
                matches = [row for row in rows
                           if row["candidate_key_sha256_128"] == candidate_hash]
                if len(matches) != 1:
                    raise RuntimeError(f"candidate key is not uniquely reproducible: {candidate_hash}")
                selected = matches[0]
                searches_before = policy.dynamic.debug().get("search_calls")
                policy.dynamic.install_candidate(selected["prepared_index"])
                env.reseed_future(future_seed)
                installed = True
            elif step > HANDOFF_STEP and mode == "discover":
                raise RuntimeError("discovery passed step288 without returning")

            actions = [None, None]
            before = policy.dynamic.debug().get("search_calls") if (
                mode == "arm" and step in (HANDOFF_STEP, HANDOFF_STEP + 24)) else None
            actions[seat] = policy(own, {})
            after = policy.dynamic.debug().get("search_calls") if before is not None else None
            other = current[1 - seat]
            actions[1 - seat] = rival(other, {}) if with_config else rival(other)
            if step < HANDOFF_STEP:
                prefix_digest.update(json.dumps(
                    actions, sort_keys=True, separators=(",", ":")).encode())
            elif mode == "arm" and step == HANDOFF_STEP:
                if not installed or before != searches_before or after != before:
                    raise RuntimeError("forced candidate was replanned on its intervention day")
            elif mode == "arm" and step == HANDOFF_STEP + 24:
                if after != before + 1:
                    raise RuntimeError(
                        f"closed-loop R1 did not resume on day13: {before}->{after}")
                closed_loop_resumed = True
            state = list(env.step(actions))

        if mode != "arm" or not closed_loop_resumed:
            raise RuntimeError("suffix arm did not complete its one-day restore check")
        cash, opponent_cash = map(float, (env.rewards[seat], env.rewards[1 - seat]))
        margin = cash - opponent_cash
        if not all(math.isfinite(value) for value in (cash, opponent_cash, margin)):
            raise RuntimeError("non-finite terminal reward")
        return {
            **common, "candidate_key": candidate_hash,
            "candidate_id": selected["candidate_id"], "future_seed": future_seed,
            "future_reseed_step": HANDOFF_STEP, "cash": cash,
            "opponent_cash": opponent_cash, "actual_margin": margin,
            "actual_win": int(margin > 0), "closed_loop_resumed_day": 13,
            "error": None, "wall_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {"mode": mode, "bot": bot, "seed": seed, "seat": seat,
                "candidate_key": candidate_hash, "future_seed": future_seed,
                "error": repr(error), "wall_seconds": time.perf_counter() - started}
    finally:
        if policy is not None:
            policy.close()


def standard_error(values):
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None


def assemble(discoveries, arms, future_seeds):
    states = []
    arm_lookup = {}
    for row in arms:
        key = (row["bot"], row["seed"], row["seat"],
               row["candidate_key"], row["future_seed"])
        if key in arm_lookup:
            raise ValueError(f"duplicate suffix arm: {key}")
        arm_lookup[key] = row
    for discovery in discoveries:
        state_key = discovery["bot"], discovery["seed"], discovery["seat"]
        candidates = discovery["normal_candidates"]
        expected_keys = {row["candidate_key_sha256_128"] for row in candidates}
        anchor_key = discovery["canonical_anchor_key"]
        outcomes = {}
        for candidate_key in expected_keys:
            for future_seed in future_seeds:
                key = (*state_key, candidate_key, future_seed)
                if key not in arm_lookup:
                    raise ValueError(f"missing suffix arm: {key}")
                row = arm_lookup.pop(key)
                if row.get("error"):
                    raise ValueError(f"failed suffix arm {key}: {row['error']}")
                for field in ("handoff_observation_hash", "prefix_transcript_hash",
                              "candidate_set_hash", "canonical_anchor_key",
                              "effective_config_sha256"):
                    if row[field] != discovery[field]:
                        raise ValueError(f"{field} mismatch for suffix arm {key}")
                if (row.get("future_reseed_step") != HANDOFF_STEP or
                        row.get("closed_loop_resumed_day") != HANDOFF_STEP // 24 + 1):
                    raise ValueError(f"invalid reseed/restore boundary for suffix arm {key}")
                outcomes[candidate_key, future_seed] = row
        candidate_results = []
        for candidate in candidates:
            candidate_key = candidate["candidate_key_sha256_128"]
            replica_rows = []
            for future_seed in future_seeds:
                row = outcomes[candidate_key, future_seed]
                anchor = outcomes[anchor_key, future_seed]
                replica_rows.append({
                    "future_seed": future_seed,
                    "future_reseed_step": row["future_reseed_step"],
                    "closed_loop_resumed_day": row["closed_loop_resumed_day"],
                    "cash": row["cash"], "opponent_cash": row["opponent_cash"],
                    "actual_margin": row["actual_margin"],
                    "actual_win": row["actual_win"],
                    "paired_margin_delta": row["actual_margin"] - anchor["actual_margin"],
                    "paired_win_delta": row["actual_win"] - anchor["actual_win"],
                })
            margin_delta = [row["paired_margin_delta"] for row in replica_rows]
            win_delta = [row["paired_win_delta"] for row in replica_rows]
            candidate_results.append({
                **candidate, "canonical_anchor": candidate_key == anchor_key,
                "replicas": replica_rows,
                "replica_group_summary": {
                    "replicas": len(replica_rows),
                    "mean_actual_margin": statistics.fmean(
                        row["actual_margin"] for row in replica_rows),
                    "mean_actual_win": statistics.fmean(
                        row["actual_win"] for row in replica_rows),
                    "mean_paired_margin_delta": statistics.fmean(margin_delta),
                    "paired_margin_delta_se": standard_error(margin_delta),
                    "mean_paired_win_delta": statistics.fmean(win_delta),
                    "paired_win_delta_se": standard_error(win_delta),
                },
            })
        states.append({
            "state_id": discovery["handoff_observation_hash"],
            "opponent": discovery["bot"], "seed": discovery["seed"],
            "seat": discovery["seat"], "handoff_step": HANDOFF_STEP,
            "handoff_observation_hash": discovery["handoff_observation_hash"],
            "prefix_transcript_hash": discovery["prefix_transcript_hash"],
            "candidate_set_hash": discovery["candidate_set_hash"],
            "canonical_anchor_key": anchor_key,
            "normal_candidate_count": len(candidates),
            "candidates": candidate_results,
        })
    if arm_lookup:
        raise ValueError(f"unexpected suffix arms: {list(arm_lookup)[:3]}")
    return states


def self_check():
    candidates = [
        {"prepared_index": 0, "candidate_id": 0,
         "candidate_key_sha256_128": "anchor", "candidate_key_int32_count": 3},
        {"prepared_index": 1, "candidate_id": 1,
         "candidate_key_sha256_128": "other", "candidate_key_int32_count": 4},
    ]
    discovery = {"bot": "bot", "seed": 1, "seat": 0,
                 "handoff_observation_hash": "observation", "prefix_transcript_hash": "prefix",
                 "candidate_set_hash": "set", "canonical_anchor_key": "anchor",
                 "effective_config_sha256": "config", "normal_candidates": candidates}
    arms = []
    for candidate in candidates:
        for future_seed in (10, 11):
            margin = future_seed + (5 if candidate["candidate_id"] else 0)
            arms.append({**discovery, "candidate_key": candidate["candidate_key_sha256_128"],
                         "candidate_id": candidate["candidate_id"], "future_seed": future_seed,
                         "future_reseed_step": HANDOFF_STEP,
                         "closed_loop_resumed_day": HANDOFF_STEP // 24 + 1,
                         "cash": margin, "opponent_cash": 0, "actual_margin": margin,
                         "actual_win": 1, "error": None})
    states = assemble([discovery], arms, (10, 11))
    result = states[0]["candidates"][1]["replica_group_summary"]
    assert result["mean_paired_margin_delta"] == 5
    broken = [dict(row) for row in arms]
    broken[-1]["handoff_observation_hash"] = "different"
    try:
        assemble([discovery], broken, (10, 11))
    except ValueError as error:
        assert "handoff_observation_hash mismatch" in str(error)
    else:
        raise AssertionError("handoff mismatch did not fail closed")
    assert len(bytes.fromhex(hashlib.sha256(struct.pack("<3i", 1, -2, 3)).hexdigest()[:32])) == 16
    print(json.dumps({"status": "PASS", "states": 1, "candidates": 2,
                      "future_replicas_are_grouped": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=entry, default=ROOT / "agent/main.py")
    parser.add_argument("--binary", type=entry)
    parser.add_argument("--opponents", default="thomas_2945")
    parser.add_argument("--start", type=int, default=2609800200)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seats", default="0")
    parser.add_argument("--future-start", type=int)
    parser.add_argument("--future-replicas", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.binary is None or args.future_start is None or args.output is None:
        parser.error("--binary, --future-start and --output are required")
    if min(args.seeds, args.future_replicas, args.workers) < 1:
        parser.error("--seeds, --future-replicas and --workers must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    names = [value for value in args.opponents.split(",") if value]
    seats = [int(value) for value in args.seats.split(",") if value]
    if (not names or len(names) != len(set(names)) or any(name not in BOTS for name in names)
            or not seats or len(seats) != len(set(seats)) or any(seat not in (0, 1) for seat in seats)):
        parser.error("opponents must be configured strong bots and seats must be unique 0/1")
    future_seeds = tuple(range(args.future_start, args.future_start + args.future_replicas))
    policy_path, binary = args.policy.resolve(), args.binary.resolve()
    fingerprints = input_fingerprints(policy_path, binary, names)
    base_tasks = [("discover", str(policy_path), str(binary), bot, seed, seat)
                  for bot in names for seed in range(args.start, args.start + args.seeds)
                  for seat in seats]
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                max_tasks_per_child=1) as pool:
        discoveries = list(pool.map(run, base_tasks))
    errors = [row for row in discoveries if row.get("error")]
    if errors:
        raise RuntimeError(f"candidate discovery failed closed: {errors[:3]}")
    config_hashes = {row["effective_config_sha256"] for row in discoveries}
    contracts = {json.dumps(row["teacher_contract"], sort_keys=True) for row in discoveries}
    if len(config_hashes) != 1 or len(contracts) != 1:
        raise RuntimeError("effective config or teacher contract changed across checkpoints")
    arm_tasks = []
    for row in discoveries:
        for candidate in row["normal_candidates"]:
            for future_seed in future_seeds:
                arm_tasks.append((
                    "arm", str(policy_path), str(binary), row["bot"], row["seed"], row["seat"],
                    candidate["candidate_key_sha256_128"], future_seed,
                    row["candidate_set_hash"],
                ))
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                max_tasks_per_child=1) as pool:
        arms = list(pool.map(run, arm_tasks))
    states = assemble(discoveries, arms, future_seeds)
    if input_fingerprints(policy_path, binary, names) != fingerprints:
        raise RuntimeError("policy, binary, config, deployment or opponent changed during labeling")
    result = {
        "format": "kaggriculture-r1-real-suffix-paired-v1",
        "diagnostic": True,
        "scope": "first warm handoff at step288; all normal candidates; one-day intervention; closed-loop R1 suffix",
        "engine": "fast_kaggriculture:FastEnv.reseed_future",
        "checkpoint_strategy": (
            "full deterministic prefix replay per arm; FastEnv.clone alone cannot clone Python opponent "
            "or native policy state"),
        "candidate_selection": "all deduplicated normal proposals; no short-score top-K; no outer",
        "proposal_key_encoding": "little_endian_signed_int32",
        "proposal_key_hash": KEY_HASH,
        "canonical_anchor": "unique candidate_id=0 normal",
        "future_replica_semantics": (
            "conditional repeats nested within one checkpoint; never independent states"),
        "runtime_overrides": {
            "R1_BINARY_PATH": str(binary), "REPLAY_HANDOFF_STEP": str(HANDOFF_STEP),
            "REPLAY_HANDOFF_LAND": "", "REPLAY_HANDOFF_MIN_STEP": "0",
            "REPLAY_HANDOFF_LAND_DELAY": "0", "REPLAY_CAPITAL_SELL_FIRST": "1",
            "R1_CONFIG_OVERRIDES": None, "REPLAY_HANDOFF_SELECTOR": None,
            "REPLAY_FORCED_OPENING": None,
        },
        "online_feature_exclusions": ["opponent", "seed", "seat", "future_seed", "opponent_private"],
        "opponent_metadata_role": "offline split and audit only; never a policy input or gate",
        "seed_range": [args.start, args.start + args.seeds],
        "seats": seats, "future_seeds": list(future_seeds),
        "fingerprints": {**fingerprints,
                         "effective_config_sha256": next(iter(config_hashes))},
        "teacher_contract": json.loads(next(iter(contracts))),
        "states": states,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      "states": len(states), "suffix_games": len(arms),
                      "normal_candidates": sum(state["normal_candidate_count"] for state in states)}))


if __name__ == "__main__":
    main()

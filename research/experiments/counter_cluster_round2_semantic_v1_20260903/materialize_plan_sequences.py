#!/usr/bin/env python3
"""Materialize Round-1 rank choices as durable Candidate8 PlanDelta intents.

This is an interface audit, not a fresh-seed outcome evaluation.  For each of
the 48 frozen discovery searches it:

1. replays the original rank sequence in the original seed/seat/opponent;
2. captures the selected PlanDelta at each checkpoint;
3. replays the same scenario by explicit semantic PlanDelta intent; and
4. requires exact reward, trace, family, signature, and delta equivalence.

The durable fist identity intentionally excludes state-derived capacity,
estimates, and Candidate8's context-dependent 64-bit signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROUND1 = ROOT / "experiments" / "counter_cluster_round1"
ROUND1_FAST_PYTHON = (
    ROOT / "experiments" / "nt_counter_fist_mvp" / "fast_kaggriculture" / "python"
)
DEFAULT_FAST_PYTHON = HERE / "native" / "fast_kaggriculture" / "python"
DEFAULT_RUNTIME_LOCK = HERE / "runtime.lock.json"
NT_ROOT = Path(
    r"D:\github\Kaggriculture_candidate8\nt\latest_20260901_candidate8_width12_mcts\workspace"
)
DEFAULT_META_AGENT_SRC = (
    NT_ROOT / "agents" / "route_clustering_switch_agent" / "src"
)
DEFAULT_SOURCE = (
    Path(r"D:\Kaggriculture\route_clustering_top40_20260831")
    / "output-v3"
    / "agent-dynamic"
    / "multifile"
    / "teammate_base.py"
)
DEFAULT_DLL_DIR = (
    Path(r"D:\Kaggriculture\route_clustering_top40_20260831")
    / ".conda_toolchain"
    / "Library"
    / "bin"
)

EXPECTED_SEARCH_SCHEMA = "kaggriculture.counter-cluster-search-results.v1"
EXPECTED_ROW_SCHEMA = "kaggriculture.counter-cluster-search-row.v1"
EXPECTED_PANEL_SCHEMA = "kaggriculture.counter-cluster-panel.v1"
EXPECTED_SEARCHES = 48
EXPECTED_OPENING_HASH = (
    "fcd500a8aaef48f001c58e85902194b37c35cfe5062f1451416141c0a7d1f28f"
)

INTENT_FIELDS = (
    "family_id",
    "target_delta",
    "effective_delay_days",
    "schedule_profile",
    "market_profile",
    "recovery_profile",
    "suffix_project",
    "market_item",
    "recovery_issue",
)
DERIVED_FIELDS = (
    "hand_delta",
    "quadrant_delta",
    "estimated_value",
    "estimated_cash_cost",
    "estimated_daily_action_load",
    "signature",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_action(value: Any) -> dict[str, Any]:
    raw = dict(value or {})
    return {
        "farmer": list(raw.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (raw.get("hands") or [])],
        "market": [list(order) for order in (raw.get("market") or [])],
    }


def route_action_hash(actions: Sequence[Mapping[str, Any]]) -> str:
    return sha256_value({"schema": "route-actions-v1", "actions": list(actions)})


def normalized_trace(trace: Sequence[Sequence[Any]]) -> list[list[dict[str, Any]]]:
    result = []
    for step, joint in enumerate(trace):
        if len(joint) != 2:
            raise ValueError(f"trace step {step} does not contain two player actions")
        result.append([normalized_action(joint[0]), normalized_action(joint[1])])
    return result


def canonical_intent(delta: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in INTENT_FIELDS if field not in delta]
    if missing:
        raise ValueError(f"PlanDelta is missing intent field: {missing[0]}")
    targets = list(delta["target_delta"])
    if len(targets) != 8:
        raise ValueError("PlanDelta target_delta must have length 8")
    result = {
        "family_id": int(delta["family_id"]),
        "target_delta": [int(value) for value in targets],
        "effective_delay_days": int(delta["effective_delay_days"]),
        "schedule_profile": int(delta["schedule_profile"]),
        "market_profile": int(delta["market_profile"]),
        "recovery_profile": int(delta["recovery_profile"]),
        "suffix_project": int(delta["suffix_project"]),
        "market_item": int(delta["market_item"]),
        "recovery_issue": int(delta["recovery_issue"]),
    }
    if not 0 <= result["family_id"] <= 8:
        raise ValueError("PlanDelta family_id must be in [0, 8]")
    return result


def full_delta(delta: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the binding payload without discarding execution fields."""
    required = (*INTENT_FIELDS, "hand_delta", "quadrant_delta", "signature")
    missing = [field for field in required if field not in delta]
    if missing:
        raise ValueError(f"PlanDelta is missing execution field: {missing[0]}")
    result = dict(delta)
    result.update(canonical_intent(delta))
    result["hand_delta"] = int(delta["hand_delta"])
    result["quadrant_delta"] = int(delta["quadrant_delta"])
    result["signature"] = int(delta["signature"])
    for field in (
        "estimated_value",
        "estimated_cash_cost",
        "estimated_daily_action_load",
    ):
        result[field] = float(delta.get(field, 0.0))
    if "family" in delta:
        result["family"] = str(delta["family"])
    return result


def semantic_sequence_payload(
    *, opening_hash: str, checkpoint: int, decision_days: Sequence[int],
    intents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "candidate8-semantic-plan-sequence.v1",
        "opening_action_sha256": opening_hash,
        "checkpoint": int(checkpoint),
        "decision_days": [int(value) for value in decision_days],
        "intents": [canonical_intent(value) for value in intents],
    }


def score(rewards: Sequence[float], seat: int) -> float:
    own, other = float(rewards[seat]), float(rewards[1 - seat])
    return 1.0 if own > other else (0.5 if own == other else 0.0)


def load_genome(path: Path, genome_names: Any, default_genome: Any) -> np.ndarray:
    names = list(genome_names())
    values = dict(zip(names, default_genome(), strict=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    values.update(payload.get("base_values", {}))
    values.update(payload["genomes"][0].get("values", {}))
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def validate_frozen_inputs(
    search_path: Path, panel_path: Path, source_path: Path,
    actions_path: Path, metadata_path: Path, genome_override: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    search = json.loads(search_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if search.get("schema") != EXPECTED_SEARCH_SCHEMA:
        raise ValueError("unsupported Round-1 search-results schema")
    if panel.get("schema") != EXPECTED_PANEL_SCHEMA:
        raise ValueError("unsupported Round-1 panel schema")

    config = search.get("config")
    rows = search.get("rows")
    if not isinstance(config, dict) or not isinstance(rows, list):
        raise ValueError("search-results is missing config/rows")
    if len(rows) != EXPECTED_SEARCHES or int(search.get("searches", -1)) != len(rows):
        raise ValueError(f"materialization requires exactly {EXPECTED_SEARCHES} rows")

    config_body = dict(config)
    claimed_config_hash = str(config_body.pop("config_sha256", "")).lower()
    observed_config_hash = hashlib.sha256(
        json.dumps(config_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if claimed_config_hash != observed_config_hash:
        raise ValueError("Round-1 config_sha256 is invalid")

    panel_body = dict(panel)
    claimed_panel_hash = str(panel_body.pop("content_sha256", "")).lower()
    if claimed_panel_hash != sha256_value(panel_body):
        raise ValueError("Round-1 panel content_sha256 is invalid")
    if str(config.get("panel_content_sha256", "")).lower() != claimed_panel_hash:
        raise ValueError("search config and panel hashes disagree")
    if str(config.get("opening_action_sha256", "")).lower() != EXPECTED_OPENING_HASH:
        raise ValueError("search config does not use the frozen v3 Opening A")

    file_checks = {
        "source_sha256": source_path,
        "discovery_actions_sha256": actions_path,
        "discovery_metadata_sha256": metadata_path,
    }
    for key, path in file_checks.items():
        if not path.is_file() or sha256_file(path) != str(config.get(key, "")).lower():
            raise ValueError(f"frozen input hash mismatch: {key}")

    genome_path = genome_override or Path(str(config.get("genome_path", "")))
    if not genome_path.is_file():
        raise ValueError(f"genome file is missing: {genome_path}")
    if sha256_file(genome_path) != str(config.get("genome_sha256", "")).lower():
        raise ValueError("frozen input hash mismatch: genome_sha256")

    train = {str(row["panel_id"]): row for row in panel["rows"] if row["split"] == "train"}
    expected_tasks = {
        f"{panel_id}:{int(seed)}:{int(seat)}"
        for panel_id in train
        for seed in config["discovery_seeds"]
        for seat in config["seats"]
    }
    observed_tasks = {str(row.get("task_id")) for row in rows}
    if len(observed_tasks) != len(rows) or observed_tasks != expected_tasks:
        raise ValueError("search rows do not match the frozen train/seed/seat task grid")
    for row in rows:
        panel_row = train.get(str(row.get("panel_id")))
        if row.get("schema") != EXPECTED_ROW_SCHEMA or panel_row is None:
            raise ValueError(f"invalid search row: {row.get('task_id')}")
        if str(row.get("config_sha256", "")).lower() != claimed_config_hash:
            raise ValueError(f"row config hash mismatch: {row['task_id']}")
        if str(row.get("opponent_action_sha256", "")).lower() != str(
            panel_row["action_sha256"]
        ).lower():
            raise ValueError(f"row opponent hash mismatch: {row['task_id']}")
        if len(row.get("selected_ranks", [])) != len(config["decision_days"]):
            raise ValueError(f"row rank count mismatch: {row['task_id']}")
    return search, panel, genome_path


def assert_isolated_native(native_path: Path, fast_python: Path) -> None:
    native = native_path.resolve()
    requested = fast_python.resolve()
    round1 = ROUND1_FAST_PYTHON.resolve()
    if not native.is_relative_to(requested):
        raise RuntimeError(
            f"loaded native extension is outside --fast-python: {native}"
        )
    if native.is_relative_to(round1):
        raise RuntimeError(
            "refusing to load the Round-1 native extension; use the isolated Round-2 build"
        )


def validate_runtime_lock(path: Path, native_path: Path, source_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"runtime lock is missing: {path}; run freeze_runtime.py before simulation"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "kaggriculture.counter-cluster-round2-runtime-lock.v1":
        raise ValueError("unsupported Round-2 runtime-lock schema")
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ValueError("runtime lock is missing its files list")
    by_role = {str(row.get("role")): row for row in rows}
    if len(by_role) != len(rows):
        raise ValueError("runtime lock has duplicate file roles")
    for required in ("v3_teammate_source", "round2_native_extension", "materializer"):
        if required not in by_role:
            raise ValueError(f"runtime lock is missing role: {required}")
    expected_paths = {
        "v3_teammate_source": source_path.resolve(),
        "round2_native_extension": native_path.resolve(),
        "materializer": Path(__file__).resolve(),
    }
    for role, expected_path in expected_paths.items():
        row = by_role[role]
        locked_path = Path(str(row.get("path", ""))).resolve()
        if locked_path != expected_path:
            raise ValueError(f"runtime lock path mismatch: {role}")
    for role, row in by_role.items():
        locked_path = Path(str(row.get("path", "")))
        if not locked_path.is_file():
            raise ValueError(f"runtime-locked file is missing: {role}")
        if sha256_file(locked_path) != str(row.get("sha256", "")).lower():
            raise ValueError(f"runtime-locked file hash mismatch: {role}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-results", type=Path, default=ROUND1 / "search_results.json")
    parser.add_argument("--panel", type=Path, default=ROUND1 / "panel24.json")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--base-actions", type=Path, default=ROUND1 / "discovery_actions.json.zlib")
    parser.add_argument("--base-metadata", type=Path, default=ROUND1 / "discovery_library.json")
    parser.add_argument("--genome", type=Path)
    parser.add_argument("--fast-python", type=Path, default=DEFAULT_FAST_PYTHON)
    parser.add_argument("--runtime-lock", type=Path, default=DEFAULT_RUNTIME_LOCK)
    parser.add_argument("--meta-agent-src", type=Path, default=DEFAULT_META_AGENT_SRC)
    parser.add_argument("--dll-dir", type=Path, action="append", default=[DEFAULT_DLL_DIR])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=HERE / "semantic_fists.json")
    parser.add_argument("--audit", type=Path, default=HERE / "materialization_audit.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be positive")
    for output in (args.output, args.audit):
        if output.exists() and not args.force:
            parser.error(f"refusing to overwrite {output}; pass --force explicitly")
    try:
        search, panel, genome_path = validate_frozen_inputs(
            args.search_results, args.panel, args.source, args.base_actions,
            args.base_metadata, args.genome,
        )
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    dll_handles = []
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in args.dll_dir:
            dll_handles.append(os.add_dll_directory(str(directory.resolve())))
    sys.path[:0] = [str(args.fast_python.resolve()), str(args.meta_agent_src.resolve())]
    try:
        import fast_kaggriculture._fast_kaggriculture as native_module
        from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
        from meta_agent.src.native_teammate_executor import NativeTeammateBundle
    except ImportError as error:
        parser.error(f"cannot import isolated Round-2 native runtime: {error}")
    native_path = Path(native_module.__file__).resolve()
    try:
        assert_isolated_native(native_path, args.fast_python)
        runtime_lock = validate_runtime_lock(args.runtime_lock, native_path, args.source)
    except RuntimeError as error:
        parser.error(str(error))
    except (json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    config = search["config"]
    decision_days = [int(value) for value in config["decision_days"]]
    checkpoint = int(config["checkpoint"])
    opening_hash = str(config["opening_action_sha256"]).lower()
    genome = load_genome(genome_path, adaptive_genome_names, adaptive_default_genome)
    bundle = NativeTeammateBundle(args.source, args.base_actions, args.base_metadata)
    if not hasattr(bundle.adaptive_executor, "candidate8_committed_sequence"):
        parser.error("native extension lacks candidate8_committed_sequence")
    opening = bundle.index("OPENING_A")
    train = {str(row["panel_id"]): row for row in panel["rows"] if row["split"] == "train"}

    def materialize(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        task_id = str(row["task_id"])
        panel_id = str(row["panel_id"])
        seed = int(row["seed"])
        seat = int(row["candidate_seat"])
        opponent = bundle.index("OPP_" + panel_id)
        ranks = [int(value) for value in row["selected_ranks"]]

        rank_run = bundle.adaptive_executor.candidate8_committed_sequence(
            genome, opponent, seed, decision_days, ranks, seat,
            bool(config["use_feasible_pool"]), opening, checkpoint, True,
        )
        rank_deltas = [full_delta(value) for value in rank_run["selected_deltas"]]
        intents = [canonical_intent(value) for value in rank_deltas]
        explicit_run = bundle.adaptive_executor.candidate8_committed_sequence(
            genome, opponent, seed, decision_days, [], seat,
            bool(config["use_feasible_pool"]), opening, checkpoint, True,
            rank_deltas,
        )
        explicit_deltas = [full_delta(value) for value in explicit_run["selected_deltas"]]

        rank_trace = normalized_trace(rank_run["trace"])
        explicit_trace = normalized_trace(explicit_run["trace"])
        baseline = bundle.executor.play(
            opening if seat == 0 else opponent,
            opponent if seat == 0 else opening,
            seed, -1, -1, -1, -1, True,
        )
        baseline_trace = normalized_trace(baseline["trace"])
        rank_rewards = [float(value) for value in rank_run["rewards"]]
        explicit_rewards = [float(value) for value in explicit_run["rewards"]]
        rank_families = [int(value) for value in rank_run["selected_family"]]
        explicit_families = [int(value) for value in explicit_run["selected_family"]]
        rank_signatures = [int(value) for value in rank_run["selected_signature"]]
        explicit_signatures = [int(value) for value in explicit_run["selected_signature"]]
        rank_matched = [bool(value) for value in rank_run["selected_matched"]]
        explicit_matched = [bool(value) for value in explicit_run["selected_matched"]]
        candidate_tape = [joint[seat] for joint in rank_trace]

        checks = {
            "rank_all_stages_selected": len(rank_matched) == len(decision_days) and all(rank_matched),
            "explicit_all_intents_matched": len(explicit_matched) == len(decision_days) and all(explicit_matched),
            "rank_decision_days_exact": [int(v) for v in rank_run["decision_day"]] == decision_days,
            "explicit_decision_days_exact": [int(v) for v in explicit_run["decision_day"]] == decision_days,
            "rank_rewards_match_round1": rank_rewards == [float(v) for v in row["oracle_rewards"]],
            "rank_families_match_round1": rank_families == [int(v) for v in row["selected_family_ids"]],
            "rank_signatures_match_round1": rank_signatures == [int(v) for v in row["selected_signatures"]],
            "rank_action_hash_matches_round1": route_action_hash(candidate_tape) == str(row["full_action_sha256"]).lower(),
            "rank_prefix_exact": rank_trace[:checkpoint] == baseline_trace[:checkpoint],
            "explicit_rewards_equal_rank": explicit_rewards == rank_rewards,
            "explicit_trace_equal_rank": explicit_trace == rank_trace,
            "explicit_families_equal_rank": explicit_families == rank_families,
            "explicit_signatures_equal_rank": explicit_signatures == rank_signatures,
            "explicit_deltas_equal_rank": explicit_deltas == rank_deltas,
            "explicit_intents_equal_input": [canonical_intent(v) for v in explicit_deltas] == intents,
        }
        passed = all(checks.values())
        sequence = semantic_sequence_payload(
            opening_hash=opening_hash, checkpoint=checkpoint,
            decision_days=decision_days, intents=intents,
        )
        sequence_hash = sha256_value(sequence)
        audit_row = {
            "task_id": task_id,
            "panel_id": panel_id,
            "opponent_action_sha256": train[panel_id]["action_sha256"],
            "seed": seed,
            "candidate_seat": seat,
            "source_selected_ranks": ranks,
            "semantic_fist_id": "semantic-fist:" + sequence_hash,
            "passed": passed,
            "checks": checks,
            "rank": {
                "rewards": rank_rewards,
                "score": score(rank_rewards, seat),
                "families": rank_families,
                "signatures": rank_signatures,
                "joint_trace_sha256": sha256_value({"schema": "joint-trace-v1", "trace": rank_trace}),
                "candidate_action_sha256": route_action_hash(candidate_tape),
                "delta_sha256": [sha256_value(value) for value in rank_deltas],
            },
            "explicit": {
                "rewards": explicit_rewards,
                "score": score(explicit_rewards, seat),
                "families": explicit_families,
                "signatures": explicit_signatures,
                "joint_trace_sha256": sha256_value({"schema": "joint-trace-v1", "trace": explicit_trace}),
                "candidate_action_sha256": route_action_hash(
                    [joint[seat] for joint in explicit_trace]
                ),
                "delta_sha256": [sha256_value(value) for value in explicit_deltas],
            },
        }
        return audit_row, rank_deltas

    rows = sorted(search["rows"], key=lambda value: str(value["task_id"]))
    materialized: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(materialize, row): str(row["task_id"]) for row in rows}
        for future in as_completed(futures):
            audit_row, deltas = future.result()
            materialized.append((audit_row, deltas))
            print(json.dumps({
                "done": len(materialized),
                "task_id": audit_row["task_id"],
                "passed": audit_row["passed"],
                "semantic_fist": audit_row["semantic_fist_id"][14:26],
            }), flush=True)
    materialized.sort(key=lambda value: value[0]["task_id"])

    failures = [row for row, _ in materialized if not row["passed"]]
    fists_by_id: dict[str, dict[str, Any]] = {}
    for audit_row, deltas in materialized:
        fist_id = audit_row["semantic_fist_id"]
        intents = [canonical_intent(value) for value in deltas]
        existing = fists_by_id.get(fist_id)
        if existing is None:
            fists_by_id[fist_id] = {
                "fist_id": fist_id,
                "semantic_sha256": fist_id.removeprefix("semantic-fist:"),
                "opening_action_sha256": opening_hash,
                "checkpoint": checkpoint,
                "decision_days": decision_days,
                "intents": intents,
                "execution_deltas": deltas,
                "representative_task_id": audit_row["task_id"],
                "source_task_ids": [audit_row["task_id"]],
            }
        else:
            if existing["intents"] != intents:
                raise RuntimeError(f"semantic fist hash collision: {fist_id}")
            existing["source_task_ids"].append(audit_row["task_id"])

    generated_at = datetime.now(timezone.utc).isoformat()
    common_inputs = {
        "round1_search_results": {
            "path": str(args.search_results.resolve()),
            "sha256": sha256_file(args.search_results),
            "config_sha256": config["config_sha256"],
        },
        "panel": {
            "path": str(args.panel.resolve()),
            "sha256": sha256_file(args.panel),
            "content_sha256": panel["content_sha256"],
        },
        "source": {"path": str(args.source.resolve()), "sha256": sha256_file(args.source)},
        "base_actions": {"path": str(args.base_actions.resolve()), "sha256": sha256_file(args.base_actions)},
        "base_metadata": {"path": str(args.base_metadata.resolve()), "sha256": sha256_file(args.base_metadata)},
        "genome": {"path": str(genome_path.resolve()), "sha256": sha256_file(genome_path)},
        "round2_native_extension": {"path": str(native_path), "sha256": sha256_file(native_path)},
        "runtime_lock": {
            "path": str(args.runtime_lock.resolve()),
            "sha256": sha256_file(args.runtime_lock),
            "schema": runtime_lock["schema"],
        },
    }
    fist_payload = {
        "schema": "kaggriculture.counter-cluster-semantic-fists.v1",
        "generated_at_utc": generated_at,
        "passed_materialization_audit": not failures,
        "source_searches": len(materialized),
        "unique_semantic_fists": len(fists_by_id),
        "identity_contract": {
            "included_fields": list(INTENT_FIELDS),
            "excluded_state_derived_fields": list(DERIVED_FIELDS),
            "sequence_anchor_fields": [
                "opening_action_sha256", "checkpoint", "decision_days"
            ],
            "execution_note": (
                "execution_deltas retain one audited full binding payload, but fresh "
                "execution matches only canonical intent and re-normalizes it in live state"
            ),
        },
        "inputs": common_inputs,
        "fists": [fists_by_id[key] for key in sorted(fists_by_id)],
    }
    audit_payload = {
        "schema": "kaggriculture.counter-cluster-plan-materialization-audit.v1",
        "generated_at_utc": generated_at,
        "passed": not failures,
        "scope": (
            "48 original discovery scenarios only; no fresh-seed train, validation, "
            "or holdout outcomes were evaluated"
        ),
        "summary": {
            "expected_rows": EXPECTED_SEARCHES,
            "audited_rows": len(materialized),
            "passed_rows": len(materialized) - len(failures),
            "failed_rows": len(failures),
            "unique_semantic_fists": len(fists_by_id),
        },
        "inputs": common_inputs,
        "rows": [row for row, _ in materialized],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, fist_payload)
    atomic_json(args.audit, audit_payload)
    del dll_handles
    print(json.dumps({
        "output": str(args.output.resolve()),
        "audit": str(args.audit.resolve()),
        **audit_payload["summary"],
        "passed": audit_payload["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

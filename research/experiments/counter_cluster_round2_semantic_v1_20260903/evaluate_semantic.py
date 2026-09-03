#!/usr/bin/env python3
"""Evaluate state-adaptive Candidate8 semantic fists on one frozen split.

Opponent tapes are rebuilt only from the selected panel split.  Fresh games
pass an empty rank sequence and the stored PlanDelta payloads; concrete actions
remain the live Candidate8 planner's responsibility.  This entry point refuses
the locked holdout split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from materialize_plan_sequences import (
    atomic_json,
    canonical_bytes,
    canonical_intent,
    load_genome,
    normalized_action,
    semantic_sequence_payload,
    sha256_file,
    sha256_value,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROUND1 = ROOT / "experiments" / "counter_cluster_round1"
ROUND1_FAST_PYTHON = (
    ROOT / "experiments" / "nt_counter_fist_mvp" / "fast_kaggriculture" / "python"
)
DEFAULT_FAST_PYTHON = HERE / "native" / "fast_kaggriculture" / "python"
NT_ROOT = Path(
    r"D:\github\Kaggriculture_candidate8\nt\latest_20260901_candidate8_width12_mcts\workspace"
)
DEFAULT_META_AGENT_SRC = (
    NT_ROOT / "agents" / "route_clustering_switch_agent" / "src"
)
DEFAULT_DLL_DIR = (
    Path(r"D:\Kaggriculture\route_clustering_top40_20260831")
    / ".conda_toolchain"
    / "Library"
    / "bin"
)

CONFIG_SCHEMA = "kaggriculture.counter-cluster-round2-semantic-config.v1"
FISTS_SCHEMA = "kaggriculture.counter-cluster-semantic-fists.v1"
PORTFOLIO_SCHEMA = "kaggriculture.counter-cluster-round2-portfolio.v1"
RUNTIME_SCHEMA = "kaggriculture.counter-cluster-round2-runtime-lock.v1"
HORIZON = 719
FROZEN_SEEDS = {
    "train": tuple(range(9107001, 9107009)),
    "validation": tuple(range(9108001, 9108009)),
}
FROZEN_SPLIT_SIZES = {"train": 12, "validation": 6}


def _decode_json(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    return json.loads(raw.decode("utf-8"))


def _route_hash(actions: Sequence[Mapping[str, Any]]) -> str:
    return sha256_value({"schema": "route-actions-v1", "actions": list(actions)})


def _load_opening(path: Path, expected_hash: str) -> tuple[str, list[dict[str, Any]]]:
    payload = _decode_json(path)
    if not isinstance(payload, Mapping) or len(payload) != 1:
        raise ValueError("opening action source must contain exactly one route")
    route_id, raw_actions = next(iter(payload.items()))
    actions = [normalized_action(value) for value in raw_actions]
    observed = _route_hash(actions)
    if len(actions) != HORIZON or observed != expected_hash:
        raise ValueError("Opening A action source does not match the frozen hash")
    if str(route_id) != "opening:" + observed:
        raise ValueError("Opening A route id/hash mismatch")
    return str(route_id), actions


def _load_replay_tape(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(row["replay_path"]))
    replay = json.loads(path.read_text(encoding="utf-8"))
    steps = replay.get("steps")
    seat = int(row["player_index"])
    if not isinstance(steps, list) or len(steps) != HORIZON + 1 or seat not in (0, 1):
        raise ValueError(f"invalid replay shape for {row['panel_id']}")
    actions: list[dict[str, Any]] = []
    for step in range(HORIZON):
        joint = steps[step + 1]
        if not isinstance(joint, list) or len(joint) != 2 or not isinstance(joint[seat], Mapping):
            raise ValueError(f"invalid replay step {step + 1} for {row['panel_id']}")
        actions.append(normalized_action(joint[seat].get("action")))
    observed = _route_hash(actions)
    if observed != str(row["action_sha256"]).lower():
        raise ValueError(f"replay action hash drift for {row['panel_id']}")
    return actions


def _validate_config_and_panel(
    config_path: Path, lock_path: Path, panel_path: Path, split: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported Round2 config schema")
    if sha256_file(config_path) != str(lock.get("config", {}).get("sha256", "")):
        raise ValueError("Round2 config differs from inputs.lock.json")
    if sha256_file(panel_path) != str(config["panel"]["file_sha256"]):
        raise ValueError("panel file hash differs from frozen config")
    body = dict(panel)
    claimed_content_hash = str(body.pop("content_sha256", ""))
    if claimed_content_hash != sha256_value(body):
        raise ValueError("panel content_sha256 is invalid")
    if claimed_content_hash != str(config["panel"]["content_sha256"]):
        raise ValueError("panel content differs from frozen config")

    expected_ids = [str(value) for value in config["panel"]["splits"][split]]
    by_id = {str(row["panel_id"]): row for row in panel.get("rows", [])}
    if len(by_id) != int(config["panel"]["count"]):
        raise ValueError("panel ids are missing or duplicated")
    rows = []
    for panel_id in expected_ids:
        row = by_id.get(panel_id)
        if row is None or str(row.get("split")) != split:
            raise ValueError(f"panel split mismatch for {panel_id}")
        rows.append(row)
    if len(rows) != FROZEN_SPLIT_SIZES[split]:
        raise ValueError(f"frozen {split} split size mismatch")
    observed_split_ids = {
        str(row["panel_id"]) for row in panel["rows"] if str(row.get("split")) == split
    }
    if observed_split_ids != set(expected_ids):
        raise ValueError(f"panel has unexpected {split} members")
    if tuple(int(value) for value in config["seed_banks"][split]) != FROZEN_SEEDS[split]:
        raise ValueError(f"frozen {split} seed bank mismatch")
    if [int(value) for value in config["seed_banks"]["seats"]] != [0, 1]:
        raise ValueError("frozen seat bank mismatch")
    return config, panel, rows


def _build_split_assets(
    directory: Path, rows: Sequence[Mapping[str, Any]], opening_id: str,
    opening_actions: list[dict[str, Any]], split: str,
) -> tuple[Path, Path, list[str], list[str], dict[str, Any]]:
    tapes: dict[str, list[dict[str, Any]]] = {opening_id: opening_actions}
    entries = [{"family": "OPENING_A", "route_id": opening_id}]
    opponent_ids: list[str] = []
    opponent_hashes: list[str] = []
    for row in rows:
        panel_id = str(row["panel_id"])
        action_hash = str(row["action_sha256"]).lower()
        route_id = "opponent:" + action_hash
        tape = _load_replay_tape(row)
        if route_id in tapes:
            raise ValueError(f"duplicate split action tape: {panel_id}")
        tapes[route_id] = tape
        entries.append({"family": "OPP_" + panel_id, "route_id": route_id})
        opponent_ids.append(route_id)
        opponent_hashes.append(action_hash)

    actions_path = directory / f"{split}_actions.json.zlib"
    metadata_path = directory / f"{split}_metadata.json"
    actions_path.write_bytes(zlib.compress(canonical_bytes(tapes), level=9))
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "kaggriculture.counter-cluster-round2-split-library.v1",
                "split": split,
                "opponent_routes": entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = {
        "construction": "opening_from_locked_blob; opponents_only_from_selected_panel_replay_path",
        "split": split,
        "opponent_count": len(rows),
        "actions_sha256": sha256_file(actions_path),
        "metadata_sha256": sha256_file(metadata_path),
        "excluded_panel_splits": sorted({"train", "validation", "holdout"} - {split}),
    }
    return actions_path, metadata_path, opponent_ids, opponent_hashes, receipt


def _validate_semantic_fists(
    path: Path, config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != FISTS_SCHEMA or payload.get("passed_materialization_audit") is not True:
        raise ValueError("semantic fists lack a passed materialization audit")
    if int(payload.get("source_searches", -1)) != int(config["search_choices"]["rows"]):
        raise ValueError("semantic fists do not cover the 48 locked search choices")
    identity = payload.get("identity_contract", {})
    if list(identity.get("included_fields", [])) != list(
        config["treatment"]["semantic_identity_fields"]
    ):
        raise ValueError("semantic identity fields differ from frozen config")
    if set(identity.get("excluded_state_derived_fields", [])) != set(
        config["treatment"]["semantic_identity_excludes"]
    ):
        raise ValueError("semantic identity exclusions differ from frozen config")

    opening_hash = str(config["opening"]["action_sha256"])
    checkpoint = int(config["opening"]["semantic_start_step"])
    days = [int(value) for value in config["candidate8"]["decision_days"]]
    fists: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload.get("fists", []):
        fist = dict(raw)
        fist_id = str(fist.get("fist_id", ""))
        if not fist_id.startswith("semantic-fist:") or fist_id in seen:
            raise ValueError("semantic fist ids are invalid or duplicated")
        if (
            str(fist.get("opening_action_sha256")) != opening_hash
            or int(fist.get("checkpoint", -1)) != checkpoint
            or [int(value) for value in fist.get("decision_days", [])] != days
        ):
            raise ValueError(f"semantic fist anchor mismatch: {fist_id}")
        intents = [canonical_intent(value) for value in fist.get("intents", [])]
        deltas = [dict(value) for value in fist.get("execution_deltas", [])]
        if len(intents) != len(days) or len(deltas) != len(days):
            raise ValueError(f"semantic fist stage count mismatch: {fist_id}")
        if [canonical_intent(value) for value in deltas] != intents:
            raise ValueError(f"semantic fist execution payload/intents disagree: {fist_id}")
        sequence = semantic_sequence_payload(
            opening_hash=opening_hash, checkpoint=checkpoint,
            decision_days=days, intents=intents,
        )
        observed = sha256_value(sequence)
        if fist_id != "semantic-fist:" + observed or str(fist.get("semantic_sha256")) != observed:
            raise ValueError(f"semantic fist identity hash mismatch: {fist_id}")
        fist["intents"] = intents
        fist["execution_deltas"] = deltas
        seen.add(fist_id)
        fists.append(fist)
    if not fists or len(fists) != int(payload.get("unique_semantic_fists", -1)):
        raise ValueError("semantic fist count mismatch")
    fists.sort(key=lambda value: value["fist_id"])
    return payload, fists


def _record_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise ValueError(f"locked file is missing: {path}")
    if sha256_file(path) != str(record.get("sha256", "")).lower():
        raise ValueError(f"locked file hash mismatch: {path}")
    return path


def _validate_runtime_lock(
    path: Path, semantic_payload: Mapping[str, Any], config_path: Path,
    input_lock_path: Path,
) -> tuple[dict[str, Any], Path, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("unsupported or missing Round2 runtime lock")
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ValueError("runtime lock lacks its files list")
    by_role = {str(row.get("role")): row for row in rows}
    if len(by_role) != len(rows):
        raise ValueError("runtime lock has duplicate file roles")
    required_roles = {
        "v3_teammate_source",
        "round2_native_extension",
        "semantic_binding",
        "semantic_executor",
        "semantic_executor_header",
        "api_contract_test",
        "materializer",
        "evaluator",
        "frozen_config",
        "input_lock",
    }
    missing = sorted(required_roles - set(by_role))
    if missing:
        raise ValueError(f"runtime lock is missing role: {missing[0]}")
    for role, record in by_role.items():
        locked = _record_path(record).resolve()
        if int(record.get("bytes", -1)) != locked.stat().st_size:
            raise ValueError(f"runtime-locked byte size mismatch: {role}")

    native_record = by_role["round2_native_extension"]
    contract_record = payload.get("api_contract_test")
    if not isinstance(contract_record, Mapping):
        raise ValueError("runtime lock lacks its API-contract receipt")
    if contract_record.get("passed") is not True:
        raise ValueError("Round2 native API contract test has not passed")
    _record_path(contract_record)
    native_path = _record_path(native_record).resolve()
    native_hash = sha256_file(native_path)
    materialized_native = semantic_payload.get("inputs", {}).get("round2_native_extension", {})
    materialized_path = _record_path(materialized_native).resolve()
    if native_path != materialized_path or native_hash != str(materialized_native.get("sha256", "")):
        raise ValueError("evaluation and materialization do not use the same Round2 pyd")

    materialized_source = semantic_payload.get("inputs", {}).get("source", {})
    expected_paths = {
        "v3_teammate_source": _record_path(materialized_source).resolve(),
        "round2_native_extension": materialized_path,
        "materializer": (HERE / "materialize_plan_sequences.py").resolve(),
        "evaluator": Path(__file__).resolve(),
        "frozen_config": config_path.resolve(),
        "input_lock": input_lock_path.resolve(),
    }
    for role, expected in expected_paths.items():
        if Path(str(by_role[role].get("path", ""))).resolve() != expected:
            raise ValueError(f"runtime lock path mismatch: {role}")
    source_hash = str(payload.get("source_sha256", ""))
    if source_hash != str(by_role["v3_teammate_source"].get("sha256", "")):
        raise ValueError("runtime lock source_sha256 convenience field mismatch")
    convenience_native = payload.get("native_extension")
    if not isinstance(convenience_native, Mapping):
        raise ValueError("runtime lock lacks native_extension convenience record")
    if (
        Path(str(convenience_native.get("path", ""))).resolve() != native_path
        or str(convenience_native.get("sha256", "")) != native_hash
    ):
        raise ValueError("runtime lock native_extension convenience record mismatch")
    return payload, native_path, native_hash


def _load_frozen_portfolio(
    path: Path, config_path: Path, fists_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != PORTFOLIO_SCHEMA
        or payload.get("frozen") is not True
        or payload.get("source_split") != "train"
    ):
        raise ValueError("validation requires a frozen Round2 train portfolio")
    if str(payload.get("config_sha256", "")) != sha256_file(config_path):
        raise ValueError("portfolio config hash mismatch")
    if str(payload.get("semantic_fists_sha256", "")) != sha256_file(fists_path):
        raise ValueError("portfolio semantic-fist hash mismatch")
    records: dict[str, Path] = {}
    for key in ("train_matrix", "train_report"):
        record = payload.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"portfolio lacks {key} provenance")
        records[key] = _record_path(record)
    ids = list(dict.fromkeys(str(value) for value in payload.get("selected_fist_ids", [])))
    if not ids or len(ids) != int(payload.get("selected_count", -1)):
        raise ValueError("portfolio has no valid selected fist ids")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_report = json.loads(records["train_report"].read_text(encoding="utf-8"))
    if payload.get("coverage_rule") != config.get("coverage_rule"):
        raise ValueError("portfolio coverage rule differs from frozen config")
    if train_report.get("split") != "train" or train_report.get("set_cover") is None:
        raise ValueError("portfolio provenance is not a train set-cover report")
    if ids != [str(value) for value in train_report["set_cover"]["selected_fist_ids"]]:
        raise ValueError("portfolio selection differs from its frozen train report")
    return payload, ids


def _select_cover(
    coverage: np.ndarray, path_score: np.ndarray, path_uplift: np.ndarray,
    fist_ids: Sequence[str], cap: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    selected: list[int] = []
    uncovered = np.ones(coverage.shape[1], dtype=bool)
    trace: list[dict[str, Any]] = []
    while bool(np.any(uncovered)) and len(selected) < cap:
        choices = []
        for fist in range(coverage.shape[0]):
            if fist in selected:
                continue
            new = uncovered & coverage[fist]
            gain = int(np.count_nonzero(new))
            members = coverage[fist]
            worst = float(np.min(path_score[fist, members])) if gain else -np.inf
            uplift = float(np.mean(path_uplift[fist, members])) if gain else -np.inf
            choices.append((gain, worst, uplift, -fist, fist, new))
        if not choices:
            break
        gain, worst, uplift, _, fist, new = max(choices, key=lambda value: value[:-1])
        if gain <= 0:
            break
        selected.append(fist)
        uncovered[new] = False
        trace.append({
            "portfolio_rank": len(selected),
            "fist_id": fist_ids[fist],
            "newly_covered_opponent_indices": np.flatnonzero(new).astype(int).tolist(),
            "newly_covered_count": gain,
            "tie_break_worst_member_score": worst,
            "tie_break_mean_uplift": uplift,
            "cumulative_covered_count": int(np.count_nonzero(~uncovered)),
            "remaining_residual_count": int(np.count_nonzero(uncovered)),
        })
    return selected, trace


def _clusters(
    selected: Sequence[int], coverage: np.ndarray, path_score: np.ndarray,
    path_margin: np.ndarray, fist_ids: Sequence[str], opponent_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[int]]:
    assignments: dict[int, list[int]] = {fist: [] for fist in selected}
    residual: list[int] = []
    for opponent in range(coverage.shape[1]):
        eligible = [fist for fist in selected if coverage[fist, opponent]]
        if not eligible:
            residual.append(opponent)
            continue
        best = max(
            eligible,
            key=lambda fist: (path_score[fist, opponent], path_margin[fist, opponent], -fist),
        )
        assignments[best].append(opponent)
    rows: list[dict[str, Any]] = []
    for fist in selected:
        members = assignments[fist]
        if members:
            rows.append({
                "cluster_id": f"C{len(rows) + 1:03d}",
                "kind": "covered",
                "representative_fist_id": fist_ids[fist],
                "member_opponent_ids": [opponent_ids[index] for index in members],
                "size": len(members),
                "minimum_member_score": float(np.min(path_score[fist, members])),
                "mean_member_score": float(np.mean(path_score[fist, members])),
                "minimum_member_margin": float(np.min(path_margin[fist, members])),
            })
    for ordinal, opponent in enumerate(residual, 1):
        rows.append({
            "cluster_id": f"R{ordinal:03d}",
            "kind": "residual",
            "representative_fist_id": None,
            "member_opponent_ids": [opponent_ids[opponent]],
            "size": 1,
        })
    return rows, residual


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
    }


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _self_test() -> None:
    coverage = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=bool)
    score = np.asarray([[.8, .7, .2], [.2, .9, .8]])
    selected, trace = _select_cover(coverage, score, score - .5, ["A", "B"], 2)
    clusters, residual = _clusters(
        selected, coverage, score, score, ["A", "B"], ["x", "y", "z"]
    )
    assert selected == [1, 0]
    assert [row["fist_id"] for row in trace] == ["B", "A"]
    assert not residual and {row["representative_fist_id"] for row in clusters} == {"A", "B"}
    assert set(FROZEN_SEEDS["train"]).isdisjoint(FROZEN_SEEDS["validation"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "holdout"))
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--input-lock", type=Path, default=HERE / "inputs.lock.json")
    parser.add_argument("--panel", type=Path, default=ROUND1 / "panel24.json")
    parser.add_argument("--opening-actions", type=Path, default=ROUND1 / "opening_actions.json.zlib")
    parser.add_argument("--semantic-fists", type=Path, default=HERE / "semantic_fists.json")
    parser.add_argument("--runtime-lock", type=Path, default=HERE / "runtime.lock.json")
    parser.add_argument("--portfolio", type=Path)
    parser.add_argument("--portfolio-out", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fast-python", type=Path, default=DEFAULT_FAST_PYTHON)
    parser.add_argument("--meta-agent-src", type=Path, default=DEFAULT_META_AGENT_SRC)
    parser.add_argument("--dll-dir", type=Path, action="append", default=[DEFAULT_DLL_DIR])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        print("self-test: PASS")
        return 0
    if args.split is None:
        parser.error("--split is required")
    if args.split == "holdout":
        parser.error("holdout is locked; evaluate_semantic.py refuses all holdout evaluation")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.split == "validation" and args.portfolio is None:
        parser.error("validation requires --portfolio from the frozen train set-cover")
    if args.split == "train" and args.portfolio is not None:
        parser.error("train fits its own set-cover and must not receive --portfolio")
    if args.split == "validation" and args.portfolio_out is not None:
        parser.error("validation cannot write or refit a portfolio")

    args.matrix = args.matrix or HERE / f"{args.split}_semantic_matrix.npz"
    args.report = args.report or HERE / f"{args.split}_semantic_report.json"
    if args.split == "train":
        args.portfolio_out = args.portfolio_out or HERE / "train_semantic_portfolio.json"
    outputs = [args.matrix, args.report]
    if args.portfolio_out is not None:
        outputs.append(args.portfolio_out)
    for output in outputs:
        if output.exists() and not args.force:
            parser.error(f"refusing to overwrite {output}; pass --force explicitly")
        output.parent.mkdir(parents=True, exist_ok=True)

    try:
        config, panel, panel_rows = _validate_config_and_panel(
            args.config, args.input_lock, args.panel, args.split,
        )
        semantic_payload, fists = _validate_semantic_fists(args.semantic_fists, config)
        runtime_payload, locked_native_path, locked_native_hash = _validate_runtime_lock(
            args.runtime_lock, semantic_payload, args.config, args.input_lock,
        )
        opening_id, opening_actions = _load_opening(
            args.opening_actions, str(config["opening"]["action_sha256"]),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))

    portfolio_payload = None
    if args.portfolio is not None:
        try:
            portfolio_payload, wanted = _load_frozen_portfolio(
                args.portfolio, args.config, args.semantic_fists,
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            parser.error(str(error))
        fist_by_id = {fist["fist_id"]: fist for fist in fists}
        unknown = [fist_id for fist_id in wanted if fist_id not in fist_by_id]
        if unknown:
            parser.error(f"portfolio contains an unknown fist: {unknown[0]}")
        fists = [fist_by_id[fist_id] for fist_id in wanted]

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
        parser.error(f"cannot import Round2 native runtime: {error}")
    actual_native_path = Path(native_module.__file__).resolve()
    requested_fast_python = args.fast_python.resolve()
    if not actual_native_path.is_relative_to(requested_fast_python):
        parser.error(f"loaded pyd is outside --fast-python: {actual_native_path}")
    if actual_native_path.is_relative_to(ROUND1_FAST_PYTHON.resolve()):
        parser.error("refusing to load the Round1 pyd")
    if actual_native_path != locked_native_path or sha256_file(actual_native_path) != locked_native_hash:
        parser.error("loaded pyd path/hash differs from the Round2 runtime lock")

    source_record = semantic_payload["inputs"]["source"]
    genome_record = semantic_payload["inputs"]["genome"]
    try:
        source_path = _record_path(source_record)
        genome_path = _record_path(genome_record)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    genome = load_genome(genome_path, adaptive_genome_names, adaptive_default_genome)
    days = [int(value) for value in config["candidate8"]["decision_days"]]
    prefix_steps = int(config["opening"]["semantic_start_step"])
    seeds = FROZEN_SEEDS[args.split]
    use_feasible_pool = bool(config["candidate8"]["use_feasible_pool"])
    fist_ids = [str(fist["fist_id"]) for fist in fists]
    stages = len(days)

    with tempfile.TemporaryDirectory(prefix=f"round2_{args.split}_") as raw_directory:
        asset_directory = Path(raw_directory)
        try:
            actions_path, metadata_path, opponent_ids, opponent_hashes, asset_receipt = (
                _build_split_assets(
                    asset_directory, panel_rows, opening_id, opening_actions, args.split,
                )
            )
            bundle = NativeTeammateBundle(source_path, actions_path, metadata_path)
        except (KeyError, TypeError, ValueError, OSError) as error:
            parser.error(str(error))
        if not hasattr(bundle.adaptive_executor, "candidate8_committed_sequence"):
            parser.error("Round2 pyd lacks candidate8_committed_sequence")
        opening_index = bundle.index("OPENING_A")
        opponent_indices = [bundle.index("OPP_" + str(row["panel_id"])) for row in panel_rows]

        shape = (len(fists), len(opponent_ids), len(seeds), 2)
        rewards = np.full(shape + (2,), np.nan, dtype=np.float64)
        baseline_rewards = np.full(shape[1:] + (2,), np.nan, dtype=np.float64)
        matches = np.zeros(shape + (stages,), dtype=bool)
        actual_family = np.full(shape + (stages,), -1, dtype=np.int8)
        actual_signature = np.zeros(shape + (stages,), dtype=np.uint64)
        reported_rank = np.full(shape + (stages,), -2, dtype=np.int16)
        actual_delta_sha256 = np.full(shape + (stages,), "", dtype="U64")
        actual_intent_sha256 = np.full(shape + (stages,), "", dtype="U64")
        expected_intent_sha256 = np.asarray([
            [sha256_value(intent) for intent in fist["intents"]] for fist in fists
        ], dtype="U64")
        source_delta_sha256 = np.asarray([
            [sha256_value(delta) for delta in fist["execution_deltas"]] for fist in fists
        ], dtype="U64")
        source_signature = np.asarray([
            [int(delta["signature"]) for delta in fist["execution_deltas"]] for fist in fists
        ], dtype=np.uint64)

        tasks: list[tuple[str, int, int, int, int]] = []
        for opponent in range(len(opponent_ids)):
            for seed_index in range(len(seeds)):
                for seat in (0, 1):
                    tasks.append(("baseline", -1, opponent, seed_index, seat))
        for fist in range(len(fists)):
            for opponent in range(len(opponent_ids)):
                for seed_index in range(len(seeds)):
                    for seat in (0, 1):
                        tasks.append(("semantic", fist, opponent, seed_index, seat))

        def play(task: tuple[str, int, int, int, int]) -> tuple[Any, ...]:
            kind, fist_index, opponent_index, seed_index, seat = task
            opponent_route = opponent_indices[opponent_index]
            seed = seeds[seed_index]
            if kind == "baseline":
                result = bundle.executor.play(
                    opening_index if seat == 0 else opponent_route,
                    opponent_route if seat == 0 else opening_index,
                    seed, -1, -1, -1, -1, False,
                )
                return (*task, [float(value) for value in result["rewards"]])
            result = bundle.adaptive_executor.candidate8_committed_sequence(
                genome, opponent_route, seed, days, [], seat,
                use_feasible_pool, opening_index, prefix_steps, False,
                fists[fist_index]["execution_deltas"],
            )
            observed_days = [int(value) for value in result["decision_day"]]
            observed_matches = [bool(value) for value in result["selected_matched"]]
            observed_ranks = [int(value) for value in result["selected_rank"]]
            families = [int(value) for value in result["selected_family"]]
            signatures = [int(value) for value in result["selected_signature"]]
            deltas = [dict(value) for value in result["selected_deltas"]]
            if not all(len(values) == stages for values in (
                observed_days, observed_matches, observed_ranks, families, signatures, deltas,
            )):
                raise RuntimeError("native semantic result has the wrong stage count")
            if observed_days != days or any(value != -1 for value in observed_ranks):
                raise RuntimeError("native semantic result used unexpected days/ranks")
            delta_hashes = [
                sha256_value(value) if matched else ""
                for value, matched in zip(deltas, observed_matches, strict=True)
            ]
            intent_hashes = [
                sha256_value(canonical_intent(value)) if matched else ""
                for value, matched in zip(deltas, observed_matches, strict=True)
            ]
            for stage, matched in enumerate(observed_matches):
                if matched:
                    if intent_hashes[stage] != expected_intent_sha256[fist_index, stage]:
                        raise RuntimeError("native exact-intent match returned a different intent")
                    if families[stage] != int(deltas[stage]["family_id"]):
                        raise RuntimeError("native family telemetry disagrees with selected delta")
                    if signatures[stage] != int(deltas[stage]["signature"]):
                        raise RuntimeError("native signature telemetry disagrees with selected delta")
                elif families[stage] != -1 or signatures[stage] != 0:
                    raise RuntimeError("unmatched semantic stage has nonempty selection telemetry")
            return (
                *task,
                [float(value) for value in result["rewards"]],
                observed_matches, observed_ranks, families, signatures,
                delta_hashes, intent_hashes,
            )

        started = time.perf_counter()
        completed = 0
        progress_step = max(1, len(tasks) // 20)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(play, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                kind, fist, opponent, seed_index, seat = result[:5]
                if kind == "baseline":
                    baseline_rewards[opponent, seed_index, seat] = result[5]
                else:
                    rewards[fist, opponent, seed_index, seat] = result[5]
                    matches[fist, opponent, seed_index, seat] = result[6]
                    reported_rank[fist, opponent, seed_index, seat] = result[7]
                    actual_family[fist, opponent, seed_index, seat] = result[8]
                    actual_signature[fist, opponent, seed_index, seat] = result[9]
                    actual_delta_sha256[fist, opponent, seed_index, seat] = result[10]
                    actual_intent_sha256[fist, opponent, seed_index, seat] = result[11]
                completed += 1
                if completed % progress_step == 0 or completed == len(tasks):
                    print(json.dumps({"completed": completed, "total": len(tasks)}), flush=True)
        elapsed = time.perf_counter() - started

    if np.isnan(rewards).any() or np.isnan(baseline_rewards).any():
        raise RuntimeError("evaluation matrix contains missing game results")
    own_rewards = np.stack((rewards[..., 0, 0], rewards[..., 1, 1]), axis=-1)
    other_rewards = np.stack((rewards[..., 0, 1], rewards[..., 1, 0]), axis=-1)
    margins = own_rewards - other_rewards
    baseline_own = np.stack((baseline_rewards[..., 0, 0], baseline_rewards[..., 1, 1]), axis=-1)
    baseline_other = np.stack((baseline_rewards[..., 0, 1], baseline_rewards[..., 1, 0]), axis=-1)
    baseline_margins = baseline_own - baseline_other
    outcome = (margins > 0).astype(np.uint8) * 2 + (margins == 0).astype(np.uint8)
    baseline_outcome = (
        (baseline_margins > 0).astype(np.uint8) * 2
        + (baseline_margins == 0).astype(np.uint8)
    )
    scores = outcome.astype(np.float32) * 0.5
    baseline_scores = baseline_outcome.astype(np.float32) * 0.5
    path_score = scores.mean(axis=(2, 3))
    path_margin = margins.mean(axis=(2, 3))
    baseline_path_score = baseline_scores.mean(axis=(1, 2))
    baseline_path_margin = baseline_margins.mean(axis=(1, 2))
    path_uplift = path_score - baseline_path_score[None, :]
    rule = config["coverage_rule"]
    coverage = (
        (path_score >= float(rule["minimum_score"]) - 1e-12)
        & (path_uplift >= float(rule["minimum_score_uplift"]) - 1e-12)
        & (path_margin > float(rule["minimum_mean_reward_margin_exclusive"]))
    )
    cell_all_stage_match_rate = matches.all(axis=-1).mean(axis=(2, 3))
    strict_full_match_coverage = coverage & (cell_all_stage_match_rate == 1.0)
    source_delta_broadcast = source_delta_sha256[:, None, None, None, :]
    source_signature_broadcast = source_signature[:, None, None, None, :]
    normalization_changed = matches & (actual_delta_sha256 != source_delta_broadcast)
    signature_changed = matches & (actual_signature != source_signature_broadcast)

    selected: list[int] = []
    cover_trace: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] | None = None
    residual: list[int] = []
    if args.split == "train":
        selected, cover_trace = _select_cover(
            coverage, path_score, path_uplift, fist_ids,
            min(int(rule["portfolio_cap"]), len(fists)),
        )
        clusters, residual = _clusters(
            selected, coverage, path_score, path_margin, fist_ids, opponent_ids,
        )
        for row in cover_trace:
            row["newly_covered_opponent_ids"] = [
                opponent_ids[index]
                for index in row.pop("newly_covered_opponent_indices")
            ]

    fist_rows = []
    for fist, fist_id in enumerate(fist_ids):
        matched_values = matches[fist]
        matched_count = int(np.count_nonzero(matched_values))
        fist_rows.append({
            "fist_id": fist_id,
            "score_rate": float(np.mean(scores[fist])),
            "raw_win_rate": float(np.mean(margins[fist] > 0)),
            "mean_margin": float(np.mean(margins[fist])),
            "minimum_path_score": float(np.min(path_score[fist])),
            "mean_uplift_vs_baseline": float(np.mean(path_uplift[fist])),
            "covered_path_count": int(np.count_nonzero(coverage[fist])),
            "strict_full_match_covered_path_count": int(
                np.count_nonzero(strict_full_match_coverage[fist])
            ),
            "stage_match_rate": float(np.mean(matched_values)),
            "all_stages_match_game_rate": float(np.mean(matched_values.all(axis=-1))),
            "match_rate_by_decision_day": {
                str(day): float(np.mean(matched_values[..., stage]))
                for stage, day in enumerate(days)
            },
            "live_normalization_changed_rate_given_match": (
                float(np.count_nonzero(normalization_changed[fist]) / matched_count)
                if matched_count else None
            ),
            "signature_changed_rate_given_match": (
                float(np.count_nonzero(signature_changed[fist]) / matched_count)
                if matched_count else None
            ),
        })
    fist_rows.sort(key=lambda row: (
        -row["covered_path_count"], -row["minimum_path_score"],
        -row["mean_uplift_vs_baseline"], row["fist_id"],
    ))

    opponent_rows = []
    for opponent, opponent_id in enumerate(opponent_ids):
        best = max(
            range(len(fists)),
            key=lambda fist: (path_score[fist, opponent], path_margin[fist, opponent], -fist),
        )
        opponent_rows.append({
            "opponent_id": opponent_id,
            "panel_id": str(panel_rows[opponent]["panel_id"]),
            "action_sha256": opponent_hashes[opponent],
            "baseline_score": float(baseline_path_score[opponent]),
            "baseline_mean_margin": float(baseline_path_margin[opponent]),
            "best_fist_id": fist_ids[best],
            "best_fist_score": float(path_score[best, opponent]),
            "best_fist_uplift": float(path_uplift[best, opponent]),
            "best_fist_mean_margin": float(path_margin[best, opponent]),
            "eligible_covering_fist_ids": [
                fist_ids[fist] for fist in np.flatnonzero(coverage[:, opponent])
            ],
        })

    _atomic_npz(
        args.matrix,
        fist_ids=np.asarray(fist_ids),
        opponent_ids=np.asarray(opponent_ids),
        opponent_action_sha256=np.asarray(opponent_hashes),
        seeds=np.asarray(seeds, dtype=np.int64),
        candidate_seats=np.asarray([0, 1], dtype=np.int8),
        decision_days=np.asarray(days, dtype=np.int16),
        rewards=rewards,
        baseline_rewards=baseline_rewards,
        margins=margins,
        baseline_margins=baseline_margins,
        outcome=outcome,
        baseline_outcome=baseline_outcome,
        scores=scores,
        baseline_scores=baseline_scores,
        path_scores=path_score,
        path_margins=path_margin,
        path_uplift=path_uplift,
        coverage=coverage,
        selected_matched=matches,
        selected_family=actual_family,
        selected_signature=actual_signature,
        selected_rank=reported_rank,
        actual_delta_sha256=actual_delta_sha256,
        actual_intent_sha256=actual_intent_sha256,
        expected_intent_sha256=expected_intent_sha256,
        source_execution_delta_sha256=source_delta_sha256,
        source_signature=source_signature,
        live_normalization_changed=normalization_changed,
        signature_changed=signature_changed,
        cell_all_stage_match_rate=cell_all_stage_match_rate,
        strict_full_match_coverage=strict_full_match_coverage,
    )
    matrix_hash = sha256_file(args.matrix)
    covered_union = np.any(coverage, axis=0)
    set_cover = None
    if args.split == "train":
        set_cover = {
            "method": "greedy uncovered gain; ties by worst covered-member score then mean uplift",
            "selected_fist_ids": [fist_ids[index] for index in selected],
            "selected_count": len(selected),
            "covered_opponent_count": len(opponent_ids) - len(residual),
            "coverage_rate": (len(opponent_ids) - len(residual)) / len(opponent_ids),
            "trace": cover_trace,
            "residual_opponent_ids": [opponent_ids[index] for index in residual],
        }

    total_matches = int(np.count_nonzero(matches))
    report = {
        "schema": "kaggriculture.counter-cluster-round2-semantic-evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "interpretation": (
            "train response clustering" if args.split == "train"
            else "paired validation comparison; not untouched confirmation"
        ),
        "execution_semantics": (
            "fixed Opening A through step 143; empty rank sequence; exact semantic "
            "PlanDelta intent match at each checkpoint; live Candidate8 action planning"
        ),
        "inputs": {
            "config": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
            "input_lock": {"path": str(args.input_lock.resolve()), "sha256": sha256_file(args.input_lock)},
            "panel": {"path": str(args.panel.resolve()), "sha256": sha256_file(args.panel)},
            "opening_actions": {
                "path": str(args.opening_actions.resolve()), "sha256": sha256_file(args.opening_actions)
            },
            "semantic_fists": {
                "path": str(args.semantic_fists.resolve()), "sha256": sha256_file(args.semantic_fists)
            },
            "runtime_lock": {
                "path": str(args.runtime_lock.resolve()), "sha256": sha256_file(args.runtime_lock)
            },
            "portfolio": None if args.portfolio is None else {
                "path": str(args.portfolio.resolve()), "sha256": sha256_file(args.portfolio)
            },
            "split_assets": asset_receipt,
            "opponent_replays": [
                {
                    "panel_id": str(row["panel_id"]),
                    "replay_path": str(Path(str(row["replay_path"])).resolve()),
                    "action_sha256": str(row["action_sha256"]).lower(),
                }
                for row in panel_rows
            ],
        },
        "dimensions": {
            "fists": len(fists),
            "opponents": len(opponent_ids),
            "seeds": len(seeds),
            "seats": 2,
            "decision_stages": stages,
            "games": int((len(fists) + 1) * len(opponent_ids) * len(seeds) * 2),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "games_per_second": len(tasks) / elapsed if elapsed else None,
            "workers": args.workers,
            "native_extension": {
                "path": str(actual_native_path), "sha256": locked_native_hash,
            },
            "native_source_sha256": runtime_payload["source_sha256"],
            "rank_inputs": [],
            "capture_trace": False,
        },
        "coverage_rule": rule,
        "baseline": {
            "route_id": opening_id,
            "action_sha256": config["opening"]["action_sha256"],
            "score_rate": float(np.mean(baseline_scores)),
            "raw_win_rate": float(np.mean(baseline_margins > 0)),
            "mean_margin": float(np.mean(baseline_margins)),
            "path_score": _summary(baseline_path_score),
        },
        "semantic_execution": {
            "stage_match_rate": float(np.mean(matches)),
            "all_stages_match_game_rate": float(np.mean(matches.all(axis=-1))),
            "match_rate_by_decision_day": {
                str(day): float(np.mean(matches[..., stage]))
                for stage, day in enumerate(days)
            },
            "live_normalization_changed_rate_given_match": (
                float(np.count_nonzero(normalization_changed) / total_matches)
                if total_matches else None
            ),
            "signature_changed_rate_given_match": (
                float(np.count_nonzero(signature_changed) / total_matches)
                if total_matches else None
            ),
            "outcome_covered_opponent_count": int(np.count_nonzero(covered_union)),
            "strict_full_match_covered_opponent_count": int(
                np.count_nonzero(np.any(strict_full_match_coverage, axis=0))
            ),
        },
        "fists": fist_rows,
        "opponents": opponent_rows,
        "set_cover": set_cover,
        "response_clusters": clusters,
        "frozen_portfolio_evaluation": None if portfolio_payload is None else {
            "selected_fist_ids": fist_ids,
            "covered_opponent_ids": [
                opponent_ids[index] for index in np.flatnonzero(covered_union)
            ],
            "residual_opponent_ids": [
                opponent_ids[index] for index in np.flatnonzero(~covered_union)
            ],
        },
        "matrix": {
            "path": str(args.matrix.resolve()),
            "sha256": matrix_hash,
            "axes": {
                "rewards": ["fist", "opponent", "seed", "candidate_seat", "physical_seat"],
                "selected_matched_and_actual_selection": [
                    "fist", "opponent", "seed", "candidate_seat", "decision_stage"
                ],
                "path_metrics_and_coverage": ["fist", "opponent"],
            },
        },
        "boundaries": {
            "opponent_actions_loaded_only_from_requested_split": True,
            "combined_round1_opponent_pack_loaded": False,
            "holdout_entry_point_refused": True,
            "fixed_action_tape_used": False,
            "selected_rank_input_used": False,
            "actual_delta_and_signature_telemetry_recorded": True,
            "set_cover_fit_on_train_only": args.split == "train",
            "validation_is_paired_not_confirmatory": args.split == "validation",
        },
        "portfolio_output": None if args.portfolio_out is None else str(args.portfolio_out.resolve()),
    }
    atomic_json(args.report, report)
    report_hash = sha256_file(args.report)

    if args.split == "train":
        portfolio = {
            "schema": PORTFOLIO_SCHEMA,
            "frozen": True,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_split": "train",
            "config_sha256": sha256_file(args.config),
            "semantic_fists_sha256": sha256_file(args.semantic_fists),
            "coverage_rule": rule,
            "train_matrix": {"path": str(args.matrix.resolve()), "sha256": matrix_hash},
            "train_report": {"path": str(args.report.resolve()), "sha256": report_hash},
            "selected_fist_ids": set_cover["selected_fist_ids"],
            "selected_count": set_cover["selected_count"],
            "covered_opponent_count": set_cover["covered_opponent_count"],
            "residual_opponent_ids": set_cover["residual_opponent_ids"],
        }
        atomic_json(args.portfolio_out, portfolio)

    print(json.dumps({
        "split": args.split,
        "matrix": str(args.matrix.resolve()),
        "report": str(args.report.resolve()),
        "games": report["dimensions"]["games"],
        "stage_match_rate": report["semantic_execution"]["stage_match_rate"],
        "selected_fist_ids": [] if set_cover is None else set_cover["selected_fist_ids"],
        "residual_opponent_ids": [] if set_cover is None else set_cover["residual_opponent_ids"],
    }, ensure_ascii=False, indent=2))
    del dll_handles
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

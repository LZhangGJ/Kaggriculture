#!/usr/bin/env python3
"""Materialize and compare full/compact direct H1 U_phase ExtraTrees features.

The runner is train-only.  It verifies the frozen eight-seed label signal and
candidate SHA order before allocating feature files.  Candidate features are
replayed from the frozen NR020 trajectory; neither candidate actions nor a
learned policy are committed.  ExtraTrees evaluation is Python-only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_contract_breadth_sweep_v2 as breadth
import run_contract_retrieval_oracle_ablation_v1 as retrieval_v1
import run_farmer_augment_union_mpc_v1 as farmer_mpc
import run_multi_farmer_path_residual_v1 as path_v1
import run_phase_breadth_h1_offsets_v1 as h1
import run_phase_challenger_8seed_labels_v1 as labels_v1
import run_phase_challenger_dataset_preflight_v1 as preflight
import run_route_residual_adapter_v0 as residual


SCHEMA = "phase-challenger-direct-u-phase-compact-et-ab-v1"
FULL_WIDTH = 2844
COMPACT_WIDTH = 1078
ROUTE_WIDTH = 147
STATE_ROUTE_WIDTH = 129
PLAN_SUMMARY_WIDTH = 18
LAYOUT_START, LAYOUT_STOP = 147, 355
ACTOR_START, ACTOR_STOP = 355, 643
TRACE_START, TRACE_STOP = 643, 2755
META_START, META_STOP = 2755, 2844
TRACE_SHAPE = (4, 16, 33)
FINAL_TREES = 96
CV_TREES = 24
DEFAULT_RANDOM_SEED = 20260829
DEFAULT_PARENT = Path(
    r"D:\Kaggriculture\cpp_route_experiments_20260827"
    r"\multi_farmer_path_residual_v1"
)
DEFAULT_LABEL_ROOT = DEFAULT_PARENT / "phase_challenger_8seed_labels_v1_formal_20260829a"
DEFAULT_SMOKE_LABEL_ROOT = DEFAULT_PARENT / "phase_challenger_8seed_labels_v1_smoke_20260829b"
DEFAULT_OUTPUT = DEFAULT_PARENT / "phase_challenger_compact_et_ab_v1"


def compact_feature_names() -> tuple[str, ...]:
    full = tuple(path_v1.FEATURE_NAMES)
    trace_components = tuple(full[TRACE_START: TRACE_START + TRACE_SHAPE[-1]])
    names = [*full[:STATE_ROUTE_WIDTH], *full[LAYOUT_START:LAYOUT_STOP]]
    names.extend(full[ACTOR_START:ACTOR_STOP])
    names.extend(full[STATE_ROUTE_WIDTH:ROUTE_WIDTH])
    for offset in range(TRACE_SHAPE[0]):
        for statistic in ("mean", "max"):
            names.extend(
                f"keep_plan_{offset}_{statistic}_{name.split('_actor_0_', 1)[-1]}"
                for name in trace_components
            )
    for statistic in ("mean", "max"):
        names.extend(
            f"candidate_h1_{statistic}_{name.split('_actor_0_', 1)[-1]}"
            for name in trace_components
        )
    names.extend(f"candidate_h1_changed_actor_{actor}" for actor in range(16))
    names.extend(full[META_START:META_STOP])
    if len(names) != COMPACT_WIDTH:
        raise AssertionError(f"compact feature width changed: {len(names)}")
    return tuple(names)


COMPACT_FEATURE_NAMES = compact_feature_names()


def _assert_shared(name: str, values: np.ndarray) -> None:
    if len(values) and not np.array_equal(values, np.broadcast_to(values[:1], values.shape)):
        raise AssertionError(f"H1 decision lost shared {name}")


def compact_feature_rows(full_rows: np.ndarray) -> np.ndarray:
    """Deterministically map one complete H1 decision from 2844D to 1078D."""

    rows = np.asarray(full_rows)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != FULL_WIDTH:
        raise ValueError(f"full H1 panel must have shape (K,{FULL_WIDTH})")
    if not np.isfinite(rows).all():
        raise ValueError("full H1 panel contains non-finite values")
    trace = rows[:, TRACE_START:TRACE_STOP].reshape((len(rows), *TRACE_SHAPE))
    state = np.concatenate((
        rows[:, :STATE_ROUTE_WIDTH], rows[:, LAYOUT_START:LAYOUT_STOP],
    ), axis=1)
    plan_summary = rows[:, STATE_ROUTE_WIDTH:ROUTE_WIDTH]
    actors = rows[:, ACTOR_START:ACTOR_STOP]
    _assert_shared("state", state)
    _assert_shared("actor slots", actors)
    _assert_shared("plan summary", plan_summary)
    _assert_shared("offsets 1..3 trace", trace[:, 1:])

    keep_trace = trace[0]
    keep_pooled = np.concatenate((
        keep_trace.mean(axis=1), keep_trace.max(axis=1),
    ), axis=1).reshape(-1)
    h1_pooled = np.concatenate((
        trace[:, 0].mean(axis=1), trace[:, 0].max(axis=1),
    ), axis=1)
    compact = np.concatenate((
        state,
        actors,
        plan_summary,
        np.broadcast_to(keep_pooled, (len(rows), len(keep_pooled))),
        h1_pooled,
        trace[:, 0, :, -1],
        rows[:, META_START:META_STOP],
    ), axis=1).astype(np.float32, copy=False)
    if compact.shape != (len(rows), COMPACT_WIDTH) or not np.isfinite(compact).all():
        raise AssertionError("invalid compact H1 feature panel")
    return compact


def canonical_candidates(
    r0: Sequence[path_v1.PathCandidate],
    phase: Sequence[path_v1.PathCandidate],
) -> list[path_v1.PathCandidate]:
    """R0 prefix followed by phase-only SHA suffix."""

    if not r0 or not phase or not r0[0].keep or not phase[0].keep:
        raise ValueError("canonical U_phase requires KEEP-first non-empty arms")
    r0_sha = {candidate.raw_sha256 for candidate in r0}
    union = [*r0, *(candidate for candidate in phase if candidate.raw_sha256 not in r0_sha)]
    shas = [candidate.raw_sha256 for candidate in union]
    if len(shas) != len(set(shas)) or shas[:len(r0)] != [c.raw_sha256 for c in r0]:
        raise AssertionError("canonical U_phase SHA ordering failed")
    return union


def iter_label_groups(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    current: str | None = None
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            state_id = str(row["state_id"])
            if current is None:
                current = state_id
            if state_id != current:
                if current in seen:
                    raise ValueError("canonical labels are not contiguous by decision")
                seen.add(current)
                yield current, rows
                current, rows = state_id, []
            if int(row["union_index"]) != len(rows):
                raise ValueError(f"label union order changed at line {line_number}")
            rows.append(row)
    if current is not None:
        if current in seen:
            raise ValueError("canonical labels repeat a completed decision")
        yield current, rows


def iter_panels(
    decisions_path: Path,
    labels_path: Path,
    max_decisions: int | None = None,
) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Stream complete decisions; a max limit is applied only between panels."""

    if max_decisions is not None and max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    groups = iter(iter_label_groups(labels_path))
    count = 0
    with decisions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_decisions is not None and count >= max_decisions:
                break
            decision = json.loads(line)
            try:
                state_id, labels = next(groups)
            except StopIteration as exc:
                raise ValueError("labels ended before decisions") from exc
            state = preflight.canonical_union_index(decision)
            expected = list(state["union_sha256"])
            observed = [str(row["candidate_sha256"]) for row in labels]
            if state_id != str(decision["state_id"]) or observed != expected:
                raise ValueError("decision and canonical label SHA/order do not align")
            for index, row in enumerate(labels):
                if (
                    int(row["union_index"]) != index
                    or str(row["state_id"]) != state_id
                    or str(row["split"]) != "train_proxy"
                    or int(row["market_diff"]) != int(state["market_by_sha"][expected[index]])
                ):
                    raise ValueError("canonical label metadata changed")
            count += 1
            yield decision, labels
    if max_decisions is None:
        try:
            extra_state, _ = next(groups)
        except StopIteration:
            pass
        else:
            raise ValueError(f"labels contain an extra decision: {extra_state}")


def _sha256_file(path: Path) -> str:
    return residual._sha256_file(path)


def load_input_contract(label_root: Path, smoke: bool) -> dict[str, Any]:
    root = label_root.resolve()
    report_path = root / ("SMOKE_REPORT.json" if smoke else "FINAL_REPORT.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != labels_v1.SCHEMA:
        raise ValueError("label report schema changed")
    scenario = report.get("scenario_contract", {})
    evidence = report.get("evidence_boundary", {})
    if (
        scenario.get("heldout_opponents_opened")
        or scenario.get("validation_seeds_opened")
        or scenario.get("sealed_seeds_opened")
        or evidence.get("heldout_opened")
        or evidence.get("validation_opened")
        or evidence.get("sealed_opened")
        or not evidence.get("train_only", False)
    ):
        raise ValueError("label artifact crossed the train-only evidence boundary")
    paths: dict[str, Path] = {"report": report_path}
    for name in ("h1_decisions.jsonl", "canonical_union_labels.jsonl"):
        artifact = report.get("artifacts", {}).get(name, {})
        path = Path(str(artifact.get("path", ""))).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError(f"label artifact escaped its frozen root: {name}")
        if _sha256_file(path) != str(artifact.get("sha256", "")):
            raise ValueError(f"label artifact SHA changed: {name}")
        paths[name] = path
    return {
        "root": root,
        "report": report,
        "paths": paths,
        "signal_passed": bool(report.get("phase_only_signal", {}).get("passed", False)),
    }


def scan_panels(
    decisions_path: Path,
    labels_path: Path,
    max_decisions: int | None,
) -> dict[str, Any]:
    rows = decisions = 0
    scenarios: list[tuple[str, int, int]] = []
    seen_scenarios: set[tuple[str, int, int]] = set()
    digest = hashlib.sha256()
    previous_scenario: tuple[str, int, int] | None = None
    previous_step = -1
    for decision, labels in iter_panels(decisions_path, labels_path, max_decisions):
        scenario = (
            str(decision["opponent"]), int(decision["seed"]), int(decision["seat"]),
        )
        step = int(decision["decision_step"])
        if scenario != previous_scenario:
            if scenario in seen_scenarios:
                raise ValueError("scenario decisions are not contiguous")
            seen_scenarios.add(scenario)
            scenarios.append(scenario)
            previous_scenario, previous_step = scenario, -1
        if step <= previous_step:
            raise ValueError("decision steps are not strictly increasing within scenario")
        previous_step = step
        digest.update(str(decision["state_id"]).encode("utf-8"))
        for label in labels:
            digest.update(str(label["candidate_sha256"]).encode("ascii"))
        decisions += 1
        rows += len(labels)
    if not decisions or not rows:
        raise ValueError("selected train panel is empty")
    return {
        "decisions": decisions,
        "rows": rows,
        "scenarios": scenarios,
        "candidate_order_sha256": digest.hexdigest(),
    }


def _metadata_arrays(rows: int) -> dict[str, np.ndarray]:
    return {
        "target_signed_log_margin": np.empty(rows, np.float32),
        "delta_margin": np.empty(rows, np.float64),
        "seed": np.empty(rows, np.int64),
        "seat": np.empty(rows, np.int8),
        "opponent": np.empty(rows, np.int16),
        "step": np.empty(rows, np.int16),
        "edit": np.empty(rows, np.int8),
        "decision": np.empty(rows, np.int32),
        "selected_by_oracle": np.empty(rows, np.bool_),
        "union_index": np.empty(rows, np.int16),
        "path_pure": np.empty(rows, np.bool_),
        "phase_only": np.empty(rows, np.bool_),
    }


def _group_panels(
    decisions_path: Path,
    labels_path: Path,
    max_decisions: int | None,
) -> Iterator[tuple[tuple[str, int, int], list[tuple[dict[str, Any], list[dict[str, Any]]]]]]:
    panels = iter_panels(decisions_path, labels_path, max_decisions)
    key = lambda pair: (
        str(pair[0]["opponent"]), int(pair[0]["seed"]), int(pair[0]["seat"]),
    )
    for scenario, group in itertools.groupby(panels, key=key):
        yield scenario, list(group)


def _replay_scenario(
    bundle: Any,
    baseline_id: str,
    tapes: Mapping[str, Sequence[Mapping[str, Any]]],
    carriers: Mapping[str, Any],
    genome: Sequence[str],
    strict: retrieval_v1.StrictLibrary,
    phase_mapping: Mapping[int, str],
    scenario: tuple[str, int, int],
    panels: Sequence[tuple[dict[str, Any], list[dict[str, Any]]]],
    full_store: np.ndarray,
    compact_store: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    row_at: int,
    decision_at: int,
) -> tuple[int, int, dict[str, int]]:
    opponent, seed, seat = scenario
    env, states, history = residual.prefix_with_history(
        bundle, baseline_id, opponent, seed, seat,
    )
    schedule = residual._full_schedule(bundle, genome, opponent, seat)
    plan_tape = residual.scheduled_tape(tapes[baseline_id], tapes, genome)
    by_step = {int(decision["decision_step"]): (decision, labels) for decision, labels in panels}
    if len(by_step) != len(panels):
        raise ValueError("scenario contains duplicate decision steps")
    anchors = {int(decision["anchor"]) for decision, _ in panels}
    pending: dict[int, dict[str, Any]] = {}
    audits = {"decisions": 0, "rows": 0, "shared_invariant_checks": 0, "sha_order_checks": 0}
    for step in range(path_v1.ANCHORS[0], path_v1.HORIZON):
        observation = dict(env.observation(seat))
        observation.update(step=step, player=seat)
        history.update(observation)
        segment = int(np.searchsorted(path_v1.STOPS, step, side="right"))
        route_id = str(genome[segment])
        if step in anchors:
            contract, layout, mask = retrieval_v1.snapshot_contract(
                observation, history, plan_tape,
            )
            indices = strict.prototype_indices[step]
            prototype_ids = list(map(str, strict.prototypes["block_ids"][indices]))
            distances = retrieval_v1.contract_distances(
                contract, layout, mask,
                strict.prototypes["entry_contracts"][indices],
                strict.prototypes["layouts"][indices],
                strict.prototypes["unlocked_masks"][indices],
                strict.scales[step],
            )
            prefixes, inspected = breadth.nearest_distinct_prefixes(
                prototype_ids, distances, breadth.KS,
            )
            pending[step] = {
                "query_sha256": h1._sha256_payload({
                    "contract": contract.tolist(), "layout": layout.tolist(),
                    "mask": mask.tolist(),
                }),
                "prefixes": prefixes,
                "raw_prototypes_inspected": inspected,
            }
        if step in by_step:
            decision, label_rows = by_step[step]
            anchor = int(decision["anchor"])
            if (
                str(decision["route_id"]) != route_id
                or int(decision["actor_count"]) != max(1, len(path_v1._positions(observation)))
                or str(decision["query_sha256"]) != str(pending[anchor]["query_sha256"])
            ):
                raise RuntimeError("frozen decision context changed on replay")
            actor_count = int(decision["actor_count"])
            tape = tapes[route_id]
            cache = breadth.build_variant_cache(
                tape, strict.by_anchor[anchor], anchor, actor_count, step,
            )
            masks = h1.h1_payload_masks(cache)
            r0_pool = tuple(donor.block_id for donor in strict.active_by_anchor[anchor])
            phase_arm = str(phase_mapping[anchor])
            phase_pool = h1._phase_pool_ids(phase_arm, anchor, pending[anchor], strict)
            r0, _ = h1.build_h1_diverse_arm(cache, r0_pool, masks)
            phase = r0 if phase_arm == "R0_active8" else h1.build_h1_diverse_arm(
                cache, phase_pool, masks,
            )[0]
            union = canonical_candidates(r0, phase)
            union_sha = [candidate.raw_sha256 for candidate in union]
            if (
                phase_arm != str(decision["phase_pool_arm"])
                or list(phase_pool) != list(decision["phase_pool_block_ids"])
                or [candidate.raw_sha256 for candidate in r0]
                != list(decision["arms"]["R0_active8"]["candidate_sha256"])
                or [candidate.raw_sha256 for candidate in phase]
                != list(decision["arms"]["phase_frozen"]["candidate_sha256"])
                or union_sha != [str(row["candidate_sha256"]) for row in label_rows]
            ):
                raise RuntimeError("U_phase regeneration changed SHA/order")
            source = path_v1._source_positions(carriers.get(route_id), step)
            full = path_v1.candidate_feature_rows(
                observation, history, plan_tape, tape, source, union, seat, step,
            )
            compact = compact_feature_rows(full)
            stop = row_at + len(union)
            full_store[row_at:stop] = full
            compact_store[row_at:stop] = compact
            keep_margin = float(label_rows[0]["margin"])
            best = max(
                range(len(label_rows)),
                key=lambda index: (
                    int(label_rows[index]["outcome"]), float(label_rows[index]["margin"]),
                ),
            )
            opponent_index = labels_v1.TRAIN_OPPONENTS.index(opponent)
            for local, (candidate, label) in enumerate(zip(union, label_rows, strict=True)):
                at = row_at + local
                delta = float(label["margin"]) - keep_margin
                metadata["target_signed_log_margin"][at] = residual._signed_log(delta)
                metadata["delta_margin"][at] = delta
                metadata["seed"][at] = seed
                metadata["seat"][at] = seat
                metadata["opponent"][at] = opponent_index
                metadata["step"][at] = step
                metadata["edit"][at] = path_v1.KIND_INDEX[candidate.kind]
                metadata["decision"][at] = decision_at
                metadata["selected_by_oracle"][at] = local == best
                metadata["union_index"][at] = local
                metadata["path_pure"][at] = bool(label["path_pure"])
                metadata["phase_only"][at] = str(label["membership"]) == "phase_only"
            row_at, decision_at = stop, decision_at + 1
            audits["decisions"] += 1
            audits["rows"] += len(union)
            audits["shared_invariant_checks"] += 4
            audits["sha_order_checks"] += 1
        route_pair = schedule[segment]
        env.step([
            bundle.executor.action_at(env, player, int(route_pair[player]), states[player])
            for player in (0, 1)
        ])
    if not env.done or int(env.step_count) != path_v1.HORIZON:
        raise RuntimeError("frozen feature replay trajectory did not complete")
    return row_at, decision_at, audits


def materialize_features(
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
    scan: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    breadth_report, lopo, phase_mapping, frozen_inputs = h1.load_frozen_phase_mapping(
        args.breadth_root, args.lopo_report,
    )
    expected_mapping = {
        int(anchor): str(arm)
        for anchor, arm in inputs["report"]["frozen_phase_mapping"].items()
    }
    if phase_mapping != expected_mapping:
        raise ValueError("phase mapping differs from frozen label report")
    load_args = labels_v1.experiment_load_args(args)
    (bundle, baseline_id, _, tapes, genomes, carriers, active_donors, experiment_inputs,
     source) = path_v1.load_experiment(load_args)
    strict = retrieval_v1.load_strict_library(args.block_library)
    genome = genomes["NR020"]
    for anchor in path_v1.ANCHORS:
        if {d.block_id for d in strict.active_by_anchor[anchor]} != {
            d.block_id for d in active_donors[anchor]
        }:
            raise ValueError(f"R0 active8 mismatch at {anchor}")

    rows = int(scan["rows"])
    full_path = output / "full_2844.npy"
    compact_path = output / "compact_1078.npy"
    full_store = np.lib.format.open_memmap(
        full_path, mode="w+", dtype=np.float32, shape=(rows, FULL_WIDTH),
    )
    compact_store = np.lib.format.open_memmap(
        compact_path, mode="w+", dtype=np.float32, shape=(rows, COMPACT_WIDTH),
    )
    metadata = _metadata_arrays(rows)
    row_at = decision_at = 0
    totals = {"decisions": 0, "rows": 0, "shared_invariant_checks": 0, "sha_order_checks": 0}
    for scenario, panels in _group_panels(
        inputs["paths"]["h1_decisions.jsonl"],
        inputs["paths"]["canonical_union_labels.jsonl"],
        args.max_decisions,
    ):
        row_at, decision_at, audit = _replay_scenario(
            bundle, baseline_id, tapes, carriers, genome, strict, phase_mapping,
            scenario, panels, full_store, compact_store, metadata, row_at, decision_at,
        )
        for key in totals:
            totals[key] += int(audit[key])
    if row_at != rows or decision_at != int(scan["decisions"]):
        raise RuntimeError("feature replay row preallocation changed")
    full_store.flush()
    compact_store.flush()
    del full_store, compact_store
    metadata_path = output / "metadata.npz"
    farmer_mpc._atomic_npz(metadata_path, metadata)
    return {
        "full_path": full_path,
        "compact_path": compact_path,
        "metadata_path": metadata_path,
        "audit": totals,
        "frozen_inputs": frozen_inputs,
        "breadth_status": breadth_report["status"],
        "lopo_status": lopo["status"],
        "experiment_inputs": experiment_inputs,
        "source": source,
    }


def _train_one(
    name: str,
    feature_path: Path,
    metadata_path: Path,
    output: Path,
    trees: int,
    cv_trees: int,
    random_seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    features = np.load(feature_path, mmap_mode="r")
    with np.load(metadata_path, allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in raw.files}
    arrays["features"] = features
    model, calibration, oof, choices = farmer_mpc.fit_a_backbone(
        arrays, trees, cv_trees, random_seed,
    )
    farmer_mpc._atomic_joblib(output / f"{name}_et.joblib", model)
    farmer_mpc._atomic_npz(output / f"{name}_oof.npz", {
        **oof, "choices": choices,
    })
    result = {
        "representation": name,
        "feature_width": int(features.shape[1]),
        "rows": int(features.shape[0]),
        "calibration": calibration,
        "python_only": True,
        "model_artifact": retrieval_v1._artifact(output / f"{name}_et.joblib"),
        "oof_artifact": retrieval_v1._artifact(output / f"{name}_oof.npz"),
    }
    del arrays, features, model, oof
    gc.collect()
    return result, choices


def direct_ab_gate(
    metadata: Mapping[str, np.ndarray],
    full_choices: np.ndarray,
    compact_choices: np.ndarray,
    compact_zero_harm: bool,
) -> dict[str, Any]:
    """Pre-registered direct-selector non-inferiority gate, all versus KEEP."""

    slices = farmer_mpc._decision_slices(np.asarray(metadata["decision"]))
    if len(slices) != len(full_choices) or len(slices) != len(compact_choices):
        raise ValueError("OOF choices do not align with complete decisions")
    full_delta: list[float] = []
    compact_delta: list[float] = []
    opponents: list[int] = []
    compact_fires: list[bool] = []
    fire_seeds: set[int] = set()
    fire_opponents: set[int] = set()
    keep_kind = path_v1.KIND_INDEX["KEEP"]
    for indices, full_choice, compact_choice in zip(
        slices, full_choices, compact_choices, strict=True,
    ):
        full_at, compact_at = int(full_choice), int(compact_choice)
        if not np.any(indices == full_at) or not np.any(indices == compact_at):
            raise ValueError("OOF choice escaped its decision panel")
        full_opponent = int(metadata["opponent"][full_at])
        compact_opponent = int(metadata["opponent"][compact_at])
        if full_opponent != compact_opponent:
            raise ValueError("full/compact OOF decisions lost opponent parity")
        full_delta.append(float(metadata["delta_margin"][full_at]))
        compact_delta.append(float(metadata["delta_margin"][compact_at]))
        opponents.append(full_opponent)
        fired = int(metadata["edit"][compact_at]) != keep_kind
        compact_fires.append(fired)
        if fired:
            fire_opponents.add(compact_opponent)
            fire_seeds.add(int(metadata["seed"][compact_at]))
    full_values = np.asarray(full_delta, np.float64)
    compact_values = np.asarray(compact_delta, np.float64)
    opponent_rows = {}
    for opponent in sorted(set(opponents)):
        mask = np.asarray(opponents) == opponent
        name = (
            labels_v1.TRAIN_OPPONENTS[opponent]
            if 0 <= opponent < len(labels_v1.TRAIN_OPPONENTS) else str(opponent)
        )
        full_sum = float(full_values[mask].sum())
        compact_sum = float(compact_values[mask].sum())
        opponent_rows[name] = {
            "full_sum_realized_delta_vs_KEEP": full_sum,
            "compact_sum_realized_delta_vs_KEEP": compact_sum,
            "compact_minus_full": compact_sum - full_sum,
            "noninferior": compact_sum + 1e-12 >= full_sum,
        }
    full_sum = float(full_values.sum())
    compact_sum = float(compact_values.sum())
    conditions = {
        "zero_harm_each_lopo_fold_vs_KEEP": bool(compact_zero_harm),
        "overall_realized_delta_retention_at_least_0.98": (
            compact_sum + 1e-12 >= 0.98 * max(0.0, full_sum)
        ),
        "compact_minus_full_nonnegative_each_opponent": all(
            row["noninferior"] for row in opponent_rows.values()
        ),
        "compact_fires_at_least_4": sum(compact_fires) >= 4,
        "compact_fires_cover_at_least_2_opponents": len(fire_opponents) >= 2,
        "compact_fires_cover_at_least_2_seeds": len(fire_seeds) >= 2,
    }
    return {
        "gate_incomplete": True,
        "missing_primary_evidence": (
            "four-fold leave-seed-fold-out predictions with each representation's "
            "LOPO-frozen threshold applied without reselection"
        ),
        "zero_harm_reference": "KEEP",
        "not_R0_residual": True,
        "realized_delta_retention_floor": 0.98,
        "full_sum_realized_delta_vs_KEEP": full_sum,
        "compact_sum_realized_delta_vs_KEEP": compact_sum,
        "compact_fires": int(sum(compact_fires)),
        "compact_fire_opponents": sorted(
            labels_v1.TRAIN_OPPONENTS[value] for value in fire_opponents
        ),
        "compact_fire_seeds": sorted(fire_seeds),
        "by_opponent": opponent_rows,
        "lopo_diagnostic_conditions": conditions,
        "passed": False,
    }


def train_ab(args: argparse.Namespace, materialized: Mapping[str, Any], output: Path) -> dict[str, Any]:
    if args.trees != FINAL_TREES or args.cv_trees != CV_TREES:
        raise ValueError("formal A/B is frozen to identical 96/24 tree counts")
    full, full_choices = _train_one(
        "full2844", materialized["full_path"], materialized["metadata_path"],
        output, args.trees, args.cv_trees, args.random_seed,
    )
    compact, compact_choices = _train_one(
        "compact1078", materialized["compact_path"], materialized["metadata_path"],
        output, args.trees, args.cv_trees, args.random_seed,
    )
    compact_zero_harm = bool(compact["calibration"]["zero_harm_each_lopo_fold"])
    with np.load(materialized["metadata_path"], allow_pickle=False) as raw:
        gate = direct_ab_gate(
            {key: raw[key] for key in raw.files},
            full_choices, compact_choices, compact_zero_harm,
        )
    return {
        "full2844": full,
        "compact1078": compact,
        "choice_agreement": float(np.mean(full_choices == compact_choices)),
        "pre_registered_compact_gate": gate,
    }


def _write_report(output: Path, report: Mapping[str, Any], smoke: bool) -> None:
    stem = "SMOKE_REPORT" if smoke else "FINAL_REPORT"
    residual._atomic_text(
        output / f"{stem}.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    panel = report.get("panel", {})
    audit = report.get("materialization_audit", {})
    residual._atomic_text(
        output / f"{stem}.md",
        "\n".join((
            "# PHASE-CHALLENGER-COMPACT-ET-AB-v1",
            "",
            f"Status: `{report['status']}`",
            "",
            f"Decisions/rows: {panel.get('decisions', 0)} / {panel.get('rows', 0)}.",
            f"Feature allocation/training: {report.get('feature_allocation_executed', False)} / "
            f"{report.get('training_executed', False)}.",
            f"SHA-order/shared checks: {audit.get('sha_order_checks', 0)} / "
            f"{audit.get('shared_invariant_checks', 0)}.",
            "Evidence boundary: train only; validation and sealed splits unopened.",
            "",
            "This is a direct U_phase candidate selector relative to KEEP, not an R0 residual gate.",
            "The 1078D representation is derived from the existing 2844D H1 panel. ",
            "All candidate SHA/order and shared H1 context invariants are asserted.",
            "ExtraTrees evaluation is Python-only; no C++ runtime was changed.",
            "",
        )),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.smoke and args.train:
        raise ValueError("smoke materializes features but never trains a one-opponent model")
    if not args.smoke and args.max_decisions is not None:
        raise ValueError("formal decisions may not be truncated")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = load_input_contract(args.label_root, args.smoke)
    base_report: dict[str, Any] = {
        "schema": SCHEMA,
        "input_label_root": str(inputs["root"]),
        "input_signal_passed": inputs["signal_passed"],
        "signal_bypassed_for_smoke": bool(args.smoke and not inputs["signal_passed"]),
        "feature_allocation_executed": False,
        "training_executed": False,
        "trees": {"final": FINAL_TREES, "lopo": CV_TREES, "random_seed": args.random_seed},
        "representations": {
            "full2844": FULL_WIDTH,
            "compact1078": COMPACT_WIDTH,
            "compact_order": [
                "state337", "actor_slots288", "plan_summary18",
                "KEEP_plan_mean_max264", "candidate_H1_mean_max66",
                "changed_actor_mask16", "candidate_meta89",
            ],
        },
        "runtime_boundary": {
            "extra_trees_evaluation": "Python only",
            "C++_modified": False,
            "validation_opened": False,
            "sealed_opened": False,
            "candidate_committed": False,
        },
        "selector_objective": {
            "kind": "direct candidate-conditioned U_phase selector A/B",
            "target": "signed-log terminal margin delta versus KEEP",
            "zero_harm_reference": "KEEP",
            "not_R0_residual": True,
            "signal_gate_role": (
                "the frozen phase-only novelty gate only decides whether U_phase "
                "is worth materializing; it does not relabel all union positives"
            ),
        },
    }
    if not args.smoke and not inputs["signal_passed"]:
        report = {
            **base_report,
            "status": "signal_insufficient_stop_before_feature_allocation",
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_report(output, report, args.smoke)
        return report

    scan = scan_panels(
        inputs["paths"]["h1_decisions.jsonl"],
        inputs["paths"]["canonical_union_labels.jsonl"],
        args.max_decisions,
    )
    materialized = materialize_features(args, inputs, scan, output)
    training = train_ab(args, materialized, output) if args.train else None
    status = "smoke_materialization_passed" if args.smoke else (
        "compact_et_ab_complete" if args.train else "formal_materialization_complete_training_not_requested"
    )
    report = {
        **base_report,
        "status": status,
        "feature_allocation_executed": True,
        "training_executed": bool(args.train),
        "panel": scan,
        "materialization_audit": materialized["audit"],
        "storage": {
            "kind": "disk-backed NumPy memmap; models are fit sequentially",
            "full_and_compact_dense_training_views_co_resident": False,
            "full2844": retrieval_v1._artifact(materialized["full_path"], scan["rows"]),
            "compact1078": retrieval_v1._artifact(materialized["compact_path"], scan["rows"]),
            "metadata": retrieval_v1._artifact(materialized["metadata_path"], scan["rows"]),
        },
        "training": training,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_report(output, report, args.smoke)
    print(json.dumps({
        "event": "phase_challenger_compact_et_ab_complete",
        "status": status, "output": str(output),
        "decisions": scan["decisions"], "rows": scan["rows"],
    }, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    result.add_argument("--frozen-run", type=Path, default=path_v1.DEFAULT_FROZEN_RUN)
    result.add_argument("--prepared-root", type=Path, action="append", default=None)
    result.add_argument("--v1-root", type=Path, default=labels_v1.continuation.routed_v2.DEFAULT_V1_ROOT)
    result.add_argument("--block-library", type=Path, default=retrieval_v1.DEFAULT_LIBRARY)
    result.add_argument("--breadth-root", type=Path, default=h1.DEFAULT_BREADTH_ROOT)
    result.add_argument("--lopo-report", type=Path, default=h1.DEFAULT_LOPO)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--trees", type=int, default=FINAL_TREES)
    result.add_argument("--cv-trees", type=int, default=CV_TREES)
    result.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    result.add_argument("--max-decisions", type=int, default=None)
    result.add_argument("--train", action="store_true")
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.smoke:
        if args.label_root == DEFAULT_LABEL_ROOT:
            args.label_root = DEFAULT_SMOKE_LABEL_ROOT
        if args.max_decisions is None:
            args.max_decisions = 4
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

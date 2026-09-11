#!/usr/bin/env python3
"""Measure adaptive unit/market layers against one fixed base route.

This is a diagnostic ablation, not an opponent router.  ``base`` is fixed for
every game before evaluation and never depends on the opponent observation or
identity.  Modes isolate where the adaptive planner loses value:

0 fully adaptive; 1 base units + adaptive market; 2 adaptive units + base
market; 3 fully base.

Modes 4 and 5 are offline causal diagnostics of the base route's public WHEAT
inventory shuttle: mode 4 removes it and mode 5 caps each order at four units.

Mode 6 executes the base route's complete step-0 action, settles the shared
market once, and then hands the resulting state to the adaptive planner.  It
tests whether the opening transaction alone explains later value loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


MODE_LABELS = {
    0: "adaptive_units__adaptive_market",
    1: "base_units__adaptive_market",
    2: "adaptive_units__base_market",
    3: "base_units__base_market",
    4: "base_units__base_market__no_wheat_shuttle",
    5: "base_units__base_market__wheat_shuttle_cap4",
    6: "base_step0__then_adaptive",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int) -> np.ndarray:
    names = list(adaptive_genome_names())
    defaults = dict(zip(names, adaptive_default_genome(), strict=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults.update({
        str(name): float(value)
        for name, value in payload.get("base_values", {}).items()
    })
    row = payload["genomes"][index]
    values = row.get("values", {})
    return np.asarray(
        [float(values.get(name, defaults[name])) for name in names],
        dtype=np.float64,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--base", default="G001")
    parser.add_argument("--opponent", default="G001")
    parser.add_argument("--modes", default="0,1,2,3")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    genomes = genome[None, :]
    base = bundle.index(args.base)
    opponent = (
        -1 if args.opponent.upper() == "PASSIVE"
        else bundle.index(args.opponent)
    )
    modes = [int(value) for value in args.modes.split(",")]
    unknown = [value for value in modes if value not in MODE_LABELS]
    if unknown:
        raise ValueError(f"unknown modes: {unknown}")
    tasks = np.asarray([
        [0, base, opponent, seed, seat, mode]
        for mode in modes
        for seed in range(args.seed_start, args.seed_start + args.seed_count)
        for seat in (0, 1)
    ], dtype=np.int64)

    started = time.perf_counter()
    rewards_raw, diagnostics_raw = bundle.adaptive_executor.play_blend_batch(
        genomes, tasks
    )
    elapsed = time.perf_counter() - started
    rewards = np.asarray(rewards_raw, dtype=np.float64)
    diagnostics = np.asarray(diagnostics_raw, dtype=np.int32)

    ranking = []
    mode_results: dict[int, dict[str, np.ndarray]] = {}
    for mode in modes:
        indices = np.flatnonzero(tasks[:, 5] == mode)
        seats = tasks[indices, 4]
        own = rewards[indices, seats]
        other = rewards[indices, 1 - seats]
        margin = own - other
        diag = diagnostics[indices]
        mode_results[mode] = {
            "indices": indices,
            "seats": seats,
            "own": own,
            "other": other,
            "margin": margin,
        }
        ranking.append({
            "mode": mode,
            "label": MODE_LABELS[mode],
            "games": int(len(indices)),
            "mean_reward": float(own.mean()),
            "p10_reward": float(np.quantile(own, 0.10)),
            "mean_opponent_reward": float(other.mean()),
            "score_rate": float(np.mean((margin > 0) + 0.5 * (margin == 0))),
            "mean_margin": float(margin.mean()),
            "avoidable_crop_losses": int(diag[:, 2].sum()),
            "avoidable_animal_losses": int(diag[:, 3].sum()),
            "end_overflow": int(diag[:, 4].sum()),
        })
    ranking.sort(key=lambda row: (-row["mean_reward"], -row["score_rate"]))

    paired_comparisons = []
    paired_states = []
    if 0 in mode_results:
        baseline = mode_results[0]
        for mode in modes:
            if mode == 0:
                continue
            current = mode_results[mode]
            own_delta = current["own"] - baseline["own"]
            opponent_delta = current["other"] - baseline["other"]
            margin_delta = current["margin"] - baseline["margin"]
            paired_comparisons.append({
                "mode": mode,
                "label": MODE_LABELS[mode],
                "games": int(len(own_delta)),
                "mean_own_cash_delta": float(own_delta.mean()),
                "median_own_cash_delta": float(np.median(own_delta)),
                "p10_own_cash_delta": float(np.quantile(own_delta, 0.10)),
                "p90_own_cash_delta": float(np.quantile(own_delta, 0.90)),
                "own_cash_improved": int(np.count_nonzero(own_delta > 0)),
                "own_cash_tied": int(np.count_nonzero(own_delta == 0)),
                "own_cash_regressed": int(np.count_nonzero(own_delta < 0)),
                "mean_opponent_cash_delta": float(opponent_delta.mean()),
                "mean_margin_delta": float(margin_delta.mean()),
            })
            for local_index, task_index in enumerate(current["indices"]):
                paired_states.append({
                    "mode": mode,
                    "seed": int(tasks[task_index, 3]),
                    "seat": int(tasks[task_index, 4]),
                    "baseline_own_cash": float(baseline["own"][local_index]),
                    "hybrid_own_cash": float(current["own"][local_index]),
                    "own_cash_delta": float(own_delta[local_index]),
                    "baseline_opponent_cash": float(baseline["other"][local_index]),
                    "hybrid_opponent_cash": float(current["other"][local_index]),
                    "opponent_cash_delta": float(opponent_delta[local_index]),
                    "baseline_margin": float(baseline["margin"][local_index]),
                    "hybrid_margin": float(current["margin"][local_index]),
                    "margin_delta": float(margin_delta[local_index]),
                })

    payload = {
        "schema": "kaggriculture.native-adaptive-blend-probe.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if np.isfinite(rewards).all() else "FAIL",
        "purpose": "diagnose execution/scheduling versus market/economic loss",
        "identity_routing_allowed": False,
        "base_route": args.base,
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "dual_seat": True,
        "games": int(len(tasks)),
        "simulation_seconds": elapsed,
        "games_per_second": float(len(tasks) / elapsed),
        "ranking": ranking,
        "paired_comparisons_vs_mode0": paired_comparisons,
        "paired_states": paired_states,
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "backbone": None if args.backbone is None else {
                "path": str(args.backbone), "sha256": sha256(args.backbone)
            },
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "games_per_second": payload["games_per_second"],
        "ranking": ranking,
        "paired_comparisons_vs_mode0": paired_comparisons,
    }, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

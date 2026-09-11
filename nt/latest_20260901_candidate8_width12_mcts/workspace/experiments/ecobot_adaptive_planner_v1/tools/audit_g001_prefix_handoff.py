#!/usr/bin/env python3
"""Compare several exact G001 opening-prefix lengths before handing off to R6."""

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int) -> np.ndarray:
    names = list(adaptive_genome_names())
    values = dict(zip(names, adaptive_default_genome(), strict=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    values.update({str(k): float(v) for k, v in payload.get("base_values", {}).items()})
    values.update({
        str(k): float(v)
        for k, v in payload["genomes"][index].get("values", {}).items()
    })
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--prefix-steps", default="0,1,2,4,8,24")
    parser.add_argument("--opponent", default="G001")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prefixes = [int(value) for value in args.prefix_steps.split(",")]
    if not prefixes or prefixes[0] != 0 or len(prefixes) != len(set(prefixes)):
        raise ValueError("prefix list must be unique and start with 0")
    if any(value < 0 or value > 719 for value in prefixes):
        raise ValueError("prefix steps must be in [0, 719]")

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    genome = load_genome(args.genomes, args.genome_index)
    genomes = genome[None, :]
    prefix_route = bundle.index("G001")
    opponent = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    tasks = np.asarray([
        [0, prefix_route, opponent, seed, seat, prefix]
        for prefix in prefixes
        for seed in range(args.seed_start, args.seed_start + args.seed_count)
        for seat in (0, 1)
    ], dtype=np.int64)

    started = time.perf_counter()
    rewards_raw, diagnostics_raw = bundle.adaptive_executor.play_prefix_batch(
        genomes, tasks
    )
    elapsed = time.perf_counter() - started
    rewards = np.asarray(rewards_raw, dtype=np.float64)
    diagnostics = np.asarray(diagnostics_raw, dtype=np.int32)

    by_prefix: dict[int, dict[str, np.ndarray]] = {}
    ranking = []
    for prefix in prefixes:
        indices = np.flatnonzero(tasks[:, 5] == prefix)
        seats = tasks[indices, 4]
        own = rewards[indices, seats]
        other = rewards[indices, 1 - seats]
        margin = own - other
        diag = diagnostics[indices]
        by_prefix[prefix] = {
            "indices": indices,
            "seats": seats,
            "own": own,
            "other": other,
            "margin": margin,
        }
        ranking.append({
            "prefix_steps": prefix,
            "games": int(len(indices)),
            "mean_own_cash": float(own.mean()),
            "median_own_cash": float(np.median(own)),
            "p10_own_cash": float(np.quantile(own, 0.10)),
            "p90_own_cash": float(np.quantile(own, 0.90)),
            "mean_opponent_cash": float(other.mean()),
            "wins": int(np.count_nonzero(margin > 0)),
            "ties": int(np.count_nonzero(margin == 0)),
            "losses": int(np.count_nonzero(margin < 0)),
            "score_rate": float(np.mean((margin > 0) + 0.5 * (margin == 0))),
            "mean_margin": float(margin.mean()),
            "avoidable_crop_losses": int(diag[:, 0].sum()),
            "avoidable_animal_losses": int(diag[:, 1].sum()),
            "end_overflow": int(diag[:, 2].sum()),
        })

    baseline = by_prefix[0]
    comparisons = []
    states = []
    for prefix in prefixes[1:]:
        current = by_prefix[prefix]
        own_delta = current["own"] - baseline["own"]
        opponent_delta = current["other"] - baseline["other"]
        margin_delta = current["margin"] - baseline["margin"]
        comparisons.append({
            "prefix_steps": prefix,
            "games": int(len(own_delta)),
            "mean_own_cash_delta": float(own_delta.mean()),
            "median_own_cash_delta": float(np.median(own_delta)),
            "mean_opponent_cash_delta": float(opponent_delta.mean()),
            "mean_margin_delta": float(margin_delta.mean()),
            "own_cash_improved": int(np.count_nonzero(own_delta > 0)),
            "own_cash_tied": int(np.count_nonzero(own_delta == 0)),
            "own_cash_regressed": int(np.count_nonzero(own_delta < 0)),
        })
        for local_index, task_index in enumerate(current["indices"]):
            states.append({
                "prefix_steps": prefix,
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

    oracle_own = []
    oracle_other = []
    oracle_prefix = []
    for local_index in range(len(baseline["own"])):
        candidates = []
        for prefix in prefixes:
            own = float(by_prefix[prefix]["own"][local_index])
            other = float(by_prefix[prefix]["other"][local_index])
            margin = own - other
            outcome = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
            candidates.append((outcome, margin, own, prefix, other))
        selected = max(candidates)
        oracle_own.append(selected[2])
        oracle_prefix.append(selected[3])
        oracle_other.append(selected[4])
    oracle_own_array = np.asarray(oracle_own, dtype=np.float64)
    oracle_other_array = np.asarray(oracle_other, dtype=np.float64)
    oracle_margin = oracle_own_array - oracle_other_array
    hindsight_oracle = {
        "warning": "clairvoyant upper bound on the evaluated prefix arms; not deployable",
        "objective": "win_then_margin_then_own_cash",
        "mean_own_cash": float(oracle_own_array.mean()),
        "mean_opponent_cash": float(oracle_other_array.mean()),
        "mean_margin": float(oracle_margin.mean()),
        "wins": int(np.count_nonzero(oracle_margin > 0)),
        "ties": int(np.count_nonzero(oracle_margin == 0)),
        "losses": int(np.count_nonzero(oracle_margin < 0)),
        "score_rate": float(np.mean((oracle_margin > 0) + 0.5 * (oracle_margin == 0))),
        "mean_own_cash_gain_vs_r6": float(
            (oracle_own_array - baseline["own"]).mean()
        ),
        "mean_margin_gain_vs_r6": float(
            (oracle_margin - baseline["margin"]).mean()
        ),
        "selected_prefix_counts": {
            str(prefix): int(sum(value == prefix for value in oracle_prefix))
            for prefix in prefixes
        },
    }

    payload = {
        "schema": "kaggriculture.g001-prefix-r6-handoff-ablation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if np.isfinite(rewards).all() else "FAIL",
        "parameters": {
            "prefix_route": "G001",
            "opponent": args.opponent,
            "prefix_steps": prefixes,
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "dual_seat": True,
        },
        "simulation_seconds": elapsed,
        "games_per_second": float(len(tasks) / elapsed),
        "ranking": ranking,
        "paired_comparisons_vs_r6": comparisons,
        "hindsight_oracle": hindsight_oracle,
        "paired_states": states,
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
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
        "paired_comparisons_vs_r6": comparisons,
        "hindsight_oracle": hindsight_oracle,
    }, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a same-seed, same-seat, same-opponent G001/Candidate8 comparison.

The Candidate8 arm is forced to replay G001 for the first 24 transitions.  Its
exact-future sequence search starts on Day 1.  This is an offline diagnostic;
the online policy never receives future information.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


FAMILY_NAMES = (
    "KEEP",
    "SCALE",
    "STOP_DEFER",
    "REBALANCE",
    "CAPACITY",
    "MARKET",
    "LAYOUT",
    "RECOVERY",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int) -> np.ndarray:
    names = list(adaptive_genome_names())
    values = dict(zip(names, adaptive_default_genome(), strict=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    values.update(payload.get("base_values", {}))
    values.update(payload["genomes"][index].get("values", {}))
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def canonical(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def direct_frozen_game(
    bundle: NativeTeammateBundle,
    own_route: int,
    opponent_route: int,
    seed: int,
    seat: int,
) -> dict:
    if seat == 0:
        return bundle.executor.play(
            own_route, opponent_route, seed, -1, -1, -1, -1, True
        )
    return bundle.executor.play(
        opponent_route, own_route, seed, -1, -1, -1, -1, True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--opponent", default="G003")
    parser.add_argument("--seed", type=int, default=3041012)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--decision-days", type=int, nargs="+", default=[1, 3, 6, 9, 12, 18, 24]
    )
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--per-node-arms", type=int, default=64)
    parser.add_argument("--prefix-steps", type=int, default=24)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    genome = load_genome(args.genomes, args.genome_index)
    g001 = bundle.index("G001")
    opponent = bundle.index(args.opponent)

    started = time.perf_counter()
    g001_native = direct_frozen_game(
        bundle, g001, opponent, args.seed, args.seat
    )
    g001_rewards = [float(value) for value in g001_native["rewards"]]
    g001_margin = g001_rewards[args.seat] - g001_rewards[1 - args.seat]
    if g001_margin <= 0:
        raise RuntimeError(
            f"selected G001 fixture is not a win: rewards={g001_rewards}"
        )

    oracle_started = time.perf_counter()
    oracle = bundle.adaptive_executor.candidate8_sequence_oracle(
        genome,
        opponent,
        args.seed,
        args.decision_days,
        args.seat,
        args.beam_width,
        args.per_node_arms,
        False,
        True,
        (),
        (),
        g001,
        args.prefix_steps,
    )
    oracle_seconds = time.perf_counter() - oracle_started
    selected_ranks = [int(value) for value in oracle["selected_rank"]]
    if any(rank < 0 for rank in selected_ranks):
        raise RuntimeError(f"oracle returned unreplayable rank sequence: {selected_ranks}")
    committed = bundle.adaptive_executor.candidate8_committed_sequence(
        genome,
        opponent,
        args.seed,
        args.decision_days,
        selected_ranks,
        args.seat,
        False,
        g001,
        args.prefix_steps,
        True,
    )
    candidate_rewards = [float(value) for value in committed["rewards"]]
    oracle_rewards = [float(value) for value in oracle["rewards"]]
    if candidate_rewards != oracle_rewards:
        raise RuntimeError(
            f"committed sequence does not reproduce oracle: "
            f"committed={candidate_rewards} oracle={oracle_rewards}"
        )

    g001_trace = canonical(g001_native["trace"])
    candidate_trace = canonical(committed["trace"])
    prefix_exact = (
        len(g001_trace) >= args.prefix_steps
        and len(candidate_trace) >= args.prefix_steps
        and g001_trace[: args.prefix_steps] == candidate_trace[: args.prefix_steps]
    )
    if not prefix_exact:
        raise RuntimeError("G001/Candidate8 joint Day0 prefix is not exact")

    families = [
        FAMILY_NAMES[int(value)] if 0 <= int(value) < len(FAMILY_NAMES) else "UNKNOWN"
        for value in committed["selected_family"]
    ]
    common = {
        "opponent": args.opponent,
        "seed": args.seed,
        "candidate_seat": args.seat,
        "shared_prefix_route": "G001",
        "shared_prefix_steps": args.prefix_steps,
    }
    games = [
        {
            **common,
            "case": "G001_FULL_ROUTE",
            "native_rewards": g001_rewards,
            "margin": g001_margin,
            "decision_days": [],
            "selected_ranks": [],
            "selected_families": [],
            "trace": g001_trace,
        },
        {
            **common,
            "case": "CANDIDATE8_SHARED_G001_DAY0_ORACLE",
            "native_rewards": candidate_rewards,
            "margin": candidate_rewards[args.seat] - candidate_rewards[1 - args.seat],
            "decision_days": [int(value) for value in committed["decision_day"]],
            "selected_ranks": selected_ranks,
            "selected_families": families,
            "selected_signatures": [
                int(value) for value in committed["selected_signature"]
            ],
            "end_overflow": int(committed["end_overflow"]),
            "avoidable_crop_losses": int(committed["avoidable_crop_losses"]),
            "avoidable_animal_losses": int(committed["avoidable_animal_losses"]),
            "trace": candidate_trace,
        },
    ]
    payload = {
        "schema": "kaggriculture.candidate8-g001-shared-day0-traces.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": {
            "candidate8_is_hindsight_oracle": True,
            "online_claim": False,
            "comparison": (
                "identical opponent, seed, seat and first 24 joint actions; "
                "Candidate8 searches only from Day 1 onward"
            ),
        },
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
        "oracle": {
            "beam_width": args.beam_width,
            "per_node_arms": args.per_node_arms,
            "use_feasible_pool": False,
            "competitive_objective": True,
            "expanded_nodes": int(oracle["expanded_nodes"]),
            "complete_continuations": int(oracle["complete_continuations"]),
            "maximum_live_beam": int(oracle["maximum_live_beam"]),
            "seconds": oracle_seconds,
        },
        "prefix_exact": prefix_exact,
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))

    receipt = {
        "schema": "kaggriculture.candidate8-g001-shared-day0-receipt.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "trace_bundle": {"path": str(args.output), "sha256": sha256(args.output)},
        "prefix_exact": prefix_exact,
        "trace_steps": [len(game["trace"]) for game in games],
        "g001_rewards": g001_rewards,
        "g001_margin": g001_margin,
        "candidate8_rewards": candidate_rewards,
        "candidate8_margin": candidate_rewards[args.seat] - candidate_rewards[1 - args.seat],
        "candidate8_cash_gain_over_g001": candidate_rewards[args.seat] - g001_rewards[args.seat],
        "candidate8_margin_gain_over_g001": (
            candidate_rewards[args.seat] - candidate_rewards[1 - args.seat] - g001_margin
        ),
        "decision_days": args.decision_days,
        "selected_ranks": selected_ranks,
        "selected_families": families,
        "oracle": payload["oracle"],
        "total_seconds": time.perf_counter() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

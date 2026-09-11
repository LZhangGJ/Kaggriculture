#!/usr/bin/env python3
"""Export frozen O1 Candidate8 joint traces from the Linux C++ engine."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
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
    values.update(payload.get("base_values", {}))
    values.update(payload["genomes"][index].get("values", {}))
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def canonical(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def select_rows(rows: list[dict], method: str, limit: int) -> list[dict]:
    arm_key = f"{method}_arm"
    active = [row for row in rows if int(row[arm_key]) != 0]
    keep = [row for row in rows if int(row[arm_key]) == 0]
    ordered = sorted(
        active,
        key=lambda row: (
            int(row["decision_day"]),
            int(row["seat"]),
            int(row["prefix_seed"]),
        ),
    )
    selected: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for row in ordered:
        key = (int(row["decision_day"]), int(row["seat"]))
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) >= limit:
            return selected
    for row in ordered:
        if row not in selected:
            selected.append(row)
        if len(selected) >= limit:
            return selected
    for row in sorted(keep, key=lambda r: (r["decision_day"], r["seat"], r["prefix_seed"])):
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=("scenario_mean", "scenario_robust"),
        default="scenario_mean",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.selection_receipt.read_text(encoding="utf-8"))
    chosen = select_rows(receipt["rows"], args.method, args.limit)
    if not chosen:
        raise RuntimeError("selection receipt contains no rows")
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    games: list[dict[str, object]] = []
    for selected in chosen:
        opponent_name = str(selected["opponent"])
        seed = int(selected["prefix_seed"])
        seat = int(selected["seat"])
        day = int(selected["decision_day"])
        arm = int(selected[f"{args.method}_arm"])
        expected_signature = int(selected[f"{args.method}_signature"])
        native = bundle.adaptive_executor.play_candidate8(
            genome,
            bundle.index(opponent_name),
            seed,
            seat,
            arm,
            day,
            False,
            True,
        )
        actual_signature = int(native["first_candidate8_signature"])
        game = {
            "opponent": opponent_name,
            "seed": seed,
            "candidate_seat": seat,
            "decision_day": day,
            "arm": arm,
            "expected_signature": expected_signature,
            "actual_signature": actual_signature,
            "signature_exact": actual_signature == expected_signature,
            "native_rewards": [float(value) for value in native["rewards"]],
            "end_overflow": int(native["end_overflow"]),
            "trace": canonical(native["trace"]),
        }
        games.append(game)
        print(
            f"export {len(games)}/{len(chosen)} day={day} seat={seat} "
            f"arm={arm} signature={game['signature_exact']}",
            flush=True,
        )
    payload = {
        "schema": "kaggriculture.candidate8-o1-trace-bundle.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "inputs": {
            "selection_receipt": {
                "path": str(args.selection_receipt),
                "sha256": sha256(args.selection_receipt),
            },
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps({"output": str(args.output), "games": len(games)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

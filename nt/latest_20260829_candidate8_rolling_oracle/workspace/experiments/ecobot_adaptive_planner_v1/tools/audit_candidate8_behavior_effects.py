#!/usr/bin/env python3
"""Prove that every Candidate8 family changes a real 719-step trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


FAMILY_NAMES = [
    "KEEP",
    "SCHEDULE_LAYOUT",
    "CONTINUOUS_SCALE",
    "UNILATERAL",
    "MULTI_PROJECT",
    "TIMING",
    "MARKET_TRANSACTION",
    "PHASE_SUFFIX",
    "LOCAL_RECOVERY",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int) -> np.ndarray:
    names = list(adaptive_genome_names())
    defaults = dict(zip(names, adaptive_default_genome(), strict=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = dict(payload.get("base_values", {}))
    values.update(payload["genomes"][index]["values"])
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
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--opponent", default="PASSIVE")
    parser.add_argument("--minimum-decision-day", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    opponent = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        seed_key = "seed" if "seed" in data.files else "prefix_seed"
        seed = np.asarray(data[seed_key])
        seat = np.asarray(data["seat"])
        family = np.asarray(data["family"])
        features = np.asarray(data["features"])
        gain = np.asarray(data["gain_vs_keep"])
        signature = np.asarray(data["signature"])

    rows = []
    for family_id in range(1, len(FAMILY_NAMES)):
        members = np.flatnonzero(family == family_id)
        if not len(members):
            rows.append(
                {
                    "family": FAMILY_NAMES[family_id],
                    "status": "NO_FEASIBLE_EXAMPLE",
                }
            )
            continue
        # Prefer a clearly consequential arm.  This is a behavior-effect test,
        # not a claim that this arm is the best policy.
        selected = int(members[np.argmax(gain[members])])
        state = int(state_id[selected])
        state_members = np.flatnonzero(state_id == state)
        local_rank = int(np.flatnonzero(state_members == selected)[0])
        current_seed = int(seed[selected])
        current_seat = int(seat[selected])
        keep = bundle.adaptive_executor.play_candidate8(
            genome,
            opponent,
            current_seed,
            current_seat,
            0,
            args.minimum_decision_day,
            True,
            True,
        )
        edited = bundle.adaptive_executor.play_candidate8(
            genome,
            opponent,
            current_seed,
            current_seat,
            local_rank,
            args.minimum_decision_day,
            True,
            True,
        )
        keep_trace = list(keep["trace"])
        edited_trace = list(edited["trace"])
        differing = [
            index
            for index, (left, right) in enumerate(
                zip(keep_trace, edited_trace, strict=True)
            )
            if left != right
        ]
        hard_errors = {
            "avoidable_crop_losses": int(edited["avoidable_crop_losses"]),
            "avoidable_animal_losses": int(edited["avoidable_animal_losses"]),
            "end_overflow": int(edited["end_overflow"]),
        }
        rows.append(
            {
                "family": FAMILY_NAMES[family_id],
                "status": "PASS"
                if differing
                and int(edited["first_candidate8_family"]) == family_id
                and not any(hard_errors.values())
                else "FAIL",
                "seed": current_seed,
                "seat": current_seat,
                "state_id": state,
                "feasible_rank": local_rank,
                "signature": int(signature[selected]),
                "candidate_features": [int(value) for value in features[selected, :20]],
                "dataset_gain_vs_keep": float(gain[selected]),
                "realized_gain_vs_keep": float(
                    edited["rewards"][current_seat]
                    - keep["rewards"][current_seat]
                ),
                "decision_step": int(edited["first_candidate8_day"]) * 24,
                "first_differing_step": differing[0] if differing else None,
                "differing_joint_steps": len(differing),
                "hard_errors": hard_errors,
            }
        )

    payload = {
        "schema": "kaggriculture.candidate8_behavior_effects.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "boundary": (
            "Each family must change the executed joint-action trajectory and "
            "remain hard-safe. Reward sign is diagnostic, not a test condition."
        ),
        "rows": rows,
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "source": args.source,
                "actions": args.actions,
                "metadata": args.metadata,
                "genomes": args.genomes,
                "dataset": args.dataset,
            }.items()
        },
    }
    if args.backbone is not None:
        payload["inputs"]["backbone"] = {
            "path": str(args.backbone),
            "sha256": sha256(args.backbone),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "rows": rows}, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

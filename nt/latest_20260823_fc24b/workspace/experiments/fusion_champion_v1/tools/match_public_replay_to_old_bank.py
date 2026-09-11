#!/usr/bin/env python3
"""Rank old frozen task routes by exact action similarity to a Replay seat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "gpu_sim/src"))
from kaggriculture_jax.codec import encode_actions, stack_actions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--seat", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = json.loads(args.episode.read_text(encoding="utf-8"))
    recorded = stack_actions(
        [
            encode_actions(
                [
                    replay["steps"][step + 1][seat].get("action") or {}
                    for seat in (0, 1)
                ]
            )
            for step in range(719)
        ]
    )
    recorded_np = [np.asarray(value)[:, args.seat] for value in recorded]
    bank = np.load(
        ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    )
    fields = list(bank.files)
    receipt = json.loads(
        (ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json").read_text(
            encoding="utf-8"
        )
    )
    names = {
        int(row["skeleton_id"]): f'{row["opponent"]}::{row["route"]}'
        for row in receipt["skeletons"]
    }
    rows = []
    for route in range(bank[fields[0]].shape[0]):
        exact = np.ones(719, dtype=bool)
        for field, target in zip(fields, recorded_np, strict=True):
            value = bank[field][route]
            if value.ndim == 1:
                exact &= value == target
            else:
                exact &= np.all(value == target, axis=tuple(range(1, value.ndim)))
        mismatch = np.flatnonzero(~exact)
        prefix = int(mismatch[0]) if mismatch.size else 719
        rows.append(
            {
                "route": route,
                "name": names.get(route, "UNKNOWN"),
                "exact_prefix": prefix,
                "rate_72": float(np.mean(exact[:72])),
                "rate_120": float(np.mean(exact[:120])),
                "rate_192": float(np.mean(exact[:192])),
                "rate_full": float(np.mean(exact)),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            row["rate_192"],
            row["rate_120"],
            row["exact_prefix"],
            row["rate_full"],
        ),
        reverse=True,
    )
    payload = {
        "schema": "kaggriculture.public-replay-old-bank-action-match.v1",
        "episode_id": int(replay["info"]["EpisodeId"]),
        "team_names": replay["info"]["TeamNames"],
        "seat": args.seat,
        "top20": ranked[:20],
        "rows": rows,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"top10": ranked[:10], "output": str(output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

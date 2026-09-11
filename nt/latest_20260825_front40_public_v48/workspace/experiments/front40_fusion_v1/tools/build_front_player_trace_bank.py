"""Compile one current Top-40 submission into a generic replay trace bank.

The authoritative seat and submission membership come from episode_rows.csv.
This deliberately avoids player-name matching and makes the artifact reusable
for every frozen Top-40 submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/hasegawa_jax_v2/tools"))
sys.path.insert(0, str(ROOT / "experiments/hasegawa_jax_v3/tools"))

from build_trace_bank_v2 import _encode  # noqa: E402
from build_trace_bank_v3 import public_arrays, shop_sequence  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-rows", type=Path, required=True)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument(
        "--team-name",
        default="",
        help="Optional audited display-name override for mojibake in episode_rows.csv.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--wins-only", action="store_true", default=True)
    parser.add_argument(
        "--all-complete",
        action="store_true",
        help="Include trusted losses as recovery-state demonstrations too.",
    )
    args = parser.parse_args()
    wins_only = not args.all_complete

    selected_rows: list[dict[str, str]] = []
    with args.episode_rows.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["submission_id"]) != args.submission_id:
                continue
            if wins_only and row["result"] != "WIN":
                continue
            # Both values refer to a locally present Episode JSON.  The latter
            # was fetched during this corpus build, while the former was
            # content-reused from the audited cache.  Full JSON/frame/reward
            # validation below is the actual compatibility gate.
            if row.get("status") not in {"downloaded", "trusted_hardlinked_cache"}:
                continue
            selected_rows.append(row)
    if not selected_rows:
        raise RuntimeError(f"no trusted rows for submission {args.submission_id}")

    # One player view per Episode.  Rows are already submission-scoped, but
    # explicit deduplication prevents repeated service rows changing route IDs.
    by_episode: dict[int, dict[str, str]] = {}
    for row in selected_rows:
        by_episode.setdefault(int(row["episode_id"]), row)

    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for episode_id, row in sorted(by_episode.items()):
        path = Path(row["path"])
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            seat = int(row["own_seat"])
            if len(document.get("steps", [])) != 720:
                raise ValueError(f"frames={len(document.get('steps', []))}")
            rewards = document.get("rewards", [0, 0])
            reward = int(rewards[seat])
            opponent_reward = int(rewards[1 - seat])
            if wins_only and reward <= opponent_reward:
                raise ValueError("row says WIN but Replay reward is not greater")
            candidates.append(
                {
                    "episode_id": episode_id,
                    "path": path,
                    "document": document,
                    "seat": seat,
                    "reward": reward,
                    "opponent_reward": opponent_reward,
                    "shops": shop_sequence(document, seat),
                    "team_name": args.team_name or row["team_name"],
                    "rank": int(row["rank"]),
                }
            )
        except Exception as exc:  # receipt keeps every rejected source auditable
            rejected.append({"episode_id": episode_id, "path": str(path), "error": repr(exc)})
    if not candidates:
        raise RuntimeError("no compatible winning Replays after validation")

    fields = (
        "unit_op", "unit_item", "unit_amount", "unit_count", "market_op",
        "market_item", "market_amount", "market_count", "expected_unit_pos",
        "expected_unit_active", "expected_money",
    )
    compiled = {field: [] for field in fields}
    own_summaries, opponent_summaries = [], []
    sheds, seeds, carried, prices, inventories = [], [], [], [], []
    sources: list[dict[str, object]] = []
    for route_id, row in enumerate(candidates):
        encoded = _encode(row["document"], int(row["seat"]))
        for field in fields:
            compiled[field].append(encoded[field])
        own, opponent, shed, seed_bank, carry, price, inventory = public_arrays(
            row["document"], int(row["seat"])
        )
        own_summaries.append(own)
        opponent_summaries.append(opponent)
        sheds.append(shed)
        seeds.append(seed_bank)
        carried.append(carry)
        prices.append(price)
        inventories.append(inventory)
        sources.append(
            {
                "route_id": route_id,
                "episode_id": int(row["episode_id"]),
                "reward": int(row["reward"]),
                "opponent_reward": int(row["opponent_reward"]),
                "margin": int(row["reward"]) - int(row["opponent_reward"]),
                "seat": int(row["seat"]),
                "shops": row["shops"].tolist(),
                "path": str(row["path"]),
                "sha256": sha256(row["path"]),
            }
        )

    arrays = {field: np.stack(value) for field, value in compiled.items()}
    arrays.update(
        expected_self_summary=np.stack(own_summaries),
        expected_opponent_summary=np.stack(opponent_summaries),
        expected_shed=np.stack(sheds),
        expected_seeds=np.stack(seeds),
        expected_carried=np.stack(carried),
        expected_market_price=np.stack(prices),
        expected_market_inventory=np.stack(inventories),
        source_shop_sequence=np.stack([row["shops"] for row in candidates]),
        source_episode_id=np.asarray([row["episode_id"] for row in sources], np.int64),
        source_reward=np.asarray([row["reward"] for row in sources], np.int32),
        source_opponent_reward=np.asarray([row["opponent_reward"] for row in sources], np.int32),
        source_margin=np.asarray([row["margin"] for row in sources], np.int32),
        source_seat=np.asarray([row["seat"] for row in sources], np.int8),
        bootstrap_route_id=np.asarray(
            int(np.argmax([row["reward"] for row in sources])), np.int16
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    receipt = {
        "schema": "kaggriculture.front40_fusion.generic_trace_bank.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "submission_id": args.submission_id,
        "rank": int(candidates[0]["rank"]),
        "team_name": str(candidates[0]["team_name"]),
        "wins_only": wins_only,
        "candidate_rows": len(selected_rows),
        "unique_episodes": len(by_episode),
        "route_count": len(candidates),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "bootstrap_route_id": int(arrays["bootstrap_route_id"]),
        "sources": sources,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "shapes": {field: list(value.shape) for field, value in arrays.items()},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "team": receipt["team_name"],
                "routes": len(candidates),
                "rejected": len(rejected),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

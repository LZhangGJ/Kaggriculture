from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "experiments" / "expert_business_agent_v2" / "tools"
sys.path.insert(0, str(TOOLS))

from generate_official_stepwise_parity_traces import generate_one  # noqa: E402


TARGET = (
    ROOT
    / "public_notebooks"
    / "recent_latest_20260825_scan_v1"
    / "kaitofukami__40-40-early-floor-39-46-top-10-v48-fast-routes"
    / "output"
    / "main.py"
)
FC24B = ROOT / "submission" / "_staging_fc24b_20260823_v1" / "main.py"
NAME = "kaito_v48"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1302001)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    receipt_path = args.receipt if args.receipt.is_absolute() else ROOT / args.receipt
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    tasks = [
        (NAME, str(FC24B), str(TARGET), str(output), seed, seat)
        for seat in (0, 1)
        for seed in seeds
    ]
    with ProcessPoolExecutor(
        max_workers=min(18, max(1, args.workers), len(tasks))
    ) as executor:
        rows = list(executor.map(generate_one, tasks, chunksize=1))

    result = {
        "schema": "kaggriculture.public-recent-20260825.kaito-v48-vs-fc24b-traces.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_package_version": importlib.metadata.version("kaggle-environments"),
        "seeds": seeds,
        "seat_swapped": True,
        "opponent": str(FC24B.resolve()),
        "opponent_sha256": sha256(FC24B),
        "agents": [
            {"name": NAME, "path": str(TARGET.resolve()), "sha256": sha256(TARGET)}
        ],
        "traces": rows,
        "ok": all(row["frames"] == 720 for row in rows),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS" if result["ok"] else "FAIL",
                "games": len(rows),
                "receipt": str(receipt_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

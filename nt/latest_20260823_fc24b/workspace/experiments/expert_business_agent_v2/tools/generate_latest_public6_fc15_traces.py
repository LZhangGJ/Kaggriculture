from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from generate_official_stepwise_parity_traces import generate_one


ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "references" / "public_latest6_20260822"
FC15 = ROOT / "experiments" / "fusion_champion_v1" / "artifacts" / "fc15_cpu_v4" / "main.py"
TARGETS = {
    name: FROZEN / name / "main.py"
    for name in (
        "boatlee_v21_latest",
        "prvsiyan_soil_v26h_latest",
        "prvsiyan_moon_v92_latest",
        "kaito_v39_history_gate_latest",
        "steven_e284_hadouken_latest",
        "salem_harvestforge_x_latest",
    )
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=822001)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--agents", nargs="*", choices=tuple(TARGETS))
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    receipt_path = args.receipt if args.receipt.is_absolute() else ROOT / args.receipt
    selected = args.agents or list(TARGETS)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    tasks = [
        (name, str(FC15), str(TARGETS[name]), str(output), seed, seat)
        for name in selected
        for seat in (0, 1)
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=min(18, args.workers, len(tasks))) as executor:
        rows = list(executor.map(generate_one, tasks, chunksize=1))

    result = {
        "schema": "kaggriculture.latest_public6_vs_fc15_official_traces.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_package_version": importlib.metadata.version("kaggle-environments"),
        "seeds": seeds,
        "seat_swapped": True,
        "opponent": str(FC15.resolve()),
        "opponent_sha256": sha256(FC15),
        "agents": [
            {"name": name, "path": str(TARGETS[name].resolve()), "sha256": sha256(TARGETS[name])}
            for name in selected
        ],
        "traces": rows,
        "ok": all(row["frames"] == 720 for row in rows),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS" if result["ok"] else "FAIL",
                "games": len(rows),
                "receipt": str(receipt_path),
            },
            indent=2,
        )
    )
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

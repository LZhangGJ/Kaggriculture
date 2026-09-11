from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_official_stepwise_parity_traces import generate_one  # noqa: E402


FROZEN = ROOT / "references" / "public_latest8_20260820"
TARGETS = {
    "deniz_v111_8c4s_latest": FROZEN / "deniz_v111_8c4s_latest" / "main.py",
    "boatlee_v20_latest": FROZEN / "boatlee_v20_latest" / "main.py",
    "kunal_2026_v1_latest": FROZEN / "kunal_2026_v1_latest" / "main.py",
    "rayk_rank_agent_latest": FROZEN / "rayk_rank_agent_latest" / "main.py",
    "kaito_v36_latest": FROZEN / "kaito_v36_latest" / "main.py",
    "x562_latest": FROZEN / "x562_latest" / "main.py",
    "tetsutani_adaptive_latest": FROZEN / "tetsutani_adaptive_latest" / "main.py",
    "flex_multi_route_latest": FROZEN / "flex_multi_route_latest" / "main.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=208201)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--agents",
        nargs="*",
        choices=tuple(TARGETS),
        help="Optional subset; omitted means all eight frozen sources.",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    receipt_path = args.receipt if args.receipt.is_absolute() else ROOT / args.receipt
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    selected_targets = args.agents or list(TARGETS)
    tasks = [
        (name, str(path), str(path), str(output), seed, seat)
        for name, path in TARGETS.items()
        if name in selected_targets
        for seat in (0, 1)
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=min(18, args.workers, len(tasks))) as executor:
        rows = list(executor.map(generate_one, tasks, chunksize=1))
    result = {
        "schema": "kaggriculture.latest_public8_mirror_traces.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_package_version": importlib.metadata.version("kaggle-environments"),
        "seeds": seeds,
        "seat_swapped": True,
        "mirror": True,
        "selected_targets": selected_targets,
        "opponents": [
            {"name": name, "path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in TARGETS.items()
            if name in selected_targets
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

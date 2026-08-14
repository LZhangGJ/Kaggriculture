"""Convert official replay JSON files into compact BC-v0 NumPy shards."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from kaggriculture_lab.replay_bc import encode_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=Path(r"D:\Kaggriculture\data\raw\replays\2026-08-13"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\Kaggriculture\data\processed\bc_v0"))
    parser.add_argument("--max-files", type=int, default=0, help="zero uses every replay")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--unit-inventory-context",
        action="store_true",
        help="store each unit's private carried inventory beside its position",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.replay_dir.glob("*.json"))
    if not files:
        parser.error(f"no replay JSON files found in {args.replay_dir}")
    random.Random(args.seed).shuffle(files)
    if args.max_files > 0:
        files = files[: args.max_files]
    if len(files) < 3:
        parser.error("at least three replay files are required")

    train_end = max(1, round(0.8 * len(files)))
    validation_end = max(train_end + 1, round(0.9 * len(files)))
    validation_end = min(validation_end, len(files) - 1)
    assignments = {
        path.name: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, path in enumerate(files)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total: Counter[str] = Counter()
    built = 0

    for index, path in enumerate(files, start=1):
        split = assignments[path.name]
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        output = split_dir / f"{path.stem}.npz"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists; pass --overwrite: {output}")
        with path.open(encoding="utf-8") as stream:
            replay = json.load(stream)
        arrays, stats = encode_replay(
            replay,
            include_unit_inventory=args.unit_inventory_context,
        )
        np.savez(output, **arrays)
        total.update(stats)
        built += 1
        print(
            f"[{index}/{len(files)}] split={split} replay={path.name} "
            f"examples={stats['examples']} unit_invalid={stats['unit_mask_invalid']} "
            f"market_invalid={stats['market_mask_invalid']}"
        )

    manifest = {
        "schema_version": 1,
        "source": str(args.replay_dir),
        "seed": args.seed,
        "unit_inventory_context": args.unit_inventory_context,
        "files": assignments,
        "stats": dict(total),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"built={built} output={args.output_dir}")
    print(json.dumps(dict(total), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

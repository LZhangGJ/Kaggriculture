"""Create a compact BC dataset by filtering and truncating episode shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples-per-shard", type=int, required=True)
    parser.add_argument(
        "--outcome",
        type=int,
        choices=(-1, 0, 1),
        help="retain only shards whose value target has this episode outcome",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_examples_per_shard <= 0:
        parser.error("--max-examples-per-shard must be positive")

    files = 0
    examples = 0
    skipped_outcome = 0
    assignments: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for source in sorted((args.input_dir / split).glob("*.npz")):
            with np.load(source) as data:
                outcome = int(np.sign(float(data["value_targets"][0])))
                if args.outcome is not None and outcome != args.outcome:
                    skipped_outcome += 1
                    continue
                size = min(args.max_examples_per_shard, len(data["features"]))
                arrays = {key: data[key][:size] for key in data.files}
            target = args.output_dir / split / source.name
            if target.exists() and not args.overwrite:
                raise FileExistsError(
                    f"output already exists; pass --overwrite: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez(target, **arrays)
            assignments[source.name] = split
            files += 1
            examples += size

    if not files:
        parser.error("no shards matched the requested filter")
    manifest = {
        "schema_version": 1,
        "kind": "filtered_bc",
        "source": str(args.input_dir),
        "max_examples_per_shard": args.max_examples_per_shard,
        "outcome": args.outcome,
        "files": files,
        "examples": examples,
        "skipped_outcome": skipped_outcome,
        "assignments": assignments,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "assignments"}, indent=2))


if __name__ == "__main__":
    main()

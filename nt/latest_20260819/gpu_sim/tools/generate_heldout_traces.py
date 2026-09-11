"""Generate the 100 unseen-seed official full-season parity corpus."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.metadata
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggle_environments import make

from generate_reference_traces import canonical_frame


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT / "reference" / "heldout"
RECEIPT = PROJECT / "receipts" / "heldout_reference_traces.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_trace(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record in records:
                    text.write(
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    )


def generate(seed: int, output_dir: Path) -> dict:
    configuration = {"episodeSteps": 720, "seed": seed}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        env = make("kaggriculture", configuration=configuration, debug=True)
        env.run(["pass", "pass"])
    frames = [canonical_frame(i, states) for i, states in enumerate(env.steps)]
    if len(frames) != 720 or frames[-1]["status"] != ["DONE", "DONE"]:
        raise RuntimeError(f"seed={seed}: incomplete official trace")
    header = {
        "record_type": "header",
        "schema": "kaggriculture_official_trace_v1",
        "package": "kaggle-environments",
        "package_version": importlib.metadata.version("kaggle-environments"),
        "scenario": "pass_pass_heldout",
        "configuration": configuration,
        "frame_count": len(frames),
    }
    records = [header, *({"record_type": "frame", **frame} for frame in frames)]
    output = output_dir / f"pass_pass_seed{seed}.jsonl.gz"
    write_trace(output, records)
    return {
        "seed": seed,
        "path": str(output.relative_to(PROJECT)),
        "frames": len(frames),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "terminal_reward": frames[-1]["reward"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-seed", type=int, default=10000)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()

    rows = []
    for index, seed in enumerate(range(args.first_seed, args.first_seed + args.count), 1):
        row = generate(seed, args.output_dir)
        rows.append(row)
        print(
            f"[{index}/{args.count}] seed={seed} frames={row['frames']} "
            f"bytes={row['bytes']}",
            flush=True,
        )
    receipt = {
        "schema": "kaggriculture_heldout_reference_traces_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_version": importlib.metadata.version("kaggle-environments"),
        "scenario": "pass_pass_heldout",
        "first_seed": args.first_seed,
        "count": args.count,
        "total_frames": sum(row["frames"] for row in rows),
        "rows": rows,
        "ok": len(rows) == args.count and all(row["frames"] == 720 for row in rows),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows"}, indent=2))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


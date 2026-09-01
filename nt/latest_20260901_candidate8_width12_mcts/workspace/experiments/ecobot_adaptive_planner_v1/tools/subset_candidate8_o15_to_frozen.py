#!/usr/bin/env python3
"""Extract O1.5 response rows in the exact frozen O1.4 row order."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROW_IDENTITY = (
    "opponent", "prefix_seed", "seat", "decision_day", "signature",
)
EXACT_AFTER_SUBSET = (
    "state_id", "opponent", "prefix_seed", "seat", "decision_day",
    "signature", "family", "features", "feature_names",
    "consequence_features", "consequence_feature_names",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def row_keys(data: np.lib.npyio.NpzFile) -> list[tuple]:
    columns = [np.asarray(data[name]) for name in ROW_IDENTITY]
    return list(zip(*(column.tolist() for column in columns), strict=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument(
        "--response",
        required=True,
        type=Path,
        nargs="+",
        help="One or more response datasets whose rows form the source pool.",
    )
    parser.add_argument("--dataset-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    responses = [np.load(path, allow_pickle=False) for path in args.response]
    try:
        if not responses:
            raise RuntimeError("at least one response dataset is required")
        response_files = set(responses[0].files)
        if any(set(response.files) != response_files for response in responses[1:]):
            raise RuntimeError("response datasets do not contain identical arrays")
        response_row_counts = [len(response["state_id"]) for response in responses]
        response_rows = sum(response_row_counts)
        combined = {}
        for name in responses[0].files:
            values = [np.asarray(response[name]) for response in responses]
            if all(
                value.ndim >= 1 and len(value) == row_count
                for value, row_count in zip(values, response_row_counts, strict=True)
            ):
                combined[name] = np.concatenate(values, axis=0)
            else:
                if not all(np.array_equal(values[0], value) for value in values[1:]):
                    raise RuntimeError(
                        f"non-row metadata differs across response datasets: {name}"
                    )
                combined[name] = values[0].copy()

        with np.load(args.frozen, allow_pickle=False) as frozen:
            frozen_rows = len(frozen["state_id"])
            frozen_keys = row_keys(frozen)
            response_keys = list(zip(*(
                np.asarray(combined[name]).tolist() for name in ROW_IDENTITY
            ), strict=True))
            response_lookup = {}
            duplicate_response_keys = 0
            for index, key in enumerate(response_keys):
                if key in response_lookup:
                    duplicate_response_keys += 1
                else:
                    response_lookup[key] = index
            missing = [key for key in frozen_keys if key not in response_lookup]
            if duplicate_response_keys or missing:
                raise RuntimeError(
                    "cannot uniquely align response rows: "
                    f"duplicates={duplicate_response_keys}, missing={len(missing)}"
                )
            selected = np.asarray(
                [response_lookup[key] for key in frozen_keys], dtype=np.int64
            )
            if len(np.unique(selected)) != frozen_rows:
                raise RuntimeError("frozen rows did not map one-to-one into response rows")

            arrays = {}
            for name, value in combined.items():
                if name == "state_id":
                    arrays[name] = np.asarray(frozen[name]).copy()
                elif value.ndim >= 1 and len(value) == response_rows:
                    arrays[name] = value[selected].copy()
                else:
                    arrays[name] = value.copy()

            exact = {
                name: bool(np.array_equal(np.asarray(frozen[name]), arrays[name]))
                for name in EXACT_AFTER_SUBSET
            }
            if not all(exact.values()):
                raise RuntimeError(f"subset differs from frozen O1.4 rows: {exact}")
            selected_opponents = sorted({str(value) for value in arrays["opponent"]})
            frozen_opponents = sorted({str(value) for value in frozen["opponent"]})
            if selected_opponents != frozen_opponents:
                raise RuntimeError("opponent set differs after frozen-row extraction")
    finally:
        for response in responses:
            response.close()

    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dataset_output, **arrays)
    payload = {
        "schema": "kaggriculture.candidate8-o15-frozen-subset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_rows": response_rows,
        "frozen_rows": frozen_rows,
        "discarded_out_of_scope_rows": response_rows - frozen_rows,
        "states": int(len(np.unique(arrays["state_id"]))),
        "opponents": selected_opponents,
        "opponent_count": len(selected_opponents),
        "duplicate_response_keys": duplicate_response_keys,
        "missing_frozen_keys": len(missing),
        "identity_and_canonical_exact": exact,
        "boundary": (
            "Only rows whose opponent/prefix_seed/seat/decision_day/signature "
            "match the frozen O1.4 corpus are retained, in frozen row order. "
            "Out-of-scope opponents are not used for O1.5 training or gates."
        ),
        "inputs": {
            "frozen": {"path": str(args.frozen), "sha256": sha256(args.frozen)},
            "response": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.response
            ],
        },
        "dataset": {
            "path": str(args.dataset_output),
            "sha256": sha256(args.dataset_output),
            "bytes": args.dataset_output.stat().st_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

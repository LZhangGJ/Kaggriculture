#!/usr/bin/env python3
"""Attach non-clairvoyant Candidate8 preview features to frozen labels.

The preview run's terminal rewards are deliberately discarded.  Row identity
must match the frozen outcome dataset exactly, so this tool cannot silently
change labels, candidate order, public context, or data splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


IDENTITY = (
    "state_id", "opponent", "prefix_seed", "seat", "decision_day",
    "signature", "family", "features", "feature_names",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--dataset-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.outcomes, allow_pickle=False) as frozen, np.load(
        args.preview, allow_pickle=False
    ) as preview:
        exact = {key: bool(np.array_equal(frozen[key], preview[key])) for key in IDENTITY}
        if not all(exact.values()):
            raise RuntimeError(f"preview rows do not match frozen labels: {exact}")
        arrays = {key: frozen[key] for key in frozen.files}
        consequences = np.asarray(preview["consequence_features"], dtype=np.int32)
        names = np.asarray(preview["consequence_feature_names"])
        if consequences.shape != (len(frozen["state_id"]), 228):
            raise RuntimeError(f"unexpected preview shape {consequences.shape}")
        arrays["consequence_features"] = consequences
        arrays["consequence_feature_names"] = names

    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dataset_output, **arrays)
    state_ids = np.asarray(arrays["state_id"])
    unique_per_state = []
    varying_per_state = []
    for state in np.unique(state_ids):
        rows = consequences[state_ids == state]
        unique_per_state.append(len(np.unique(rows, axis=0)))
        varying_per_state.append(int(np.sum(np.ptp(rows, axis=0) > 0)))
    payload = {
        "schema": "kaggriculture.candidate8-o14-exact-consequence-merge.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "identity_arrays_exact": exact,
        "rows": int(len(state_ids)),
        "states": int(len(np.unique(state_ids))),
        "consequence_width": int(consequences.shape[1]),
        "globally_varying_features": int(np.sum(np.ptp(consequences, axis=0) > 0)),
        "mean_unique_consequences_per_state": float(np.mean(unique_per_state)),
        "minimum_unique_consequences_per_state": int(np.min(unique_per_state)),
        "mean_varying_features_per_state": float(np.mean(varying_per_state)),
        "minimum_varying_features_per_state": int(np.min(varying_per_state)),
        "boundary": (
            "Only consequence_features and their names are copied from the "
            "canonical preview. All labels and identity arrays come unchanged "
            "from the frozen outcome dataset."
        ),
        "inputs": {
            "outcomes": {"path": str(args.outcomes), "sha256": sha256(args.outcomes)},
            "preview": {"path": str(args.preview), "sha256": sha256(args.preview)},
        },
        "dataset": {"path": str(args.dataset_output), "sha256": sha256(args.dataset_output)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

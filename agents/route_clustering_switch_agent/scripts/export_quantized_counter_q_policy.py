#!/usr/bin/env python3
"""Export a fitted backward route-Q forest to the sklearn-free runtime format."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))

from meta_agent.src.search_route_policy import QuantizedExtraTreesRouteQ


def _pack_model(bundle: dict, bits: int) -> tuple[bytes, dict[str, int]]:
    if bits not in (8, 16):
        raise ValueError("quantization bits must be 8 or 16")
    targets = np.asarray(bundle["targets"], dtype=str)
    checkpoints = np.asarray(bundle["checkpoints"], dtype=np.int16)
    models = bundle["models"]
    if len(models) != len(checkpoints):
        raise ValueError("model/checkpoint counts differ")

    checkpoint_tree_offsets = [0]
    tree_node_offsets: list[int] = []
    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    feature_chunks: list[np.ndarray] = []
    threshold_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    node_offset = 0
    scale = (1 << bits) - 1
    value_dtype = np.uint8 if bits == 8 else np.uint16

    for model in models:
        if int(model.n_outputs_) != len(targets):
            raise ValueError("target count does not match fitted model outputs")
        for estimator in model.estimators_:
            tree = estimator.tree_
            tree_node_offsets.append(node_offset)
            local_left = np.asarray(tree.children_left, dtype=np.int32)
            local_right = np.asarray(tree.children_right, dtype=np.int32)
            left_chunks.append(np.where(local_left < 0, -1, local_left + node_offset))
            right_chunks.append(np.where(local_right < 0, -1, local_right + node_offset))
            feature_chunks.append(np.asarray(tree.feature, dtype=np.int16))
            threshold_chunks.append(np.asarray(tree.threshold, dtype=np.float64))
            raw_values = np.asarray(tree.value[:, :, 0], dtype=np.float64)
            value_chunks.append(np.rint(np.clip(raw_values, 0.0, 1.0) * scale).astype(
                value_dtype
            ))
            node_offset += int(tree.node_count)
        checkpoint_tree_offsets.append(len(tree_node_offsets))

    output = io.BytesIO()
    np.savez_compressed(
        output,
        targets=targets,
        checkpoints=checkpoints,
        checkpoint_tree_offsets=np.asarray(checkpoint_tree_offsets, dtype=np.int16),
        tree_node_offsets=np.asarray(tree_node_offsets, dtype=np.int32),
        left=np.concatenate(left_chunks).astype(np.int32, copy=False),
        right=np.concatenate(right_chunks).astype(np.int32, copy=False),
        feature=np.concatenate(feature_chunks).astype(np.int16, copy=False),
        threshold=np.concatenate(threshold_chunks).astype(np.float64, copy=False),
        leaf_values=np.concatenate(value_chunks).astype(value_dtype, copy=False),
    )
    packed = output.getvalue()
    return packed, {
        "bits": bits,
        "scale": scale,
        "checkpoints": len(checkpoints),
        "trees": len(tree_node_offsets),
        "nodes": node_offset,
        "features": int(models[0].n_features_in_),
        "targets": len(targets),
        "npz_bytes": len(packed),
    }


def _load_feature_panel(
    path: Path, checkpoints: list[int], expected_features: int,
) -> tuple[list[np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as saved:
        names = saved["feature_names"].astype(str).tolist()
        features = [
            np.ascontiguousarray(
                saved[f"features_{checkpoint}"].reshape(-1, expected_features),
                dtype=np.float32,
            )
            for checkpoint in checkpoints
        ]
    if len(names) != expected_features:
        raise ValueError("feature panel schema differs from fitted model")
    return features, names


def _validate_predictions(
    bundle: dict, policy: dict, feature_panel: Path,
) -> tuple[list[dict], list[str]]:
    checkpoints = [int(value) for value in bundle["checkpoints"]]
    expected_features = int(bundle["models"][0].n_features_in_)
    features, names = _load_feature_panel(
        feature_panel, checkpoints, expected_features
    )
    runtime = QuantizedExtraTreesRouteQ(policy)
    rows = []
    for checkpoint, model, matrix in zip(checkpoints, bundle["models"], features):
        reference_indices = np.argmax(model.predict(matrix), axis=1)
        runtime_targets = np.asarray([
            runtime.predict(checkpoint, vector) for vector in matrix
        ])
        runtime_indices = np.asarray([
            list(bundle["targets"]).index(value) for value in runtime_targets
        ])
        mismatches = int(np.count_nonzero(reference_indices != runtime_indices))
        rows.append({
            "checkpoint": checkpoint,
            "samples": len(matrix),
            "mismatches": mismatches,
            "agreement_rate": float(1.0 - mismatches / len(matrix)),
            "reference_counts": dict(Counter(
                np.asarray(bundle["targets"])[reference_indices].tolist()
            )),
            "runtime_counts": dict(Counter(runtime_targets.tolist())),
        })
    return rows, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(8, 16), default=16)
    parser.add_argument("--max-mismatches", type=int, default=0)
    parser.add_argument("--output-policy", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    if bundle.get("schema") != "backward-counter-q-model-v1":
        raise ValueError("unsupported fitted route-Q model schema")
    packed, model_stats = _pack_model(bundle, args.bits)
    encoded = base64.b64encode(packed).decode("ascii")
    policy = {
        "schema_version": 1,
        "kind": "backward_counterfactual_route_q",
        "feature_schema": "semantic_route_switch_v1",
        "targets": [str(value) for value in bundle["targets"]],
        "openings": [str(bundle["opening"])],
        "checkpoints": [int(value) for value in bundle["checkpoints"]],
        "nodes": [
            {"selected": {
                "opening": str(bundle["opening"]),
                "checkpoint": int(checkpoint),
                "enabled": True,
            }}
            for checkpoint in bundle["checkpoints"]
        ],
        "q_model": {
            "schema": "quantized-extra-trees-route-q-v1",
            **model_stats,
            "source_model": str(args.model.resolve()),
        },
        "q_model_npz_base64": encoded,
    }
    prediction_rows, feature_names = _validate_predictions(
        bundle, policy, args.validation_features
    )
    mismatch_count = sum(row["mismatches"] for row in prediction_rows)
    policy["feature_names"] = feature_names
    policy["quantization_validation"] = {
        "feature_panel": str(args.validation_features.resolve()),
        "mismatches": mismatch_count,
        "rows": prediction_rows,
    }
    if mismatch_count > args.max_mismatches:
        raise RuntimeError(
            f"quantized selector has {mismatch_count} prediction mismatches; "
            f"limit is {args.max_mismatches}"
        )

    args.output_policy.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_policy.write_text(
        json.dumps(policy, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": "quantized-counter-q-export-report-v1",
        "source_model": str(args.model.resolve()),
        "validation_features": str(args.validation_features.resolve()),
        "output_policy": str(args.output_policy.resolve()),
        "policy_bytes": args.output_policy.stat().st_size,
        "model": model_stats,
        "prediction_validation": prediction_rows,
        "prediction_mismatches": mismatch_count,
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

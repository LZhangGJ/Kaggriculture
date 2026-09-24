#!/usr/bin/env python3
"""Compare two aligned native JobBatch NPZ rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


IDENTITY = ("seed", "seat", "opponent", "route", "policy_seed")
REQUIRED = (*IDENTITY, "own_cash", "rival_cash")
OPPONENTS = {
    1: "thomas", 2: "meta", 3: "replay_clean", 4: "salemali",
    5: "fieldcraft",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(set(archive.files)) != len(archive.files):
                raise RuntimeError(f"duplicate arrays in {path}")
            missing = set(REQUIRED) - set(archive.files)
            if missing:
                raise RuntimeError(f"missing arrays in {path}: {sorted(missing)}")
            arrays = {name: np.asarray(archive[name]) for name in REQUIRED}
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"invalid native JobBatch NPZ {path}: {error}") from error

    if arrays["seed"].ndim != 1:
        raise RuntimeError(f"invalid identity array seed in {path}")
    size = len(arrays["seed"])
    if not size:
        raise RuntimeError(f"empty native JobBatch NPZ: {path}")
    for name in IDENTITY:
        value = arrays[name]
        if value.ndim != 1 or len(value) != size or not np.issubdtype(value.dtype, np.integer):
            raise RuntimeError(f"invalid identity array {name} in {path}")
    for name in ("own_cash", "rival_cash"):
        value = arrays[name]
        if (value.ndim != 1 or len(value) != size or
                not (np.issubdtype(value.dtype, np.integer) or
                     np.issubdtype(value.dtype, np.floating)) or
                not np.all(np.isfinite(value))):
            raise RuntimeError(f"invalid cash array {name} in {path}")
    if not set(np.unique(arrays["seat"])).issubset({0, 1}):
        raise RuntimeError(f"invalid seat identity in {path}")
    unknown = set(map(int, np.unique(arrays["opponent"]))) - set(OPPONENTS)
    if unknown:
        raise RuntimeError(f"unknown opponent identity in {path}: {sorted(unknown)}")
    if np.any(arrays["seed"] < 0) or np.any(arrays["policy_seed"] < 0):
        raise RuntimeError(f"negative seed identity in {path}")
    opponent, route = arrays["opponent"], arrays["route"]
    if np.any(((opponent != 3) & (route != -1)) |
              ((opponent == 3) & ((route < 0) | (route >= 245)))):
        raise RuntimeError(f"invalid route identity in {path}")
    identities = set(zip(*(map(int, arrays[name]) for name in IDENTITY)))
    if len(identities) != size:
        raise RuntimeError(f"duplicate game identity in {path}")
    return arrays


def _summary(parent_margin: np.ndarray, child_margin: np.ndarray) -> dict:
    parent_win = parent_margin > 0
    child_win = child_margin > 0
    rescue = int(np.count_nonzero(~parent_win & child_win))
    hurt = int(np.count_nonzero(parent_win & ~child_win))
    return {
        "games": len(parent_margin),
        "parent_wins": int(parent_win.sum()),
        "child_wins": int(child_win.sum()),
        "delta_wins": int(child_win.sum() - parent_win.sum()),
        "parent_mean_margin": float(parent_margin.mean()),
        "child_mean_margin": float(child_margin.mean()),
        "delta_mean_margin": float((child_margin - parent_margin).mean()),
        "rescue": rescue,
        "hurt": hurt,
        "mcnemar_exact_p": float(
            binomtest(rescue, rescue + hurt, 0.5).pvalue
            if rescue + hurt else 1.0),
    }


def _bootstrap(seed: np.ndarray, delta_margin: np.ndarray,
               delta_win: np.ndarray, samples: int, rng_seed: int) -> dict:
    unique, inverse = np.unique(seed, return_inverse=True)
    counts = np.bincount(inverse)
    margin_units = np.bincount(inverse, weights=delta_margin) / counts
    win_units = np.bincount(inverse, weights=delta_win) / counts
    rng = np.random.default_rng(rng_seed)
    selected = rng.integers(len(unique), size=(samples, len(unique)))
    margin_estimates = margin_units[selected].mean(axis=1)
    win_estimates = win_units[selected].mean(axis=1)
    return {
        "clusters": len(unique),
        "delta_mean_margin": np.quantile(
            margin_estimates, [0.025, 0.975]).tolist(),
        "delta_win_rate": np.quantile(
            win_estimates, [0.025, 0.975]).tolist(),
    }


def build_report(parent_path: Path, child_path: Path, label: str,
                 bootstrap_samples: int = 20_000,
                 bootstrap_seed: int = 20260923) -> dict:
    if not label.strip() or bootstrap_samples < 1:
        raise ValueError("label must be non-empty and bootstrap samples positive")
    for path in (parent_path, child_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent_sha, child_sha = _sha256(parent_path), _sha256(child_path)
    parent, child = _load(parent_path), _load(child_path)
    for name in IDENTITY:
        if (parent[name].dtype != child[name].dtype or
                not np.array_equal(parent[name], child[name])):
            raise RuntimeError(f"parent/child identity mismatch: {name}")

    parent_margin = parent["own_cash"] - parent["rival_cash"]
    child_margin = child["own_cash"] - child["rival_cash"]
    delta_margin = child_margin - parent_margin
    delta_win = (child_margin > 0).astype(np.float64) - (parent_margin > 0)
    codes = sorted(map(int, np.unique(parent["opponent"])))
    masks = {OPPONENTS[code]: parent["opponent"] == code for code in codes}
    return {
        "status": "PASS",
        "schema": "native_job_npz_paired_eval_v1",
        "label": label,
        "games": len(parent_margin),
        "environment_seed_clusters": len(np.unique(parent["seed"])),
        "seed_start": int(parent["seed"].min()),
        "seed_end": int(parent["seed"].max()),
        "identity_exact": {name: True for name in IDENTITY},
        "parent_arrays": {"path": str(parent_path), "sha256": parent_sha},
        "child_arrays": {"path": str(child_path), "sha256": child_sha},
        "overall": _summary(parent_margin, child_margin),
        "by_opponent": {
            name: _summary(parent_margin[mask], child_margin[mask])
            for name, mask in masks.items()
        },
        "cluster_bootstrap_95": {
            "unit": "environment_seed",
            "samples": bootstrap_samples,
            "rng_seed": bootstrap_seed,
            "overall": _bootstrap(parent["seed"], delta_margin, delta_win,
                                  bootstrap_samples, bootstrap_seed),
            "by_opponent": {
                name: _bootstrap(parent["seed"][mask], delta_margin[mask],
                                 delta_win[mask], bootstrap_samples,
                                 bootstrap_seed)
                for name, mask in masks.items()
            },
        },
    }


def _write_exclusive(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", delete=False) as target:
            temporary = Path(target.name)
            json.dump(report, target, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.chmod(0o644)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite: {args.output}")
    report = build_report(args.parent, args.child, args.label,
                          args.bootstrap_samples, args.bootstrap_seed)
    _write_exclusive(args.output, report)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()

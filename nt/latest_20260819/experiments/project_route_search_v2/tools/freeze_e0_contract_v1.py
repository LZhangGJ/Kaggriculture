"""Freeze and validate the M0 E0 contract and cumulative seed panels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"
DERIVED_EVENT_BANK = PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz"
RECEIPT_PATH = PROJECT_DIR / "receipts" / "m0_contract_freeze_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _seed_hash(seeds: list[int]) -> str:
    array = np.asarray(seeds, dtype="<i4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ with stable member order and timestamps."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with zipfile.ZipFile(
            raw_handle,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in ("event_seeds", "weed_spawn", "shop_choice"):
                buffer = io.BytesIO()
                np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compresslevel=9)


def _expand_segments(contract: dict) -> dict[str, list[int]]:
    segments: dict[str, list[int]] = {}
    for name, spec in contract["seed_segments"].items():
        start = int(spec["start"])
        stop = int(spec["stop"])
        seeds = list(range(start, stop + 1))
        if len(seeds) != int(spec["count"]):
            raise ValueError(f"{name}: count does not match inclusive range")
        segments[name] = seeds
    names = list(segments)
    for index, left_name in enumerate(names):
        left = set(segments[left_name])
        for right_name in names[index + 1 :]:
            overlap = left.intersection(segments[right_name])
            if overlap:
                raise ValueError(
                    f"seed segments overlap: {left_name}, {right_name}: {sorted(overlap)}"
                )
    return segments


def _compose_panels(contract: dict, segments: dict[str, list[int]]) -> dict[str, list[int]]:
    panels: dict[str, list[int]] = {}
    for name, parts in contract["cumulative_panels"].items():
        panels[name] = [seed for part in parts for seed in segments[part]]
    expected_counts = {
        "E0_SEARCH_16": 16,
        "E0_PROMOTE_32": 32,
        "E0_AUDIT_128": 128,
        "E0_OFFICIAL_HOLDOUT_64": 64,
    }
    for name, count in expected_counts.items():
        if len(panels.get(name, [])) != count:
            raise ValueError(f"{name}: expected {count} seeds")
        if len(set(panels[name])) != count:
            raise ValueError(f"{name}: duplicate seeds")
    if panels["E0_PROMOTE_32"][:16] != panels["E0_SEARCH_16"]:
        raise ValueError("E0_PROMOTE_32 is not a fixed extension of E0_SEARCH_16")
    if panels["E0_AUDIT_128"][:32] != panels["E0_PROMOTE_32"]:
        raise ValueError("E0_AUDIT_128 is not a fixed extension of E0_PROMOTE_32")
    if set(panels["E0_AUDIT_128"]).intersection(
        panels["E0_OFFICIAL_HOLDOUT_64"]
    ):
        raise ValueError("official holdout overlaps the 128-seed search panel")
    return panels


def _validate_configs() -> dict[str, bool]:
    budgets = _json_load(CONFIG_DIR / "search_budgets_v1.json")
    rows = budgets["family_domain_budgets"]
    row_totals_ok = all(
        row["native"] + row["causal_hybrid"] == row["total"] for row in rows
    )
    native = sum(row["native"] for row in rows)
    hybrid = sum(row["causal_hybrid"] for row in rows)
    total = sum(row["total"] for row in rows)
    global_totals = budgets["global_domain_totals"]
    budget_total_ok = (
        row_totals_ok
        and native == global_totals["native"] == 66000
        and hybrid == global_totals["causal_hybrid"] == 94000
        and total
        == global_totals["total"]
        == budgets["initial_complete_candidate_total"]
        == 160000
    )
    generation_mix_ok = abs(sum(budgets["candidate_generation_mix"].values()) - 1.0) < 1e-12
    offspring_mix_ok = abs(sum(budgets["offspring_mix"].values()) - 1.0) < 1e-12

    families = _json_load(CONFIG_DIR / "family_domains_v1.json")["families"]
    family_ids_ok = [family["family_id"] for family in families] == list(range(8))
    family_names_ok = [family["name"] for family in families] == [
        row["family"] for row in rows
    ]

    coverage = _json_load(CONFIG_DIR / "species_capacity_and_coverage_v1.json")
    species_ok = coverage["crops"] == [
        "WHEAT",
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
        "MELON",
    ] and coverage["animals"] == ["GOOSE", "COW", "SHEEP"]

    validations = {
        "budget_totals_equal_160000": budget_total_ok,
        "candidate_generation_mix_sums_to_one": generation_mix_ok,
        "offspring_mix_sums_to_one": offspring_mix_ok,
        "family_ids_are_dense_zero_to_seven": family_ids_ok,
        "family_budget_rows_match_domain_config": family_names_ok,
        "species_lists_cover_all_official_crops_and_animals": species_ok,
    }
    failures = [name for name, passed in validations.items() if not passed]
    if failures:
        raise ValueError(f"config validation failed: {failures}")
    return validations


def freeze(event_bank: Path, output: Path) -> dict:
    event_bank = event_bank.resolve()
    output = output.resolve()
    contract_path = CONFIG_DIR / "e0_contract_v1.json"
    contract = _json_load(contract_path)
    segments = _expand_segments(contract)
    panels = _compose_panels(contract, segments)
    config_validations = _validate_configs()

    with np.load(event_bank, allow_pickle=False) as source:
        required_keys = {"event_seeds", "weed_spawn", "shop_choice"}
        if set(source.files) != required_keys:
            raise ValueError(f"unexpected event-bank keys: {source.files}")
        source_seeds = np.asarray(source["event_seeds"], dtype=np.int64)
        source_index = {int(seed): index for index, seed in enumerate(source_seeds)}
        ordered_segment_names = [
            "E0_CANONICAL_2",
            "S1_INITIAL_16",
            "S2_ADDITIONAL_16",
            "S3_ADDITIONAL_96",
            "OFFICIAL_HOLDOUT_64",
        ]
        derived_seeds = [seed for name in ordered_segment_names for seed in segments[name]]
        missing = [seed for seed in derived_seeds if seed not in source_index]
        if missing:
            raise ValueError(f"source event bank is missing seeds: {missing}")
        indices = np.asarray([source_index[seed] for seed in derived_seeds], dtype=np.int64)
        arrays = {
            "event_seeds": np.asarray(derived_seeds, dtype=np.int64),
            "weed_spawn": np.asarray(source["weed_spawn"])[indices],
            "shop_choice": np.asarray(source["shop_choice"])[indices],
        }

    _write_deterministic_npz(DERIVED_EVENT_BANK, arrays)
    manifest = {
        "contract_id": contract["contract_id"],
        "official_package_version": contract["official_package_version"],
        "source_event_bank": _repo_relative(event_bank),
        "source_event_bank_sha256": _sha256(event_bank),
        "derived_event_bank": _repo_relative(DERIVED_EVENT_BANK),
        "derived_event_bank_sha256": _sha256(DERIVED_EVENT_BANK),
        "derived_event_shapes": {
            name: list(array.shape) for name, array in arrays.items()
        },
        "segments": {
            name: {
                "count": len(seeds),
                "minimum": min(seeds),
                "maximum": max(seeds),
                "seed_sha256": _seed_hash(seeds),
                "seeds": seeds,
            }
            for name, seeds in segments.items()
        },
        "panels": {
            name: {
                "count": len(seeds),
                "seed_sha256": _seed_hash(seeds),
                "segments": contract["cumulative_panels"][name],
                "seeds": seeds,
            }
            for name, seeds in panels.items()
        },
        "validations": {
            "all_primary_segments_pairwise_disjoint": True,
            "search_16_prefix_of_promote_32": True,
            "promote_32_prefix_of_audit_128": True,
            "official_64_disjoint_from_audit_128": True,
            "all_selected_seeds_exist_in_source_event_bank": True,
        },
    }
    _json_write(output, manifest)

    hashed_inputs = [
        REPO_ROOT / contract["design_document"],
        PROJECT_DIR / "README.md",
        PROJECT_DIR / "pyproject.toml",
        CONFIG_DIR / "e0_contract_v1.json",
        CONFIG_DIR / "family_domains_v1.json",
        CONFIG_DIR / "species_capacity_and_coverage_v1.json",
        CONFIG_DIR / "performance_gates_v1.json",
        CONFIG_DIR / "search_budgets_v1.json",
        output,
        PROJECT_DIR / "src" / "project_route_search_v2" / "__init__.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "constants.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "schema.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "null_opponent.py",
        Path(__file__).resolve(),
        event_bank,
        DERIVED_EVENT_BANK,
    ]
    hashed_inputs.extend(sorted((PROJECT_DIR / "tests").glob("test_*.py")))
    receipt = {
        "receipt_id": "M0_CONTRACT_FREEZE_V1",
        "contract_id": contract["contract_id"],
        "status": "PASS",
        "scope": "M0_CONTRACT_AND_EMPTY_PACKAGE_ONLY",
        "not_claimed": [
            "project_executor_implemented",
            "route_search_run",
            "high_potential_route_found",
            "competitive_strength_validated",
        ],
        "seed_validations": manifest["validations"],
        "config_validations": config_validations,
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
        },
        "files": {
            _repo_relative(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in hashed_inputs
        },
    }
    _json_write(RECEIPT_PATH, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze(args.event_bank, args.output)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

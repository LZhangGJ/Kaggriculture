"""Run a small crop-only M2.6 search smoke and verify route diversity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for source_dir in (
    PROJECT_DIR / "src", REPO_ROOT / "gpu_sim" / "src",
    REPO_ROOT / "experiments" / "strategic_v5" / "src",
):
    sys.path.insert(0, str(source_dir))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kaggriculture_jax.state import load_event_bank, load_tables  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from project_route_search_v2.m26_genome import validate_m26_crop_genome_v2  # noqa: E402
from project_route_search_v2.m26_rollout import (  # noqa: E402
    initialize_m26_rollout_carry_v2, make_m26_crop_rollout_v2, summarize_m26_rollout_v2,
)
from project_route_search_v2.m26_search import sample_m26_crop_genomes_v2  # noqa: E402


def _signature(parts: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        value = np.ascontiguousarray(part)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--seeds-per-candidate", type=int, default=4)
    parser.add_argument("--sampler-seed", type=int, default=260618)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--event-bank", type=Path,
        default=PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_DIR / "receipts" / "m26_crop_search_smoke_v1.json",
    )
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    sampled = sample_m26_crop_genomes_v2(args.candidates, args.sampler_seed)
    validation = validate_m26_crop_genome_v2(sampled)
    genome = jax.tree.map(
        lambda value: jnp.repeat(value, args.seeds_per_candidate, axis=0), sampled
    )
    panel = json.loads((PROJECT_DIR / "configs" / "seed_panels_v1.json").read_text(encoding="utf-8"))
    base_seeds = np.asarray(
        panel["panels"]["E0_SEARCH_16"]["seeds"][: args.seeds_per_candidate], dtype=np.int64
    )
    seeds = np.tile(base_seeds, args.candidates)
    source_seeds, bank = load_event_bank(args.event_bank)
    source_index = {int(seed): index for index, seed in enumerate(source_seeds)}
    event_index = np.asarray([source_index[int(seed)] for seed in seeds], dtype=np.int32)
    events = Events(bank.weed_spawn[event_index], bank.shop_choice[event_index])
    carry = initialize_m26_rollout_carry_v2(jnp.asarray(seeds, dtype=jnp.int32), genome)
    rollout = jax.jit(make_m26_crop_rollout_v2())
    started = time.perf_counter()
    final, _ = rollout(carry, events, load_tables(), genome)
    jax.block_until_ready(final)
    elapsed = time.perf_counter() - started
    summary = jax.device_get(summarize_m26_rollout_v2(final))
    coverage = {name: np.asarray(value) for name, value in jax.device_get(final.coverage)._asdict().items()}
    metrics = {
        name: np.asarray(value)
        for name, value in jax.device_get(final.metrics)._asdict().items()
    }
    arrays = {name: np.asarray(value) for name, value in summary._asdict().items() if name != "coverage"}
    state = jax.device_get(final.environment_state)
    lanes = args.seeds_per_candidate
    signatures = []
    mean_banks = []
    for candidate in range(args.candidates):
        sl = slice(candidate * lanes, (candidate + 1) * lanes)
        signatures.append(_signature([
            arrays["final_bank"][sl], coverage["crop_plant_actions"][sl],
            coverage["crop_harvest_actions"][sl], coverage["layout_plant_actions"][sl],
            coverage["fertilizer_actions"][sl], coverage["sell_orders_by_product"][sl],
            np.asarray(state.tile_kind)[sl, 0], np.asarray(state.tile_crop)[sl, 0],
        ]))
        mean_banks.append(float(np.mean(arrays["final_bank"][sl])))
    unique_signatures = len(set(signatures))
    plant_total = np.sum(coverage["crop_plant_actions"], axis=0)
    checks = {
        "genome_contract_valid": not validation,
        "all_719_step_seasons_done": bool(np.all(arrays["done"])),
        "hard_failure_zero": bool(np.all(arrays["unexplained_failure_count"] == 0)),
        "same_day_water_zero": bool(np.all(arrays["plant_without_same_day_water"] == 0)),
        "terminal_shed_zero": bool(np.all(arrays["terminal_sellable_shed_value"] == 0)),
        "avoidable_liquidation_loss_zero": bool(np.all(arrays["avoidable_liquidation_loss"] == 0)),
        "all_five_crops_executed": bool(np.all(plant_total > 0)),
        "at_least_16_distinct_route_signatures": unique_signatures >= min(16, args.candidates),
    }
    diagnostic_fields = (
        "invalid_raw_action_count", "unexpected_pass_count", "effect_mismatch_count",
        "owner_inactive_count", "deadline_missed_count", "resource_unavailable_count",
        "unit_compiler_overlap_count", "market_compiler_overlap_count",
        "project_cap_hits", "obligation_cap_hits",
    )
    failed_lanes = np.flatnonzero(arrays["unexplained_failure_count"] > 0)
    best = np.argsort(np.asarray(mean_banks))[::-1][:10]
    receipt = {
        "receipt_id": "M26_SMALL_CROP_ONLY_SEARCH_SMOKE_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
        "candidate_count": args.candidates, "seeds_per_candidate": lanes,
        "full_seasons": len(seeds), "steps_per_season": 719,
        "sampler_seed": args.sampler_seed, "evaluation_seeds": base_seeds.tolist(),
        "compile_and_run_seconds": elapsed, "checks": checks,
        "unique_route_signatures": unique_signatures,
        "aggregate_crop_plant_actions": plant_total.astype(int).tolist(),
        "hard_diagnostic_totals": {
            field: int(np.sum(metrics[field])) for field in diagnostic_fields
        },
        "failed_lanes": [
            {
                "candidate_id": int(lane // lanes), "seed": int(seeds[lane]),
                "unexplained_failure_count": int(arrays["unexplained_failure_count"][lane]),
                "diagnostics": {
                    field: int(metrics[field][lane])
                    for field in diagnostic_fields if int(metrics[field][lane]) != 0
                },
            }
            for lane in failed_lanes.tolist()
        ],
        "mean_bank_distribution": {
            "minimum": float(np.min(mean_banks)), "median": float(np.median(mean_banks)),
            "maximum": float(np.max(mean_banks)),
        },
        "top10_diagnostic_only": [
            {"candidate_id": int(index), "mean_bank": mean_banks[index], "signature": signatures[index]}
            for index in best.tolist()
        ],
        "boundary": "SEARCH_PIPELINE_AND_ROUTE_DIVERSITY_SMOKE_NOT_ROUTE_QUALITY_OR_PROMOTION",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

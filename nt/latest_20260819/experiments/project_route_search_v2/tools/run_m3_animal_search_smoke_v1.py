"""Run a 256-season M3 animal-genome search and diversity smoke."""

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
    PROJECT_DIR / "src",
    REPO_ROOT / "gpu_sim" / "src",
    REPO_ROOT / "experiments" / "strategic_v5" / "src",
):
    sys.path.insert(0, str(source_dir))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kaggriculture_jax.state import load_event_bank, load_tables  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from project_route_search_v2.m3_genome import validate_m3_animal_genome_v2  # noqa: E402
from project_route_search_v2.m3_rollout import (  # noqa: E402
    initialize_m3_rollout_carry_v2,
    make_m3_animal_rollout_v2,
    summarize_m3_rollout_v2,
)
from project_route_search_v2.m3_search import sample_m3_animal_genomes_v2  # noqa: E402


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
    parser.add_argument("--sampler-seed", type=int, default=360618)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--event-bank",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "receipts" / "m3_animal_search_smoke_v1.json",
    )
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    sampled = sample_m3_animal_genomes_v2(
        args.candidates, seed=args.sampler_seed
    )
    validation = validate_m3_animal_genome_v2(sampled)
    genome = jax.tree.map(
        lambda value: jnp.repeat(value, args.seeds_per_candidate, axis=0),
        sampled,
    )
    panel = json.loads(
        (PROJECT_DIR / "configs" / "seed_panels_v1.json").read_text(encoding="utf-8")
    )
    base_seeds = np.asarray(
        panel["panels"]["E0_SEARCH_16"]["seeds"][: args.seeds_per_candidate],
        dtype=np.int64,
    )
    seeds = np.tile(base_seeds, args.candidates)
    source_seeds, bank = load_event_bank(args.event_bank)
    source_index = {int(seed): index for index, seed in enumerate(source_seeds)}
    event_index = np.asarray(
        [source_index[int(seed)] for seed in seeds], dtype=np.int32
    )
    events = Events(bank.weed_spawn[event_index], bank.shop_choice[event_index])
    carry = initialize_m3_rollout_carry_v2(
        jnp.asarray(seeds, dtype=jnp.int32), genome
    )
    rollout = jax.jit(make_m3_animal_rollout_v2())
    started = time.perf_counter()
    final, _ = rollout(carry, events, load_tables(), genome)
    jax.block_until_ready(final)
    elapsed = time.perf_counter() - started
    summary = jax.device_get(summarize_m3_rollout_v2(final))
    arrays = {
        name: np.asarray(value)
        for name, value in summary._asdict().items()
        if name != "coverage"
    }
    coverage = {
        name: np.asarray(value)
        for name, value in jax.device_get(final.coverage)._asdict().items()
    }
    metrics = {
        name: np.asarray(value)
        for name, value in jax.device_get(final.metrics)._asdict().items()
    }
    state = jax.device_get(final.environment_state)
    lanes = args.seeds_per_candidate
    signatures = []
    mean_banks = []
    for candidate in range(args.candidates):
        sl = slice(candidate * lanes, (candidate + 1) * lanes)
        signatures.append(
            _signature(
                [
                    arrays["final_bank"][sl],
                    coverage["build_actions"][sl],
                    coverage["purchase_orders"][sl],
                    coverage["feed_actions"][sl],
                    coverage["care_actions"][sl],
                    coverage["harvest_actions"][sl],
                    coverage["fertilizer_actions"][sl],
                    arrays["active_animals_by_species"][sl],
                    np.asarray(state.tile_animal)[sl, 0],
                ]
            )
        )
        mean_banks.append(float(np.mean(arrays["final_bank"][sl])))
    unique_signatures = len(set(signatures))
    species_actions = {
        field: np.sum(coverage[field], axis=0)
        for field in (
            "build_actions",
            "purchase_orders",
            "place_actions",
            "feed_actions",
            "care_actions",
            "harvest_actions",
            "fertilizer_actions",
        )
    }
    hard_summary_fields = (
        "unexplained_failure_count",
        "unplanned_animal_escape",
        "animal_capacity_loss",
        "feed_hard_deadline_miss",
        "care_bonus_forfeited_unexplained",
        "care_bonus_capacity_clipped_unexplained",
        "animal_bought_without_place_plan",
        "cow_sheep_pasture_conflict",
        "terminal_sellable_shed_value",
        "terminal_unit_inventory_value",
        "avoidable_liquidation_loss",
        "animals_stranded_in_shed_at_terminal",
        "animals_stranded_in_unit_inventory_at_terminal",
    )
    checks = {
        "genome_contract_valid": not validation,
        "all_719_step_seasons_done": bool(np.all(arrays["done"])),
        "all_hard_diagnostics_zero": all(
            int(np.sum(arrays[field])) == 0 for field in hard_summary_fields
        ),
        "all_species_build_purchase_place_feed_harvest": all(
            bool(np.all(species_actions[field] > 0))
            for field in (
                "build_actions",
                "purchase_orders",
                "place_actions",
                "feed_actions",
                "harvest_actions",
            )
        ),
        "care_and_fertilizer_paths_exercised": int(
            np.sum(species_actions["care_actions"])
        )
        > 0
        and int(np.sum(species_actions["fertilizer_actions"])) > 0,
        "at_least_16_distinct_route_signatures": unique_signatures
        >= min(16, args.candidates),
    }
    failed_candidates = []
    for candidate in range(args.candidates):
        sl = slice(candidate * lanes, (candidate + 1) * lanes)
        hard = {
            field: int(np.sum(arrays[field][sl])) for field in hard_summary_fields
        }
        if not any(hard.values()):
            continue
        lane_rows = []
        for offset in range(lanes):
            lane = candidate * lanes + offset
            lane_hard = {
                field: int(arrays[field][lane]) for field in hard_summary_fields
            }
            if any(lane_hard.values()):
                lane_rows.append(
                    {
                        "seed": int(seeds[lane]),
                        "final_bank": int(arrays["final_bank"][lane]),
                        "hard": lane_hard,
                    }
                )
        phases = int(np.asarray(sampled.phase_count)[candidate])
        failed_candidates.append(
            {
                "candidate_id": candidate,
                "hard": hard,
                "lanes": lane_rows,
                "genome": {
                    "phase_start_step": np.asarray(
                        sampled.phase_start_step[candidate, :phases]
                    ).astype(int).tolist(),
                    "animal_target": np.asarray(
                        sampled.animal_target[candidate, :phases]
                    ).astype(int).tolist(),
                    "land_target": np.asarray(
                        sampled.land_target[candidate, :phases]
                    ).astype(int).tolist(),
                    "hand_target": np.asarray(
                        sampled.hand_target[candidate, :phases]
                    ).astype(int).tolist(),
                    "animal_investment_stop_step": np.asarray(
                        sampled.animal_investment_stop_step[candidate]
                    ).astype(int).tolist(),
                    "animal_place_wave_size": np.asarray(
                        sampled.animal_place_wave_size[candidate]
                    ).astype(int).tolist(),
                    "feed_stock_horizon_days": np.asarray(
                        sampled.feed_stock_horizon_days[candidate]
                    ).astype(int).tolist(),
                    "care_policy": np.asarray(
                        sampled.care_policy[candidate]
                    ).astype(int).tolist(),
                    "maintenance_utilization_cap": float(
                        np.asarray(sampled.maintenance_utilization_cap)[candidate]
                    ),
                    "cash_floor": int(np.asarray(sampled.cash_floor)[candidate]),
                },
            }
        )
    best = np.argsort(np.asarray(mean_banks))[::-1][:10]
    receipt = {
        "receipt_id": "M3_ANIMAL_SEARCH_SMOKE_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "candidate_count": args.candidates,
        "seeds_per_candidate": lanes,
        "full_seasons": len(seeds),
        "steps_per_season": 719,
        "sampler_seed": args.sampler_seed,
        "evaluation_seeds": base_seeds.tolist(),
        "compile_and_run_seconds": elapsed,
        "checks": checks,
        "unique_route_signatures": unique_signatures,
        "aggregate_species_actions": {
            key: value.astype(int).tolist() for key, value in species_actions.items()
        },
        "hard_summary_totals": {
            field: int(np.sum(arrays[field])) for field in hard_summary_fields
        },
        "failed_candidate_diagnostics": failed_candidates,
        "metric_totals": {
            field: int(np.sum(value)) for field, value in metrics.items()
        },
        "mean_bank_distribution": {
            "minimum": float(np.min(mean_banks)),
            "median": float(np.median(mean_banks)),
            "maximum": float(np.max(mean_banks)),
        },
        "top10_diagnostic_only": [
            {
                "candidate_id": int(index),
                "mean_bank": mean_banks[index],
                "signature": signatures[index],
            }
            for index in best.tolist()
        ],
        "validation_errors": validation,
        "boundary": "SEARCH_PIPELINE_AND_ROUTE_DIVERSITY_SMOKE_NOT_ROUTE_QUALITY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

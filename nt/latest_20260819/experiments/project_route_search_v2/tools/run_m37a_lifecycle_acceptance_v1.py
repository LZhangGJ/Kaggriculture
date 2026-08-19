"""Accept M3.7A detailed diagnostics plus the small GPU aggregate carry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
for source in (
    PROJECT / "src",
    ROOT / "gpu_sim" / "src",
    ROOT / "experiments" / "strategic_v5" / "src",
):
    sys.path.insert(0, str(source))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kaggriculture_jax.state import reset  # noqa: E402
from project_route_search_v2.lifecycle import (  # noqa: E402
    initialize_project_controller_v2,
)
from project_route_search_v2.m25_rollout import (  # noqa: E402
    empty_m25_metrics_v2,
)
from project_route_search_v2.m26_rollout import (  # noqa: E402
    empty_m26_coverage_v2,
)
from project_route_search_v2.m3_rollout import (  # noqa: E402
    empty_m3_coverage_v2,
    empty_m3_metrics_v2,
)
from project_route_search_v2.m35_rollout import (  # noqa: E402
    empty_m35_flow_v2,
)
from project_route_search_v2.m35_schema import M35RolloutCarryV2  # noqa: E402
from project_route_search_v2.m36_replay_calendar import (  # noqa: E402
    compile_gold_replay_calendar_v3,
)
from project_route_search_v2.m36_rollout import (  # noqa: E402
    M36CRolloutCarryV3,
    _empty_aggregates,
)
from project_route_search_v2.m37_lifecycle import (  # noqa: E402
    NUM_M37_BLOCKERS_V1,
    empty_m37_lifecycle_aggregates_v1,
    update_m37_daily_lifecycle_v1,
)


DEFAULT_REPLAY = (
    ROOT
    / "replay"
    / "gold_top20_latest_2026-08-18_082723"
    / "latest_per_gold"
    / "episode-94051618-replay.json"
)
DEFAULT_DETAILED = PROJECT / "receipts" / "m37a_detailed_lifecycle_v4.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--detailed", type=Path, default=DEFAULT_DETAILED)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "receipts" / "m37a_lifecycle_acceptance_v1.json",
    )
    return parser.parse_args()


def make_carry(states) -> M36CRolloutCarryV3:
    batch_size = states.step.shape[0]
    farm = M35RolloutCarryV2(
        states,
        initialize_project_controller_v2(states, 1),
        empty_m25_metrics_v2(batch_size),
        empty_m3_metrics_v2(batch_size),
        empty_m26_coverage_v2(batch_size),
        empty_m3_coverage_v2(batch_size),
        empty_m35_flow_v2(batch_size),
    )
    return M36CRolloutCarryV3(farm, _empty_aggregates(batch_size))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    detailed = json.loads(args.detailed.read_text(encoding="utf-8"))
    calendar_one, calendar_diagnostic = compile_gold_replay_calendar_v3(
        replay, player=1, candidate_id=3_702
    )
    calendar = jax.tree.map(
        lambda value: jnp.repeat(value, args.batch_size, axis=0), calendar_one
    )
    seeds = jnp.arange(args.batch_size, dtype=jnp.int32) + jnp.int32(19_052_633)
    states = jax.vmap(reset)(seeds)
    carry = make_carry(states)
    aggregate = empty_m37_lifecycle_aggregates_v1(args.batch_size)
    update = jax.jit(
        lambda current, life, plan: update_m37_daily_lifecycle_v1(
            current, life, plan, player=1
        )
    )
    started = time.perf_counter()
    first = update(carry, aggregate, calendar)
    jax.block_until_ready(first)
    first_elapsed = time.perf_counter() - started
    duplicate = update(carry, first, calendar)
    jax.block_until_ready(duplicate)
    day_two_states = states._replace(
        step=jnp.full((args.batch_size,), 24, dtype=jnp.int16)
    )
    day_two = update(make_carry(day_two_states), duplicate, calendar)
    jax.block_until_ready(day_two)

    host_first, host_duplicate, host_day_two = jax.device_get(
        (first, duplicate, day_two)
    )
    leaves = jax.tree.leaves(host_day_two)
    bytes_total = int(sum(np.asarray(value).nbytes for value in leaves))
    field_shapes = {
        name: list(np.asarray(getattr(host_day_two, name)).shape)
        for name in host_day_two._fields
    }
    detailed_checks = detailed.get("checks", {})
    checks = {
        "official_version_1327": replay.get("module_version") == "1.32.7",
        "detailed_status_pass": detailed.get("status") == "PASS",
        "detailed_all_stages_present": bool(
            detailed_checks.get("all_required_stages_present")
        ),
        "detailed_stages_monotonic": bool(
            detailed_checks.get("all_observed_stage_steps_monotonic")
        ),
        "detailed_trace_is_719_actions": bool(
            detailed_checks.get("trace_is_719_actions")
        ),
        "detailed_trace_excluded_from_gpu_carry": bool(
            detailed_checks.get("detailed_trace_not_in_gpu_carry")
        ),
        "gpu_aggregate_leaf_rank_at_most_2": all(
            np.asarray(value).ndim <= 2 for value in leaves
        ),
        "gpu_aggregate_has_no_719_axis": all(
            719 not in np.asarray(value).shape for value in leaves
        ),
        "blocker_vocabulary_count_10": int(
            host_day_two.blocker_day_counts.shape[-1]
        )
        == NUM_M37_BLOCKERS_V1,
        "same_day_update_idempotent": all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                jax.tree.leaves(host_first),
                jax.tree.leaves(host_duplicate),
                strict=True,
            )
        ),
        "next_day_debt_age_advances": bool(
            np.max(np.asarray(host_day_two.max_crop_debt_age_days)) >= 2
        ),
        "calendar_event_overflow_zero": int(
            calendar_diagnostic.event_overflow_count
        )
        == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "receipt_id": "M37A_LIFECYCLE_ACCEPTANCE_V1",
        "status": status,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "official_version": replay.get("module_version"),
        "batch_size": args.batch_size,
        "aggregate_bytes_total": bytes_total,
        "aggregate_bytes_per_environment": bytes_total / args.batch_size,
        "aggregate_field_shapes": field_shapes,
        "first_compile_and_execute_seconds": first_elapsed,
        "detailed_receipt": args.detailed.resolve().relative_to(ROOT.resolve()).as_posix(),
        "checks": checks,
        "raw_replay_action_payload_stored": False,
        "calendar_schema_changed": False,
        "hard_error_gate_relaxed": False,
        "reproduce_command": (
            "wsl.exe -d Ubuntu-24.04 -- bash -lc \"cd "
            "/mnt/e/ai_coding/kaggle/kaggriculture && "
            "/home/mitubant/kaggriculture-envs/jax0101-cuda12/bin/python "
            "experiments/project_route_search_v2/tools/"
            "run_m37a_lifecycle_acceptance_v1.py --require-gpu "
            f"--batch-size {args.batch_size} --output "
            "experiments/project_route_search_v2/receipts/"
            "m37a_lifecycle_acceptance_v1.json\""
        ),
        "boundary": "M37A_DIAGNOSTIC_INFRASTRUCTURE_ONLY_NOT_ECONOMIC_ACCEPTANCE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

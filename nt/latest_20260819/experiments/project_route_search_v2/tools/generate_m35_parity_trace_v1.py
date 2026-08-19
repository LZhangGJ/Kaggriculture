"""Generate all M3.5 branch trajectories for official 1.32.7 replay."""

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
from kaggriculture_jax.types import Action, Events, State  # noqa: E402
from project_route_search_v2.m35_candidates import m35_branch_coverage_panel_v2  # noqa: E402
from project_route_search_v2.m35_rollout import (  # noqa: E402
    initialize_m35_rollout_carry_v2,
    make_m35_farm_rollout_v2,
    summarize_m35_rollout_v2,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--event-bank",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "traces" / "m35_parity_trace_v1.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "receipts" / "m35_parity_trace_generation_v1.json",
    )
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    genome, names = m35_branch_coverage_panel_v2()
    panel = json.loads(
        (PROJECT_DIR / "configs" / "seed_panels_v1.json").read_text(encoding="utf-8")
    )
    seeds = np.asarray(
        panel["panels"]["E0_AUDIT_128"]["seeds"][: len(names)], dtype=np.int64
    )
    source_seeds, bank = load_event_bank(args.event_bank)
    source_index = {int(seed): index for index, seed in enumerate(source_seeds)}
    event_indices = np.asarray(
        [source_index[int(seed)] for seed in seeds], dtype=np.int32
    )
    events = Events(bank.weed_spawn[event_indices], bank.shop_choice[event_indices])
    tables = load_tables()
    payload: dict[str, np.ndarray] = {
        "seeds": seeds,
        "candidate_indices": np.arange(len(names), dtype=np.int32),
        "candidate_names": np.asarray(names, dtype=f"U{max(map(len, names))}"),
    }
    timings = []
    for seat in (0, 1):
        carry = initialize_m35_rollout_carry_v2(
            jnp.asarray(seeds, dtype=jnp.int32), genome, player=seat
        )
        rollout = jax.jit(make_m35_farm_rollout_v2(player=seat, trace=True))
        started = time.perf_counter()
        final, (actions, trajectory, _) = rollout(carry, events, tables, genome)
        jax.block_until_ready(final)
        timings.append(time.perf_counter() - started)
        actions, trajectory, summary = jax.device_get(
            (actions, trajectory, summarize_m35_rollout_v2(final, player=seat))
        )
        for name, value in zip(Action._fields, actions, strict=True):
            payload[f"seat{seat}_action_{name}"] = np.asarray(value)
        for name, value in zip(State._fields, trajectory, strict=True):
            payload[f"seat{seat}_state_{name}"] = np.asarray(value)
        payload[f"seat{seat}_summary_final_bank"] = np.asarray(summary.final_bank)

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.trace, **payload)
    receipt = {
        "receipt_id": "M35_ALL_BRANCH_PARITY_TRACE_GENERATION_V1",
        "status": "PASS",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "candidate_indices": list(range(len(names))),
        "candidate_names": list(names),
        "seeds": seeds.tolist(),
        "seats": [0, 1],
        "steps": 719,
        "compile_and_execute_seconds_by_seat": timings,
        "trace": args.trace.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "trace_sha256": _sha(args.trace),
        "trace_bytes": args.trace.stat().st_size,
        "boundary": "ALL_M35_FEED_AND_RECYCLE_BRANCHES_NOT_EXHAUSTIVE_GENOME_PARITY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

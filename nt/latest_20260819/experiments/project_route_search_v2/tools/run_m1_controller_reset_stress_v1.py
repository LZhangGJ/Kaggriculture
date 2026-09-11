"""Run the M1 1,000-episode batched controller carry contamination audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
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

from kaggriculture_jax.state import reset  # noqa: E402
from project_route_search_v2.lifecycle import (  # noqa: E402
    controller_equal_per_batch_v2,
    initialize_project_controller_v2,
    snapshot_project_controller_v2,
)
from project_route_search_v2.scan_carry import (  # noqa: E402
    advance_controller_scan_carry_v2,
    initialize_controller_scan_carry_v2,
    poison_controller_for_reset_test_v2,
)
from project_route_search_v2.schema import ControllerScanCarryV2  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def make_stress_function(
    *,
    batch_size: int,
    episodes: int,
    steps_per_episode: int,
    player: int,
    base_seed: int,
):
    lane = jnp.arange(batch_size, dtype=jnp.int32)
    all_episode_start = jnp.ones((batch_size,), dtype=jnp.bool_)
    step_indices = jnp.arange(steps_per_episode, dtype=jnp.int16)

    def state_for_episode(episode_index):
        seeds = jnp.asarray(base_seed, dtype=jnp.int32) + (
            episode_index.astype(jnp.int32) * batch_size + lane
        )
        states = jax.vmap(reset)(seeds)
        episode_money = jnp.asarray(3000, dtype=jnp.int32) + episode_index.astype(
            jnp.int32
        )
        return states._replace(
            money=states.money.at[:, player].set(episode_money)
        )

    initial_states = state_for_episode(jnp.asarray(0, dtype=jnp.int32))
    initial_carry = initialize_controller_scan_carry_v2(initial_states, player)
    initial_carry = initial_carry._replace(
        controller=poison_controller_for_reset_test_v2(initial_carry.controller)
    )

    def run():
        def one_episode(carry, episode_index):
            episode_states = state_for_episode(episode_index)
            expected_fresh = initialize_project_controller_v2(
                episode_states, player
            )
            prior_dirty = ~controller_equal_per_batch_v2(
                carry.controller, expected_fresh
            )
            reset_carry, reset_diag = advance_controller_scan_carry_v2(
                carry,
                episode_states,
                all_episode_start,
                player,
            )
            reset_clean = controller_equal_per_batch_v2(
                reset_carry.controller, expected_fresh
            )

            def one_step(step_carry, step_index):
                states = episode_states._replace(
                    step=jnp.full((batch_size,), step_index, dtype=jnp.int16),
                    money=episode_states.money.at[:, player].set(
                        episode_states.money[:, player]
                        + step_index.astype(jnp.int32)
                    ),
                    shed=episode_states.shed.at[:, player, 0].set(
                        (step_index % 97).astype(jnp.int16)
                    ),
                    seeds=episode_states.seeds.at[:, player, 0].set(
                        (step_index % 23).astype(jnp.int16)
                    ),
                    tile_yield=episode_states.tile_yield.at[
                        :, player, 0, 0
                    ].set(step_index.astype(jnp.int16)),
                )
                controller = snapshot_project_controller_v2(
                    states, step_carry.controller, player
                )
                return ControllerScanCarryV2(states, controller), None

            final_carry, _ = jax.lax.scan(
                one_step,
                reset_carry,
                step_indices,
            )
            expected_final = snapshot_project_controller_v2(
                final_carry.environment_state,
                expected_fresh,
                player,
            )
            snapshot_clean = controller_equal_per_batch_v2(
                final_carry.controller, expected_final
            )
            poisoned = poison_controller_for_reset_test_v2(
                final_carry.controller
            )
            next_carry = final_carry._replace(controller=poisoned)
            metrics = (
                prior_dirty.astype(jnp.int32),
                (~reset_clean).astype(jnp.int32),
                reset_diag.cross_episode_contamination.astype(jnp.int32),
                (~snapshot_clean).astype(jnp.int32),
            )
            return next_carry, metrics

        final_carry, metrics = jax.lax.scan(
            one_episode,
            initial_carry,
            jnp.arange(episodes, dtype=jnp.int32),
        )
        return final_carry, tuple(jnp.sum(value) for value in metrics)

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps-per-episode", type=int, default=719)
    parser.add_argument("--player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--base-seed", type=int, default=910000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "receipts" / "m1_controller_reset_stress_v1.json",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.episodes <= 0 or args.steps_per_episode <= 0:
        parser.error("batch-size, episodes, and steps-per-episode must be positive")

    stress = make_stress_function(
        batch_size=args.batch_size,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        player=args.player,
        base_seed=args.base_seed,
    )
    compiled = jax.jit(stress)
    compile_start = time.perf_counter()
    _, first_metrics = compiled()
    jax.block_until_ready(first_metrics)
    compile_and_first_seconds = time.perf_counter() - compile_start

    warm_start = time.perf_counter()
    _, second_metrics = compiled()
    jax.block_until_ready(second_metrics)
    warm_seconds = time.perf_counter() - warm_start

    metric_values = [int(np.asarray(value)) for value in second_metrics]
    prior_dirty, reset_mismatch, contamination, snapshot_mismatch = metric_values
    expected_dirty = args.batch_size * args.episodes
    passed = (
        prior_dirty == expected_dirty
        and reset_mismatch == 0
        and contamination == 0
        and snapshot_mismatch == 0
    )
    source_paths = [
        REPO_ROOT
        / "gpt_review"
        / "gpt"
        / "KAGGRICULTURE_E0_THREE_LAYER_PROJECT_SEARCH_DESIGN_V1_1_ZH.md",
        PROJECT_DIR / "pyproject.toml",
        PROJECT_DIR / "src" / "project_route_search_v2" / "constants.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "schema.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "lifecycle.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "scan_carry.py",
        PROJECT_DIR / "tests" / "test_episode_reset.py",
        Path(__file__).resolve(),
        PROJECT_DIR / "receipts" / "m0_contract_freeze_v1.json",
    ]
    receipt = {
        "receipt_id": f"M1_CONTROLLER_RESET_STRESS_V1_SEAT{args.player}",
        "status": "PASS" if passed else "FAIL",
        "scope": "M1_CONTROLLER_RESET_RECONCILE_AND_SCAN_CARRY",
        "configuration": {
            "batch_size": args.batch_size,
            "episodes": args.episodes,
            "steps_per_episode": args.steps_per_episode,
            "player": args.player,
            "base_seed": args.base_seed,
            "controller_steps": args.batch_size
            * args.episodes
            * args.steps_per_episode,
        },
        "results": {
            "poisoned_prior_lane_count": prior_dirty,
            "expected_poisoned_prior_lane_count": expected_dirty,
            "post_reset_mismatch_count": reset_mismatch,
            "cross_episode_contamination_count": contamination,
            "post_full_episode_snapshot_mismatch_count": snapshot_mismatch,
        },
        "timing_seconds": {
            "jit_compile_and_first_run": compile_and_first_seconds,
            "warm_run": warm_seconds,
        },
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "devices": [str(device) for device in jax.devices()],
            "backend": jax.default_backend(),
        },
        "not_claimed": [
            "crop_project_lifecycle_complete",
            "animal_project_lifecycle_complete",
            "simulator_throughput_measured",
            "route_search_run",
        ],
        "files": {
            _relative(path): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        },
    }
    _write_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

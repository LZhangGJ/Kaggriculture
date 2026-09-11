#!/usr/bin/env python3
"""Daily public-state diagnosis for FC22 against the frozen local PRT.

The output canonicalizes FC22 as the own side in both seat orientations and
stores only decision-time public rival facts plus FC22's own private state.
Opponent private shed/inventory is deliberately excluded from per-game rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/fusion_champion_v1/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from diagnose_pair_trajectories import (  # noqa: E402
    extract_snapshot,
    serialize_public_per_game,
    summarize_snapshots,
)
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc22_feed_value_guard_player_action_v1,
    initialize_fusion_champion_feed_value_carry_v1,
)
from kaggriculture_jax.constants import ANIMALS, CROPS, PRODUCTS  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    build_router_arrays,
    load_bank,
)
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


SNAPSHOT_AFTER_ACTION = tuple(sorted(set((0, 7, 15) + tuple(range(23, 719, 24)) + (718,))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=594001)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "experiments/fusion_champion_v1/receipts/"
            "fc22_prt_daily_trajectory_seed594001_n128x2_v1.json"
        ),
    )
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    old_receipt = json.loads(
        (
            ROOT
            / "experiments/expert_business_agent_v2/receipts/"
            "jax_full37_mixed_exact_proxy_bank_v1.json"
        ).read_text(encoding="utf-8")
    )
    resources = {
        "old_bank": load_bank(
            ROOT
            / "experiments/expert_business_agent_v2/artifacts/"
            "jax_full37_mixed_exact_proxy_bank_v1.npz"
        ),
        "latest_bank": load_bank(
            ROOT
            / "experiments/expert_business_agent_v2/artifacts/"
            "latest_public8_route_bank_v1.npz"
        ),
        "runtime": load_high_potential_runtime_tables_v1(
            ROOT
            / "experiments/expert_business_agent_v2/artifacts/"
            "latest_public8_runtime_tables_v1.npz"
        ),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    names = [row["name"] for row in rr.ROSTER]
    opponent_id = names.index("local_prt_v6")
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    simulator = rr.make_simulator_step(resources["tables"])
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry):
            return fc22_feed_value_guard_player_action_v1(
                states,
                resources["tables"],
                resources["latest_bank"],
                resources["old_bank"],
                resources["runtime"],
                carry,
                player,
            )

        return policy

    orientation_money: list[np.ndarray] = []
    orientation_snapshots: list[list[tuple[int, dict[str, np.ndarray]]]] = []
    started = perf_counter()
    for candidate_seat in (0, 1):
        states = jax.vmap(reset)(jnp.asarray(seeds))
        candidate_carry = initialize_fusion_champion_feed_value_carry_v1(len(seeds))
        opponent_carry = rr.initialize_agent_carry(
            opponent_id, len(seeds), resources["router"]
        )
        candidate_policy = make_candidate_policy(candidate_seat)
        opponent_policy = rr.make_agent_policy(
            opponent_id, 1 - candidate_seat, **resources
        )
        snapshots: list[tuple[int, dict[str, np.ndarray]]] = []
        for action_index in range(719):
            candidate_action, candidate_carry = candidate_policy(
                states, candidate_carry
            )
            opponent_action, opponent_carry = opponent_policy(
                states, opponent_carry
            )
            states = (
                simulator(states, candidate_action, opponent_action, events)
                if candidate_seat == 0
                else simulator(states, opponent_action, candidate_action, events)
            )
            if action_index in SNAPSHOT_AFTER_ACTION:
                snapshot = extract_snapshot(states, own=candidate_seat)
                k320 = candidate_carry.base.base.k320
                snapshot.update(
                    {
                        "route_own": np.asarray(
                            jax.device_get(k320.ray_route_id), dtype=np.int16
                        ),
                        "route_rival": np.full(
                            (len(seeds),), opponent_id, dtype=np.int16
                        ),
                        "route_locked_own": np.asarray(
                            jax.device_get(k320.ray_route_locked), dtype=bool
                        ),
                        "route_locked_rival": np.ones((len(seeds),), dtype=bool),
                        "route_family_own": np.full(
                            (len(seeds),), -1, dtype=np.int8
                        ),
                        "route_family_rival": np.full(
                            (len(seeds),), -1, dtype=np.int8
                        ),
                        "sheep_pressure_own": np.asarray(
                            jax.device_get(candidate_carry.base.base.sheep_pressure),
                            dtype=bool,
                        ),
                        "wheat_credit_own": np.asarray(
                            jax.device_get(candidate_carry.wheat_credit),
                            dtype=np.int16,
                        ),
                    }
                )
                snapshots.append((action_index, snapshot))
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError("not all games DONE")
        if any(
            int(np.sum(np.asarray(getattr(terminal, name))))
            for name in ("hand_cap_hits", "market_loop_cap_hits", "price_lut_oob")
        ):
            raise AssertionError("simulator safety counter hit")
        orientation_money.append(np.asarray(terminal.money, dtype=np.int64))
        orientation_snapshots.append(snapshots)

    first, second = orientation_money
    candidate_cash = np.concatenate((first[:, 0], second[:, 1]))
    opponent_cash = np.concatenate((first[:, 1], second[:, 0]))
    margins = candidate_cash - opponent_cash
    combined_snapshots = []
    for (step0, snapshot0), (step1, snapshot1) in zip(
        orientation_snapshots[0], orientation_snapshots[1], strict=True
    ):
        if step0 != step1:
            raise AssertionError("snapshot step mismatch")
        combined_snapshots.append(
            (
                step0,
                {
                    key: np.concatenate((snapshot0[key], snapshot1[key]), axis=0)
                    for key in snapshot0
                },
            )
        )

    per_game = serialize_public_per_game(combined_snapshots, margins, seeds)
    # Add the two FC22-only diagnostics omitted by the generic serializer.
    for game_index, game in enumerate(per_game):
        for day_index, (_, snapshot) in enumerate(combined_snapshots):
            game["daily"][day_index]["route_locked_own_diagnostic"] = bool(
                snapshot["route_locked_own"][game_index]
            )
            game["daily"][day_index]["sheep_pressure_own_diagnostic"] = bool(
                snapshot["sheep_pressure_own"][game_index]
            )
            game["daily"][day_index]["wheat_credit_own_diagnostic"] = int(
                snapshot["wheat_credit_own"][game_index]
            )

    payload = {
        "schema": "kaggriculture.fc22.prt_daily_trajectory.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "truth_boundary": (
            "rival public board and market only; no rival private inventory or identity "
            "is available to an online decision"
        ),
        "crop_order": list(CROPS),
        "animal_order": list(ANIMALS),
        "product_order": list(PRODUCTS),
        "seed_start": int(seeds[0]),
        "seed_count": int(seeds.size),
        "games": int(margins.size),
        "wins": int(np.sum(margins > 0)),
        "losses": int(np.sum(margins < 0)),
        "score_rate": float(np.mean(margins > 0)),
        "mean_margin": float(np.mean(margins)),
        "daily_summary": summarize_snapshots(combined_snapshots, margins),
        "per_game": per_game,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "games": payload["games"],
                "wins": payload["wins"],
                "score_rate": payload["score_rate"],
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

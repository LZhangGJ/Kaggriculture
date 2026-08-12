"""Measure pure-tensor Kaggriculture transition throughput."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from kaggriculture_lab.gpu_engine import (
    CudaKaggricultureEnv,
    GpuEngineConfig,
    M_BUY_PRODUCT,
    M_BUY_SEED,
    M_SELL,
    U_DIG,
    U_DROP,
    U_EAST,
    U_PICKUP,
    U_PLACE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--market-quantity", type=int, default=16)
    parser.add_argument(
        "--profile",
        choices=("pass", "move", "mixed", "interact", "pickup", "drop", "place", "market", "buyproduct", "sell"),
        default="move",
    )
    parser.add_argument("--hands", type=int, default=0, help="active hired hands per player")
    parser.add_argument("--warmup-steps", type=int, default=None, help="override full phase-specialization warmup")
    parser.add_argument("--repeats", type=int, default=1, help="timed repetitions reported by median")
    parser.add_argument("--no-triton", action="store_true", help="disable fused Triton interaction kernels")
    parser.add_argument("--no-compile-routing", action="store_true", help="skip torch.compile for faster cold start")
    parser.add_argument("--fuse-market-kernel", action="store_true", help="process every market order in one kernel")
    args = parser.parse_args()
    if not 0 <= args.hands <= 16:
        parser.error("--hands must be between 0 and 16")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    config = GpuEngineConfig(
        max_market_orders=2,
        max_market_quantity=args.market_quantity,
        starting_money=1_000_000_000 if args.profile == "buyproduct" else 3000,
        shed_capacity=100_000 if args.profile == "buyproduct" else 100,
        use_triton=not args.no_triton,
        compile_action_routing=not args.no_compile_routing,
        fuse_market_kernel=args.fuse_market_kernel,
    )
    env = CudaKaggricultureEnv(args.envs, device=args.device, config=config)
    actions = env.empty_actions()
    if args.profile in ("move", "market", "buyproduct", "sell"):
        actions.unit_ops[:, :, : args.hands + 1] = U_EAST
    elif args.profile == "mixed":
        actions.unit_ops[:, :, 0] = U_DIG
        actions.unit_ops[:, :, 1 : args.hands + 1] = U_EAST
    elif args.profile == "interact":
        actions.unit_ops[:, :, : args.hands + 1] = U_DIG
    elif args.profile == "pickup":
        actions.unit_ops[:, :, : args.hands + 1] = U_PICKUP
    elif args.profile == "drop":
        actions.unit_ops[:, :, : args.hands + 1] = U_DROP
    elif args.profile == "place":
        actions.unit_ops[:, :, : args.hands + 1] = U_PLACE
    if args.profile == "market":
        actions.market_ops[:, :, 0] = M_BUY_SEED
        actions.market_quantities[:, :, 0] = 1
    elif args.profile == "buyproduct":
        actions.market_ops[:, :, 0] = M_BUY_PRODUCT
        actions.market_quantities[:, :, 0] = args.market_quantity
    elif args.profile == "sell":
        actions.market_ops[:, :, 0] = M_SELL
        actions.market_quantities[:, :, 0] = args.market_quantity
    def prepare_state() -> None:
        if args.hands:
            env.state.hands_count.fill_(args.hands)
            env.state.unit_active[:, :, : args.hands + 1] = True
        if args.profile in ("pickup", "drop", "place"):
            env.state.positions[:, :, : args.hands + 1] = 4
        if args.profile == "pickup":
            env.state.shed[:, :, 0] = config.shed_capacity
        elif args.profile in ("drop", "place"):
            env.state.unit_inventory[:, :, : args.hands + 1, 0] = config.shed_capacity
        elif args.profile == "sell":
            env.state.shed[:, :, 0] = args.steps * args.market_quantity + 64

    prepare_state()
    # Exercise the no-demand, shop-demand, and town-center phases before timing;
    # Triton specializes these branches and caches each variant separately.
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = max(
            config.town_center_sell_interval + 1,
            config.turns_per_day * config.town_shop_unlock_interval + 1,
        )
    for _ in range(warmup_steps):
        env.step(actions)
    elapsed_samples = []
    for _ in range(args.repeats):
        env.reset()
        prepare_state()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.steps):
            env.step(actions)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_samples.append(time.perf_counter() - started)
    elapsed = statistics.median(elapsed_samples)
    joint_turns = args.envs * args.steps
    print(
        f"device={args.device} triton={env.use_triton} compiled_routing={config.compile_action_routing} "
        f"fused_market_kernel={config.fuse_market_kernel} profile={args.profile} "
        f"hands={args.hands} envs={args.envs} steps={args.steps} warmup={warmup_steps} repeats={args.repeats}"
    )
    if args.repeats > 1:
        print("sample_seconds=" + ",".join(f"{sample:.4f}" for sample in elapsed_samples))
    print(f"seconds={elapsed:.4f} joint_turns/s={joint_turns / elapsed:,.0f}")
    print(f"full_720_games/s={(joint_turns / elapsed) / 720:,.1f}")


if __name__ == "__main__":
    main()

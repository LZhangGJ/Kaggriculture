"""Measure pure-tensor Kaggriculture transition throughput."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--no-triton", action="store_true", help="disable fused Triton interaction kernels")
    args = parser.parse_args()
    if not 0 <= args.hands <= 16:
        parser.error("--hands must be between 0 and 16")

    config = GpuEngineConfig(
        max_market_orders=2,
        max_market_quantity=args.market_quantity,
        starting_money=1_000_000_000 if args.profile == "buyproduct" else 3000,
        shed_capacity=100_000 if args.profile == "buyproduct" else 100,
        use_triton=not args.no_triton,
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
    warmup_steps = max(
        config.town_center_sell_interval + 1,
        config.turns_per_day * config.town_shop_unlock_interval + 1,
    )
    for _ in range(warmup_steps):
        env.step(actions)
    env.reset()
    prepare_state()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
        env.step(actions)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    joint_turns = args.envs * args.steps
    print(
        f"device={args.device} triton={env.use_triton} profile={args.profile} "
        f"hands={args.hands} envs={args.envs} steps={args.steps}"
    )
    print(f"seconds={elapsed:.4f} joint_turns/s={joint_turns / elapsed:,.0f}")
    print(f"full_720_games/s={(joint_turns / elapsed) / 720:,.1f}")


if __name__ == "__main__":
    main()

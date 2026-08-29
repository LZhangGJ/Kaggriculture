# Licensed under the Apache License, Version 2.0.
import argparse
import time
import numpy as np

from fast_kaggriculture import Config, FastBatchEnv, FastEnv, Op, Item


def timed(label, games, turns, fn):
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    print(f"{label:28s} {games:7d} games  {games/elapsed:10.1f} games/s  {games*turns/elapsed:12.0f} steps/s  ({elapsed:.3f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--official-episodes", type=int, default=5)
    args = ap.parse_args()
    turns = 719
    cfg = Config()
    pass_actions = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2

    def fast_raw():
        env = FastEnv(cfg, 0)
        for episode in range(args.episodes):
            env.reset_raw(episode)
            for _ in range(turns):
                env.step_raw(pass_actions)

    timed("C++ string/raw", args.episodes, turns, fast_raw)

    b = args.batch
    units = np.zeros((b, 2, 1, 3), dtype=np.int32)
    units[..., 0] = int(Op.PASS)
    units[..., 1] = int(Item.NONE)
    unit_counts = np.ones((b, 2), dtype=np.int32)
    market = np.zeros((b, 2, 1, 3), dtype=np.int32)
    market_counts = np.zeros((b, 2), dtype=np.int32)
    batch = FastBatchEnv(b, cfg, 0)

    def fast_batch():
        rounds = max(1, args.episodes // b)
        for r in range(rounds):
            batch.reset(list(range(r * b, (r + 1) * b)))
            for _ in range(turns):
                batch.step_packed(units, unit_counts, market, market_counts)

    batch_games = max(1, args.episodes // b) * b
    timed("C++ packed/OpenMP", batch_games, turns, fast_batch)

    segment_units = np.broadcast_to(units[:, None], (b, turns, 2, 1, 3)).copy()
    segment_uc = np.broadcast_to(unit_counts[:, None], (b, turns, 2)).copy()
    segment_market = np.broadcast_to(market[:, None], (b, turns, 2, 1, 3)).copy()
    segment_mc = np.broadcast_to(market_counts[:, None], (b, turns, 2)).copy()

    def fast_segment():
        rounds = max(1, args.episodes // b)
        for r in range(rounds):
            batch.reset(list(range(r * b, (r + 1) * b)))
            batch.run_packed_segment(segment_units, segment_uc, segment_market, segment_mc)

    timed("C++ segment/OpenMP", batch_games, turns, fast_segment)

    if args.official_episodes:
        from kaggle_environments import make

        def official():
            for episode in range(args.official_episodes):
                env = make("kaggriculture", configuration={"seed": episode})
                env.run([lambda obs: pass_actions[0], lambda obs: pass_actions[1]])

        timed("official Kaggle Python", args.official_episodes, turns, official)


if __name__ == "__main__":
    main()

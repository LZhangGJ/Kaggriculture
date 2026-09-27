"""Prepare official Python-RNG events before collection, for arbitrary seeds.

Conditional draws are indexed by the GPU simulator. No host RNG is called in a
turn. Call this ahead of the next rollout to overlap preparation with updates.
"""
import random
import numpy as np


def seed_events(seed):
    weed=np.empty((30,200),dtype=np.bool_)
    shops=np.empty((30,201),dtype=np.int8)
    clone=random.Random(0)
    for day in range(30):
        rng=random.Random((int(seed)*1_000_003)^day)
        for draws in range(201):
            clone.setstate(rng.getstate())
            shops[day,draws]=clone.randrange(8)
            if draws<200:weed[day,draws]=rng.random()<.005
    return weed,shops


def prepare_events(seeds,workers=1):
    if workers>1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=min(workers,len(seeds)),mp_context=multiprocessing.get_context('spawn')) as pool:
            rows=list(pool.map(seed_events,seeds,chunksize=4))
    else:rows=[seed_events(seed) for seed in seeds]
    return tuple(np.stack(v) for v in zip(*rows))

# Native Salemali7 2900 opponent

Offline-only C++ port of `opponents/salemali7_2900/main.py`. It reuses the
repository's typed simulator/action ABI and packs the authoritative 720-frame
tape once; rollout does not import Python or parse JSON.

Implemented semantics are exactly the source's fixed tape, hand alignment,
weed transaction, and one-step `MELON/MILK/STRAWBERRY/WOOL` front-run/debt
repayment. Asset decoding and non-monotonic calls fail closed.

```bash
PYTHONPATH=../../.. /root/miniforge3/envs/torch-npu/bin/python generate_assets.py
./build.sh
PYTHONPATH=../../.. /root/miniforge3/envs/torch-npu/bin/python check_parity.py \
  --seed-count 16 --output parity_report.json
PYTHONPATH=../../.. /root/miniforge3/envs/torch-npu/bin/python check_parity.py \
  --active-self --seed-count 16 --output parity_active_holdout.json
```

Do not add this opponent to the RL pool unless the frozen source hash, asset
hash, native build hash, both seats, all 719 steps, and disjoint holdout seeds
are recorded with zero action mismatches.

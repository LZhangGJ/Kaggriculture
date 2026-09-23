# Native v3 actor kernel B

Dependency-free float32 CPU fallback for the fixed action-event v3 actor. It
loads binary A from [`../native_student_actor/FORMAT.md`](../native_student_actor/FORMAT.md),
including its normalization tensors, and implements token means, embeddings,
PyTorch `GRUCell` gate order `r,z,n`, legal masking, softmax/log-probability,
and deterministic counter-based sampling.

```bash
OMP_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /root/miniforge3/envs/torch-npu/bin/python \
  experiments/native_student_actor/export_frozen_binary.py \
  --output work/student-v1/native-r3-parent.bin

OMP_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /root/miniforge3/envs/torch-npu/bin/python \
  experiments/native_student_v3/export_fixture.py \
  --output work/student-v1/native-r3-parent.fixture.bin --events 256

g++ -O3 -std=c++17 -ffp-contract=off -Wall -Wextra -Werror \
  experiments/native_student_v3/actor.cpp \
  experiments/native_student_v3/parity_bench.cpp \
  -o experiments/native_student_v3/parity_bench

experiments/native_student_v3/parity_bench \
  work/student-v1/native-r3-parent.bin \
  work/student-v1/native-r3-parent.fixture.bin 10000
```

R3 parent result on this host: 256 real rollout events pass with zero greedy
or counter-RNG mismatches; max absolute error is `6.92e-6` for logits and
`7.19e-6` for next hidden. Scalar kernel throughput is about 1,437 events/s
(`696 us/event`, one core). This directory deliberately has no environment
runner or production integration.

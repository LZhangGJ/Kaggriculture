# 最新公开 8 方案 JAX 逐个迁移最终验收

日期：2026-08-20
官方裁判：`kaggle-environments==1.32.7`
GPU：NVIDIA GeForce RTX 3090 24GB
结论：**8/8 PASS**

## 最终结果

| 最新页面身份 | 官方逐步一致 | 动态覆盖步 | transitions/s | games/s | 首编译+首跑 | JAX峰值 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| V111 \| 8C4S Economic Core Premium Lead | 4/4 | 382 | 243,423 | 338.6 | 22.5s | 149.6 MiB | PASS |
| V20-Adaptive-R1 \| Multi-Route Agent | 4/4 | 237 | 172,990 | 240.6 | 38.7s | 148.8 MiB | PASS |
| Kaggriculture 2026 V1 | 4/4 | 237 | 167,609 | 233.1 | 15.4s | 152.1 MiB | PASS |
| Kaggriculture Rank Your Agent | 4/4 | 303 | 164,774 | 229.2 | 47.4s | 152.3 MiB | PASS |
| 106/130 Multi-Generation \| v36 Robust Hybrid | 4/4 | 908 | 165,446 | 230.1 | 383.3s | 156.2 MiB | PASS |
| Kaggriculture X544 - Nah, I'd Win. | 16/16 | 850 | 176,841 | 246.0 | 43.8s | 156.4 MiB | PASS |
| 🌾Adaptive Farming Strategy for Kaggriculture | 4/4 | 237 | 179,971 | 250.3 | 14.1s | 160.7 MiB | PASS |
| Kaggriculture \| Multi-Route Farming Agent | 4/4 | 76 | 238,638 | 331.9 | 24.5s | 160.7 MiB | PASS |

说明：

- `官方逐步一致` 同时要求 719 个动作的全部字段、每帧公开/私有状态、终局现金/奖励一致，并做双座位。
- `动态覆盖步` 是官方 Python 动作不等于任何允许原始路线的步数，证明不是只迁移固定 tape。
- X562 额外验证 8 个种子 × 双座位，其中 4 个上下文进入 high route、12 个进入 low route。
- 2048 batch 使用 1024 套事件各复制一份；每个 Agent 都通过 lane 一致和同进程三次 fresh-reset 一致，未发现跨局污染。
- 显存列是 JAX 分配器可观测峰值，不含 CUDA 驱动上下文；物理 GPU 为 24GB，JAX 当前 WSL 分配器上限约 18GB。

## 关键工程结论

1. 8 个页面实际是 7 份唯一源码：Kunal 与 Boatlee `main.py` SHA-256 完全相同，因此共享精确 JAX 实现是正确复用，不是漏迁移。
2. Kaito V36 首次编译最重（约 383 秒），但稳态仍约 165k transitions/s；应依赖持久编译缓存，避免频繁改图。
3. Deniz/Flex 的图较轻，约 239k–243k transitions/s；其余复杂动态路线约 165k–180k/s。
4. 所有方案都存在实际运行时覆盖动作；未把“只迁移路线 tape”冒充完整迁移。
5. 历史 Exact5 五个身份重新执行官方双座位逐步回归，5/5 PASS；新增最新版 mode 没有覆盖旧语义。
6. 本报告不比较跨日期 Public 分数。被动对手现金只用于运行正确性和吞吐诊断，不能替代同事件 bank、双座位、当前对手池 Arena。

## 主要文件

- JAX 新控制器：`experiments/strategic_v5/src/strategic_v5/latest_public8_gpu.py`
- 共享动态控制器：`experiments/strategic_v5/src/strategic_v5/high_potential_v20_gpu.py`
- 路线 bank：`experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz`
- runtime tables：`experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz`
- 总机器凭据：`experiments/expert_business_agent_v2/receipts/latest_public8_jax_acceptance_v1.json`
- 历史 Exact5 回归：`experiments/expert_business_agent_v2/receipts/high_potential_exact5_historical_regression_after_latest8_v1.json`
- 每方案凭据：`experiments/expert_business_agent_v2/receipts/latest_public8_jax/`
- 每方案报告：`experiments/expert_business_agent_v2/reports/latest_public8_jax/`

## 结论边界

状态 `EXACT_PARITY_ACCEPTED_ON_FROZEN_CORPUS` 表示冻结验收语料上逐字段严格一致，
并不声称用有限随机种子数学证明全部可达状态。修改源码、路线 bank、runtime tables、
控制器或官方环境版本后，必须重跑全套验收。

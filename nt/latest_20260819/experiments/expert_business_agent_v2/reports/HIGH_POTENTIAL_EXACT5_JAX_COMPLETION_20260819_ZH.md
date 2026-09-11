# 五个高潜力公开方案 JAX 精确复现验收报告

日期：2026-08-19  
官方裁判版本：`kaggle-environments 1.32.7`  
GPU：RTX 3090 24GB  
结论：**PASS，可作为 GPU 冻结对手和离线路线搜索锚点。**

## 1. 本次复现对象

| 方案 | JAX 实现 | 核心动态能力 |
|---|---|---|
| Boatlee V20 | `high_potential_v20_player_action_v1 / MODE_BOATLEE` | 商店路由、布局回退、杂草恢复、仓库腾挪、卖单排序、短期 premium 抢卖 |
| Ray K320 | `high_potential_v20_player_action_v1 / MODE_RAY_K320` | 路由锁定、1/2/4 步预占与偿还、终局种子裁剪 |
| Tetsutani | `high_potential_v20_player_action_v1 / MODE_TETSUTANI` | 动态商店路由、premium queue 提前量 |
| Kaito V27 | `public_v27_player_action_exact_v1` | 固定经营路线、RC5 杂草恢复、源方案旧价格公式卖单排序 |
| Flex V59 | `public_rc5_weed_player_action_v1` | 固定经营路线、RC5 杂草恢复 |

五个方案共用十条去重后的原始动作路线，但动态控制器、内部状态和市场动作顺序均在 GPU 上执行；不是把官方 Python Agent 放在 CPU 上逐步调用。

## 2. 精确一致性验收

验收场景为同方案镜像对局。官方 Python 1.32.7 先运行双方完整 Agent；复验时保留一侧官方动作流，另一侧替换为 JAX 控制器。这样会真实触发对手观察、市场竞争、动态路由、预占/偿还和恢复分支。

覆盖：5 个方案 × 2 个独立种子 × 2 个座位 = 20 局；每局 719 个决策步，共 14,380 步。

| 方案 | 上下文 | 动作张量 | 每帧状态 | 终局奖励 | 结论 |
|---|---:|---:|---:|---:|---|
| Boatlee V20 | 4/4 | 100% | 100% | 100% | PASS |
| Ray K320 | 4/4 | 100% | 100% | 100% | PASS |
| Tetsutani | 4/4 | 100% | 100% | 100% | PASS |
| Kaito V27 | 4/4 | 100% | 100% | 100% | PASS |
| Flex V59 | 4/4 | 100% | 100% | 100% | PASS |

严格门槛是：每个单位动作、每个市场订单、全部公开/私有状态字段和终局奖励全部相同。只要一个字段不同，该局即失败。

关键修正：公开 Agent 对同一步多个卖单排序时使用了冻结的旧价格计算式，而非官方 1.32.7 价格表。两者有时现金相同但动作顺序不同。现已预计算源 Agent 的精确价格 LUT，Boatlee、Ray、Tetsutani 和 Kaito 均按源代码原样排序。

## 3. RTX 3090 批量吞吐

测试条件：2048 个环境、完整 719 步、固定静止对手；数字为 JIT 完成后的第二次 steady-state 运行，不含事件生成和首次编译。

| 方案 | 完整局/秒 | transitions/秒 | 全部完成 |
|---|---:|---:|---|
| Boatlee V20 | 256.3 | 184,283 | 是 |
| Ray K320 | 256.9 | 184,726 | 是 |
| Tetsutani | 266.7 | 191,780 | 是 |
| Kaito V27 | 359.3 | 258,362 | 是 |
| Flex V59 | 349.8 | 251,514 | 是 |

复杂的前三个动态控制器仍超过 18 万 transitions/s；较轻的 Kaito/Flex 超过 25 万 transitions/s。五个方案均满足大规模 GPU 搜索和批量对战要求。

## 4. 可复现实物

- JAX 动态控制器：`experiments/strategic_v5/src/strategic_v5/high_potential_v20_gpu.py`
- Kaito 精确控制器：`experiments/strategic_v5/src/strategic_v5/public_v25_gpu.py`
- 十条去重路线：`experiments/expert_business_agent_v2/artifacts/high_potential_route_bank_v1.npz`
- 动态运行表：`experiments/expert_business_agent_v2/artifacts/high_potential_runtime_tables_v1.npz`
- 官方镜像轨迹清单：`experiments/expert_business_agent_v2/receipts/high_potential_exact5_mirror_traces_v1.json`
- 严格一致性收据：`experiments/expert_business_agent_v2/receipts/high_potential_exact5_mirror_jax_parity_v1.json`
- 性能收据：`experiments/expert_business_agent_v2/reports/HIGH_POTENTIAL_EXACT5_GPU_BENCHMARK_B2048_20260819.json`

## 5. 结论边界

本验收证明这五个 JAX Agent 在选定的官方随机种子、双座位和动态镜像轨迹上逐步精确一致，并证明 2048 batch 的 GPU 吞吐。它不是对数学意义上全部可能状态的穷举证明；后续若修改官方环境、源 Agent、路线库或动态运行表，必须重新运行一致性验收。

下一阶段可直接把五个方案作为冻结对手池，在 GPU 上批量寻找高潜力路线；最终入选路线仍需用官方 Python 1.32.7 裁判复验。

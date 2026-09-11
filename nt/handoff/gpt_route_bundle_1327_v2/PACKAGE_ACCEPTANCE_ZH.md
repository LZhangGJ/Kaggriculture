# Kaggriculture 1.32.7 V2 交接包验收记录

日期：2026-08-15

状态：`PASS`

## 正式结果

| 检查 | 结果 | 证据 |
|---|---:|---|
| R6/R7 各 8,000 个原始候选 | PASS | 共 16,000 个；4→16→32→128 seed 漏斗完成 |
| 固定形状与重复编译 | PASS | 每阶段 JIT cache 始终为 1 |
| 清仓安全修复 | PASS | 只用搜索域 84101..84228 选出每族 5 条，不读取最终 holdout 收益 |
| 未见 seed 双座位 JAX holdout | PASS | 10 候选 × 64 seed × 2 seat = 1,280 局 |
| 官方 Python 1.32.7 复验 | PASS | 1,280/1,280 局完成；请求 seed 与实际 seed 全部一致 |
| JAX / 官方终局现金一致 | PASS | 逐局完全一致；最大绝对误差 0 |
| 最终拳法库 | PASS | R6 牛奶 3 条、R7 羊毛 3 条；各含两种生成方向 |
| Full-core 回归 | PASS | `test_e4_full_core.py`：19 passed |

官方收据：`experiments/route_playbook_v1/receipts/route_genome_r6_r7_official_holdout_v3.json`。

最终收据：`experiments/route_playbook_v1/receipts/route_genome_r6_r7_final_library_v1.json`。

## 4 核 CPU 验收入口

GPT 首选运行：

```bash
bash RUN_CPU_FINAL_RECEIPT_CHECK.sh
```

该入口不安装 JAX、不重跑正式搜索，只用 Python 标准库复核固定 SHA256 和所有关键验收字段。需要功能 smoke 时才运行 `SETUP_CPU_GPT.sh` 与 `RUN_CPU_GPT_SMOKE.sh`。

## 结论边界

最终结论仅为 `BEST_OBSERVED_WITHIN_FROZEN_DOMAIN`，对手为 `NullAgent`。这不是对强手胜率、BC、RL、在线路由或 Kaggle 排行榜成绩的证明。

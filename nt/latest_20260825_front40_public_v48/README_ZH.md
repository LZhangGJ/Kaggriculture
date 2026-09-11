# Kaggriculture Rank1–40 复原与 2026-08-25 公开方案快照

这是 NT 工作区截至 2026-08-25 的可审计协作快照，覆盖：

- 天梯 Rank1–40 当前高分 submission 的逐名分析和路线复原；
- 对 FC24B 的筛选、路由搜索、留出评测和拒绝/入池依据；
- 2026-08-25 按 `Recently Run` 下载的 34 份公开 Notebook；
- Kaito V48 六条新路线的精确 JAX 迁移；
- Kaito V48 的官方 Python 1.32.7 / JAX 双座位逐步一致性；
- Kaito V48 对 FC24B 独立双座位 1,024 局正式强度验收。

## 1. 当前最重要的新增结论

Kaito V48 是本轮公开扫描中确认的新强力家族：

| 项目 | 结果 |
|---|---:|
| 独立对局 | 1,024 |
| 胜负 | 851–173 |
| 胜率 | 83.11% |
| 95% Wilson 区间 | 80.69%–85.28% |
| 平均现金差 | +6,237 |
| 完成/硬错误 | 1,024局完成，硬错误为0 |
| 官方/JAX一致性 | 双座位动作、状态、终局奖励完全一致 |

它达到融合供体门，但内部两条稀有分支明显较弱；后续应融合其强分支，不能直接假设六条路线都优于 FC24B。

## 2. 目录

- `workspace/experiments/front40_fusion_v1/`
  - 41 份逐名/总览报告，覆盖 Rank1–40；
  - 356 个根级冻结路线库、路由表和模型资产；
  - 674 个机器可读实验回执；
  - 完整配置、分析和 JAX Arena 工具；
- `workspace/experiments/hasegawa_jax_v1/`、`v2/`、`v3/`
  - Rank1 Hasegawa 复原迭代代码和验收资产；
- `workspace/public_notebooks/recent_latest_20260825_scan_v1/`
  - 本轮 34 份原始公开 Notebook、导出源码和 Kaggle 元数据；
- `workspace/experiments/public_recent_20260825/`
  - 去重、官方初筛、V48路线库、官方 Trace、严格一致性和正式强度工具/回执；
- `workspace/experiments/strategic_v5/src/strategic_v5/recent_public_20260825_gpu.py`
  - Kaito V48 的 JAX 动态控制器。

## 3. 先读这些文件

1. `workspace/experiments/front40_fusion_v1/README.md`
2. `workspace/experiments/front40_fusion_v1/reports/RANK01...RANK40...md`
3. `workspace/experiments/public_recent_20260825/reports/RECENTLY_RUN_PUBLIC_SCAN_20260825_ZH.md`
4. `workspace/experiments/public_recent_20260825/receipts/kaito_v48_jax_parity_seed1302001_n2x2_v3.json`
5. `workspace/experiments/public_recent_20260825/receipts/kaito_v48_vs_fc24b_seed1304001_n512x2_v1.json`

## 4. 快照边界

此包保留最终和中间可复验的根级路线资产，但有意排除：

- Rank1 原始可视化 Replay 缓存（约 6GB）；
- JAX 编译缓存（约 1.2GB）；
- Python `__pycache__` / `.pyc`；
- 大型前缀状态临时缓存目录。

这些内容不属于 Agent 语义，也不应进入 Git。所有 Rank1–40 的最终报告、根级路线 Bank、路由配置、模型和实验回执均已保留；Rank30 是 FC24B 自身，所以报告中明确按 identity skip 处理。

本快照是对 `nt/latest_20260823_fc24b/` 的增量层。运行 V48 对 FC24B 的 GPU 工具时，需要该旧快照已冻结的官方 JAX 核、FC24B源码和路线依赖。把本目录的 `workspace/` 覆盖到同一工作区即可。

## 5. 复现实验

在完整工作区、WSL CUDA JAX 环境中运行：

```text
python experiments/public_recent_20260825/tools/run_kaito_v48_vs_fc24b_jax.py \
  --batch 512 \
  --seed-start 1304001 \
  --output experiments/public_recent_20260825/receipts/reproduced.json
```

验收要求：GPU 后端、两个座位均719步完成、硬错误计数为0，并同时报告点胜率与 Wilson 区间。

## 6. 完整性

`MANIFEST_SHA256.tsv` 记录包内除自身外所有文件的相对路径、字节数和 SHA-256。任何协作者复制或解压后都可据此校验。


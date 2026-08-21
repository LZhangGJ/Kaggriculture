# Kaggriculture 28-Agent Arena、融合开发与 FC2A 修复快照

冻结日期：2026-08-21
官方规则口径：`kaggle-environments==1.32.7`
状态：Arena 已完成；FC2A 卡死已修复；最强融合 Agent 的全池逐对手 90% 目标仍在开发中。

## 本快照包含什么

1. **28 个严格验收 JAX Agent 的本地 Arena**
   - 28 个去重后的完整 Agent；
   - 378 组两两对战；
   - 每组 50 个事件种子并双方换座，共 100 局；
   - 合计 37,800 局、每局 719 步；
   - 完整排名、逐对手矩阵、八个 GPU 分片回执、事件库与合并结果均已保留。
2. **Fusion Champion 的开发源码和实验记录**
   - 规则主导的 K320、X562、PRT、Rank14 压力分支与反镜像能力；
   - 所有当前工具、配置、小型模型工件和不超过 2 MB 的逐实验回执；
   - 大型逐步轨迹/反事实 JSON 没有重复放入 Git，但其结论、哈希边界和复跑工具已记录。
3. **FC2A Public 卡死故障与修复版**
   - 原故障提交 `55662157`；
   - Shadow-PRT 修复提交 `55663355`；
   - 两版 `main.py`、提交包、Public 审计、已知故障种子双座位复验和详细技术说明。

## 先看哪些文件

- `docs/ALL_EXACT_JAX_ROUND_ROBIN_100_GAMES_20260820_ZH.md`：28-Agent Arena 总结果。
- `docs/FUSION_DEVELOPMENT_STATUS_20260821_ZH.md`：从 K320 基线到当前 FC2B/反事实/LGBM 实验的完整开发进度。
- `docs/FC2A_SHADOW_PRT_FIX_TECHNICAL_NOTE_20260821_ZH.md`：FC2A 修复版为什么从第 1 步计算 PRT、何时真正输出 PRT。
- `docs/FC2A_PUBLIC_STALL_ROOT_CAUSE_20260821_ZH.md`：Public 卡死的证据链。
- `workspace/experiments/fusion_champion_v1/README.md`：融合项目的最终验收纪律。

## 目录

```text
latest_20260821_arena_fusion_fc2a/
├─ docs/                         人读报告
├─ workspace/
│  ├─ gpu_sim/src/              官方 1.32.7 语义的 JAX 模拟器源码
│  ├─ experiments/
│  │  ├─ expert_business_agent_v2/  28-Agent Arena、冻结路由库和结果
│  │  ├─ fusion_champion_v1/         当前融合 Agent 源码、工具、配置与回执
│  │  ├─ strategic_v5/               28-Agent 热路径策略模块
│  │  ├─ route_playbook_v1/          事件库和轨迹执行基础
│  │  └─ kawashigi_counterfactual_ranker_v2/ 旧 Agent 动态路由依赖
│  └─ submission/
│     ├─ 55662157_*_BROKEN/      原始卡死版
│     └─ 55663355_*/             Shadow-PRT 修复版
└─ MANIFEST_SHA256.tsv           全文件大小和 SHA-256
```

## WSL + RTX 3090 复跑

从本快照的 `workspace` 目录执行。下列命令沿用主项目已配置的 WSL JAX 环境；若仓库单独克隆到其他位置，需要把 Python 路径换成自己的 CUDA JAX 环境。

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/e/ai_coding/kaggle/kaggriculture/.publish/LZhangGJ_Kaggriculture/nt/latest_20260821_arena_fusion_fc2a/workspace && export XLA_PYTHON_CLIENT_PREALLOCATE=false && /mnt/e/ai_coding/kaggle/kaggriculture/gpu_sim/.venv-wsl/bin/python experiments/expert_business_agent_v2/tools/run_all_exact_jax_round_robin.py --seeds 1 --pair-limit 1 --steps 719 --event-bank-output /tmp/nt_arena_smoke_events.npz --receipt-output /tmp/nt_arena_smoke.json --report-output /tmp/nt_arena_smoke.md'
```

完整 28-Agent 两两 100 局应优先使用八分片 batched runner；历史结果已包含在本快照中，不建议仅为确认文件完整性重复消耗约 1.7 小时。

本快照已经从自身目录完成一次 `seed=570001`、1 组、双方换座、完整 719 步 GPU 烟测：2/2 对局终局完成，`integrity_pass=true`，hand cap、market loop cap、price LUT 越界均为 0。回执见：

- `workspace/experiments/expert_business_agent_v2/receipts/package_self_contained_smoke_seed570001_n1x2_v1.json`
- `workspace/experiments/expert_business_agent_v2/reports/PACKAGE_SELF_CONTAINED_SMOKE_SEED570001_N1X2_V1.md`

## 真实性边界

- `gold_proxy_*` 是按金牌 Replay 复现并通过官方/JAX 一致性验收的本地 Agent，不是原选手源码。
- Arena 排名只代表冻结事件库和本地对手池，不等于 Public 榜分。
- FC2A 修复只证明消除了确定性的 431 帧全 PASS 卡死，不等于已经满足全池逐对手 90%。
- 当前最强融合研究仍有 `PRT`、`Rank12` 和近镜像 K320 等短板；失败实验没有包装成成功结论。

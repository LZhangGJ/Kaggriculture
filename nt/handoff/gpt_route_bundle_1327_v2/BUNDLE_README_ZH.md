# Kaggriculture 1.32.7：JAX 路线搜索与官方裁判交接包 V2

## 给只有 4 核 CPU 的 GPT

你不需要、也不应该重跑正式 JAX 搜索。正式搜索已在 RTX 3090 上完成；本包保留源码、配置、候选、逐局 trace、验收报告和机器收据，供代码审查与结果核验。

解压后先执行不需要安装依赖的收据检查：

```bash
bash RUN_CPU_FINAL_RECEIPT_CHECK.sh
```

它只使用 Python 标准库，通常几秒内完成，核对最终 6 条路线、官方复验收据和 trace 清单的 SHA256，并检查以下硬条件：

- 官方环境精确为 `kaggle-environments==1.32.7`；
- 10 条决赛候选 × 64 个未见 seed × 双座位 = 1,280 局全部完成；
- 请求 seed 与官方实际 seed 全部一致；
- 每局均为 720 帧并正常 `DONE`；
- JAX 与官方 Python 的终局现金逐局完全一致，最大误差为 0；
- 最终入库牛奶 3 条、羊毛 3 条，且各自覆盖两种生成方向。

建议先读：

1. `experiments/route_playbook_v1/reports/ROUTE_GENOME_R6_R7_FINAL_LIBRARY_ACCEPTANCE_ZH.md`
2. `experiments/route_playbook_v1/reports/ROUTE_GENOME_R6_R7_OFFICIAL_1_32_7_HOLDOUT_ACCEPTANCE_ZH.md`
3. `experiments/route_playbook_v1/reports/ROUTE_GENOME_R6_R7_FULL_PILOT_ACCEPTANCE_ZH.md`
4. `experiments/route_playbook_v1/artifacts/route_genome_v1/final_playbook_v1/accepted_routes_v1.jsonl`

若要确认程序确实能在 CPU 上启动，可再运行：

```bash
bash SETUP_CPU_GPT.sh
bash RUN_CPU_GPT_SMOKE.sh
```

这只是极小可运行性检查，不用于重现正式搜索。4 核 CPU 不适合重跑 16,000 个候选或 1,280 局官方正式复验。

## 已完成的正式流程

```text
RTX 3090：R6 牛奶 / R7 羊毛各 8,000 个候选
  -> 4 seed
  -> 16 seed
  -> 32 seed
  -> 128 seed
  -> 清仓安全修复与冻结选择
  -> 64 个未见 seed、双座位 JAX holdout
  -> 官方 Python 1.32.7 对同一动作 trace 逐局复验
  -> 最终拳法库：牛奶 3 条 + 羊毛 3 条
```

正式搜索共 16,000 个原始候选，约 21.4 分钟；各阶段固定形状、JIT cache 始终为 1。最终官方复验为 1,280 局，JAX/官方终局现金最大误差 0。

## 本机 RTX 3090 才运行的内容

正式入口：

`experiments/route_playbook_v1/tools/run_route_genome_r6_r7_full_pilot_v1.py`

正式配置：

`experiments/route_playbook_v1/configs/route_genome_r6_r7_full_pilot_v1.json`

完整搜索会写入实验目录，建议在本机项目副本而不是审查 ZIP 内执行。需要的 Python 路径为：

- `gpu_sim/src/`
- `experiments/strategic_v5/src/`
- `experiments/route_playbook_v1/src/`

包内原有 `RUN_ROUTE_SEARCH.ps1` / `RUN_CPU_ROUTE_SEARCH.sh` 是旧的、小规模通用路线探针，不等同于本次 R6/R7 正式漏斗。

## 关键目录

- `gpu_sim/src/`：官方 1.32.7 语义的 JAX 模拟器。
- `gpu_sim/reference/static_tables_v2.npz`：冻结规则表。
- `official_1_32_7/`：官方 1.32.7 源码与 wheel。
- `experiments/strategic_v5/src/`：Full-core / Full-econ 依赖。
- `experiments/route_playbook_v1/src/`：Route Genome、动态账本与路线执行代码。
- `experiments/route_playbook_v1/artifacts/route_genome_v1/full_pilot_v1/`：4→16→32→128 seed 漏斗产物。
- `.../liquidation_repair_v1/`：只在训练域做的清仓安全修复候选。
- `.../official_holdout_v3/`：未见 seed 的动作 trace 与清单。
- `.../final_playbook_v1/`：最终 6 条拳法。
- `experiments/route_playbook_v1/receipts/`：完整机器收据。
- `experiments/route_playbook_v1/reports/`：人类可读验收报告。

## 结论边界

最终标签只能是 `BEST_OBSERVED_WITHIN_FROZEN_DOMAIN`：

- 对手是 `NullAgent`；
- 证明的是路线可执行、可清仓、在冻结离线域内具有赚钱潜力；
- 不证明对公开强手胜率；
- 不包含 BC、RL、在线路由或排行榜上分结论；
- JAX 用于大规模筛选，官方 1.32.7 才是最终裁判。

校验整个交接包内容：

```bash
python3 scripts/verify_bundle.py
```

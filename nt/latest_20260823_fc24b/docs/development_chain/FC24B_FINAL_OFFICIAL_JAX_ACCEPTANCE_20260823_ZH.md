# FC24B 最终官方/JAX验收与Goal收口报告（2026-08-23）

## 1. 最终结论

本轮目标已在冻结语料与固定竞技场协议下全部通过：

- 2026-08-22 选定的 6 个最新公开强 Agent 已下载、静态解包、哈希校验并通过 Python 编译；
- 6/6 已迁入 JAX，并在官方 `kaggle-environments==1.32.7` 轨迹上通过动作、逐帧状态和终局现金严格一致性；
- FC24B 对最新 6 个与旧 28 个本地 Agent 均完成每个对手 256 局双座位测试；
- 34 个对手的单项胜率全部达到至少 90%；
- FC24B 的自包含 CPU Agent 与 JAX 实现通过官方逐步一致性、CPU 时限和跨局状态重入验收；
- 最佳配置、CPU 源码、JAX 源码、原始回执、哈希与机器可读冻结清单均已保存。

最终机器清单：`experiments/fusion_champion_v1/receipts/fc24b_frozen_acceptance_manifest_v1.json`，状态为 `PASS`，14/14 项硬检查全部为 `true`。

## 2. 原Goal逐项审计

| 原要求 | 验收证据 | 结果 |
|---|---|---|
| 拉取 Boatlee V21、Soil V26-H、Moon V92、Kaito V39、Steven E284、Salem 最新版 | 原始 notebook、静态解包源码、嵌入哈希和 `references/public_latest6_20260822/manifest.json` | PASS |
| 官方 1.32.7 验收 | 6 个方案各 2 seed × 双座位，共 24 个完整官方轨迹上下文 | PASS |
| 精确 JAX 迁移 | 6/6 动作、逐帧状态、终局现金全部 4/4 严格一致 | PASS |
| 每个新 Agent 与最新方案双座位共 256 局 | 最新 6 个各 128 seed × 双座位 | PASS |
| 新旧全部本地 Agent 单项胜率至少 90% | 34 个对手、每个 256 局，共 8,704 局 | PASS |
| 防止整体退化 | 相对 FC22：4 局败转胜、0 局胜转败；1,313 局现金增加、0 局现金下降 | PASS |
| Python/JAX 严格一致 | FC24B 标准 32 上下文 + 针对性 4 上下文，动作/状态/终局均 36/36 | PASS |
| 保存最优配置、源码、报告和可复现凭证 | 冻结配置、当前最佳指针、CPU/JAX源码、回执和最终清单 | PASS |

## 3. 最新6个公开方案的冻结与JAX迁移

### 3.1 来源完整性

冻结过程只静态读取 notebook 中的 payload，不执行 notebook 代码；每份解包后的 `main.py` 都满足：

1. notebook SHA-256 与下载快照一致；
2. 解包源码 SHA-256 与 notebook 内声明值一致；
3. Python 编译通过；
4. 原始 notebook 与解包源码都被保留。

| 方案 | Kaggle ref | 源码 SHA-256 |
|---|---|---|
| Boatlee V21 | `boatlee/v21-r1-public-state-route-portfolio` | `C6F96A8521DC9AA369B6F27E5B36B9D481E5C1688F50C8CA215C3BB53F1F9EB8` |
| Soil V26-H | `prvsiyan/kaggriculture-frontier-the-soil-remembers-rain` | `F4B8C16395382060D23F70FA65DD376FD567AB55C430D7AEE41BBE0BC117A88C` |
| Moon V92 | `prvsiyan/kaggriculture-frontier-the-moon-counts-melons` | `CA4F2C0925C38C1CACBB7756EF791F04A45F65660CE4914AD41271AD5798E09F` |
| Kaito V39 | `kaitofukami/103-126-untouched-future-top-30-v39-history-gate` | `C0298C82C2A2F9B41CD48657B63B16A9C6F07EA3EE02DC85062F807E54E8C54A` |
| Steven E284 | `stevenleehans/kaggriculture-e284-hadouken` | `B5C2E156689B41CEA5F2E1A0A1CAE2E18FD70423931BBAC0885BBA8B2142C497` |
| Salem最新版 | `salemali7/3094-score-kaggriculture` | `D39DBA50793D9777C990347443BF0C481C78ADAEA86055F6F6B0600DCFCD9F2E` |

### 3.2 官方/JAX逐步一致性

验收口径不是只比较输赢，而是逐上下文检查全部 719 个动作、每个官方帧的状态，以及终局现金。

| JAX Agent | 上下文 | 动作完全一致 | 状态完全一致 | 终局完全一致 | 后端 |
|---|---:|---:|---:|---:|---|
| Boatlee V21 | 4 | 4 | 4 | 4 | RTX 3090 GPU |
| Soil V26-H | 4 | 4 | 4 | 4 | RTX 3090 GPU |
| Moon V92 | 4 | 4 | 4 | 4 | RTX 3090 GPU |
| Kaito V39 | 4 | 4 | 4 | 4 | RTX 3090 GPU |
| Steven E284 | 4 | 4 | 4 | 4 | RTX 3090 GPU |
| Salem最新版 | 4 | 4 | 4 | 4 | RTX 3090 GPU |

JAX控制器：`experiments/strategic_v5/src/strategic_v5/latest_public6_20260822_gpu.py`。

说明：这里的“精确”是对冻结的官方轨迹语料严格一致，不宣称有限 seed 可以数学证明所有可达状态。

## 4. FC24B全Agent竞技场结果

固定协议：每个对手使用 128 个随机事件，双方换座，共 256 局；所有局都必须完成，且运行错误诊断为零。

### 4.1 最新6个公开Agent

| 对手 | 胜场/256 | 胜率 |
|---|---:|---:|
| Boatlee V21 | 248 | 96.88% |
| Soil V26-H | 235 | 91.80% |
| Moon V92 | 245 | 95.70% |
| Kaito V39 | 253 | 98.83% |
| Steven E284 | 240 | 93.75% |
| Salem最新版 | 246 | 96.09% |

最新6个最低胜率为 Soil V26-H 的 91.80%，均通过 90% 硬门。

### 4.2 旧28-Agent完整名单

- 28/28 单项胜率至少 90%；
- 最低为 Local PRT V6：231/256，90.23%；
- 其次为 Public Soil G04 与 Flex V59：均为 235/256，91.80%；
- 相对 FC22，旧名单产生 4 局败转胜、0 局胜转败。

完整逐对手结果见：`experiments/fusion_champion_v1/reports/FC24B_FULL_LOCAL_NON_REGRESSION_GATE_20260823_ZH.md`。

## 5. FC24B相对FC22的有效改动

FC24B 不改变主经营路线，只加入一次通用的“终局最后可行补收”：

1. 仅在第30天、`step >= 696`；
2. 单位站在成熟且有正产量的作物上；
3. 原动作准备移动离开；
4. 当前必须恰好是完成 `HARVEST -> 最短回仓 -> DROP` 的最后可行时刻；
5. 作物当前清算价值必须至少为随身货物价值的两倍；
6. 选择当前价值最高的候选，随后粘性执行到入库，并只合并新增产品的终局销售。

该规则不读取对手身份、Replay ID、seed、固定坐标或未来隐藏事件。

关键 PRT 败局 `seed=594122, candidate_seat=0`：

- FC22：154,468 对 154,609，落后 141；
- FC24B 在第709步补收成熟草莓；
- FC24B：154,776 对 154,609，领先 167。

初版 FC24 没有两倍价值门，曾延迟高价牛奶/羊毛出售并造成现金下降，已明确拒绝；FC24B 在全部 8,704 个固定配对局中没有任何候选现金下降。

## 6. FC24B Python、官方裁判和JAX一致性

自包含 CPU Agent：`experiments/fusion_champion_v1/artifacts/fc24b_cpu_v1/main.py`。

| 验收 | 上下文/调用数 | 结果 |
|---|---:|---|
| 标准官方1.32.7双座位逐步一致性 | 32上下文 | 动作/状态/终局 32/32 |
| PRT与K320针对性逐步一致性 | 4上下文 | 动作/状态/终局 4/4 |
| 合计严格一致性 | 36上下文 | 动作/状态/终局 36/36 |
| CPU动作Replay | 23,008次调用 | 全部一致 |
| 同一模块连续正序+倒序重入 | 64局、46,016次调用 | 全部一致，无跨局污染 |

CPU性能：

- 平均 import：0.316 秒；
- 平均动作：0.523 ms；
- P95：0.922 ms；
- P99：1.350 ms；
- 最大：5.0035 ms；
- 验收上限：1,000 ms/动作。

因此本地 Python 实现与高速 JAX 实现不是两套近似逻辑，而是在冻结官方语料上逐步一致。

## 7. 冻结资产

| 资产 | 路径 |
|---|---|
| 当前最佳配置 | `experiments/fusion_champion_v1/configs/fc24b_frozen_best_v1.json` |
| 当前最佳指针 | `experiments/fusion_champion_v1/configs/CURRENT_BEST_RULE_CONFIG.json` |
| JAX策略源码 | `experiments/fusion_champion_v1/src/fusion_champion_v1/policy_gpu.py` |
| CPU构建器 | `experiments/fusion_champion_v1/tools/build_fc24b_cpu_submission.py` |
| 自包含CPU Agent | `experiments/fusion_champion_v1/artifacts/fc24b_cpu_v1/main.py` |
| 最新6源码清单 | `references/public_latest6_20260822/manifest.json` |
| 最新6 JAX控制器 | `experiments/strategic_v5/src/strategic_v5/latest_public6_20260822_gpu.py` |
| 最新6路线库 | `experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz` |
| 最终机器清单 | `experiments/fusion_champion_v1/receipts/fc24b_frozen_acceptance_manifest_v1.json` |

关键冻结哈希：

- FC24B JAX策略：`EE9B46453B1AB673E197AFD682C7691ECDBB1596B7C9233BAE8FBE80BCD3E843`
- CPU构建器：`952728E22EBD5226BC85D020EB53CD154FB98309BEF1BD9CF5A166A2E38B9869`
- CPU Agent：`E7ABC6C215CF5A2FB2CF8FEDE83157568123DDBA0E086571ECD82A24751D9E6B`
- 冻结配置：`DE81A9F34DE7B9F64486F50A24B44AA9E314C95F84FDA7065708AD4B5E4C0ACF`

最终清单还保存了所有依赖、回执、原始轨迹和来源文件的完整 SHA-256。

## 8. 可复现命令

以下命令均从项目根目录执行。

### 8.1 重新构建CPU Agent

```powershell
E:\ai_coding\kaggle\kaggriculture\.venv\python.exe experiments\fusion_champion_v1\tools\build_fc24b_cpu_submission.py --output experiments\fusion_champion_v1\artifacts\fc24b_cpu_v1\main.py
```

### 8.2 重新生成标准官方轨迹

```powershell
E:\ai_coding\kaggle\kaggriculture\.venv\python.exe experiments\expert_business_agent_v2\tools\generate_official_stepwise_parity_traces.py --pool experiments\fusion_champion_v1\configs\fc15_official_parity_pool_v1.json --candidate experiments\fusion_champion_v1\artifacts\fc24b_cpu_v1\main.py --output experiments\fusion_champion_v1\artifacts\fc24b_official_parity_seed594001_n8x2_v1 --receipt experiments\fusion_champion_v1\receipts\fc24b_cpu_v1_official_stepwise_traces_seed594001_n8x2_v1.json --seed-start 594001 --seeds 8 --workers 16
```

### 8.3 在WSL RTX3090上重跑JAX逐步一致性

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd /mnt/e/ai_coding/kaggle/kaggriculture && gpu_sim/.venv-wsl/bin/python experiments/fusion_champion_v1/tools/accept_fc12g_jax_stepwise_parity.py --trace-receipt experiments/fusion_champion_v1/receipts/fc24b_cpu_v1_official_stepwise_traces_seed594001_n8x2_v1.json --output experiments/fusion_champion_v1/receipts/fc24b_cpu_v1_jax_stepwise_parity_seed594001_n8x2_v1.json --policy fc24b"
```

### 8.4 校验跨局重入与最终冻结清单

```powershell
E:\ai_coding\kaggle\kaggriculture\.venv\python.exe experiments\fusion_champion_v1\tools\validate_fc24b_reentry.py --agent experiments\fusion_champion_v1\artifacts\fc24b_cpu_v1\main.py --trace-receipt experiments\fusion_champion_v1\receipts\fc24b_cpu_v1_official_stepwise_traces_seed594001_n8x2_v1.json --output experiments\fusion_champion_v1\receipts\fc24b_cpu_v1_reentry_forward_reverse_v1.json
E:\ai_coding\kaggle\kaggriculture\.venv\python.exe experiments\fusion_champion_v1\tools\freeze_fc24b_acceptance.py
```

## 9. 结论边界

本报告证明的是：在冻结官方 `1.32.7`、指定随机事件库、双座位协议和当前34-Agent本地池内，FC24B满足全部硬门，并且CPU/JAX语义一致。

它不等于：

- 数学证明所有未来状态都无误；
- 保证未来新增Agent仍有90%胜率；
- 保证Public天梯分数；
- 已向Kaggle提交。

本轮没有执行新的 Kaggle Public 提交。

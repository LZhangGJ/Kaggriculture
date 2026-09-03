# 2026-09-03：NT 语义克制拳法 Round2 重测记录

## 文件位置

- 本轮工作目录：`C:\Users\DHU_Z\OneDrive\文档\ChatGPT\kaggleculture\experiments\counter_cluster_round2_semantic_v1`
- D 盘独立归档：`D:\Kaggriculture\counter_cluster_round2_semantic_v1_20260903`
- Git 仓库内目录：`research/experiments/counter_cluster_round2_semantic_v1_20260903`
- 相关 Agent 源码：`D:\Kaggriculture\route_clustering_top40_20260831\source\agents\route_clustering_switch_agent`
- 详细实验协议：`PROTOCOL.md`
- 完整结果摘要：`RESULT.md`
- 可执行分析：`round2_analysis.ipynb`
- 机器可读训练结果：`train_semantic_report.json`、`train_semantic_matrix.npz`

归档只保留本轮有效代码、锁文件和最终结果，排除 `__pycache__`、失败尝试目录 `attempts` 和重复的 `runtime.compat.lock.json`。
D 盘归档保留了本轮精确执行所用的 Windows `.pyd`；Git 副本遵循仓库的 `*.py[cod]` 忽略规则，不提交平台相关二进制，只保留完整原生源码、构建配置和 `runtime.lock.json` 中的二进制哈希。

## 实验目标

验证 NT 搜索得到的拳法在换 seed、换座位和盘面变化后，能否作为实时自适应策略继续克制历史 Replay 路线，并据此形成“同一套拳法可稳定击败”的 response cluster。

本轮特意排除了旧的固定动作带和 rank 重放。拳法保存为 day 6、9、12、18、24 的五段语义 `PlanDelta` 意图；每个检查点根据实时盘面、库存和商店重新生成可行候选、匹配稳定意图，再由 Candidate8 规划具体动作。

## 执行过程

1. 冻结 v3 G003 开局 `103928643:1`、24 条 Replay 的 train/validation/holdout 切分、seed、门槛和全部输入哈希。固定开局执行到 step 143，语义拳法从 day 6 开始。
2. 在本轮原生仿真器副本中加入显式 `PlanDelta` 接口；新对局传入空 rank 序列，只允许语义意图驱动实时规划。
3. 运行 `test_semantic_api_contract.py`，检查 Python/pybind 接口、空 rank 和语义参数。
4. 运行 `materialize_plan_sequences.py --workers 4 --force`，把以前 48 个 W12/A64 单场景搜索选择反解为语义拳法。
5. 通过 discovery 等价门：48/48 个选择的 reward、719-step 联合轨迹、family、signature、delta、动作哈希和开局前缀完全一致；去重后得到 25 套语义拳法。
6. 运行 `audit_signature_inversion.py`：79/79 个 `family + signature` 唯一反解为稳定意图。
7. 运行 `evaluate_semantic.py --split train --workers 4 --force`：25 套拳法 × 12 条训练路线 × 8 个新 seed × 双座位，共 4,800 个 treatment 对局；另运行 192 个 Opening A baseline 对局。
8. 独立审计发现 unmatched 阶段曾写入默认 delta 哈希。修正为 unmatched 哈希为空后，冻结新 runtime，并用相同 4,992 局完整重跑。新旧 31 个数组中只有两个 telemetry 哈希数组按预期变化；其余 29 个数组逐元素一致。
9. 从原始 rewards 独立复算 margins、scores、baseline、uplift、coverage 和 set-cover；运行 `round2_analysis.ipynb` 的 22 项断言。未运行 validation 或 holdout。

## 初步结果

- 所有 24,000 个 `selected_rank` 均为 `-1`，确认没有按 rank 或固定动作带重放。
- PlanDelta 阶段匹配率为 `18,083 / 24,000 = 75.3458%`；五段全部匹配的对局占 `26.0833%`。
- 在已匹配阶段中，`16,293 / 18,083 = 90.1012%` 的实时完整 delta 相对 discovery 状态发生变化，说明执行器确实在根据当前状态调整。
- 300 个拳法–对手单元中，score 门槛通过 1 个，uplift 门槛通过 0 个，正 mean-margin 门槛通过 0 个；三门同时通过为 `0/300`。
- response set-cover 选择 0 套拳法、覆盖 0 条训练路线；12 条训练路线全部是 residual，因此尚无可供前缀分类器学习的有效克制类别。
- 最接近门槛的组合对 P02 得分 `0.625`，但 Opening A baseline 为 `1.0`，uplift 为 `-0.375`，mean margin 为 `-5,866.5625`。
- 14 个单元在全部 16 局中五段均匹配，但其中最好 score 只有 `0.25`、最好 mean margin 为 `-10,641.125`。失败不能主要归因于语义匹配失败。

初步判断：实时语义执行方式成立，但以前在单一 discovery seed/座位/对手状态上找到的 48 个搜索选择明显缺乏跨随机条件稳健性。本结果不能推出“NT 无法生成克制拳法”，只能否定这批旧候选可直接泛化的假设。

## 可能的问题与边界

1. **搜索目标过拟合。** 候选来自单 seed、单座位的 discovery 搜索，本轮只重新执行和评估，没有按多 seed、双座位目标重新运行昂贵的 NT 搜索。
2. **候选池太窄。** 48 个搜索选择仅形成 25 套语义拳法，不足以说明更大的 NT 搜索空间没有可用解。
3. **中期意图可用性较弱。** day 9 和 day 12 的匹配率分别只有 `62.4167%` 和 `62.3125%`；后续可考虑更稳定的语义身份或允许受约束的近邻匹配。不过完全匹配单元同样失败，所以这不是当前首要瓶颈。
4. **覆盖范围有限。** 目前只测试 v3 G003 开局和 12 条训练路线；G032/G040 尚未进入本轮实验。
5. **对手模型有限。** 对手按历史 Replay 动作执行，并非会根据我方变招继续自适应的在线对手。
6. **尚未外推。** 因训练 coverage 为 0，按协议没有读取 validation/holdout 结果；不能报告泛化性能。
7. **归档不是搬目录即运行的发行包。** 部分 lock 和脚本仍引用原工作区 Round1 输入、D 盘 Candidate8 genome/meta-agent 源码及原始 Replay；D 盘归档足以审阅本轮代码和结果，但精确重跑仍需这些外部输入及 CPython 3.13/MinGW 运行时。

## 下一步实验门

先对 12 条 residual 路线重新运行 NT，将多 seed、双座位下的 score、相对 Opening A 的 uplift 和 reward margin 直接作为搜索目标。每产生一套新拳法就重算 response coverage，并合并能被同一拳法稳定击败的路线。只有训练集得到非空 portfolio 后，才训练基于前缀后验的分类/切换器并进入 validation。

# NT 语义克制拳法 Round2（2026-09-03）

## 一句话结论

旧 NT 候选改成实时语义 PlanDelta 后可以随盘面重新规划，但在新 seed、双座位训练中
没有任何一套稳定克制一条路线：response coverage 为 `0/300`。

## 状态

`实验失败`：训练门失败，按预注册协议停止；validation 和 holdout 均未运行。

这只否定“旧单场景候选可直接泛化”，不否定 NT 重新按稳健目标搜索后可能找到拳法。

## 问题

固定当前 v3 G003 到 step 143 后，以 day 6、9、12、18、24 的语义 PlanDelta 调用实时
Candidate8 规划器，旧的 48 个 NT 搜索选择能否在新状态下稳定克制 Replay 对手，并让
一套拳法覆盖多条路线？

预注册协议和停止门见 `PROTOCOL.md`，机器可读权威配置为 `config.json`。

## 代码位置

| 内容 | 仓库路径 |
|---|---|
| 语义选择物化 | `materialize_plan_sequences.py` |
| 语义反解审计 | `audit_signature_inversion.py` |
| 训练矩阵评估 | `evaluate_semantic.py` |
| API 合约检查 | `test_semantic_api_contract.py` |
| 分析 Notebook 生成器 | `build_round2_analysis_notebook.py` |
| 本轮原生仿真器副本 | `native/fast_kaggriculture/` |

本轮没有把失败尝试目录、Python 缓存或重复运行时副本提交 Git。

## 数据位置

- Git 内实验目录：`research/experiments/counter_cluster_round2_semantic_v1_20260903/`。
- D 盘完整归档：`D:\Kaggriculture\counter_cluster_round2_semantic_v1_20260903`。
- 原工作目录：`C:\Users\DHU_Z\OneDrive\文档\ChatGPT\kaggleculture\experiments\counter_cluster_round2_semantic_v1`。
- 输入版本和 SHA-256：`inputs.lock.json`；运行时版本和二进制哈希：`runtime.lock.json`。
- 小型结果：`train_semantic_report.json`、`train_semantic_portfolio.json`、
  `train_semantic_matrix.npz`、`validation_receipt.json`。
- 拳法定义和审计：`semantic_fists.json`、`materialization_audit.json`、
  `signature_inversion_audit.json`。

精确复跑还依赖锁文件记录的 Round1 输入、Candidate8 genome/meta-agent 源码、原始
Replay 以及 CPython 3.13/MinGW 运行时。Git 目录可审计，但不是搬目录即运行的发行包。

## 方法

1. 固定 v3 G003 `103928643:1` 至 step 143；语义拳法从 step 144 开始。
2. 将 48 个旧 W12/A64 选择反解为 day 6/9/12/18/24 的稳定语义意图；禁止新对局
   使用候选 rank 或固定动作带。
3. 通过 48/48 discovery 完整等价门后，去重得到 25 套拳法；79/79 个
   `family + signature` 能唯一反解。
4. 用 12 条训练路线、8 个新 seed、双座位评估 25 套拳法，并用 Opening A 做配对基线。
5. 覆盖要求同时满足：score ≥ 0.625、相对基线 uplift ≥ 0.125、mean reward margin > 0。
6. 训练 coverage 为空时停止，不查看 validation/holdout。

稳定意图只保存有意义的 PlanDelta 字段。容量、估值、预计成本、动作负载和 signature
在每局根据真实状态重新计算，不作为跨局身份。

## 结果

`已执行`：共 4,992 局，其中 `25 × 12 × 8 × 2 = 4,800` 个 treatment 对局，另有
192 个 Opening A baseline 对局。

| 指标 | 结果 |
|---|---:|
| 新对局 rank telemetry | 24,000 / 24,000 为 `-1` |
| PlanDelta 阶段匹配 | 18,083 / 24,000 = 75.3458% |
| 五段全部匹配的对局 | 26.0833% |
| 匹配后实时 delta 改变 | 16,293 / 18,083 = 90.1012% |
| 通过 score 门的拳法–对手单元 | 1 / 300 |
| 通过 uplift 门 | 0 / 300 |
| 通过正 mean-margin 门 | 0 / 300 |
| 同时通过三门、可用于聚类 | 0 / 300 |

14 个单元在全部 16 局中五阶段均匹配，但最好 score 仍只有 0.25，最好 mean margin 为
-10,641.125。因此主要失败不是语义匹配，而是旧候选跨随机条件不稳健。完整解释见
`RESULT.md`，可复核计算见 `round2_analysis.ipynb`。

## 复现

从本目录执行，先核对 `inputs.lock.json` 和 `runtime.lock.json` 中的外部路径及哈希：

```bash
python test_semantic_api_contract.py
python materialize_plan_sequences.py --workers 4 --force
python audit_signature_inversion.py
python evaluate_semantic.py --split train --workers 4 --force
python build_round2_analysis_notebook.py
```

上述命令会消耗大量计算并引用本机外部输入。不要在输入不齐时声称复现成功。Notebook
包含 22 项断言；结果还从原始 rewards 独立复算过 margins、scores、baseline、uplift、
coverage 和 set-cover。

## 局限

- 25 套候选来自单 discovery seed、座位和对手状态，没有用多 seed、双座位目标搜索。
- 仅测试 G003 和 12 条训练路线；G032/G040 未进入本轮。
- day 9/day 12 匹配率约 62%，但完全匹配单元同样失败，所以不是首要瓶颈。
- 对手按历史 Replay 动作执行，不是会根据我方变招继续响应的在线模型。
- validation 没有运行；holdout 保持锁定，不能报告泛化表现。

## 下一步

只对 12 条 residual 路线重新运行 NT，并直接优化多 seed、双座位下的 score、相对
Opening A uplift 和 reward margin。每得到新拳法就重算 response coverage；训练集得到
非空 portfolio 后，才训练前缀后验分类器并打开 validation。

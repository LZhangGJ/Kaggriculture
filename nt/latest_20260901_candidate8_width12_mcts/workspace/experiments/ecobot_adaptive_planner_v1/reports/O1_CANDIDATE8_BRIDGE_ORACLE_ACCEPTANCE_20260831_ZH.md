# O1 Candidate8 桥接 Oracle 验收报告

日期：2026-08-31  
规则：官方 `kaggle-environments==1.32.7`  
结论：**工程 PASS，价值门 FAIL；不允许直接进入全面 O2。**

## 1. 本实验回答什么

此前 O0 会针对每一局的真实未来，事后挑出最优 Candidate8 候选。它证明候选语言有表达上限，但线上无法知道真实未来，因此 O0 不能部署。

O1 改为：

1. 在 Day0、Day1、Day6 冻结当前公开状态；
2. 所有候选共享相同的随机未来 Bank A；
3. 只根据 Bank A 选择候选；
4. 用完全不重叠的 Bank B 盲评；
5. Bank B 和当前真实比赛的未来从不进入选择特征。

因此 O1 测量的是：**不知道本局真实未来时，只知道一组可能未来分布，Candidate8 还能保留多少稳定价值。**

## 2. 冻结设置

- 基座：R6；
- 候选：Candidate8 shortlist，单状态最多 64 条；
- 对手：冻结 G001；
- 决策日：Day0、Day1、Day6；
- 双座位；
- 每个状态 64 套共同随机未来：Bank A 32、Bank B 32；
- 两次完全独立实验，每次 8 个 prefix seed；
- 合计 96 个状态、357,760 次完整续跑；
- C++ 续跑 0 overflow，实测 446.5 / 452.1 complete continuations/s。

冻结配置见：[o1_bridge_oracle_g001_v1.json](E:/ai_coding/kaggle/kaggriculture/experiments/ecobot_adaptive_planner_v1/configs/o1_bridge_oracle_g001_v1.json)。

## 3. 三种结果必须分开看

| 名称 | 是否看真实未来 | 选择方式 | 是否可部署 |
|---|---:|---|---:|
| KEEP / R6 | 否 | 不换候选 | 是 |
| O1 scenario-mean | 否 | Bank A 平均胜率→分差→现金最优 | 只能作为乐观桥接诊断 |
| O1 robust | 否 | 候选相对 KEEP 的 Bank A 胜率退化≤5%，且分差增益 q10>0 | 是，但允许放弃换拳 |
| O0 clairvoyant | **是** | Bank B 每条真实未来分别事后选最优 | **否** |

## 4. 两次盲评结果

### 4.1 独立实验分别看

| 实验 | O1 mean 胜率增益 | O1 mean 分差增益 | O1 mean 我方现金增益 | 激活后胜率退化率 | O0 胜率增益 | O0 分差增益 |
|---|---:|---:|---:|---:|---:|---:|
| Run 1 | +1.50pp | +1,215 | +2,178 | 21.3% | +37.73pp | +35,897 |
| Run 2（全新 seeds） | +2.15pp | **-382** | +2,475 | 10.4% | +36.26pp | +35,589 |

Run 1 的 O1 mean 分差增益 95% 区间为 `[-843, +3,015]`；Run 2 为 `[-2,448, +1,480]`，均跨过 0。

### 4.2 两次等权汇总

| 方法 | score rate | 相对 KEEP | 平均分差 | 相对 KEEP | 我方现金 | 相对 KEEP |
|---|---:|---:|---:|---:|---:|---:|
| KEEP / R6 | 3.19% | — | -41,812 | — | 63,837 | — |
| O1 scenario-mean | 5.01% | **+1.82pp** | -41,395 | **+417** | 66,163 | **+2,326** |
| O1 robust | 3.19% | **0.00pp** | -41,812 | **0** | 63,837 | **0** |
| O0 clairvoyant | 40.19% | **+37.00pp** | -6,069 | **+35,743** | 71,435 | **+7,598** |

O1 mean 只保留了 O0：

- 约 **4.9%** 的胜率增益；
- 约 **1.2%** 的分差增益；
- 约 **30.6%** 的我方现金增益。

换句话说，从 O0 到 O1，失去了约 **35.17pp score rate** 和 **35,326 平均分差**。此前 Oracle 的主要优势来自知道“这局具体会发生什么”，不是候选在未知未来下普遍更优。

这里的 score rate 是冻结决策状态在独立未来 Bank 上的完整续跑胜/平/负得分率，不是 Public Leaderboard 分数。

## 5. 分日期归因

| 决策日 | O1 mean 胜率增益 | 分差增益 | 我方现金增益 | 解读 |
|---|---:|---:|---:|---|
| Day0 | 约 +4pp | **-666** | +1,853 | 能碰到更多小胜，但整体对抗分差变差；开局选择高度依赖具体未来 |
| Day1 | 约 0pp | +244 | +1,345 | 基本没有稳定对抗价值 |
| Day6 | 约 +2pp | **+1,672** | +3,781 | 三个节点中最有希望，但仍未达到 +5pp 门槛 |

保守 O1 在两次实验中均 0% 激活，全部回退 KEEP。这不是程序没生成候选，而是没有候选同时通过“跨未来低退化 + q10 分差为正”的安全门。

## 6. 验收门

预先冻结的主门：

- Bank B 胜率至少 +5pp；
- Bank B 平均分差至少 +1,500；
- bootstrap 胜率下界不为负；
- bootstrap 分差下界大于 0；
- 激活后胜率退化率不超过 5%；
- 激活后分差为负的状态不超过 20%。

结果：

- O1 robust：0 激活，+0pp、+0 分差；**FAIL**；
- O1 mean：合计仅 +1.82pp、+417 分差，且两个独立实验的分差区间均跨 0；**FAIL**。

不能因为 O1 mean 的我方现金约 +2,326 就判定成功：对抗目标是胜负和双方分差，不是只提高自己的账面现金。

## 7. 官方 1.32.7 抽样一致性

从 O1 mean 选中的候选中抽取 8 局，覆盖 Day0/1/6 和双座位：

- 8/8 候选签名从相同公开前缀精确重建；
- 8/8 动作流完整 719 步；
- 8/8 C++ 与官方 Python 终局奖励逐值相同；
- 0 end overflow。

回执：[candidate8_o1_g001_scenario_mean_official_spot8_v1.json](E:/ai_coding/kaggle/kaggriculture/experiments/ecobot_adaptive_planner_v1/receipts/candidate8_o1_g001_scenario_mean_official_spot8_v1.json)。

边界：官方 Python 无法在中局替换成反事实 RNG 后缀，因此官方复验确认的是“候选重建和完整动作语义”，Bank A/B 的反事实价值仍由已冻结的 C++ 引擎测量。

## 8. 最终判断与 O2 门控

### 结论

**不偷看真实未来后，Candidate8 仍有少量价值，但远不足以证明可以依靠一个无条件多未来平均选择器稳定击败 G001。**

最重要的现象是：O1 更容易提高我方现金，却不能稳定提高双方分差。这说明当前候选的收益高度依赖：

- 对手之后的真实出售和扩产；
- 市场未来库存与价格路径；
- 该候选与后续阶段候选的组合；
- 当前可见偏差是否真的对应某类未来。

### O2 决定

**NO-GO：不得直接用当前 O1 标签训练一个全日期、全候选的 O2 选择器。** 标签不够稳定，直接学习大概率只会学到“增加我方现金”，而不是“稳定赢”。

只允许先做一个缩小版 O1.1：

1. 聚焦 Day6 或真实偏差触发点，不在 Day0 无条件换拳；
2. 输入当前可见的对手产能、预计上市量、市场库存、价格和我方计划偏差；
3. 目标使用配对胜率与双方分差，不以我方现金单独做主标签；
4. 允许模型 abstain / KEEP；
5. 在多对手独立 Bank 上重新达到 `+5pp / +1,500` 后，才进入 O2。

这不是说 Candidate8 没有上限；O0 已证明它有很高的事后表达上限。失败的是：**当前状态信息和无条件未来平均，无法可靠判断哪条高上限路线适合眼前这一局。**

## 9. 复现资产

- 一键脚本：[run_candidate8_o1_bridge_g001_v1.ps1](E:/ai_coding/kaggle/kaggriculture/experiments/ecobot_adaptive_planner_v1/tools/run_candidate8_o1_bridge_g001_v1.ps1)
- O1 评估器：[audit_candidate8_o1_bridge.py](E:/ai_coding/kaggle/kaggriculture/experiments/ecobot_adaptive_planner_v1/tools/audit_candidate8_o1_bridge.py)
- C++ trace 导出：[export_candidate8_o1_traces.py](E:/ai_coding/kaggle/kaggriculture/experiments/ecobot_adaptive_planner_v1/tools/export_candidate8_o1_traces.py)
- 官方重放：[verify_candidate8_o1_trace_bundle_official.py](E:/ai_coding/kaggle/kaggriculture/experiments/ecobot_adaptive_planner_v1/tools/verify_candidate8_o1_trace_bundle_official.py)
- Run 1 盲评：[candidate8_o1_bridge_g001_n8x2_days016_future64_eval_v1.json](E:/ai_coding/kaggle/kaggriculture/experiments/ecobot_adaptive_planner_v1/receipts/candidate8_o1_bridge_g001_n8x2_days016_future64_eval_v1.json)
- Run 2 盲评：[candidate8_o1_bridge_g001_n8x2_days016_future64_repro_eval_v1.json](E:/ai_coding/kaggle/kaggriculture/experiments/ecobot_adaptive_planner_v1/receipts/candidate8_o1_bridge_g001_n8x2_days016_future64_repro_eval_v1.json)


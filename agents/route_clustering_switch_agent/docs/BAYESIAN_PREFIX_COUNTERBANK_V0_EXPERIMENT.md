# BAYESIAN-PREFIX-COUNTERBANK-v0 实验规格

## 1. 实验目的

验证以下完整因果链是否成立：

```text
前几天公开轨迹
  -> 可校准的对手类别后验
  -> 少量类别反制路线
  -> 在不知道未来结果时优于全局路线混合
```

本实验不是为了证明分类准确率高，也不是为了再寻找一条全局冠军路线。最终判断对象是
完整 agent 的封闭对局收益和不退化表现。

## 2. 假设

### H1：存在可利用的前缀信息

第 2/4/6/8/12 天的公开轨迹对后续“需要哪种反制”有预测力，而且不是团队名、提交 ID
或市场 seed 泄漏造成的。

### H2：反制类别比行为类别更可压缩

当前 200+ 行为类别中，有多个类别可以共享同一组反制路线；共享后不会显著增加类别内
最坏 regret。

### H3：软后验优于硬分类

在早期路线不可辨认时，按后验概率混合候选的策略收益不低于硬选择一个类别，并优于
当前全局混合。

### H4：每类少量路线能够覆盖

每个反制类别最多三条共享路线，可以覆盖类别内大多数路线，并在困难路线上的胜率下界
不低于当前全局策略。

## 3. 冻结范围

实验开始前保存以下快照及 SHA-256：

- replay 清单、文件时间和 episode 去重结果；
- RouteGenome 聚合库和路线族映射；
- 当前生产 agent、7 条活跃路线和 selector；
- 候选路线库；
- C++ 引擎二进制和源码提交；
- 对手池入口文件。

实验过程中新 replay 可以进入下一轮增量池，但不能回写本轮封闭测试。

## 4. 数据切分

禁止按单局随机切分。至少使用四层隔离：

| 切分 | 用途 | 隔离要求 |
|---|---|---|
| train | 估计似然、反制类别和 GA 适应度 | 可见路线与 seed |
| dev | 选择 `n`、阈值、类别数和风险系数 | 新 seed，尽量分提交 |
| route holdout | 测试已知风格的新路线/提交 | submission/team 不与 train 重叠 |
| temporal holdout | 测试新到 replay 与分布漂移 | 时间晚于冻结点，全程封闭 |

同一 episode 的双座位、副本或压缩副本必须在同一切分。一个 submission 的重复 replay
不能跨 train 和 holdout。所有评测同时报告按 seed、对手、团队和座位的分层结果。

## 5. 对局与前缀样本生成

### 5.1 真实 replay

用于：

- 提取真实执行前缀；
- 校准路线频率和 OOD 阈值；
- 验证市场干扰、动作失败和非理想执行是否被覆盖。

### 5.2 C++ 模拟

用于：

- 让同一对手路线运行在多个市场 seed、双座位和不同我方共同开局下；
- 构造候选路线 × 对手路线 payoff 矩阵；
- 生成反事实标签和 GA 适应度；
- 完整运行 719 回合，检查执行错误。

前缀识别数据和策略评测数据使用不同 seed。不得用评测胜负反向修改同一轮类别和先验。

## 6. 处理规模：先小后大

### Stage A：可辨认性预检

- 对手：从当前池分层选 32～64 条路线；
- 候选：当前 7 条生产路线，加少量代表 donor；
- checkpoint：第 2/4/6/8/12 天；
- 目标：计算后验校准和 payoff signature，不运行 GA。

若 oracle 知道类别后相对全局混合没有增益，说明分类没有策略价值，停止本类别方向。

### Stage B：固定候选路由

- 扩展到 128～256 条对手路线；
- 只使用现有候选，比较硬分类、贝叶斯后验和 oracle；
- 目标：证明贝叶斯路由本身能利用前缀，不把生成和路由问题混在一起。

### Stage C：类别 counterbank GA

- 只选择 Stage B 中存在明显 oracle gap 的类别；
- 每类生成候选，但允许最终路线跨类别共享；
- 每类最多保留三条引用，控制全局活跃候选预算。

### Stage D：全池冻结验证

- 当前最强 agent；
- 现有 652 路线池及更新后的冻结对手池；
- route holdout 和 temporal holdout；
- 新 seed、双座位、完整 719 回合。

## 7. 反制类别构建

1. 对每条对手路线计算现有候选面板的 payoff signature。
2. 对胜率和 margin 使用分层平滑，保存标准误或 bootstrap 下界。
3. 先按 payoff signature 形成反制类别。
4. 对每个 checkpoint 检查这些类别是否能由公开前缀区分。
5. 若两个反制类别前期不可区分，不强制合并，也不硬分类；在线保留混合后验。
6. 若多个行为类别共享相同反制集合，合并其反制类别引用。

类别数通过 dev 集上的策略 regret、后验校准和路线预算共同选择，不使用封闭测试调参。

## 8. v0 贝叶斯模型

实现两个低复杂度基线：

### M1：收缩对角密度

- 连续特征：标准化后的 diagonal Student-t 或 Gaussian；
- 二值/计数事件：经过稳定变换后进入同一向量；
- 每个 checkpoint 独立拟合类条件参数；
- 缺失特征通过 mask 排除，不填成真实的 0。

### M2：kNN/kernel likelihood

- 对每个类别保存有限 medoid/prototype；
- 由到原型的距离估计似然；
- 通过 dev 集温度缩放校准后验。

选择标准是 dev 策略收益，其次才是 NLL/Brier。若 M1 与 M2 都不优于先验，停止，不
进入神经网络实验。

## 9. 对照组

| 组 | 描述 | 作用 |
|---|---|---|
| A | 当前生产 selector / 全局稳健混合 | 必须击败的基线 |
| B | 前 `n` 天硬分类后选择类别路线 | 验证软后验是否必要 |
| C | 贝叶斯后验加权 counterbank | 主实验 |
| D | 已知完整真实路线的 oracle | 可达到上界 |
| E | 随机打乱类别标签 | 排除数据泄漏和偶然相关 |
| F | 只用市场/seed 上下文 | 排除模型靠环境识别 |

线上特征中不得包含 team、submission、opponent ID、seed、最终奖励和未来轨迹。

## 10. 候选选择和运行时决策

对每类先用贪心覆盖从现有候选中选 `K <= 3`，再进行 GA。集合目标以类别内每个对手
被集合中最佳路线覆盖的结果计算，而不是将三条路线平均：

```text
coverage(S,C) = aggregate_r max_{a in S} U_LCB(a,r)
```

运行时对每条满足 continuation contract 的候选计算：

```text
expected_lcb = sum_c posterior[c] * U_LCB(candidate,c)
gain = expected_lcb - expected_lcb(KEEP) - switch_cost
```

只有 `gain` 超过 dev 预注册门槛才切换。v0 最多切换一次；每天可以更新信念，但切换
后保持锁定，以便与当前生产执行层公平对比。

## 11. 指标

### 11.1 信念质量

- posterior NLL；
- Brier score；
- calibration/ECE；
- top-k posterior mass；
- 后验熵随天数的变化；
- OOD AUROC 或已知/未知分离率。

分类准确率不是单独的验收指标。

### 11.2 策略质量

- 原始胜率、得分率和平均 margin；
- 按对手的最低胜率；
- 按 seed 配对的收益下界；
- 相对 A 的增益；
- 相对 D 的 oracle regret；
- 座位差异和执行完成率；
- KEEP、切换和 OOD 回退比例。

### 11.3 复杂度

- 观察原型数、反制类别数；
- 共享路线总数与线上实际引用数；
- 模型文件和动作库大小；
- 每回合推理耗时；
- 新 replay 的增量更新时间。

## 12. Go / No-Go 门槛

以下门槛在看封闭测试前写入 run manifest：

### 进入 Stage C

- D 相对 A 存在稳定正 oracle gap；
- C 在 dev 上优于 A，且不弱于 B；
- 标签打乱和只看环境的对照不能复制 C 的增益；
- 后验校准没有明显崩坏。

### 进入生产集成候选

- C 在 route/temporal holdout 的配对收益下界为正；
- 最低单对手表现不发生超过预注册容忍度的退化；
- OOD 子集不弱于当前回退策略；
- 100% 完成，轨迹长度和动作合法性通过；
- 总体胜率至少 90%，单对手至少 80%；
- 活跃路线和推理成本满足 manifest 中的部署预算。

若 C 只提高分类准确率而不提高对局收益，实验判失败。若 D 没有明显优于 A，说明该
前缀阶段没有足够可利用的专用反制空间，应推迟 checkpoint 或改进候选生成。

## 13. 预期产物

建议输出根目录：

```text
artifacts/bayesian_prefix_counterbank_v0/<run-id>/
```

每次 run 至少保存：

- `RUN_MANIFEST.json`：代码、引擎、数据、seed、门槛和哈希；
- `prefix_features.npz` 与 `prefix_index.jsonl.gz`；
- `payoff_matrix.npz` 与 `payoff_index.json`；
- `response_class_map.json`；
- `belief_model.npz` 与 `calibration.json`；
- `counterbank.json`；
- `dev_report.json`；
- `sealed_holdout_report.json`；
- `REPORT.md`。

字段定义见 [数据/接口契约](OPPONENT_PREFIX_BELIEF_DATA_CONTRACT_V0.md)。

## 14. 计划实现文件

以下为计划文件名，不表示当前已经存在：

- `scripts/extract_opponent_prefix_features.py`
- `scripts/build_route_payoff_signatures.py`
- `scripts/cluster_response_equivalent_routes.py`
- `scripts/train_bayesian_prefix_router.py`
- `scripts/evolve_cluster_counterbanks.py`
- `scripts/evaluate_bayesian_prefix_counterbank.py`
- `src/meta_agent/src/opponent_belief.py`
- `src/meta_agent/src/counterbank_policy.py`

实现顺序必须保持 Stage A → B → C → D；不得先运行大规模 GA 再寻找能够解释其结果的
类别。
